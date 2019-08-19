# -*- coding: utf-8 -*-
# NB: in lilypond, edit auto-beam.scm to add
# ((end * * 4 4) . ,(ly:make-moment 1 4))
# ((end * * 4 4) . ,(ly:make-moment 3 4))
# the footer "music engraving..." resides in titling-init.ly, tagline

import os
import tempfile
from emlib import xmlprinter
from emlib import lib
from emlib.iterlib import pairwise
from emlib.midi.generalmidi import get_midi_program
from . import pack
from . import tools
from .config import *
from .definitions import *
from .note import *
from .scorefuncs import *
from .conversions import *
from .voice import Voice
from .staff import Staff
from .envir import logger
from . import typehints as t


class Score(object):
    def __init__(self, config: RenderConfig=None, title=None):
        """

        """
        if isinstance(config, ConfigDict):
            raise TypeError("config should be a RenderConfig, call RenderConfig(config, ...)")
        assert isinstance(config, RenderConfig)

        pagelayout = config['pagelayout']
        staffsize = config['staffsize']
        pagesize = config['pagesize']
        timesig = config.timesig
        tempo = config.tempo

        assert istimesig(timesig)
        assert isinstance(tempo, (int, float, Fraction)) and 30 <= tempo < 200, f"tempo: {tempo} ({type(tempo)})"
        assert pagelayout in ('portrait', 'landscape')
        assert isinstance(staffsize, int) and staffsize > 0

        self.timesig = timesig
        self.tempo = tempo
        self.staffsize = staffsize
        self.pagesize = pagesize
        self.pagelayout = pagelayout
        self.transient_mask = None
        self.renderconfig = config
        self.dyncurve = self.renderconfig.dyncurve
        self.staffs = []
        self.weighter = pack.new_weighter(config)

        self.title = title

    def addstaff(self, staff: Staff) -> None:
        """
        Add a Staff to this Score
        """
        assert isinstance(staff, Staff)
        self.staffs.append(staff)

    def addvoice(self, voice:Voice, name:str=None, clef:str=None) -> None:
        """
        Create a staff out of a given voice and append it to this score
        """
        s = Staff(voice,
                  timesig=self.timesig,
                  tempo=self.tempo,
                  possible_divs=self.renderconfig['divisions'],
                  name=name,
                  clef=clef
                  )
        self.addstaff(s)

    def render(self) -> None:
        """
        Render the voices into staffs->measures->pulses

        pitch_resolution defines the pitch resolution in fractions of
        a semitone:
            1: chromatic
            0.5: 1/4 tones
            0.25: 1/8 tones

        notes having the same resulting pitch will be tied together
        NB: lilypond does output only until 1/4 tones, so use 1/8 tones
        if you want to "know" when two notes, which in the score look the same,
        are 1/8 tone apart (although you will not know which is higher!)
        maybe someday lilypond will be able to represent 1/8 tones...
        They are output correctly in the MusicXML file, though.
        """
        logger.info("\n~~~~~~~~ Rendering Score ~~~~~~~\n")
        self.staffs.sort(key=lambda x: x.meanpitch(), reverse=True)
        # check if we have to fill the notes with their transient value
        if self.transient_mask is not None:
            transient_mask = self.transient_mask
            for staff in self.staffs:
                for note in staff.unrendered_notes():
                    note.transient = transient_mask(note.start)

        for i, staff in enumerate(self.staffs):
            logger.debug("Rendering staff %d/%d" % (i+1, len(self.staffs)))
            staff.render(renderconfig=self.renderconfig)
        logger.debug('rendering done, checking data integrity...')
        if self.verify_render():
            logger.info('\nVerify ok')
        else:
            logger.info('\n**found errors**')
        if self.renderconfig['notesize_follows_dynamic']:
            self.change_note_size_in_function_of_dynamic()
        if self.renderconfig['show_transients']:
            self.generate_articulations_from_transient_values()

    def iternotes(self):
        """returns an iterator over all the notes in this score"""
        if not self.rendered:
            for staff in self.staffs:
                for voice in staff.voices:
                    for note in voice.notes:
                        yield note
        else:
            for staff in self.staffs:
                for voice in staff.voices:
                    for measure in voice.measures:
                        for pulse in measure.pulses:
                            for note in pulse.notes:
                                yield note

    @property
    def rendered(self):
        return all(staff.rendered for staff in self.staffs)
    
    def verify_render(self) -> bool:
        for staff in self.staffs:
            for voice in staff.voices:
                lastnote = Note(10, start=-1, dur=1, color="verify")
                for measure in voice.measures:
                    for pulse in measure.pulses:
                        assert all(n0.end == n1.start for n0, n1 in pairwise(pulse.notes))
                        # assert not hasholes(pulse.notes)
                        # assert not hasoverlap(pulse.notes)
                        assert almosteq(sum(ev.dur for ev in pulse), pulse.pulse_dur), \
                            "Notes in pulse should sum to the pulsedur: %s" % pulse.notes
                        div = pulse.subdivision
                        possibledurs = {R(i+1, div) for i in range(div)}
                        assert all(ev.dur in possibledurs for ev in pulse), \
                            "The durations in a pulse should fit in the subdivision (%d)" \
                            "Durs: %s  Poss.Durs: %s" % (
                                div, [n.dur for n in pulse.notes], possibledurs)
                        if div > 1:
                            assert not all(ev.isrest() for ev in pulse), \
                                "div:{0} notes:{1}".format(div, pulse.notes)
                        for note in pulse.notes:
                            if note.isrest() or lastnote.isrest():
                                continue
                            # Note | Note
                            if almosteq(note.pitch, lastnote.pitch):
                                if not lastnote.tied:
                                    lastnote.tied = True
                            lastnote = note
        return True

    def dump(self):
        staff_strings = [f"~~~~~~~~~~~~~~~~ Staff {i+1} ~~~~~~~~~~~~~~~~\n" + staff.dump()
                         for i, staff in enumerate(self.staffs)]
        return "\n".join(staff_strings)

    def write(self, outfile: str) -> None:
        """
        Write the score to outfile

        Possible formats: 
            * xml (musicXML)
            * pdf (using the lilypond backend)
            * ly (lilypond)

        The format is deducted from the extension of outfile
        """
        outfile = lib.normalize_path(outfile)
        ext = os.path.splitext(outfile)[1].lower()
        if ext == ".xml":
            self._writexml(outfile)
        elif ext == ".pdf":
            return self.writepdf(outfile=outfile, method='lilypond')
        elif ext == ".ly":
            from . import renderlily
            renderlily.score_to_lily(self, outfile=outfile)
        else:
            raise ValueError("Format not supported")

    def writepdf(self, outfile, method='lilypond', openpdf=False) -> None:
        """

        Methods:
            * lilypond: a lilypond file is generated and lilypond is
              called to produce the pdf file (lilypond needs to be installed)
            * musicxml: a musicxml file is generated, musescore is called
              to produce the pdf file (musescore needs to be installed)

        Args:
            outfile: the generated pdf file
            method: possible methods are 'lilypond' and 'musicxml'
            openpdf: if True, open the resulting pdf file in the standard application

        """
        outfile = lib.normalize_path(outfile)
        if method == 'lilypond':
            lyfile = tempfile.mktemp(suffix='.ly')
            self.write(lyfile)
            lilytools.lily2pdf(lyfile, outfile)
        elif method == 'musicxml':
            xmlfile = tempfile.mktemp(suffix='.xml')
            self.write(xmlfile)
            tools.musicxml2pdf(xmlfile, outfile, backend='musescore')
        else:
            raise KeyError(f"method should be 'lilypond' or 'musicxml', got {method}")

        if openpdf:
             lib.open_with_standard_app(outfile)

    def show(self):
        self.writepdf(tempfile.mktemp(suffix=".pdf"), openpdf=True, method='lilypond')

    def _writexml(self, outfile: str) -> None:
        """
        Writes the already rendered score to a musicXML file.

        To output the score as a pdf, see .writepdf
        """
        outfile = lib.normalize_path(outfile)
        base, ext = os.path.splitext(outfile)
        assert ext == '.xml'
        logger.debug(f"toxml: writing musicXML to file {outfile}")
        
        out = open(outfile, 'w')
        _ = xmlprinter.xmlprinter(out)  # parser

        page_height, page_width = lib.page_dinsize_to_mm(self.pagesize, self.pagelayout)

        unit_converter = LayoutUnitConverter.from_staffsize(self.staffsize)

        _.startDocument()
        out.write('<!DOCTYPE score-partwise PUBLIC\n')
        out.write('   "-//Recordare//DTD MusicXML 1.1 Partwise//EN"\n')
        out.write('   "http://www.musicxml.org/dtds/partwise.dtd">')
        
        T, T1 = _.tag, _.tag1
        with T('score-partwise', version='2.0'):
            with T('defaults'):
                with T('scaling'):
                    T1('millimeters', "%.2f" % unit_converter.millimeters)
                    T1('tenths', unit_converter.tenths)
                with T('page-layout'):
                    T1('page-height', unit_converter.to_tenths(page_height))
                    T1('page-width', unit_converter.to_tenths(page_width))
                # TODO: include margins?
            with T('part-list'):
                # each staff has an id
                for i, staff in enumerate(self.staffs):
                    part_number = i + 1
                    part_id = "P%d" % part_number
                    name = staff.name or str(i+1)
                    with T('score-part', id=part_id):  # <score-part id="P1">
                        T1('part-name', name)
                        midi_instrument_name = self.renderconfig.get('midi_global_instrument') or staff.midi_instrument
                        if midi_instrument_name is not None:
                            midiprog = get_midi_program(midi_instrument_name)
                            midi_instrument_name = midi_instrument_name.lower()
                            id2 = part_id + ("-I%d" % part_number)
                            with T('score-instrument', id=id2):
                                T1('instrument-name', midi_instrument_name)
                            with T('midi-instrument', id=id2):
                                T1('midi-channel', 1)
                                T1('midi-program', midiprog)
            # ------ Staffs ------
            numstaffs = len(self.staffs)
            for i, staff in enumerate(self.staffs):
                with T('part', id="P%d" % (i+1)):
                    logger.debug('parsing staff {numstaff}/{numstaffs}'.format(numstaff=i, numstaffs=numstaffs))
                    # Each staff renders itself as XML
                    staff.toxml(_)
        _.endDocument()

    def change_note_size_in_function_of_dynamic(self):
        dyncurve = self.renderconfig.dyncurve
        for note in self.iternotes():
            if not note.isrest():
                dyn = dyncurve.amp2dyn(note.amp)
                note.size = dyn2lilysize(dyn)
    
    def generate_articulations_from_transient_values(self):
        for note in self.iternotes():
            if not note.isrest():
                tr = note.transient
                articulation = transient2articulation(tr)
                if articulation:
                    note.add_articulation(articulation)

    def density(self):
        return float(sum(staff.density() for staff in self.staffs)/len(self.staffs))

    def meanweight(self):
        totalweight = 0
        for note in self.iternotes():
            if not note.isrest():
                totalweight += self.weighter.weight(m2f(note.pitch), amp2db(note.amp), note.dur)
        return totalweight/self.end

    @property
    def end(self):
        return max(staff.end for staff in self.staffs)
