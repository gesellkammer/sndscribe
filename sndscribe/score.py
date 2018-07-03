# -*- coding: utf-8 -*-
from __future__ import division, print_function
# NB: in lilypond, edit auto-beam.scm to add
# ((end * * 4 4) . ,(ly:make-moment 1 4))
# ((end * * 4 4) . ,(ly:make-moment 3 4))
# the footer "music engraving..." resides in titling-init.ly, tagline

from __future__ import absolute_import
import os
from emlib import xmlprinter
from emlib.iterlib import pairwise
from emlib.midi.generalmidi import get_midi_program
from . import pack
from . import dynamics
from .tools import *
from .config import *
from .definitions import *
from .note import *
from .reduction import (reduce_breakpoints, simplify_notes, partial_simplify)
from .scorefuncs import *
from .conversions import *
from .voice import Voice
from .staff import Staff
from .envir import logger
from . import typehints as t


class Score(object):
    def __init__(self,
                 timesig=(4, 4),      # type: t.Tup[int, int]
                 tempo=60,            # type: t.Number
                 dyncurve=None,       # type: t.U[str, DynamicsCurve, None]
                 transient_mask=None,
                 config=None,         # type: ConfigDict
                 **kws):
        """
        * timesig: (num, den)
        * dyncurve: a dynamics.DynamicsCurve, or a dict defining one (see config.defaults)
        * transient_mark:
            - If None, transients are not taken into account.
            - Else, a bpf is expected which returns either
              0 or 1 for in time t
        * config: a configuration file as defined in config.defaultconfig or as a result
                  of calling config.makeconfig
        * any other keywords are used to override the config

        NB: Use addstaff to append Staffs to the Score
        """
        pagelayout = config['pagelayout']
        staffsize = config['staffsize']
        pagesize = config['pagesize']
        assert istimesig(timesig)
        assert isinstance(tempo, int) and 30 <= tempo < 200
        assert pagelayout in ('portrait', 'landscape')
        assert isinstance(staffsize, int) and staffsize > 0
        config = config if config is not None else getdefaultconfig()
        if kws:
            config = config.copy()
            config.update(kws)
        self.timesig = timesig
        self.tempo = tempo
        self.staffsize = staffsize
        self.pagesize = pagesize
        self.pagelayout = pagelayout
        self.transient_mask = transient_mask
        self.renderconfig = RenderConfig(tempo=tempo, timesig=timesig, config=config)
        if dyncurve is not None:
            assert dyncurve is self.renderconfig.dyncurve
        self.dyncurve = self.renderconfig.dyncurve 
        self.staffs = []
        self.weighter = pack.new_weighter(config)

    def addstaff(self, staff):
        """
        :type staff: Staff
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
                  possible_divs=self.renderconfig['divisions'])
        self.addstaff(s)

    def render(self):
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
        logger.info("\n================ Rendering Score ================\n")
        self.staffs.sort(key=lambda x: x.meanpitch(), reverse=True)
        # check if we have to fill the notes with their transient value
        if self.transient_mask is not None:
            transient_mask = self.transient_mask
            for staff in self.staffs:
                for note in staff.unrendered_notes():
                    t = note.start
                    note.transient = transient_mask(t)

        for i, staff in enumerate(self.staffs):
            logger.info("Rendering staff %d/%d" % (i+1, len(self.staffs)))
            staff.render(renderconfig=self.renderconfig)
        logger.info('rendering done, checking data integrity...')
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
    
    def verify_render(self):
        # dyncurve = self.renderconfig.dyncurve
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
                                    logger.debug("bad ties found, fixing")
                                    lastnote.tied = True
                            lastnote = note
        return True

    def dump(self):
        staff_strings = ["================= Staff %d ==============\n" %
                         (i+1) + staff.dump() for i, staff in enumerate(self.staffs)]
        return "\n".join(staff_strings)

    def write(self, outfile):
        """
        Write the score to outfile.

        Possible formats: 
            * musicXML
            * pdf

        The format is deducted from the extension of outfile
        """
        ext = os.path.splitext(outfile.lower())[1]
        if ext == ".xml":
            return self.toxml(outfile)
        elif ext == ".pdf":
            xml = self.toxml(outfile)
            pdf = musicxml2pdf(xml)
            return pdf
        else:
            raise ValueError("Format not supported")

    def toxml(self, outfile):
        """
        Writes the already rendered score to a musicXML file.

        Returns the path of the outfile (can differ from the outfile given)

        To output the score as a pdf, call tools.musicxml2pdf
        """
        base, ext = os.path.splitext(outfile)
        outfile = base + '.xml'
        logger.info("writing musicXML: " + outfile)
        
        out = open(outfile, 'w')
        _ = xmlprinter.xmlprinter(out)  # parser

        def get_page_size(pagesize, pagelayout):
            page_height, page_width = {
                'a3': (420, 297),    # in millimeters
                'a4': (297, 210)
            }[pagesize.lower()]
            assert pagelayout in ('landscape', 'portrait')
            if pagelayout == 'landscape':
                page_width, page_height = page_height, page_width
            return page_height, page_width
        
        page_height, page_width = get_page_size(self.pagesize, self.pagelayout)
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
        return outfile
    
    def change_note_size_in_function_of_dynamic(self):
        dyncurve = self.renderconfig.dyncurve
        assert isinstance(dyncurve, dynamics.DynamicsCurve)
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
                # totalweight += note.amp * note.dur
                # totalweight += calculate_note_weight(note)
                # totalweight += noteweight(note.pitch, amp2db(note.amp), note.dur)
                # totalweight += self.weighter.noteweight(note.pitch, amp2db(note.amp), note.dur)
                totalweight += self.weighter.weight(m2f(note.pitch), amp2db(note.amp), note.dur)
        return totalweight/self.end

    @property
    def end(self):
        return max(staff.end for staff in self.staffs)
