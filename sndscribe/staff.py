from __future__ import absolute_import
from .definitions import (DEFAULT_POSSIBLE_DIVISIONS, MUSICXML_DIVISIONS,
                          REGULAR_NOTETYPES)
from .scorefuncs import infer_clef, get_notated_duration
from .voice import Voice
from .tools import R, timesig2pulsedur, previous_power_of_two
from .conversions import (dyn2fontsize, dur2xml, pitch2xml,
                          micro_to_xmlaccidental, bw_to_xmlnotehead)
from .config import RenderConfig
from .typehints import Number, Opt


class Staff(object):
    def __init__(self, voices, name=None, clef=None, timesig=(4, 4),
                 tempo=60, possible_divs=DEFAULT_POSSIBLE_DIVISIONS,
                 size=18, midi_instrument=None):
        if isinstance(voices, Voice):
            voices = [voices]
        assert all(isinstance(voice, Voice) for voice in voices)
        notes = voices[0].notes
        assert notes, "no notes in this Staff!"
        assert any(not note.isrest() for note in notes), "only rests in this Staff!"
        clef = clef or infer_clef(notes)
        assert clef in ('F', 'F8', 'G', 'G8')
        assert isinstance(timesig, tuple) and len(timesig) == 2 and all(isinstance(x, int) for x in timesig)
        assert isinstance(size, int) and size > 0
        self.voices = voices
        self.measures = []
        self.name = name
        self.timesig = timesig
        self.tempo = tempo
        self.possible_divs = [R(division) for division in possible_divs]
        self.rendered = False
        self.size = size
        self.midi_instrument = midi_instrument
        self.clef = clef
        self.renderconfig = None  # type: Opt[RenderConfig]
    
    @property
    def end(self):
        return max((voice.end for voice in self.voices))

    def unrendered_notes(self):
        for voice in self.voices:
            for note in voice.unrendered_notes():
                yield note

    def render(self, renderconfig:RenderConfig, end:Number=None):
        """
        Staff.render. Called by Score. Calls each voice to render

        """
        if end is None:
            end = self.end
        self.renderconfig = renderconfig
        for voice in self.voices:
            voice.render(renderconfig=renderconfig, end=end)
        self.rendered = True

    def meanpitch(self):
        voices = self.voices
        pitches = [voice.meanpitch() for voice in voices
                   if voice.meanpitch() is not None]
        if not pitches:
            return None
        return sum(pitches)/len(pitches)

    def density(self):
        return sum(voice.density() for voice in self.voices)/len(self.voices)
    
    def dump(self):
        lines = []
        dyncurve = self.renderconfig.dyncurve
        for i, voice in enumerate(self.voices):
            lines.append("Voice %d    ---------------------------" % i)
            for m, measure in enumerate(voice.measures):
                lines.append("\tMeasure %d" % m)
                for p, pulse in enumerate(measure.pulses):
                    lines.append("\t  Pulse %d" % p)
                    lines.append('\t    Offset: %.3f  Div: %d' %
                                 (pulse.offset, pulse.subdivision))
                    for note in pulse.notes:
                        dyn = "" if note.isrest() else dyncurve.amp2dyn(note.amp)
                        lines.append("\t      {note:<40} {dyn}".format(
                            note=note, dyn=dyn))
        return "\n".join(lines)
        
    def toxml(self, parser):
        include_dynamics = self.renderconfig['show_dynamics']
        include_sizes = self.renderconfig['notesize_follows_dynamic']
        use_css_sizes = self.renderconfig['use_css_sizes']
        """genera XML code para el propio staff"""
        dyncurve = self.renderconfig.dyncurve
        num_measures = len(self.voices[0].measures)
        pulse_dur = timesig2pulsedur(self.timesig, self.tempo)
        last_dynamic = current_dynamic = ''
        close_tie = [False for _ in range(len(self.voices))]
        clef_dict = {'G':'2', 'F':'4', 'C':'3'}
        _, T, T1 = parser, parser.tag, parser.tag1
        for imeasure in range(num_measures):
            # encabezado del cada measure
            with T('measure', number=imeasure+1):
                # el primer measure define los atributos
                if imeasure == 0:
                    with T('attributes'):
                        # with T('tempo'):
                        #     T1('denominator', self.timesig[1])
                        #     T1('value', self.tempo)
                        with T('divisions'):
                            _(str(MUSICXML_DIVISIONS))
                        with T('key'):
                            T1('fifths', 0)
                        with T('time'):
                            T1('beats', self.timesig[0])
                            T1('beat-type', self.timesig[1])
                        clef = self.clef[0]
                        with T('clef'):
                            T1('sign', clef)
                            T1('line', clef_dict[clef])
                            if len(self.clef) == 2:
                                T1('clef-octave-change', {'G': 1, 'F': -1}[clef])
                    with T('direction', placement='above'):
                        with T('direction-type'):
                            with T('metronome'):
                                T1('beat-unit', 'quarter')
                                T1('per-minute', self.tempo)
                        _.empty('sound', tempo=self.tempo)
                # se fija en cada voice para incluir en el measure
                for voice_number, voice in enumerate(self.voices):
                    measure = voice.measures[imeasure]
                    for pulse in measure.pulses:
                        pulse.join_tied_notes()
                        actual_notes = pulse.subdivision
                        normal_notes = previous_power_of_two(pulse.subdivision)
                        normal_type = REGULAR_NOTETYPES[self.timesig[1]*normal_notes]
                        current_numbeams = 0
                        pulse.verify()
                        for note_number, note in enumerate(pulse.notes):
                            duration = dur2xml(note.dur)
                            durtype, numbeams, numdots = get_notated_duration(
                                note.dur, pulse.subdivision, pulse_dur, self.timesig)
                            is_lastnote = note_number == len(pulse.notes)-1
                            noteattrs = {}
                            if not note.isrest():
                                dyn = dyncurve.amp2dyn(note.amp, nearest=True)
                                current_dynamic = dyn
                                if include_sizes:
                                    noteattrs['font-size'] = dyn2fontsize(dyn, use_css_sizes)
                            with T('note', **noteattrs):
                                if note.isrest():
                                    _.empty('rest')
                                else:
                                    step, alter, octave = pitch2xml(note.pitch)
                                    with T("pitch"):
                                        T1('step', step)
                                        T1('alter', alter)
                                        T1('octave', octave)
                                T1('duration', duration)
                                # se fija si hay que agregar ligaduras
                                if close_tie[voice_number]:
                                    _.empty('tie', type='stop')
                                if note.tied:
                                    _.empty('tie', type='stop')
                                with T('voice'):
                                    _(voice_number+1)
                                typeattrs = {}
                                if not note.isrest() and include_sizes:
                                    typeattrs['symbol-size'] = note.size
                                with T('type', **typeattrs):
                                    _(durtype)
                                if numdots > 0:
                                    for dot in range(numdots):
                                        _.empty('dot')
                                if not note.isrest() and alter != 0:
                                    T1('accidental', micro_to_xmlaccidental(alter))
                                do_tuplet = pulse.subdivision in (1, 2, 4, 8, 16, 32, 64)
                                if do_tuplet:
                                    with T('time-modification'):
                                        T1('actual-notes', actual_notes)
                                        T1('normal-notes', normal_notes)
                                        T1('normal-type', normal_type)
                                do_beams = False  # <-------- DON'T DO BEAMS
                                if do_beams:
                                    if note_number != len(pulse.notes)-1 and note_number != 0:
                                        for beam in range(numbeams):
                                            with T('beam', number=beam+1):
                                                _('continue')
                                    if numbeams != current_numbeams:
                                        if numbeams > current_numbeams:
                                            # agrega beams y continua los que habia
                                            for beam in range(current_numbeams, numbeams):
                                                with T('beam', number=beam+1):
                                                    _('begin')
                                        if numbeams < current_numbeams:
                                            for beam in range(numbeams, current_numbeams):
                                                with T('beam', number=beam+1):
                                                    _('end')
                                    # se fija si es la ultima nota del pulso
                                    # (para cerrar los beams)
                                    if note_number == len(pulse.notes)-1:
                                        # for beam in range(current_numbeams):
                                        for beam in range(numbeams):
                                            with T('beam', number=beam+1):
                                                _('end')
                                current_numbeams = numbeams
                                #  -- NOTEHEADS --
                                if not note.isrest() and self.renderconfig['show_noteshapes']:
                                    xml_notehead = bw_to_xmlnotehead(note.bw)
                                    attrs = {'font-size': 8}
                                    if xml_notehead.filled:
                                        attrs['filled'] = True
                                    with T('notehead', **attrs):
                                        _(xml_notehead.shape)
                                # -- NOTATIONS --
                                do_dynamics = (current_dynamic != last_dynamic) and not note.isrest()
                                
                                # if close_tie[voice_number] or note.tied or do_tuplet or do_dynamics or articulations:
                                with T('notations'):
                                    if close_tie[voice_number] and not note.tied:
                                        _.empty('tied', type='stop')
                                    if note.tied:
                                        _.empty('tied', type='start')
                                    if do_tuplet:
                                        if note_number == 0:
                                            _.empty('tuplet', bracket='yes', number=1,
                                                    type='start')
                                        if is_lastnote:
                                            _.empty('tuplet', number=1, type='stop')
                                    if include_dynamics and do_dynamics:
                                        with T('dynamics'):
                                            _.empty(current_dynamic)
                                        last_dynamic = current_dynamic
                                    if not note.isrest():
                                        articulations = [a for a in note.articulations if a is not None]
                                        if articulations:
                                            with T('articulations'):
                                                for articulation in articulations:
                                                    # TODO: define our o
                                                    _.empty(articulation)
                                close_tie[voice_number] = note.tied
                                # finish note
                # finish measure
