from __future__ import absolute_import
from .note import Event
from .tools import *
from .scorefuncs import *
from .quantization import find_best_division
from .config import RenderConfig


class Pulse(object):
    def __init__(self,
                 notes:  List[Event],
                 offset: Fraction,
                 renderconfig: RenderConfig,
                 pulsedur: Fraction=R(1)
                 ):
        assert isinstance(notes, list)
        assert len(notes) > 0
        assert all(isinstance(note, Event) for note in notes)
        assert isinstance(offset, Fraction) and offset >= 0
        assert isinstance(pulsedur, Fraction) and pulsedur > 0
        pitch_resolution: float = renderconfig['pitch_resolution']
        dyncurve = renderconfig.dyncurve
        self.original_notes = notes
        self.notes = [snapnote(note, pitch_resolution, dyncurve=dyncurve)
                      for note in notes]
        self.offset = offset
        self.renderconfig = renderconfig
        self.pulse_dur = asR(pulsedur)
        self.subdivision = None
        self._division_info = None
        # Puts the division in self.subdivision.
        # The notes according to the division are in self.notes
        self.quantize()
        self.verify()
        for note in self.notes:
            num, den = note.dur.numerator, note.dur.denominator
            assert (num, den) not in IRREGULAR_TUPLES
            assert note.dur*self.subdivision not in (5, 9, 13)
            assert isR(note.dur)
        # assert len(self.notes) <= max(renderconfig['divisions'])
        assert len(self.notes) <= max(renderconfig['divisions'])
        assert self.notes[0].start == self.offset
        assert almosteq(self.notes[-1].end, self.offset + self.pulse_dur)

    def __iter__(self):
        return iter(self.notes)

    def __len__(self):
        return len(self.notes)

    def quantize(self):
        """
        Pulse.find_best_division
        After calling this function, the current Pulse should contain
        a list of notes in self.notes which fit in a grid of
        self.subdivision ticks, which itself is one of self.possible_divs

        NB1: the old notes are not deleted, they are preserved in
             self.original_notes
        NB2: notes which have not been fitted are saved in self.unfitted_notes
        """
        timesig = self.renderconfig.timesig
        tempo = self.renderconfig.tempo
        pulse_dur = timesig2pulsedur(timesig, tempo)
        possible_divs = self.renderconfig['divisions']
        # possible_divs = self.renderconfig['divisions']
        
        if not self.notes:
            logger.debug("No notes in Pulse, filling with silence")
            div, assigned_notes = 1, [Rest(self.offset, pulse_dur)]
            info = None
        else:
            assert all(isinstance(note, Event) for note in self.notes)
            div, assigned_notes, info = find_best_division(
                self.notes, possible_divs, pulsestart=self.offset,
                pulsedur=pulse_dur, dyncurve=self.renderconfig.dyncurve,
                config=self.renderconfig)
            assert div in possible_divs
            possible_durs = [R(i, div) for i in range(1, div + 1)]
            assert all(event.dur in possible_durs for event in assigned_notes)
        self.subdivision = div
        self.notes = assigned_notes
        self.join_tied_notes()
        self.notes = break_irregular_tuples(self.notes, self.subdivision, pulse_dur)
        self.notes.sort(key=lambda n: n.start)
        self._division_info = info
        assert len(self.notes) >= 1
        assert self.subdivision >= 1
        t0, t1 = self.offset, self.offset+self.pulse_dur
        assert all(t0 <= note.start and note.end <= t1 for note in self.notes)
        
    def join_tied_notes(self):
        assert len(self.notes) >= 1
        joint_notes = notes_join_tied(self.notes)
        pulsedur = timesig2pulsedur(self.renderconfig.timesig, self.renderconfig.tempo)
        newnotes = break_irregular_tuples(joint_notes, self.subdivision, pulsedur)

        def simplify_pulse(notes, pulsedur):
            if len(notes) > 1 and all(note.isrest() for note in notes):
                t0, t1 = notes[0].start, notes[-1].end
                div, notes = 1, [Rest(t0, t1-t0)]
            elif len(notes) == 1:
                assert almosteq(notes[0].dur, pulsedur)
                div, notes = 1, [notes[0].clone(dur=pulsedur)]
            else:
                div = None
            return div, notes

        division, newnotes = simplify_pulse(newnotes, pulsedur)
        if division is not None and self.subdivision != division:
            # logger.debug(f"join_tied_notes: changing div. from {self.subdivision} to {division}")
            self.subdivision = division
        assert all(isinstance(note, Event) for note in newnotes)
        assert newnotes[0].start == self.notes[0].start
        assert newnotes[-1].end == self.notes[-1].end
        assert not(self.subdivision > 1 and all(isinstance(n, Rest) for n in self.notes)), (self.subdivision, self.notes)
        self.notes = newnotes

    def verify(self, correct=False):
        notes = self.notes
        maxdivisions = max(self.renderconfig['divisions'])
        if maxdivisions > 16:
            logger.error("#### maxdivisions: %d" % maxdivisions)
        assert len(self.notes) <= maxdivisions
        assert almosteq(sum(n.dur for n in notes), self.pulse_dur), notes
        if not all(note.isrest() or note.pitch > 0 and note.amp > 0 for note in notes):
            if not correct:
                logger.error("Either pitched notes with amp 0 or silences with amp>0")
                raise ValueError("malformed Note")
            else:
                raise NotImplemented("correction is not implemented")
