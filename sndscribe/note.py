# coding: latin-1
from __future__ import absolute_import
from __future__ import print_function
from .tools import *
from .fraction import *
from emlib.pitch import amp2db
from emlib.iterlib import pairwise
from .envir import logger
from . import typehints as t


class NonMutableError(Exception): 
    pass


class Event(object):
    def __init__(self, start:t.Number, dur:t.Number):
        self._start = asR(start)   # type: Fraction
        self._dur = asR(dur)       # type: Fraction
        self._mutable = True       # type: bool

    def _check_mutable(self):
        if not self._mutable:
            raise NonMutableError("Attempted to modify this object!")

    def freeze(self) -> None:
        self._mutable = False

    @property
    def end(self):
        # type: () -> Fraction
        return self.start + self.dur

    @end.setter
    def end(self, newend:t.Number):
        self._check_mutable()
        assert newend > self.start
        self.dur = asR(newend) - self.start

    @property
    def dur(self) -> Fraction:
        return self._dur

    @dur.setter
    def dur(self, dur:t.Number):
        self._check_mutable()
        self._dur = asR(dur)

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value:t.Number):
        self._check_mutable()
        self._start = asR(value)

    @staticmethod
    def isrest() -> bool:
        return False

    def clone(self, *args, **keys):
        raise NotImplementedError()

    def break_at_pulse(self, pulsedur:Fraction, mindur:t.Rat=0) -> List['Event']:
        pass


class Note(Event):
    def __init__(self,
                 pitch: float,
                 start: t.Number,
                 dur: t.Opt[t.Number]=None,
                 amp: float = 1,
                 bw: float  = 0.,
                 color: t.Opt[str] = None,
                 size: float = 0,
                 tied=False,
                 articulations: t.Opt[t.List[str]] = None,
                 transient:float = 0,
                 end:t.Opt[Number] = None
                 ) -> None:
        """
        pitch: the midinumber of this note
        start: start time
        dur: the duration of this Note (you can either set the dur or set the end)
        amp: the amplitude of this note
        bw: the bandwidth (noisyness)
        color: ??
        size: the notated size
        tied: is this a tied note?
        articulations: an optional list of arituclations (each articulation is a string)
        transient: the "attackness" of this note
        end: set either end or dur
        """
        assert isinstance(pitch, (float, int))
        start = asR(start)
        if end is not None:
            dur = asR(end) - start
        else:
            assert dur is not None
            dur = asR(dur)
        super().__init__(start, dur)
        if 0 < pitch < 10:
            logger.debug("Low pitch")
        if amp == 0 and pitch > 0:
            pitch = 0
            logger.info(f"warning: pitched note with amp 0 (pitch={pitch}")
        if (amp > 0 and pitch > 0) or (amp == pitch == 0):
            raise ValueError(f"Malformed Note: amp={amp} pitch={pitch}")
        self._pitch = pitch      # type: float
        self.amp = amp           # type: float
        self.start = asR(start)  # type: Fraction
        self.bw = bw             # type: float
        self.color = color       # type: str
        self.size = float(size)  # type: float
        self.tied = tied         # type: bool
        self.articulations = articulations if articulations is not None else []  # type: t.List[str]
        self.transient = transient  # type: float

    @property 
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, pitch):
        self._check_mutable()
        assert pitch > 0
        self._pitch = pitch            
    
    @property
    def linearamp(self):
        return linearamp(self.amp)

    def clone(self, **keys) -> 'Note':
        """
        Create a new Note with the given keys modified
        """
        out = Note(self.pitch, self.start, self.dur, self.amp, self.bw,
                   self.color, size=self.size, tied=self.tied,
                   articulations=self.articulations, transient=self.transient)
        for key, value in keys.items():
            setattr(out, key, value)
        assert out.pitch > 0 and out.amp > 0
        return out

    def add_articulation(self, articulation:str) -> None:
        assert isinstance(articulation, str)
        if self.articulations is not None:
            self.articulations.append(articulation)
        else:
            self.articulations = [articulation]
    
    def __repr__(self):
        """Note.__repr__"""
        if self.articulations and any(self.articulations):
            artstr = " art:%s" % str(self.articulations)
        else:
            artstr = ""
        trans = " %.2f" % self.transient if self.transient else ""
        tied = " TIE" if self.tied else ""
        pitch = ("%.1f" % self.pitch if not self.isrest() else "REST").rjust(4)
        ampstr = " %.0fdB" % amp2db(self.amp) if not self.isrest() else ""
        color = " " + str(self.color) if self.color is not None else ""
        return f"({pitch} {self.start:.3f}-{self.end:.3f}{ampstr}{tied}{artstr}{trans}{color})"

    def break_at_pulse(self, pulsedur:Fraction, mindur:t.Rat=0.0) -> t.List['Note']:
        if self.dur < mindur:
            raise ValueError("Note too short!")
        newnotes = []  # type: t.List[Note]
        for t0, t1 in break_dur_at_pulses(self.start, self.dur, pulsedur=pulsedur):
            if t1 - t0 > mindur:
                cloned = self.clone(start=t0, dur=t1-t0,
                                    articulations=None, transient=0, tied=True)
                newnotes.append(cloned)
        assert newnotes
        newnotes[0].articulations = self.articulations
        newnotes[-1].tied = self.tied
        rendered_events_make_contiguous(newnotes, pulsedur)
        return newnotes


class Rest(Event):
    def __init__(self, start:t.Number, dur:t.Number) -> None:
        super().__init__(start=start, dur=dur)

    @staticmethod
    def isrest() -> bool:
        return True

    @property
    def tied(self) -> bool:
        return False

    def __repr__(self) -> str:
        s = "(REST {start:.3f}-{end:.3f})".format(
            start=float(self.start), end=float(self.end))
        return s

    def break_at_pulse(self, pulsedur:Fraction, mindur:t.Number=0) -> List['Rest']:
        if self.dur < mindur:
            raise ValueError("Rest too short!")
        newrests = []
        for t0, t1 in break_dur_at_pulses(self.start, self.dur, pulsedur=pulsedur):
            if t1 - t0 > mindur:
                newrests.append(Rest(t0, dur=t1-t0))
        assert newrests
        rendered_events_make_contiguous(newrests, pulsedur)
        return newrests

    def clone(self, start:Opt[Fraction]=None, dur:Opt[Fraction]=None,
              end:Opt[Fraction]=None) -> 'Rest':
        if dur is not None and end is not None:
            raise ValueError("One one of dur or end can be specified")
        start = start if start is not None else self.start
        if end is not None:
            dur = asR(end) - start
        elif dur is None:
            dur = self.dur
        dur = dur if dur is not None else self.dur
        return Rest(start, dur)


def meanpitch(events:List[Event]) -> float:
    """
    Returns the mean weighted pitch of notes,
    or 0 if no pitched notes (only rests)
    """
    if not any(isinstance(e, Note) for e in events):
        return 0
    notes = [event for event in events if isinstance(event, Note)]
    energy = sum(note.linearamp * note.dur for note in notes)
    if not energy:
        return sum(note.pitch for note in notes) / len(notes)
    return sum(note.pitch*note.linearamp*note.dur for note in notes) / energy


def rendered_events_make_contiguous(events:List[Event], pulsedur:Fraction) -> List[Event]:
    """
    Remove any gaps between rendered events (in place)
    It does not create new silences, but makes sure that the end of any
    event is the same as the start of the next event

    **NB**: this function should be called after filling with silences
            and cutting any overlap
    """
    if not all(e.dur <= pulsedur for e in events):
        invalid_events = [ev for ev in events if ev.dur > pulsedur]
        raise ValueError(f"This should be called on rendered events: {invalid_events}")
    if not all(e1.start - e0.end < pulsedur for e0, e1 in pairwise(events)):
        raise ValueError("The gap between two notes should be less than the pulse dur.")
    for e0, e1 in pairwise(events):
        if e0.end == e1.start:
            # no gap, OK
            continue
        time_of_next_pulse = next_pulse(e0.end, pulsedur)
        if time_of_next_pulse < e1.start:
            # e0 ends before the pulse and e1 starts after the pulse?
            logger.debug(f"make_contiguous: changing end from {e0.end} to {time_of_next_pulse}")
            logger.debug(f"make_contiguous: changin start from {e1.start} to {time_of_next_pulse}")
            e0.end = e1.start = time_of_next_pulse
        else:
            e0.end = e1.start
    assert all(e1.start == e0.end for e0, e1 in pairwise(events))
    return events