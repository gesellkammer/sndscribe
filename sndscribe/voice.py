from __future__ import absolute_import
import numpy as np
from emlib.pitch import f2m
import sndtrck
from .tools import *
from .config import RenderConfig
from .scorefuncs import *  
from .note import *
from .measure import Measure
from .envir import logger
from .typehints import *


class Voice(object):
    def __init__(self, notes:List[Note]=None):
        """
        A voice is a general container for non overlapping Notes.

        * No rests should be added, they are generated during rendering.
        * A voice generates a list of measures, each measure generates a list of pulses
        """
        self._notes = []          # type: List[Note]
        self._start = R(-1)       # type: Fraction
        self._end = R(-1)         # type: Fraction
        self._meanpitch = -1      # type: float
        self.measures = []        # type: List[Measure]
        self.divided_notes = []   # type: List[Note]
        if notes:
            self.addnotes(notes)

    def __repr__(self) -> str:
        if self.isempty:
            return "Voice()"
        else:
            return "Voice(%.3f:%.3f)" % (self.start.__float__(), self.end.__float__())

    def isrendered(self) -> bool:
        return len(self.measures) > 0

    @property
    def notes(self) -> List[Note]:
        return self._notes

    def addnote(self, note:Note) -> None:
        return self.addnotes([note])
        
    def addnotes(self, notes:List[Note]) -> None:
        assert isinstance(notes, (list, tuple))
        for note in notes:
            assert isinstance(note, Note)
            if note.pitch == 0:
                assert note.amp == 0
            if note.dur < 1e-10:
                logger.debug("adding a very short note: %s" % note)
            self._notes.append(note)
        self.sort_by_time()

    def _invalidate_cache(self) -> None:
        self._start = R(-1)
        self._end = R(-1)
        self._meanpitch = -1

    def unrendered_notes(self) -> List[Note]:
        return self.notes

    def sort_by_time(self) -> None:
        self._notes.sort(key=lambda x: x.start)
        self._invalidate_cache()

    def density(self) -> float:
        notes = [note for note in self.notes if not note.isrest()]
        return sum(note.dur for note in notes)/(self.end - self.start)

    @property
    def start(self) -> Fraction:
        """Returns the start time of the voice"""
        if self._start >= 0:
            return self._start
        else:
            if self.notes:
                self._start = min(note.start for note in self.notes)
                return self._start
            raise ValueError("voice is empty")

    def isempty(self) -> bool:
        return len(self.notes) == 0

    @property
    def end(self) -> Fraction:
        """Returns the end time of the voice"""
        if self._end >= 0:
            return self._end
        elif self.notes:
            self._end = max(note.end for note in self.notes)
            return self._end
        else:
            raise ValueError("voice is empty")

    def render(self, renderconfig:RenderConfig, end:Opt[Fraction]=None) -> None:
        """
        Voice::render

        crea compases segun la timesig y el tempo dados
        se asegura de que al final el compas este completo con silencios

        Prerender:
            * cut overlap
            * fill silences
            * divide long notes

        * render
        """
        assert not self.isrendered()
        assert renderconfig is not None and isinstance(renderconfig, RenderConfig)
        self.quantize_pitches(renderconfig['pitch_resolution'])
        timesig = renderconfig.timesig
        tempo = renderconfig.tempo
        if end is None:
            end = self.end
        self.measures = []
        pulse = timesig2pulsedur(timesig, tempo)
        measuredur = timesig2measuredur(timesig, tempo)
        silence_at_end = measuredur - (end % measuredur)
        cutnotes = cut_overlap(self.notes)
        fillednotes = fill_silences(cutnotes, R(0), cutnotes[-1].end + silence_at_end)
        notes = divide_long_notes(fillednotes, pulse, mindur=1e-8)
        self.divided_notes = notes
        num_measures = int(math.ceil(end / float(measuredur)))
        for imeasure in range(num_measures):
            t0 = asR(imeasure * measuredur)
            t1 = asR(t0 + measuredur)
            notes_in_meas = []
            for note in notes:
                if note.start >= t0 and note.end <= t1:
                    notes_in_meas.append(note)
            assert not notes_in_meas or abs(notes_in_meas[-1].end - t1) < 1e-8
            measure = Measure(notes_in_meas, t0, renderconfig=renderconfig)
            self.measures.append(measure)

    def meanpitch(self) -> float:
        if self._meanpitch > 0:
            return self._meanpitch
        if not self.notes or all(note.isrest() for note in self.notes):
            return -1
        mean = meanpitch(self.notes)
        self._meanpitch = mean
        return mean
        
    def is_free_between(self, time0, time1) -> bool:
        return qintersection(self.start, self.end, time0, time1) is not None

    def addpartial(self, partial:sndtrck.Partial) -> bool:
        """
        Returns True if partial could be added
        """
        assert not self.isrendered()
        # partialdata: 2D numpy with columns [time, freq, amp, phase, bw]
        partialdata: np.ndarray = partial.toarray()
        amps = partialdata[:, 2]
        assert np.all(amps[1:-1] > 0)
        freqs = partialdata[:, 1]
        assert np.all(freqs[1:-1] > 0)
        max_overlap = R(1, 16)
        # check overlap
        # lastnote_dur = self.renderconfig['last_note_duration']
        if not self.isempty:
            diff = partial.t0 - self.end
            if diff < 0:
                # There is some overlap, check if it is small
                if abs(diff) <= max_overlap:
                    array_fitbetween(partialdata[:, 0], float(self.end), float(partial.t1))
                else:
                    raise ValueError("Overlap detected: %.3f" % diff)
        if len(partialdata) < 2:
            logger.error("Trying to add an empty partial, skipping")
            return False
        # TODO: hacer minamp configurable
        minamp = db2amp(-90)
        # The 1st and last bp can have amp=0, used to avoid clicks. Should we include them?
        if len(partialdata) > 2 and partialdata[0, 2] == 0 and partialdata[0, 1] == partialdata[1, 1]:
            partialdata = partialdata[1:]
        notes = [] # type: List[Note]
        for i in range(len(partialdata)-1):
            t0, freq0, amp0, phase0, bw0 = partialdata[i, 0:5]
            t1, freq1, amp1 = partialdata[i+1, 0:3]
            dur = t1 - t0
            if dur < 1e-12:
                logger.error("small note: " + str((t0, t1, freq0, amp0)))
            pitch = f2m(freq0)
            # amp = (amp0 + amp1) * 0.5
            amp = amp0
            note = Note(pitch, t0, dur, max(amp, minamp), bw0, tied=False, color="addpartial")
            notes.append(note)
        # The last breakpoint was not added: add it if it would make a 
        # difference in pitch and is not just a closing bp (with amp=0)
        t0, f0, a0 = partialdata[-2, 0:3]    # butlast
        t1, f1, a1 = partialdata[-1, 0:3]    # last
        if a1 > 0 and abs(f2m(f1) - f2m(f0)) > 0.5:
            lastnote_dur = min(R(1, 16), notes[-1].dur)
            notes.append(Note(f2m(f1), start=t1, dur=lastnote_dur, amp=a1, color="lastnote"))
        notes.sort(key=lambda n: n.start)
        mindur = R(1, 128)
        if has_short_notes(notes, mindur):
            logger.error(">>>>> short notes detected")
            logger.error("\n        ".join(str(n) for n in notes if n.dur < mindur))
            raise ValueError("short notes detected")
        if any(n.amp == 0 for n in notes):
            logger.error("Notes with amp=0 detected: ")
            logger.error("\n        ".join(str(n) for n in notes if n.amp == 0))
            raise ValueError("notes with amp=0 detected")
        self.addnotes(notes)
        return True

    def quantize_pitches(self, step=0.5) -> None:
        assert not self.isrendered()
        halfstep = step * 0.5
        for note in self.notes:
            note.pitch = int((note.pitch + halfstep) / step) * step
