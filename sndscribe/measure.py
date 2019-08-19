from .note import Event
from .tools import *
from .pulse import Pulse
from .envir import logger
from .typehints import *
from .fraction import Fraction
from .config import RenderConfig


class Measure:
    """
    Measures are created by each voice when they are rendered.
    They themselves create Pulses and distribute the notes among them
    This ends the rendering process.
    """
    def __init__(self, notes:List[Event], offset:Fraction, renderconfig:RenderConfig):
        assert isinstance(notes, (list, tuple))
        assert all(isinstance(note, Event) for note in notes)
        assert isinstance(offset, Fraction) and offset >= 0
        self.notes = notes
        self.offset = offset
        self.renderconfig = renderconfig
        self.pulses = []   # type: List[Pulse]
        self.rendered = True
        self.create_pulses()

    def __iter__(self):
        return iter(self.pulses)

    def create_pulses(self) -> None:
        timesig = self.renderconfig.timesig
        tempo = self.renderconfig.tempo
        pulsedur = timesig2pulsedur(timesig, tempo)
        numpulses = timesig[0]
        pulses = []
        dyncurve = self.renderconfig.dyncurve
        for pulsenum in range(numpulses):
            t0 = asR(self.offset) + pulsenum*pulsedur
            t1 = t0 + pulsedur
            notes_in_pulse = []
            for note in self.notes:
                if note.start > t1:
                    break
                if t0 <= note.start < t1:
                    assert note.end <= t1, "notes across boundaries!"
                    notes_in_pulse.append(note)
            maxdivisions = max(self.renderconfig['divisions'])
            if len(notes_in_pulse) > maxdivisions+2:
                logger.debug(f"maxdivisions: {maxdivisions}, notes in pulse: {len(notes_in_pulse)}")
                for note in notes_in_pulse:
                    dyn = dyncurve.amp2dyn(note.amp) if not note.isrest() else "-"
                    logger.debug(f"note: {note}   dyn: {dyn}")
                logger.error("too many notes")
            pulse = Pulse(notes_in_pulse, t0, self.renderconfig)
            pulses.append(pulse)
        self.pulses = pulses
