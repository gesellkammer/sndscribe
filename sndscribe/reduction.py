from __future__ import absolute_import
from numbers import Number
import warnings
import numpy as np
import sndtrck
from emlib.pitch import db2amp, amp2db
from emlib.lib import nearest_element
from . import dynamics
from .tools import almosteq
from .note import Note
from .envir import logger



def reduce_breakpoints(partials, pitch_grid=0.5, db_grid=0, time_grid=0):
    """
    Reduce the breakpoints of this Spectrum. After evaluation, each breakpoint
    represents the variation of at least one of the parameters given

    - partials: a Spectrum or a list of Partials
    - pitch_grid: the minimal difference between two breakpoints, in pitch-steps
                  (midi steps). It can be fractional or a list of all possible
                  pitches
    - db_grid: the minimal difference between two breakpoints' amplitude.
               It can be a dB value, or a list of possible dB values.
    - time_grid: if given, the partials are resampled at this resolution, and any
                 redundant breakpoints are removed

    :rtype : list[sndtrck.Partial]

    NB: see em.comp.dynamics.DynamicsCurve.todbs to use dynamics as db_grid
    """
    assert isinstance(pitch_grid, (int, float))
    assert 0 < pitch_grid <= 1
    assert isinstance(db_grid, (Number, list))
    if isinstance(db_grid, list):
        assert all(-200 < db <= 0 for db in db_grid)

    newpartials = []
    for p in partials:
        p2 = partial_simplify(p, pitch_grid, db_grid, time_grid=time_grid)
        if len(p2) < 2:
            logger.debug("partial has too few breakpoints, skipping")
            continue
        elif p2.duration < time_grid:
            logger.debug("Partial too short, skipping. numbr: %d  duration: %.4f"
                  % (len(p2), p2.duration))
            continue
        if time_grid > 0:
            density = (len(p2)-1)/p2.duration
            maxdensity = 1/time_grid
            # assert maxdensity < 20
            assert density < maxdensity + 1e-4, (density, maxdensity)
        else:
            warnings.warn("no time quantisation will be performed")
        newpartials.append(p2)
        times = p2.times
        durs = np.diff(times)
        if (durs < 1/1000.).any():
            logger.debug("small durations in partial: {0}".format(durs[durs < 1e-12]))
    return newpartials
    # return sndtrck.Spectrum(newpartials)


def partial_simplify(partial, pitch_grid=0.5, db_grid=None, time_grid=0):
    """
    :param partial: the partial to simplify
    :param pitch_grid: a number indicating a minimum resolution, or
                       a list of possible pitches
    :param db_grid: a number indicating a minimum resolution (in dB)
                    or a list of possible dBs
    :param time_grid: a number indicating a minimum time resolution
                      (0 to leave as is)
    """
    assert isinstance(partial, sndtrck.Partial)
    assert isinstance(pitch_grid, (list, Number))
    assert isinstance(db_grid, (list, Number))
    assert isinstance(time_grid, Number)

    newpartial = partial.quantized(pitch_grid=pitch_grid, db_grid=db_grid, time_grid=time_grid)
    # newpartial = newpartial.simplified(pitch_grid)
    newpartial = newpartial.simplify(pitchdelta=0.5, dbdelta=db_grid, bwdelta=1)
    density = (len(newpartial)-1)/newpartial.duration
    if len(newpartial) == len(partial):
        logger.debug("Could not simplify partial. Numbr=%d" % len(partial))
    else:
        logger.debug("Reduced {numbr}-->{newnumbr} dur:{dur:.3f} dens:{density}".format(
            numbr=len(partial), newnumbr=len(newpartial),
            dur=partial.duration, density=density))
    if time_grid > 0 and len(newpartial) > 2:
        maxdensity = 1/time_grid
        assert density <= maxdensity+1e-10, ("numbr:%d dur:%f dens.:%f maxdens.: %.1f times: %s"
             % (len(newpartial), newpartial.duration, density, maxdensity, newpartial.times))
    assert isinstance(newpartial, sndtrck.Partial)
    return newpartial


def simplify_notes(notes, minpitch, dyncurve):
    """
    Join notes together which would not show any difference
    when notated

    :type notes: list[Note]

    :param minpitch: pitch resolution
    :type minpitch: float

    :param dyncurve: DynamicsCurve
    :type dyncurve: dynamics.DynamicsCurve

    :return: list of Notes
    :rtype : list[Note]
    """
    assert isinstance(notes, list) and all(isinstance(note, Note) for note in notes)
    assert all(note.pitch > 0 and note.amp > 0 for note in notes)
    assert isinstance(minpitch, float)
    assert isinstance(dyncurve, dynamics.DynamicsCurve)

    def snap(note, pitches, dbs):
        """
        :rtype : Note
        """
        amp = db2amp(nearest_element(amp2db(note.amp), dbs))
        return note.clone(pitch=nearest_element(note.pitch, pitches), amp=amp)

    dbs = dyncurve.asdbs()
    pitches = np.arange(0, 130, minpitch)
    notes = [snap(note, pitches, dbs) for note in notes]
    gap = 1e-8
    newnotes = []
    noteopen = notes[0]

    for note in notes[1:]:
        if noteopen is None or note.start - noteopen.end > gap:
            # there is a gap
            if noteopen is not None:
                newnotes.append(noteopen)
            noteopen = note
        elif not almosteq(note.pitch, noteopen.pitch) or not almosteq(note.amp, noteopen.amp):
            # no gap and note has new information
            newnotes.append(noteopen)
            noteopen = note
        else:
            # a continuation of the open note
            noteopen.end = note.end
    if noteopen is not None:
        newnotes.append(noteopen)
    assert all(note.pitch > 0 and note.amp > 0 for note in newnotes)
    assert newnotes[0].start == notes[0].start and newnotes[-1].end == notes[-1].end
    assert len(newnotes) <= len(notes)
    return newnotes
