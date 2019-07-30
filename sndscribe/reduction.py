from __future__ import absolute_import
import numpy as np
import sndtrck
from emlib.pitchtools import db2amp, amp2db, f2m
from emlib.lib import nearest_element, isiterable
from emlib.iterlib import pairwise
from . import dynamics
from .tools import almosteq
from .note import Note
from .envir import logger
from .typehints import  List, Seq, Opt


_Number = (int, float)

def reduce_breakpoints(partials, pitch_grid=0.5, db_grid=0, time_grid=0) -> List[sndtrck.Partial]:
    """
    Reduce the breakpoints of this Spectrum. After evaluation, each breakpoint
    represents the variation of at least one of the parameters given

    - partials: a Spectrum or a list of Partials
    - pitch_grid: the minimal difference between two breakpoints, in pitch-steps
                  (midi steps). It can be fractional or a list of all possible
                  pitches
    - db_grid: the minimal difference between two breakpoints' amplitude.
               It can be a dB value, or a list of possible dB values.
    - time_grid: if given, the breakpoints are snapped to this resolution, and any
                 redundant breakpoints are removed

    """
    assert isinstance(pitch_grid, (int, float))
    assert 0 < pitch_grid <= 1
    assert isinstance(db_grid, (int, float, list))
    if isinstance(db_grid, list):
        assert all(-200 < db <= 0 for db in db_grid)
    logger.debug("**************** reduce_breakpoints **************")
    logger.debug(f"** pitch_grid: {pitch_grid}, time_grid: {time_grid}")
    newpartials = []
    for p in partials:
        p2 = partial_simplify(p, pitch_grid, db_grid, time_grid=time_grid)
        if len(p2) < 2:
            logger.debug("partial has too few breakpoints, skipping")
            continue
        elif p2.duration < time_grid:
            logger.debug(f"Partial too short, skipping."
                         f" Num breakpoints: {len(p2)}, dur.: {p2.duration:.3f}")
            continue
        if time_grid > 0:
            density = (len(p2)-1)/p2.duration
            maxdensity = 1/time_grid
            # assert maxdensity < 20
            assert density < maxdensity + 1e-4, (density, maxdensity)
        else:
            logger.info("no time quantisation will be performed")
        newpartials.append(p2)
        times = p2.times
        durs = np.diff(times)
        if (durs < 1/1000.).any():
            logger.info(f"reduce_breakpoints: small durations in partial: {durs[durs < 1e-12]}")
    logger.debug(f"reduce_breakpoints: before {len(partials)} partials / after {len(newpartials)}")
    return newpartials


def fix_quantization_overlap(partials: Seq[sndtrck.Partial], threshold=0.001) -> Opt[List[sndtrck.Partial]]:
    """
    After calling reduce_breakpoints on a series of contiguous partials it can happen that,
    due to time quantization, a partial start stays behind from a previous partials's end

    Here we fix such circumstancies.

    We assume that partials are sorted

    :param partials: the partials to fix
    :param threshold: only time mismatches smaller than this will be fixed
    :return: a list of fixed partials, or None if an error ocurred
    """
    newpartials = [partials[0]]
    for p0, p1 in pairwise(partials):
        if p1.t0 > p0.t1:
            newpartials.append(p1)
        elif p0.t1 - p1.t0 < threshold:
            assert p1.times[1] > p0.t1
            p1 = p1.crop(p0.t1, p1.t1)
            newpartials.append(p1)
        else:
            logger.error("fix_quantization_overlap: can't fix, overlap is too big")
            logger.error(f"   p0: {p0}, p1: {p1}")
            return None
    return newpartials
    

def partial_simplify(partial: sndtrck.Partial, pitch_grid=0.25, db_grid=1., time_grid=0) -> sndtrck.Partial:
    """
    Simplify a partial to remove breakpoints which do not result in actual deviations

    partial:
        the partial to simplify
    pitch_grid:
        a number indicating a minimum resolution, or a list of possible pitches (in midinotes)
    db_grid:
        a number indicating a minimum resolution (in dB) or a list of possible dBs,
        in which case breakpoints are snapped to these values (0=leave as is)
    time_grid:
        a number indicating a minimum time resolution (0 to leave as is)
        NB: currently not used
    """
    assert isinstance(partial, sndtrck.Partial)
    assert isinstance(pitch_grid, (list, int, float))
    assert isinstance(db_grid, (list, int, float))
    assert isinstance(time_grid, (int, float))

    logger.debug(f"starting partial_simplify: numbps: {len(partial)}")
    newpartial = partial.quantized(pitchgrid=pitch_grid, dbgrid=db_grid, timedelta=time_grid)
    logger.debug(f"numbps after quantization: {len(newpartial)}")

    pitch_delta = pitch_grid if not isiterable(pitch_grid) else 0.25
    db_delta = db_grid if not isiterable(db_grid) else 1.0
    if db_delta == 0 and time_grid > 0:
        db_delta = 0.5

    newpartial = newpartial.simplify_notes(pitchdelta=pitch_delta, dbdelta=db_delta)
    density = (len(newpartial)-1) / newpartial.duration

    if len(newpartial) == len(partial):
        logger.debug(f"Partial not simplified ({len(partial)} bps)")
    else:
        logger.debug(f"Reduced partial from {len(partial)} bps -> {len(newpartial)} bps"
                     f"(dur:{partial.duration:.3f}, dens:{density:3f})")

    if time_grid > 0 and len(newpartial) > 2:
        maxdensity = 1/time_grid
        assert density <= maxdensity+1e-10, \
            ("numbr:%d dur:%f dens.:%f maxdens.: %.1f times: %s" % 
                (len(newpartial), newpartial.duration, density, maxdensity, newpartial.times))
    assert isinstance(newpartial, sndtrck.Partial)

    if len(newpartial) > len(partial):
        logger.warn(f"partial_simplify failed: old partial: {len(partial)} breakpoints / new partial: {len(newpartial)}")
    return newpartial


def snap_note(note: Note, pitches: Seq[float], dbs: List[float]) -> Note:
    amp = db2amp(nearest_element(amp2db(note.amp), dbs))
    return note.clone(pitch=nearest_element(note.pitch, pitches), amp=amp)


def simplify_notes(notes: List[Note], minpitch:float, dyncurve: dynamics.DynamicsCurve) -> List[Note]:
    """
    Join notes together which would not show any difference when notated
    Returns the simplified notes

    :param notes: the notes to simplify
    :param minpitch: pitch resolution
    :param dyncurve: DynamicsCurve

    :return: list of Notes
    """
    assert isinstance(notes, list) and all(isinstance(note, Note) for note in notes)
    assert all(note.pitch > 0 and note.amp > 0 for note in notes), notes
    assert isinstance(minpitch, float)
    assert isinstance(dyncurve, dynamics.DynamicsCurve)


    dbs = dyncurve.asdbs()
    pitches = np.arange(0, 130, minpitch)
    notes = [snap_note(note, pitches, dbs) for note in notes]
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
    assert all(note.pitch > 0 and note.amp > 0 for note in newnotes), f"newnotes: {newnotes}"
    assert newnotes[0].start == notes[0].start and newnotes[-1].end == notes[-1].end
    assert len(newnotes) <= len(notes)
    return newnotes
