from __future__ import absolute_import
from emlib.lib import returns_tuple, issorted, snap_to_grid
from .note import *
from .envir import logger
from .dynamics import DynamicsCurve
from .tools import Fraction, R, asR, isR
from .conversions import typestr2numbeams
from . import typehints as t


def break_irregular_tuples(notes_in_pulse: t.List[Event], div: int, pulsedur: Fraction
                           ) -> t.List[Event]:
    assert isinstance(div, int)
    assert 1 <= div < 32
    assert all(isinstance(note, Event) for note in notes_in_pulse)
    assert len(notes_in_pulse) > 0
    assert isR(pulsedur)
    newnotes = []
    notes_in_pulse = sorted(notes_in_pulse, key=lambda x: x.start)
    pulsedur = asR(pulsedur)
    # grid = arange(start, end, R(start-end)/R(division))
    for note in notes_in_pulse:
        slotdur = pulsedur / div
        numslots = int(note.dur/slotdur + 1e-8)
        if numslots in (5, 9, 10, 11, 13):
            if numslots in (5, 9, 13):
                slots0, slots1 = numslots - 1, 1
            elif numslots == 11:
                slots0, slots1 = numslots - 3, 3
            elif numslots == 10:
                slots0, slots1 = numslots - 2, 2
            else:
                raise ValueError("numslots should be one of 5, 9, 10, 11, 13, got {numslots}")
            if isinstance(note, Note):
                note0 = note.clone(dur=slotdur*slots0, tied=True)
                note1 = note.clone(start=note0.end, dur=slotdur*slots1, tied=note0.tied)
            else:
                note0 = note.clone(dur=slotdur*slots0)
                note1 = note.clone(start=note0.end, dur=slotdur*slots1)
            assert almosteq(note0.dur+note1.dur, note.dur)
            newnotes.append(note0)
            newnotes.append(note1)
        else:
            newnotes.append(note)
    assert len(newnotes) >= 1
    assert newnotes[0].start == notes_in_pulse[0].start and newnotes[-1].end == notes_in_pulse[-1].end
    return newnotes

@returns_tuple("typestr numbeams numdots")
def _notated_duration(dur: Fraction, div: int, pulsedur: Fraction, timesig: t.Tup[int, int]
                     ) -> t.Tup[str, int, int]:
    """
    This function is always called after break_irregular_tuples, so the passed dur should be
    a regular duration, meaning that it is either an 8th, quarter, 16th note, etc (possibly
    with a dot)

    Args:
        dur:
        div:
        pulsedur:
        timesig:

    Returns:

    """
    assert isinstance(dur, Fraction)
    assert isinstance(div, int)
    assert isinstance(pulsedur, Fraction)
    assert isinstance(timesig, tuple) and len(timesig) == 2 and isinstance(timesig[0], int) and isinstance(timesig[1], int)
    denominator = R(timesig[1])


@returns_tuple("typestr numbeams numdots")
def notated_duration(dur: Fraction, div: int, pulsedur: Fraction, timesig: t.Tup[int, int]
                     ) -> t.Tup[str, int, int]:
    """
    Given a duration within its context (division of the pulse, duration of the pulse, time
    signature, returns elements needed to notate the duration: the duration type ("eigth", "quarter", etc),
    the corresponding number of beams (1 for 8th note, 2 for 16th note, etc) and the corresponding
    number of dots

    Args:
        dur: the duration of the note
        div: the division of the pulse
        pulsedur: the duration of the pulse
        timesig: the time signature

    Returns:
        A tuple (typestr, numbeams, numdots)

    NB: Raises ValueError if it cannot determine a duration
    """
    # dx = 0.00000001
    dur = asR(dur)
    pulsedur = asR(pulsedur)
    denominator = R(timesig[1])
    _type = dur * pulsedur * 4 / denominator
    num, den = _type.numerator, _type.denominator
    coef = R(1) if den == div else R(div)/R(den)
    den = int(den * coef)
    num = int(num * coef)
    # Is it an irregular notetype?
    typestr, numdots = IRREGULAR_NOTETYPES.get((num, den), (None, None))
    if not typestr:
        _type = R(num, den).limit_denominator(128)
        num, den = _type.numerator, _type.denominator
        if (num, den) in IRREGULAR_NOTETYPES:
            typestr, numdots = IRREGULAR_NOTETYPES[(num, den)]
        else:
            raise ValueError("Can't find notation for duration: " + str((num, den)))
    numbeams = typestr2numbeams(typestr)
    assert isinstance(typestr, str) and typestr in NOTETYPES, f"typestr: {typestr} ({type(typestr)})"
    assert isinstance(numbeams, int) and 0 <= numbeams <= 6, f"numbeams: {numbeams}"
    assert isinstance(numdots, int) and 0 <= numdots <= 2, f"numdots: {numdots}"
    return typestr, numbeams, numdots


def infer_clef(notes: t.List[Event]) -> str:
    """
    Infer a clef from the pitch of the notes

    :type notes: list[Event]
    :rtype : str
    :return: str, one of "G", "G8", "F", "F8"
    """
    assert all(isinstance(n, Event) for n in notes)
    assert len(notes) > 0
    pitch = meanpitch(notes)
    if pitch < 36:
        clef = 'F8'
    elif 36 <= pitch < 60:
        clef = 'F'
    elif 60 <= pitch < 87:
        clef = 'G'
    else:
        clef = 'G8'
    assert clef in ('F8', 'F', 'G', 'G8')
    return clef


def snapnote(note: Event, pitch_resolution: float, dyncurve: DynamicsCurve) -> Event:
    """
    Snaps the values of this Note (pitch, amplitude) to the grids given

    * note: a Note
    * pitch_resolution: the resolution, in semitones
    * Returns: a clone of this note, snapped to the nearest
               values

    :type note: Note
    :type pitch_resolution: float
    :type dyncurve: dynamics.DynamicsCurve
    :rtype : Note
    """
    if isinstance(note, Rest):
        return note
    elif isinstance(note, Note):
        normalized_pitch = snap_to_grid(note.pitch, pitch_resolution)
        normalized_amp = dyncurve.dyn2amp(dyncurve.amp2dyn(note.amp))
        assert normalized_pitch > 0, (note, pitch_resolution)
        assert normalized_amp > 0, (note, dyncurve.amp2dyn(note.amp))
        return note.clone(pitch=normalized_pitch, amp=normalized_amp)
    else:
        raise TypeError(f"Expecting an Event (a Note, a Rest), got {type(note)}")

def notes_join_tied(notes):
    # type: (t.List[Event]) -> t.List[Event]
    """
    Merge tied notes
    """
    if not notes:
        raise ValueError("notes should not be empty")
    for n in notes:
        if isinstance(n, Note):
            assert n.amp > 0 and n.pitch > 0
        else:
            assert isinstance(n, Rest)
    if len(notes) == 1:
        return notes
    last_note = notes[0]
    groups = []
    accum = [last_note]
    for note in notes[1:]:
        if isinstance(note, Rest) and isinstance(last_note, Rest):
            accum.append(note)
            last_note = note
        elif note.isrest() != last_note.isrest():
            groups.append(accum)
            last_note = note
            accum = [last_note]
        else:
            assert isinstance(note, Note) and isinstance(last_note, Note)
            if note.pitch == last_note.pitch and note.amp == last_note.amp and last_note.tied:
                accum.append(note)
                last_note = note
            else:
                groups.append(accum)
                last_note = note
                accum = [last_note]
    groups.append(accum)
    outnotes = []
    for group in groups:
        if len(group) in (5, 9, 13):
            note0 = group[0].clone(end=group[:-2].end, tied=True)
            note1 = group[-1].clone()
            outnotes.append(note0)
            outnotes.append(note1)
        else:
            joint_note = group[0].clone(end=group[-1].end)
            outnotes.append(joint_note)
    return outnotes


def divide_long_notes(notes: t.List[Event], pulsedur: Fraction, mindur=1e-8
                      ) -> t.List[Event]:
    """
    Divide las notas en fracciones no mayores que un pulso, uniendo
    las notas largas con ligaduras
    """
    assert all(isinstance(n, Event) for n in notes)
    newnotes = []
    for note in notes:
        fragments = note.break_at_pulse(pulsedur, mindur=mindur)
        newnotes.extend(fragments)
    rendered_events_make_contiguous(newnotes, pulsedur=pulsedur)
    assert almosteq(newnotes[0].start, notes[0].start, mindur+1e-12), (newnotes[0].start, notes[0].start)
    assert almosteq(newnotes[-1].end, notes[-1].end, mindur+1e-12)
    assert not hasholes(newnotes)
    return newnotes


def has_short_notes(notes: t.List[Event], mindur=R(1, 64)) -> bool:
    maxnotes = 5
    shortnotes = 0
    for n in notes:
        if n.dur < mindur:
            logger.debug(f"has_short_notes: small duration: {n}")
            shortnotes += 1
            if shortnotes >= maxnotes:
                break
    return shortnotes > 0


def cut_overlap(notes: t.List[Event]) -> t.List[Event]:
    """
    New notes cut previous ones.

    Returns: the new notes
    """
    assert all(isinstance(n, Event) for n in notes)
    for n0, n1 in pairwise(notes):
        if n0.start >= n1.start:
            raise AssertionError(f"Notes not sorted: {n0}, {n1}")
    if not hasoverlap(notes):
        return notes
    newnotes = []
    for note0, note1 in pairwise(notes):
        if note0.end<=note1.start:
            newnotes.append(note0)
        else:
            overlap = note0.end-note1.start
            if overlap>1e-12:
                logger.debug("cutting overlap of: {0:f}".format(float(overlap)))
            newdur = note1.start-note0.end
            if newdur<=0:
                raise ValueError(f"negative overlap: {note0} {note1}")
            newnotes.append(note0.clone(dur=newdur))
    newnotes.append(notes[-1])
    durnotes = sum(n.dur for n in notes)
    durnewnotes = sum(n.dur for n in newnotes)
    assert almosteq(durnotes, durnewnotes, 1e-8), \
        f"original notes dur: {float(durnotes)}, new notes: {float(durnewnotes)}"
    assert not hasoverlap(newnotes)
    return newnotes


def hasoverlap(notes):
    """
    :type notes: list[Event]
    """
    if len(notes) < 2:
        return False
    return any(note0.end > note1.start for note0, note1 in pairwise(notes))


def hasholes(notes):
    """
    :type notes: list[Event]
    """
    assert all(isinstance(n, Event) for n in notes)
    return any(n0.end < n1.start for n0, n1 in pairwise(notes))


def fill_silences(notes:t.List[Event],
                  start: Fraction = None,
                  end: Fraction   = None,
                  mindur:t.Rat = 0.001) -> t.List[Event]:
    """ 
    Fill gaps between notes with silences (Note with pitch=amp=0)
    
    Returns: filled notes
    """
    start = start if start is not None else notes[0].start
    end = end if end is not None else notes[-1].end
    assert all(isinstance(n, Event) for n in notes)
    assert isinstance(start, Fraction)
    assert isinstance(end, Fraction)
    assert all(note.dur >= mindur for note in notes)
    assert issorted(notes, key=lambda n: n.start)
    if end is not None:
        assert notes[-1].end <= end

    newnotes = []  # t.List[Event]
    first_note_start = notes[0].start
    if first_note_start > start:
        # newnotes.append(Note.newrest(start, first_note_start - start))
        newnotes.append(Rest(start, first_note_start - start))
    for note0, note1 in pairwise(notes):
        gap = note1.start - note0.end
        assert gap >= 0   # make sure that there are no overlaps
        if gap == 0:
            newnotes.append(note0)
        elif gap < mindur:
            # the gap is too short for a silence, extend the note
            newnotes.append(note0.clone(dur=note0.dur+gap))
        else:
            # The gap is longer than the minimum duration for a silence, so add one
            newnotes.append(note0)
            newnotes.append(Rest(note0.end, gap))
    newnotes.append(notes[-1])
    last_note_end = notes[-1].end
    if last_note_end < end:
        # newnotes.append(Note.newrest(last_note_end, end-last_note_end))
        newnotes.append(Rest(last_note_end, end-last_note_end))
    assert len(newnotes) >= len(notes)
    assert all(note.dur >= mindur for note in newnotes)
    assert issorted(newnotes, key=lambda note: note.start)
    assert almosteq(newnotes[0].start, start) and almosteq(newnotes[-1].end, end), \
        "start:{0} end:{1} newnotes:{2}".format(start, float(end), newnotes)
    assert not hasoverlap(newnotes)
    for n0, n1 in pairwise(newnotes):
        assert almosteq(n0.end, n1.start), str((n0, n1))
    return newnotes


