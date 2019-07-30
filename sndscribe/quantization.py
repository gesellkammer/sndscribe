from __future__ import absolute_import
from emlib.lib import checktype

from emlib import lib
import bisect
from fractions import Fraction
from math import sqrt


from .config import RenderConfig
from .note import *
from .tools import *
from .typehints import *
from .dynamics import DynamicsCurve


class Result(NamedTuple):
    division: int
    penalty: float
    assigned_notes: List[Event]
    grid: List[Fraction]
    info: Opt[dict] = None

    def __repr__(self):
        gridstr = " ".join("{x.numerator}/{x.denominator}".format(x=x) for x in self.grid)
        gridstr = "[%s]" % gridstr
        L = [f"\n Division: {self.division}",
             f"   Total Penalty: {self.penalty:.1f}",
             f"   Notes: {self.assigned_notes}",
             f"   Grid : {gridstr}"
        ]
        if self.info:
            info = " ".join(f"{key}={float(value):.1f}" for key, value in self.info.items())
            L.append("  Info: %s" % info)
        return "\n".join(L)


def calculate_weight_note(note:Event, config:RenderConfig) -> float:
    if not isinstance(note, Note):
        amp = linearamp(db2amp(config['silence_db']))
        transient = 0.0
    else:
        amp = linearamp(note.amp)
        transient = note.transient
    return weightedsum(
        (amp*float(note.dur), config['weight_ampdur']),
        (transient,           config['weight_transient'])
    )


def calculate_note_score(note:Event, time_shift:float, config:RenderConfig) -> float:
    if isinstance(note, Rest):
        return 0
    noteweight, timingweight = config['weight_note'], config['weight_time_accuracy']
    # TODO: check that time_shift is always 0..1 independent of pulse
    score = weightedsum(
        (calculate_weight_note(note, config), noteweight),
        (1 - time_shift, timingweight)
    )
    return score


def calculate_average_note(notes:List[Note], config:RenderConfig) -> Note:
    """Average notes into one note, discarding silences"""
    notes = [n for n in notes if not n.isrest()]
    notes.sort(key=lambda n:n.start)
    weights = [calculate_weight_note(n, config=config) for n in notes]
    pitches = [n.pitch for n in notes]
    amps = [n.amp for n in notes]
    pitch = iweightedsum(zip(pitches, weights))
    amp = iweightedsum(zip(amps, weights))
    start, end = notes[0].start, notes[-1].end
    assert pitch > 0
    assert amp > 0
    return Note(pitch, amp=amp, start=start, end=end, color="avg")


def pulse_join_slots(events:List[Event]) -> List[Event]:
    pulsedur = events[-1].end - events[0].start
    assert all(isinstance(event, Event) for event in events)
    assert almosteq(sum(event.dur for event in events), pulsedur)
    assert all(e0.end == e1.start for e0, e1 in pairwise(events))
    newevents = []  # type: List[Event]
    openevent = events[0]
    for event in events[1:]:
        if isinstance(event, Rest):
            if openevent.isrest():
                # Rest | Rest
                openevent = openevent.clone(end=event.end)
            else:
                # Note | Rest
                assert isinstance(openevent, Note)
                newevents.append(openevent)
                openevent = event
        elif isinstance(event, Note):
            if isinstance(openevent, Rest):
                # Note | Rest
                newevents.append(openevent)
                openevent = event
            elif isinstance(openevent, Note):
                # Note | Note
                if openevent.pitch == event.pitch and almosteq(openevent.amp, event.amp):
                    openevent = openevent.clone(end=event.end)
                else:
                    newevents.append(openevent)
                    openevent = event
            else:
                raise TypeError("expecting either a Rest or a Note")
    newevents.append(openevent)
    assert newevents[0].start == events[0].start
    assert newevents[-1].end == events[-1].end
    assert almosteq(sum(n.dur for n in newevents), pulsedur)
    assert all(n0.end == n1.start for n0, n1 in pairwise(newevents))
    return newevents


def find_best_fit_for_slot(notes_in_slot:List[Event], slot_start:Fraction, config:RenderConfig) -> Note:
    bestscore, bestnote = max((calculate_note_score(note, abs(note.start-slot_start), config), note)
                              for note in notes_in_slot)
    assert isinstance(bestnote, Note)
    # assert not bestnote.isrest()
    return bestnote


def calculate_penalty_timeshift(assigned_notes:Seq[Opt[Event]],
                                grid:List[Fraction],
                                config:RenderConfig) -> float:
    """
    assigned_notes is a list with ONE assigned note per slot
    The assigned note is the real note, so its beginning does not
    need to fit the grid. It is this distance that we measure here
    """
    assert all(isinstance(note, Event) or note is None for note in assigned_notes)
    accum = 0.0
    for note, slotstart in zip(assigned_notes, grid):
        if note is None:
            continue
        if isinstance(note, Note):
            amp = note.amp
        else:
            amp = db2amp(config['silence_db'])
        noteweight = linearamp(amp)
        timeshift = abs(note.start - slotstart)
        penalty = timeshift * noteweight
        accum += penalty
    return accum


def jointnote_lost_info(avgnote:Note,
                        notes:List[Event],
                        dyncurve:DynamicsCurve,
                        normalized=False) -> float:
    """
    Calculates how much information was lost by averaging the notes
    into avgnote
    """
    assert isinstance(notes, (tuple, list)) and all(isinstance(note, Event) for note in notes)

    def amp2index(amp):
        return dyncurve.dyn2index(dyncurve.amp2dyn(amp))

    norests = [note for note in notes if isinstance(note, Note)]  # type: List[Note]
    assert all(note.amp > 0 for note in norests)
    lostmidi = sum(abs(avgnote.pitch - note.pitch) for note in norests)  # type: float
    lostamp = sum(abs(amp2index(avgnote.amp) - amp2index(note.amp)) for note in norests)  # type: float
    midiweight, ampweigth = 4, 1
    info = weightedsum((lostmidi, midiweight),
                       (lostamp, ampweigth))
    if not normalized:
        return info
    maxmidi = 2 * len(notes)
    maxamp = 1 * len(notes)
    maxinfo = weightedsum((maxmidi, midiweight),
                          (maxamp, ampweigth))
    return info/maxinfo


class Division(NamedTuple):
    division: int
    assignednotes: List[Event]
    info: List[Result]


def find_best_division(notes: List[Event],
                       possible_divs: List[int], 
                       pulsestart: Fraction,
                       pulsedur: Fraction, 
                       config: RenderConfig, 
                       dyncurve: DynamicsCurve) -> Division:
    """
    Return the division and the assigned notes

    assigned_notes must be a list of notes which fill the given pulse completely and without any gaps
    silences should be created (amp=0) for any gap between notes.
    """
    # dyncurve = dyncurve or get_default_dynamicscurve()
    join_small = config['join_small_notes']
    debug = config['debug']
    results = []
    # maxcomplexity = max(config['divcomplexity'].values())
    complexitydict = config['divcomplexity']
    maxcomplexity = max(complexitydict[div] for div in possible_divs)
    for division in possible_divs:
        dt = pulsedur * R(1, division)  # type: Fraction
        start = asR(pulsestart)
        grid = [asR(start + dt*i).limit_denominator(1000) for i in range(division)]
        notes_in_slots = [[] for _ in range(division)]    # type: List[List[Event]]
        indexes_in_slots = [[] for _ in range(division)]  # type: List[List[int]]
        for inote, note in enumerate(notes):
            # slot = bisect.bisect(grid, note.start) - 1
            slot = min(lib.nearest_index(note.start, grid), len(grid)-1)
            notes_in_slots[slot].append(note)
            indexes_in_slots[slot].append(inote)
        # Assigned notes is a list of Events|None
        assigned_notes = [None] * division      # type: List[Opt[Event]]
        unassigned_notes = [None] * division    # type: List[Opt[List[Event]]]
        for islot, notes_in_slot in enumerate(notes_in_slots):
            if len(notes_in_slot) == 0:
                pass
            elif len(notes_in_slot) == 1:  # only one note, assign it
                assigned_notes[islot] = notes_in_slot[0]
            else:
                # more than one note in slot
                if not join_small:
                    # get the best one
                    best_note = find_best_fit_for_slot(notes_in_slot, pulsestart, config)
                    assigned_notes[islot] = best_note
                    unassigned_notes[islot] = [note for note in notes_in_slot if note is not best_note]
                else:
                    # join notes in slot into one note
                    note = calculate_average_note(notes_in_slot, config)
                    assigned_notes[islot] = note
                    unassigned_notes[islot] = notes_in_slot
        divcomplexity = config['divcomplexity'].get(division, maxcomplexity * 2)

        timeshift = calculate_penalty_timeshift(assigned_notes, grid, config)
        predicted_durs = []  # type: List[Opt[Fraction]]
        for note in assigned_notes:
            if note is None:
                predicted_durs.append(None)
            else:
                dur = max(1, int(note.dur / dt + 0.5)) * dt  # snap to a grid of dt, with a minimum of dt
                predicted_durs.append(dur)
        predicted_incorrect_durs = []  # type: List[Fraction]
        for predicted_dur, note in zip(predicted_durs, assigned_notes):
            if predicted_durs is not None:
                durdiff = abs(predicted_dur - note.dur) if note is not None else 0
                predicted_incorrect_durs.append(durdiff)
        # unassigned_notes: similar to assigned_notes, but holds either None, or the
        # unassigned notes for each slot
        if join_small:
            timeshift_weight = config['penalty_merged_notes']
            # the thing to measure is how much information has been lost
            # by joining the notes in one slot into one note
            total_lost_info = 0.0
            for jointnote, original_notes in zip(assigned_notes, unassigned_notes):
                if original_notes is not None:
                    total_lost_info += jointnote_lost_info(jointnote, original_notes, dyncurve)
            leftout = total_lost_info * config['penalty_leftout']
        else:
            timeshift_weight = config['penalty_timeshift']
            assert all(checktype(slot, [Event]) or slot is None for slot in unassigned_notes)
            unassigned_flat = sum((notes for notes in unassigned_notes if notes is not None), [])
            unassigned_notes_weights = [calculate_weight_note(n, config) for n in unassigned_flat]
            leftout = sum(unassigned_notes_weights) * config['penalty_leftout']
            # leftout = sum(map(calculate_weight_note, unassigned_flat)) * config['penalty_leftout']

        penalties = {
            # 'leftout': sum(calculate_weight_note(note) for note in unassigned_notes) * quantization.penalty_leftout,
            'leftout': leftout,
            'complexity': divcomplexity/maxcomplexity * config['penalty_complexity'],
            'notelength': float(sum(predicted_incorrect_durs)) * config['penalty_incorrect_duration'],
            'timeshift': timeshift * timeshift_weight
        }

        def calculate_penalty(penalties, method='euclidean') -> float:
            if method == 'euclidean':
                values = list(penalties.values())
                return sqrt(sum(value**2 for value in values))
            else:
                raise NotImplementedError()

        totalpenalty = calculate_penalty(penalties)
        info = lib.dictmerge(penalties, {'total': totalpenalty, 'division': division})
        results.append(Result(division, totalpenalty, assigned_notes, grid, info))
    
    # -- Get result with lowest penalty
    results.sort(key=lambda result: result.penalty)
    best_result = results[0]
    best_division = best_result.division
    assigned_notes = best_result.assigned_notes
    grid = best_result.grid
    # Take care of silences and long notes. 
    # * If a slot has a None it means that it is either a silence or the rest of a previous note. 
    # * If a note extends over the next slot and the next slot is empty (None) the slot becomes 
    #   an extension of the previous note if it would cover at least half of the slot, 
    #   else it remains a silence.
    events = calculate_events_for_slots(assigned_notes, grid, pulsedur)
    assigned_notes = pulse_join_slots(events)
    if debug:
        print("\n--------------------- pulse start: %2.3f ------------------------" % pulsestart)
        print("best division:      ", best_division)
        print("assigned notes:     ", best_result.assigned_notes)
        print("notes and silences: ", assigned_notes)

    # ~~~~~~~~~~~ Post Conditions ~~~~~~~~~~
    possible_durs = [R(i, best_division) for i in range(1, best_division + 1)]
    assert all(note.dur in possible_durs for note in assigned_notes)
    assert all(isinstance(note, Event) for note in assigned_notes)
    assert all(almosteq(n0.end, n1.start) for n0, n1 in pairwise(assigned_notes)), assigned_notes
    assert all(note.start in best_result.grid for note in assigned_notes)
    return Division(best_division, assigned_notes, results)


def calculate_events_for_slots(assigned_notes: List[Opt[Event]],
                               grid:List[Fraction],
                               pulsedur:U[Fraction, int]) -> List[Event]:
    """
    Calculates the events that go in each slot

    assigned_notes:
        a list of len=division, with one (Note | None) representing
        each slot. A None in a slot means that either it is empty (a Rest)
        or the previous note continues on this slot (depending on the duration)
    grid:
        a list of len=division, with the time of each slot. grid[0] is the beginning of
        ohe Pulse, grid[-1] + slotdur is the end of the pulse
        (slotdur = grid[1] - grid[0], we assume that all slots are the same duration)
    pulsedur:
        the duration of this pulse


    """
    division: int = len(grid)
    slot_dur: Fraction = R(pulsedur, division)

    assert len(assigned_notes) == len(grid)
    assert all(isinstance(note, Event) or note is None for note in assigned_notes), assigned_notes
    assert all(isR(t) for t in grid)
    # assert all(t0 <= n.start < t0+slot_dur for n, t0 in zip(assigned_notes, grid) if n is not None)

    # events = list(range(division))
    events = [None] * division   # type: List[Opt[Event]]
    last_note = None             # type: Event
    onlynotes = [n for n in assigned_notes if n is not None]  # type: List[Event]
    if len(onlynotes) > 1:
        assert all(n0.end <= n1.start for n0, n1 in pairwise(onlynotes))
    for i in range(len(assigned_notes)):
        note = assigned_notes[i]   # type: Event
        slot_start = grid[i]       # type: Fraction
        if note is not None:   # ---- Assigned slot
            # fitted_note = Note(note).override(start=slot_start, dur=slot_dur)
            events[i] = note.clone(start=slot_start, dur=slot_dur)
            last_note = note
        elif last_note is None or last_note.isrest():
            last_note = Rest(slot_start, slot_dur)
            events[i] = last_note
        elif i > 0 and last_note is not None:
            # The previous note is a Note. See if it is longer than its slot
            if last_note.end > slot_start:
                if (last_note.end - slot_start)/slot_dur < 0.5:
                    # It does not justify an extention, we insert a silence and
                    # we end the open note
                    last_note = Rest(slot_start, slot_dur)
                    events[i] = last_note
                else:
                    if isinstance(last_note, Rest):
                        extended = Rest(slot_start, slot_dur)
                    elif isinstance(last_note, Note):
                        tied = last_note.end > slot_start + slot_dur
                        extended = last_note.clone(start=slot_start, dur=slot_dur, tied=tied)
                    else:
                        raise TypeError(f"expected Note or Rest, got {last_note}")
                    # logger.debug("extended note: %s" % extended)
                    events[i] = extended
                    # we don't update last_note, since this is only an extension
            else:
                assert last_note is not None
                # the last note has already finished, so this is a silence
                last_note = Rest(slot_start, slot_dur)
                events[i] = last_note
        else:
            raise AssertionError("wtf!!??")
    assert len(events) == len(assigned_notes)
    assert all(isinstance(event, Event) for event in events), \
        "??? "+str([ev for ev in events if not isinstance(ev, Event)])
    assert all(almosteq(event.dur, slot_dur) for event in events)
    assert all(almosteq(event.start, slotstart) for event, slotstart in zip(events, grid))
    assert all(almosteq(e0.end, e1.start) for e0, e1 in pairwise(events))
    return events
