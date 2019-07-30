import bisect
from .definitions import *
from emlib.lib import returns_tuple, roundres
from emlib.pitchtools import parse_midinote
from .tools import points_to_millimeters, bpm2beattime
from .typehints import *
from .tools import Fraction


def micro_to_xmlaccidental(alter:float, resolution=0.25) -> str:
    """
    alter: XML `alter` element, a float indicating the midinote 
           variation of the indicated step
           0   = no variation
           0.5 = quarter note up
    """
    # TODO: implement all accidentals (not only sharps)
    if resolution not in (0.25, 0.5, 1):
        raise ValueError(f"Resolution must be 0.25, 0.5 or 1, got {resolution}")
    alter = roundres(alter, resolution)
    return MUSICXML_ACCIDENTALS[alter]


def bw_to_xmlnotehead(bw:float) -> XmlNotehead:
    """
    bw is a value between 0-1, where 0 is normal
    """
    index = int(round(bw_to_noteheadindex(bw)))
    assert 0 <= index < len(MUSICXML_NOTEHEADS)
    return MUSICXML_NOTEHEADS[index]


def bw_to_noteheadindex(bw:float) -> float:
    return bw * (len(MUSICXML_NOTEHEADS) - 1)


def staffsize_to_xmlscaling(staffsize_in_points:float) -> Tup[float, int]:
    """
    Given a staff size in points (20 points = 7 mm)
    returns a tuple (millimeters, tenths) according
    to the scaling system used in musicxml

    musicxml seems to fix tenths=40
    """
    staffsize_in_mm = points_to_millimeters(staffsize_in_points)
    return staffsize_in_mm, MUSICXML_TENTHS


def transient2articulation(transient):
    """
    valid musicXML articulations:
        accent, strong-accent, staccato, tenuto, 
        detached-legato, staccatissimo, spiccato
    """
    x = ((0, None), (1, 'accent'))
    i = bisect.bisect(x, (transient, 'zzzzzzzzz')) - 1
    i = max(i, 0)
    if i > len(x) - 1:
        i = -1
    return x[i][1]


@returns_tuple("pitchclass octave microtone")
def pitch2note(pitch:float) -> Tup[int, int, float]:
    """
    pitch: a midi pitch (can be fractional to express microtonal deviations)

    Returns: pitchclass, octave, microtone
    """
    midinote = int(pitch)
    micro = pitch - midinote
    pitchclass = midinote % 12
    octave = int(midinote / 12.) - 1
    return pitchclass, octave, micro


def pitch2xml(pitch:float, allowflats=True) -> Tup[int, int, int]:
    """
    Convert a midinote to its musicxml representation

    * pitch: a midi pitch (can be fractional to express microtonal deviations)
    * allowflats: if False, only sharps are returned

    Returns: step, alteration, octave
    """
    pitchclass, octave, micro = pitch2note(pitch)
    step, alter = NOTE_DICT[pitchclass]
    alter = float(alter) + micro
    alter = int(alter * 4) / 4.0
    if alter == int(alter):
        alter = int(alter)
    if allowflats and alter >= 1.5:
        alter = -2 + alter
        step, _ = NOTE_DICT[(pitchclass + 1) % 12]
        if pitchclass == 12:
            octave += 1
    return step, alter, octave


def dur2xml(seconds:U[Number, Fraction], tempo:Number=60, divisions_per_quarter:int=MUSICXML_DIVISIONS) -> int:
    pulsedur = bpm2beattime(tempo)
    return int((seconds / pulsedur) * divisions_per_quarter)


def dyn2lilysize(dyn:int) -> int:
    return DYNAMIC_TO_RELATIVESIZE[dyn]


def dyn2fontsize(dyn:int, css=False) -> int:
    if css:
        raise ValueError(
            "this is not supported by the conversion stage, use css=False")
    return DYNAMIC_TO_CSSFONTSIZE[dyn] if css else DYNAMIC_TO_FONTSIZE[dyn]


class LayoutUnitConverter:

    def __init__(self, millimeters, tenths=MUSICXML_TENTHS):
        self.millimeters = millimeters
        self.tenths = tenths

    def to_tenths(self, mm:float) -> int:
        # tenths should always be an integer
        return int(mm / self.millimeters * self.tenths)

    def to_millimeters(self, tenths:int) -> float:
        # millimeters are floats
        return tenths / float(self.tenths) * self.millimeters

    @classmethod
    def from_staffsize(cls, staffsize_in_points):
        staffsize_in_mm = points_to_millimeters(staffsize_in_points)
        return cls(staffsize_in_mm)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def lily_clef(clef: str) -> str:
    lily = {
        'G':'treble',
        'F':'bass',
        'G8':'treble^8',
        'F8':'bass_8'
    }.get(clef)
    if clef is None:
        raise KeyError("clef {clef} unknown, should be one of F, G, F8, G8")
    return lily


def lily_pitch(midinote: float, microtone=True, force_accidental=False) -> str:
    """
    midinote: a midinote to convert to a lilypond pitch

    """
    if microtone:
        return _lilypitch_microtone(midinote, force_accidental=force_accidental)
    raise NotADirectoryError("only microtonal pitch implemented in lilypond")


_micro2suffix = {
     0    : "",
     0.25 : "iq",
     0.5  : "ih",
    -0.25 : "eq",
    -0.5  : "eh"
}


def lily_duration(notetype: str, numdots: int) -> str:
    """

    Example


    >>> lilydur("eigth", 2)
    "8.."

    >>> lilydur("16th", 1)
    "16."

    :param notetype: as returned by `get_notated_duration`, one of
        'eighth', 'quarter', etc.
    :param numdots: number of dots
    :return: the duration in lilypond format
    """
    numericdur = NOTETYPE_TO_NUMERIC_DURATION[notetype]
    if numdots == 0:
        return str(numericdur)
    return str(numericdur)+'.'*numdots


def _lilypitch_microtone(pitch: float, force_accidental=False) -> str:
    pitchidx, micro, octave, pitchname = parse_midinote(pitch)
    root = pitchname[0].lower()
    if len(pitchname) == 2:
        chromatic_accidental = pitchname[1]
        if chromatic_accidental == "#":
            root += "is"
        else:
            root += "es"
    snappedmicro = roundres(micro, 0.25)
    suffix = _micro2suffix.get(snappedmicro)
    if suffix is None:
        raise ValueError(f"Could not parse alteration: {micro} (rounded to {snappedmicro})")
    lilyoct = LILYPOND_ABSOLUTE_OCTAVE[octave]
    force = "!" if force_accidental else ""
    return f"{root}{suffix}{lilyoct}{force}"


def typestr2numbeams(typestr: str) -> int:
    numbeams = {
        'eighth': 1,
        'quarter': 0,
        'half': 0,
        '16th': 2,
        '32nd': 3,
        '64th': 4
    }.get(typestr)
    if numbeams is None:
        raise KeyError(typestr)
    return numbeams


