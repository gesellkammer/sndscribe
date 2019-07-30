import math
import collections
import numpy
from emlib.pitchtools import m2f, n2m, amp2db, db2amp
from emlib import lib
from emlib.iterlib import pairwise

from .definitions import *
from . import lilytools
from .fraction import *
from . import typehints as t


def logfreqs(delta_midi=0.5) -> numpy.ndarray:
    """
    return an array of frequencies corresponding to the pitches of all notes of the audible spectrum
    delta_midi indicates the definition of the pitch matrix: 1=semitones, 0.5, quartertones, etc.
    """
    return numpy.fromiter((m2f(x) for x in numpy.arange(0, 139, delta_midi)), dtype=numpy.float64)


def semitones():
    """
    generate an array of the frequencies of all semitones
    """
    return logfreqs(1)


def quartertones():
    return logfreqs(0.5)


def piano_freqs(start='A0', stop='C8') -> numpy.ndarray:
    """
    generate an array of the frequencies representing all the piano keys
    """
    keys = range(int(n2m(start)), int(n2m(stop))+1)
    return numpy.fromiter((m2f(key) for key in keys), dtype=numpy.float64)


def grid_floor(x, dx):
    return x - (x % dx)


def grid_round(x, dx):
    dif = x % dx
    if dif > (dx/2):
        return x + dif
    return x - dif


def linearamp(amp:float, dbfloor:float=-100.0) -> float:
    """
    Converts an amplitude to a value between 0-1 following the dB scale

    dbfloor: min. amplitude as db
    """
    dbrange = abs(dbfloor)
    posdb = max(dbfloor, amp2db(amp)) + dbrange
    return posdb / dbrange


def points_to_millimeters(points:float) -> float:
    return points * 7 / 20.


def issequence(obj) -> bool:
    return isinstance(obj, collections.Sequence) and not isinstance(obj, str)


# def snap_to_guides(guides, elements):
#     """
#     Snap each element to one of the guides.
#
#     Returns
#     (snapped, diffs)
#
#     >>> snap_to_guides([1, 2, 3], [1.4, 4])
#     ([1, 3], [0.4, 1])
#
#     The difference values will be positive if the element is
#     snapped to the right, negative if it is snapped to the left
#
#     diffs = [x - snapped for x, snapped in zip(X, snap_to_guides(guides, X))]
#     """
#     assert issequence(guides) and issequence(elements)
#     snapped, diffs = [], []
#     for n in elements:
#         snapped_element = nearest_element(n, guides)
#         snapped.append(snapped_element)
#         diffs.append(n - snapped_element)
#     return snapped, diffs


def previous_power_of_two(x:t.Number) -> int:
    """
    Return the power of two either equal or lower to x

    >>> previous_power_of_two(8)
    8
    >>> previous_power_of_two(7)
    4
    """
    return 2 ** int(math.log(float(x), 2))


def array_fitbetween(A:numpy.ndarray, x0:float, x1:float) -> None:
    """
    fit A between x0 and x1, in-place
    
    (A-A0)/A1 * (x1-x0) + x0
    """
    assert A.dtype in (float, int)
    assert isinstance(x0, (int, float)), "x0 should be a number but it is {}".format(x0.__class__)
    assert isinstance(x1, (int, float)), "x1 should be a number but it is {}".format(x1.__class__)
    A0, A1 = A[0], A[-1]
    A -= A0
    A /= A1 - A0
    A *= (x1-x0)
    A += x0
    eps = 1e-12
    assert abs(A[0] - x0) < eps and abs(A[-1] - x1) < eps, \
        "array: %s, x0, x1: %.3f, %.3f" % (str(A), x0, x1)
    assert A.dtype == float


def almosteq(a:t.Number, b:t.Number, eps=1e-12) -> bool:
    assert isinstance(a, (int, float, Fraction)) and isinstance(b, (int, float, Fraction))
    return abs(a-b) < eps


def time2bpm(timeperbeat:Fraction) -> Fraction:
    assert isinstance(timeperbeat, Fraction)
    out = R(60) / asR(timeperbeat)
    assert isinstance(out, Fraction)
    return out


def bpm2beattime(bpm:t.U[int, Fraction]) -> Fraction:
    """
    return value in seconds
    """
    assert isinstance(bpm, (int, Fraction))
    out = asR(60) / asR(bpm)
    assert isinstance(out, Fraction)
    return out


def timesig2pulsedur(timesig:t.Tup[int, int], tempo:t.U[int, Fraction]) -> Fraction:
    """
    * timesig: (n, d)
    * tempo: the beats per minute. Uses the denominator of the time-signature
             as a reference, so if the timesig=(3, 8), the tempo refers to
             an 1/8 note

    Returns: the duration of a pulse (in seconds)
    """
    assert istimesig(timesig)
    assert isinstance(tempo, (int, Fraction))

    den = R(timesig[1])
    tempo = asR(tempo)
    out = (60/tempo) * (4/den)
    assert isinstance(out, Fraction)
    return out


def istimesig(timesig:t.Tup[int, int]) -> bool:
    isa = isinstance
    return isa(timesig, tuple) and isa(timesig[0], int) and isa(timesig[1], int)


def timesig2measuredur(timesig:t.Tup[int, int], tempo:t.U[int, Fraction]) -> Fraction:
    """
    
    * timesig: (num, den)
    * tempo: the beats per minute. Uses the denominator of the time-signature
             as a reference, so if the timesig=(3, 8), the tempo refers to
             an 1/8 note
    """
    assert lib.checktype(timesig, (int, int))
    assert isinstance(tempo, (int, Fraction))
    out = timesig2pulsedur(timesig, tempo) * timesig[0]
    assert isinstance(out, Fraction)
    return out


def break_dur_at_pulses(start:Fraction, dur:Fraction, pulsedur:Fraction) -> t.List[t.Tup[Fraction, Fraction]]:
    """
    Break a duration into multiple durations equal or shorter
    than `pulse`, at `pulse`

    Returns a list of (start, end) positions
    """
    # TODO: take pulse offset into account
    assert dur > 1e-8, "VERY small duration!"
    pulsedur = asR(pulsedur)
    t = asR(start)
    endt = asR(t + dur)
    times = []
    cant_elem = (pulseceil(endt, pulsedur) - pulseint(t, pulsedur)) / pulsedur
    t_offs = t % pulsedur
    endt_offs = endt % pulsedur
    if cant_elem < 2:
        times.append((t, endt))
    else:
        # quitar los extremos
        if t_offs:
            times.append((t, pulseceil(t, pulsedur)))
            t = pulseceil(t, pulsedur)
        if endt_offs:
            times.append((pulseint(endt, pulsedur), endt))
            endt = pulseint(endt, pulsedur)
        cant_elem = int((endt + pulsedur - t) / pulsedur)
        for i in range(cant_elem - 1):
            times.append((t + pulsedur * i, t + pulsedur * (i + 1)))
    times.sort()
    assert almosteq(dur, sum(t1-t0 for t0, t1 in times))
    assert len(times) >= 1, "wtf!! start:{start}, dur:{dur}, pulse:{pulse}".format(
        start=float(start), dur=float(dur), pulse=pulsedur)
    assert almosteq(times[0][0], start) and almosteq(times[-1][1], start+dur)
    assert lib.checktype(times, [(Fraction, Fraction)])
    return times


def next_pulse(time:t.Rat, pulsedur:Fraction) -> Fraction:
    return R(math.ceil(asR(time)/pulsedur))


def prev_pulse(time:t.Rat, pulsedur:Fraction) -> Fraction:
    time = asR(time)
    return time - (time % pulsedur)


def pulseceil(n:t.Number, pulsedur:Fraction) -> Fraction:
    """
    like ceil(x), but in reference to a pulse duration
    
    returns a multiple of pulsedur equal or higher than n
    """
    n = asR(n)
    pulsedur = asR(pulsedur)
    ent = int(n/pulsedur)
    tmp0 = ent * pulsedur
    offs = n % pulsedur
    tmp1 = int(math.ceil(float(offs)))
    tmp2 = tmp1 * pulsedur
    tmp3 = tmp0 + tmp2
    assert isR(tmp3)
    return tmp3
   

def pulseint(n:t.Number, pulsedur:Fraction) -> Fraction:
    """
    Like int(x), but in reference to a pulse duration
    
    :param n: the number to be rounded to a multiple of pulsedur
    :param pulsedur: the pulse duration
    :return: a multiple of pulsedur equal or lower than n
    """
    pulsedur = asR(pulsedur)
    return int(n/pulsedur) * pulsedur


def weightedsum(*values:t.Tup[t.Number, t.Number]) -> float:
    """
    Each `value` is a tuple (value, weight)

    Returns the weighted sum of these values
    """
    assert all(isinstance(value, tuple) for value in values)
    weights = [w for v, w in values]
    return sum(v*w for v, w in values)/sum(weights)


def iweightedsum(seq:t.Iter[t.Tup[t.Number, t.Number]]) -> float:
    """
    seq: an iterator of (value, weight)
    """
    totalweight = 0
    totalvalue = 0
    for v, w in seq:
        totalvalue += v*w
        totalweight += w
    return totalvalue / totalweight


def musicxml2pdf(xmlfile:str, outfile:str=None, backend='lilypond') -> str:
    """
    Convert an xml file to pdf, return the path of the pdf file
    
    :param xmlfile: the path to the musicxml file 
    :param outfile: the path of the pdf file. If not given, the xmlfile with a pdf extension is used
    :param backend: possible backends: lilypond, musescore
    :return: the path of the generated pdf file 
    """
    backend = backend.lower()
    if backend in ('lily', 'lilypond'):
        lilyfile = lilytools.xml2lily(xmlfile)
        if not os.path.exists(lilyfile):
            raise RuntimeError("could not convert to lilypond (file does note exist: %s)" % lilyfile)
        pdf = lilytools.lily2pdf(lilyfile, outfile)
        if not pdf:
            raise RuntimeError("could not convert lilypond to pdf")
        return pdf
    elif backend in ('muse', 'musescore'):
        raise NotImplementedError("Ooops...")
    else:
        raise NotImplementedError("Backend not supported")


def xor(b1:bool, b2:bool) -> bool:
    return b1 != b2


def check_sorted(objs, key=None):
    if key is None:
        key = lambda x:x
    for x0, x1 in pairwise(objs):
        if key(x0) >= key(x1):
            raise ValueError(f"Not sorted: {x0} ({key(x0)} >= {x1} ({key(x1)})")


def parse_timesig(timesig: str) -> Opt[Tup[int, int]]:
    """
    Convert a string timesignature of the form "5/8" in a tuple (5, 8).

    Args:
        timesig: a time signature as string, like "3/4"

    Returns:
        A tuple (num, den) representing the given timesignature, or
        None if the time signature can't be parsed
    """
    try:
        numstr, denstr = timesig.split("/")
        return int(numstr), int(denstr)
    except:
        return None


def as_timesig_tuple(timesig) -> Tup[int, int]:
    if isinstance(timesig, tuple):
        if len(timesig) == 2:
            return timesig
        raise ValueError(f"Expected a tuple(num, den) but got {timesig}")
    elif isinstance(timesig, str):
        timesigtup = parse_timesig(timesig)
        if timesigtup is None:
            raise ValueError(f"Can't convert string {timesig} to a time signature")
        return timesigtup
    else:
        raise TypeError(f"timesig should be a tuple like (3, 4) or a string '3/4' but got {timesig}")