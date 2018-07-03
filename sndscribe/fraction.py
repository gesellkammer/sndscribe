
from fractions import Fraction
from . import typehints as t


def R(*args, **kws) -> Fraction:
    return Fraction(*args, **kws).limit_denominator(9999999)


def asR(n: t.U[Fraction, int, float]) -> Fraction:
    if isinstance(n, Fraction):
        return n
    return R(n)


def isR(n) -> bool:
    return isinstance(n, Fraction)


def qrange(start:t.Number, stop:t.Number, step:t.Number) -> t.Gen[Fraction]:
    start = asR(start)
    stop = asR(stop)
    step = asR(step)
    while start<stop:
        yield start
        start+=step


def qintersection(u1, u2, v1, v2):
    # type: (Fraction, Fraction, Fraction, Fraction) -> t.Opt[t.Tup[Fraction, Fraction]]
    """
    return the intersection of (u1, u2) and (v1, v2) or None if no intersection
    """
    x0 = u1 if u1 > v1 else v1
    x1 = u2 if u2 < v2 else v2
    if x0 < x1:
        return x0, x1
    return None