from fractions import Fraction
from typing import (
    Union as U,
    Optional as Opt,
    Tuple as Tup, List,
    Sequence as Seq,
    Iterator as Iter,
    TypeVar,
    Dict,
    cast,
    NamedTuple,
    Any,
    Iterable as Gen,
)

Number = U[int, float, Fraction]
Rat = U[float, Fraction]


T = TypeVar("T")


def remove_type_imports():
    g = globals()
    for key in "_T Number U Opt Tup List Seq Iter TypeVar Dict cast NamedTuple Any Gen".split():
        del g[key]
    