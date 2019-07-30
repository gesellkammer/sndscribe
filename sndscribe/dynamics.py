#  Functions to convert between dB and musical dynamics
#  also makes a representation of the amplitude in terms of musical dynamics

from bisect import bisect as _bisect
import bpf4 as _bpf
from bpf4 import BpfInterface
from emlib.pitchtools import db2amp, amp2db
from . import typehints as t

_DYNAMICS = ['pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff']


class DyncurveDescr(t.NamedTuple):
    shape: str          # a str describing the interpolation, like "expon(2.5)"
    mindb: float        # the lowest amplitude, in dB
    maxdb: float        # the highest amplitude, in dB
    dynamics: t.List[str]  # a list of dynamics, like ["pppp", "ppp", "pp", ..., "fff"]


def _aslist(obj) -> t.List:
    if isinstance(obj, list):
        return obj
    return list(obj)


class DynamicsCurve(object):
    def __init__(self, bpf:BpfInterface, dynamics:t.List[str]=None) -> None:
        """
        shape: a bpf mapping a dynamic between 0-1 to amplitude (0-1)
        dynamics: a list of possible dynamics, or None to use the default

        NB: see .fromdescr

        Example: define a curve with an amplitude floor of -80 dB

        c = bpf.linear(0, -80, 1, 0).db2amp()
        d = DynamicsCurve(c)
        """
        self.dynamics = _aslist(dynamics) if dynamics else _DYNAMICS   # type: t.List[str]
        fitbpf = bpf.fit_between(0, len(self.dynamics)-1)              # type: BpfInterface
        amps2dyns, dyns2amps = _create_dynamics_mapping(fitbpf, dynamics=self.dynamics)
        self._amps2dyns:t.List[t.Tup[float, str]] = amps2dyns
        self._dyns2amps:t.Dict[str, float]        = dyns2amps
        self._amps = [amp for amp, dyn in amps2dyns]
        assert len(self._amps2dyns) == len(self.dynamics)

    @classmethod
    def getdefault(cls) -> 'DynamicsCurve':
        shape = create_shape("expon(4.0)", -80, 0)
        return cls(shape, _DYNAMICS)

    @classmethod
    def fromdescr(cls, shape:str, mindb:int=-90, maxdb:int=0, dynamics:t.List[str]=None) -> 'DynamicsCurve':
        """
        shape: the shape of the mapping ('linear', 'expon(2)', etc)
        mindb, maxdb: db value of minimum and maximum amplitude
        dynamics: the list of possible dynamics, ordered from soft to loud

        Example:

        DynamicsCurve.fromdescr('expon(3)', mindb=-80, dynamics='ppp pp p mf f ff'.split())
        """
        curve = create_shape(shape, mindb, maxdb)
        return cls(curve, dynamics)

    def amp2dyn(self, amp:float, nearest=True) -> str:
        """
        Convert amplitude to a string representation of its corresponding
        musical dynamic as defined in DYNAMIC_TABLE
        
        amp: an amplitude 0-1
        nearest: if True, find the nearest dynamic (can round up), 
            otherwise, the next lower dynamic
        """
        amps = self._amps
        dyns = self.dynamics
        if amp < amps[0]:
            return dyns[0]
        if amp > amps[-1]:
            return dyns[-1]
        insert_point = _bisect(amps, amp)
        if not nearest:
            floor = max(0, amps[insert_point - 1])
            return dyns[floor]
        if insert_point == 0:
            return dyns[0]
        if insert_point >= len(amps):
            return dyns[-1]
        assert 1 <= insert_point < len(amps)
        amp0, dyn0 = amps[insert_point - 1], dyns[insert_point - 1]
        amp1, dyn1 = amps[insert_point], dyns[insert_point]
        db = amp2db(amp)
        dyn = dyn0 if abs(db-amp2db(amp0)) < abs(db-amp2db(amp1)) else dyn1
        assert isinstance(dyn, str)
        return dyn

    def dyn2amp(self, dyn:str) -> float:
        """
        convert a dynamic expressed as a string to its 
        corresponding amplitude
        """
        amp = self._dyns2amps.get(dyn.lower())
        if amp is None:
            raise ValueError("dynamic %s not known" % dyn)
        return amp

    def dyn2db(self, dyn:str) -> float:
        return amp2db(self.dyn2amp(dyn))

    def db2dyn(self, db:float) -> str:
        return self.amp2dyn(db2amp(db))

    def dyn2index(self, dyn:str) -> int:
        """
        Convert the given dynamic to an integer index into the
        possible dynamics given
        """
        try:
            return self.dynamics.index(dyn)
        except ValueError:
            raise ValueError("dynamic not defined, it should be one of %s" % str(self.dynamics))

    def index2dyn(self, idx:int) -> str:
        return self.dynamics[idx]

    def amp2index(self, amp:float) -> int:
        return self.dyn2index(self.amp2dyn(amp))

    def index2amp(self, idx:int) -> float:
        return self.dyn2amp(self.index2dyn(idx))

    def asdbs(self, step=1) -> t.List[float]:
        """
        Convert the dynamics defined in this curve to dBs
        """
        indices = range(0, len(self.dynamics), step)
        dbs = [self.dyn2db(self.index2dyn(index)) for index in indices]
        assert dbs 
        return dbs


def _validate_dynamics(dynamics:t.List[str]):
    assert not set(dynamics).difference(_DYNAMICS), \
        "Dynamics not understood"


def _create_dynamics_mapping(bpf:BpfInterface, dynamics:t.List[str]=None) \
        -> t.Tup[t.List[t.Tup[float, str]], t.Dict[str, float]]:
    """
    Calculate the global dynamics table according to the bpf given

    * bpf: a bpf from dynamic-index to amp
    * dynamics: a list of dynamics

    Returns:

    a tuple (amps2dyns, dyns2amps), where 
        - amps2dyns is a List of (amp, dyn)
        - dyns2amps is a dict mapping dyn -> amp
    """
    if dynamics is None:
        dynamics = _DYNAMICS
    assert isinstance(bpf, BpfInterface)
    _validate_dynamics(dynamics)
    dynamics_table = [(bpf(i), dyn) for i, dyn in enumerate(dynamics)]
    dynamics_dict = {dyn: ampdb for ampdb, dyn, in dynamics_table}
    return dynamics_table, dynamics_dict


def create_shape(shape='expon(3)', mindb=-90, maxdb=0) -> BpfInterface:
    """
    Return a bpf mapping 0-1 to amplitudes, as needed to be passed
    to DynamicsCurve

    * descr: a descriptor of the curve to use to map amplitude to dynamics
    * mindb, maxdb: the maximum and minimum representable amplitudes (in dB)
    * dynamics: - a list of dynamic-strings,
                - None to use the default (from pppp to ffff)
    
    If X is dynamic and Y is amplitude, an exponential curve with exp > 1
    will allocate more dynamics to the soft amplitude range, resulting in more
    resolution for small amplitudes.
    A curve with exp < 1 will result in more resolution for high dynamics

    import bpf4 as bpf
    shape = bpf.util.makebpf("expon(3)", [0, 1], [-90, 0]).db2amp()
    """
    minamp, maxamp = db2amp(mindb), db2amp(maxdb)
    return _bpf.util.makebpf(shape, [0, 1], [minamp, maxamp])
    

def get_default_curve():
    return DynamicsCurve.getdefault()
 

def as_dyncurve(curve:t.U[DynamicsCurve, dict]) -> DynamicsCurve:
    if isinstance(curve, DynamicsCurve):
        return curve
    elif isinstance(curve, dict):
        return DynamicsCurve.fromdescr(**curve)
    else:
        raise ValueError("could not convert curve %s to DynamicsCurve" % str(curve))

