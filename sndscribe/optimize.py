from __future__ import absolute_import
from __future__ import print_function
import itertools
from .transcribe import generate_score
from .config import get_default_config
from math import sqrt
from emlib import lib
from .dynamics import DynamicsCurve


def _ascurve(curve):
    if isinstance(curve, str):
        return curve
    return "expon({0})".format(curve)


@lib.returns_tuple("dyncurve results")
def best_dyncurve(spectrum, curves=(0.5, 1, 2.5, 4), mindbs=(-90, -60), maxdbs=(-12, -6), 
                  dynamics='ppp pp p mp mf f ff fff',
                  weightfactor=1, densityfactor=1, config=None, **kws):
    """
    Test the parameter space and find the dynamic curve which maximizes
    the weight and density in the spectrum

    spectrum: a sdntrck.Spectrum
    curves: either a list of curvedescriptions ('linear', 'expon(2)', etc)
            or a list of exponentials to be passed to expon
    dynamics: seq. of possible dynamics
    mindbs: mindbs to test
    maxdbs: maxdbs to test
    config:
    kws: keywords passed to generate_score (numvoices, minfreq, etc)

    NB: this function can take a LONG time to finish since all combinations are tested
    """
    results = []
    curves = [_ascurve(curve) for curve in curves]
    config = config or get_default_config()
    debug = config.get('debug', True)
    if isinstance(dynamics, str):
        dynamics = dynamics.split()
    for curve, mindb, maxdb in itertools.product(curves, mindbs, maxdbs):
        if debug:
            print("analyzing dynamic curve for params: %s" % str((curve, mindb, maxdb)))
        dyncurve = DynamicsCurve.fromdescr(curve, mindb=mindb,
                                           maxdb=maxdb, dynamics=dynamics)
        sc = generate_score(spectrum, dyncurve=dyncurve, **kws)
        result = dict(curve=curve, mindb=mindb, maxdb=maxdb, density=sc.score.density(),
                      weight=sc.score.meanweight(), dyncurve=dyncurve)
        results.append(result)
    maxdensity = max(r['density'] for r in results)
    mindensity = min(r['density'] for r in results)
    maxweight = max(r['weight'] for r in results)
    minweight = min(r['weight'] for r in results)

    # sort by euclidian distance

    def delta(x, x0, x1):
        return (x-x0)/(x1-x0)

    def euclidian(result):
        w = result['weight']
        d = result['distance']
        return sqrt((delta(w, minweight, maxweight)*weightfactor)**2 +
                    (delta(d, mindensity, maxdensity)*densityfactor)**2)

    results.sort(key=euclidian, reverse=True)
    return results[0]['dyncurve'], results
