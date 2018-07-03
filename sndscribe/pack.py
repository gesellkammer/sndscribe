from __future__ import division, print_function
from __future__ import absolute_import
from math import *
import bisect
from collections import namedtuple

import sndtrck
import bpf4 as bpf
from emlib.pitch import *
from emlib.lib import intersection, returns_tuple
from emlib.iterlib import pairwise, flatten1
from emlib.distribute import dohndt

from .config import RenderConfig
from .envir import logger
from .typehints import *

from functools import reduce, lru_cache

Region = namedtuple("Region", "t0 t1")

_default = {
    'freq2points': bpf.linear(
        0, 0.00001,
        20, 5,
        50, 100,
        400, 100,
        800, 80,
        2800, 80,
        3800, 50,
        4500, 10,
        5000, 0
    ),
    'amp2points': bpf.linear(
        -90, 0.00001,
        -60, 2,
        -35, 40,
        -12, 95,
        0, 100
    ),
    'dur2points': bpf.linear(
        0, 0.0001,
        0.1, 5,
        0.3, 20,
        0.5, 40,
        2, 70,
        5, 100
    ),
    'harmonicity_gain': bpf.linear(
        0.7, 1,
        1, 2
    ),
    'overtone_gain': bpf.linear(
        1, 1.5,
        5, 1
    )
}


def _asbpf(obj):
    if isinstance(obj, dict):
        return bpf.util.dict_to_bpf(obj)
    elif isinstance(obj, list):
        # can be either a flat list of [x0, y0, x1, y1, ...] or [[x0, y0], [x1, y1], ...]
        if isinstance(obj[0], (tuple, list)):
            obj = list(flatten1(obj))
        assert len(obj) % 2 == 0
        Xs = obj[::2]
        Ys = obj[1::2]
        return bpf.core.Linear(Xs, Ys)
    else:
        return bpf.util.asbpf(obj)

#
# class Cache(OrderedDict):
#     def __init__(self, maxlen=None):
#         self.maxlen = maxlen
#         OrderedDict.__init__(self)
#
#     def __setitem__(self, key, value):
#         OrderedDict.__setitem__(self, key, value)
#         if self.maxlen is not None and len(self) > self.maxlen * 1.25:
#             while len(self) > self.maxlen:
#                 self.popitem(last=False)


Curve = U[float, bpf.BpfInterface, List[Tup[float, float]]]


class PartialWeighter(object):
    def __init__(self,
                 freq2points: Opt[Curve]=None,
                 amp2points:  Opt[Curve]=None,
                 dur2points:  Opt[Curve]=None,
                 freqweight=2.0,
                 ampweight=1.0,
                 durweight=2.0,
                 spectrum:    Opt[sndtrck.Spectrum]=None,
                 f0_gain=2.0,
                 f0_threshold=0.1,
                 notf0_gain=1.0,
                 prefer_overtones=False,
                 harmonicity_gain: Opt[Curve]=None,
                 overtone_gain:    Opt[Curve]=None,
                 voiced_gain:      Opt[Curve]=None
                 ):
        """
        * f0_gain: ratio to apply to the weight if a partial is the fundamental
                   Use a big value to make sure that the fundamental always has
                   a high weight in comparison to other partials
        """
        if spectrum is not None:
            assert isinstance(spectrum, sndtrck.Spectrum)
        
        def arg_or_default(arg, name):
            if arg is None:
                arg = _default[name]
                logger.debug("PartialWeighter: using default {name}".format(name=name))
            return arg

        self.freq2points = _asbpf(arg_or_default(freq2points, 'freq2points'))
        self.amp2points = _asbpf(arg_or_default(amp2points, 'amp2points'))
        self.dur2points = _asbpf(arg_or_default(dur2points, 'dur2points'))
        self.harmonicity_gain = _asbpf(arg_or_default(harmonicity_gain, 'harmonicity_gain'))
        self.overtone_gain = _asbpf(arg_or_default(overtone_gain, 'overtone_gain'))
        self.ampweight = ampweight
        self.durweight = durweight
        self.freqweight = freqweight
        self.spectrum = spectrum
        self.f0_gain = f0_gain
        self.f0_theshold = f0_threshold
        self.notf0_gain = notf0_gain
        self.prefer_overtones = prefer_overtones
        self.voiced_gain = _asbpf(voiced_gain or 1)
        # ---------------
        self._spectrum_f0 = None

    @lru_cache(maxsize=10000)
    def partialweight(self, partial:sndtrck.Partial) -> float:
        weight = self.weight(partial.meanfreq_weighted, partial.meanamp, partial.duration)
        gain = 1
        if self.spectrum is not None:
            try:
                spectral_factor = self.spectral_factor(partial)
            except ValueError:
                spectral_factor = 1
            gain *= spectral_factor
        out = weight * gain
        return out

    def spectrum_f0(self) -> sndtrck.io.EstimateF0:
        """
        :return: A tuple(freq_bpf, confidence_bpf) 
        """
        if self._spectrum_f0 is not None:
            return self._spectrum_f0
        if self.spectrum is None:
            raise ValueError("spectrum was not set for this PartialWeighter, "
                             "so f0 can't be calculated")
        self._spectrum_f0 = self.spectrum.estimatef0(minfreq=30, maxfreq=3000, interval=0.05)
        return self._spectrum_f0

    def spectral_factor(self, partial: sndtrck.Partial) -> float:
        """
        Returns a `gain` factor to apply to a Partial's weight,
        representing the importance of this partial in the structure 
        of the spectrum (is it the fundamental? is it an overtone? which??)
        
        NB: This is only of relevance if a spectrum was set in this
            PartialWeighter
        """
        factor = 1.0
        if self.spectrum is not None:
            f0 = self.spectrum_f0()
            # average confidence
            t0, t1 = partial.t0, partial.t1
            avgconf = f0.confidence[t0:t1].map(1000).mean()
            factor *= self.voiced_gain(avgconf)
            
            minconfidence = 0.5
            if avgconf > minconfidence:
                f0_ratio = _partial_f0_ratio(partial, f0, confidence=minconfidence)
                if abs(1 - f0_ratio) <= self.f0_theshold:
                    factor *= self.f0_gain
                elif self.prefer_overtones:
                    frac, N = normalize_harmonicity(f0_ratio)
                    factor *= self.harmonicity_gain(frac) * self.overtone_gain(N)
                else:
                    factor *= self.notf0_gain
        return factor

    def weight(self, freq: float, amp: float, dur: float) -> float:
        freqpoints = self.freq2points(freq)/100.
        amppoints = self.amp2points(amp2db(amp))/100.
        durpoints = self.dur2points(dur)/100.
        freqw, ampw, durw = self.freqweight, self.ampweight, self.durweight
        total = sqrt((freqpoints*freqw)**2 + (amppoints*ampw)**2 + (durpoints*durw)**2)
        maxtotal = sqrt(freqw**2 + ampw**2 + durw**2)
        total /= maxtotal
        assert total > 0
        return total


_PARTIALWEIGHTER = None


def normalize_harmonicity(ratio:float) -> Tup[float, int]:
    """
    ratio: the ratio between the freq. and a fundamental

    N: 1-x, where 1 is the fundamental, 2 is the octave, etc.
       Always an integer
    frac: the distance to the overtone, always a positive number between 0 and 1

    3.2 --> 3, 0.2
    2.9 --> 3, 0.1
    1.8 --> 2, 0.2
    2.3 --> 2, 0.3
    """
    N = round(ratio)
    frac = abs((ratio % 1) - round(ratio % 1))
    harmonicity = _linlin(frac, 0, 0.5, 1, 0)
    return harmonicity, N  # we return in the same order as modf


def _linlin(x:float, n0:float, n1:float, m0:float, m1:float) -> float:
    return ((x-n0)/(n1-n0))*(m1-m0)+m0


def _partial_f0_ratio(partial:sndtrck.Partial, f0:sndtrck.io.EstimateF0, confidence=0.85) -> float:
    """
    Returns the ratio of the partial to the f0
    1 -> the partial is the f0
    2 -> the partial is the first harmonic
    etc ...
    """
    freq = (f0.freq * (f0.confidence > confidence))[partial.t0:partial.t1]  # type: bpf.BpfInterface
    div = (freq > 0).integrate()
    if div == 0:
        return 0
    f0_at_partial = freq.integrate()/div
    return partial.meanfreq / f0_at_partial


def new_weighter(config: RenderConfig, spectrum: Opt[sndtrck.Spectrum]=None):
    d = {}

    pack_keys = {
        'f0_gain', 'notf0_gain', 'f0_threshold', 'freqweight',
        'ampweight', 'durweight', 'freq2points',
        'amp2points', 'dur2points', 'prefer_overtones',
        'harmonicity_gain', 'overtone_gain', 'voiced_gain'
    }
    d = {key: config['pack_' + key] for key in pack_keys}
    return PartialWeighter(spectrum=spectrum, **d)


def setdefaultweights(**keys):
    """See PartialWeighter for possible keys"""
    global _PARTIALWEIGHTER
    weighter = PartialWeighter(**keys)
    _PARTIALWEIGHTER = weighter


def getdefaultweighter():
    global _PARTIALWEIGHTER
    if _PARTIALWEIGHTER is None:
        _PARTIALWEIGHTER = PartialWeighter()
    return _PARTIALWEIGHTER


def track_margin(track:'Track', partial:sndtrck.Partial) -> Opt[float]:
    """
    Returns margin between closest partial in this track and the given partial,
    or None if the partial does not fit

    * If the partial does not fit, it returns a negative margin
    * If the track is empty, returns a margin of 0
    * the margin is the minimal distance to the next partial in track
    """
    if len(track) == 0:   # The track is empty
        return 0
    inf = 9999999999
    regions = [Region(-inf, -inf)] + [Region(p.t0, p.t1) for p in track] + [Region(inf, inf)]
    t0s, t1s = list(zip(*regions))
    i = bisect.bisect(t1s, partial.t0)
    intersect_post = intersection(partial.t0, partial.t1, regions[i].t0, regions[i].t1)
    if intersect_post is not None:
        # They overlap, calculate negative margin
        margin = intersect_post[0] - intersect_post[1]
        return margin
    intersect_pre = intersection(partial.t0, partial.t1, regions[i-1].t0, regions[i-1].t1)
    if intersect_pre is not None:
        # they overlap, return negative margin
        margin = intersect_pre[0] - intersect_pre[1]
        return margin
    if not regions[i-1].t1 <= partial.t0 < regions[i].t0:
        print(regions[i - 1], regions[i], partial)
        raise ValueError("WTF: ")
    if partial.t1 <= regions[i].t0:
        left = partial.t0 - regions[i-1].t1
        right = regions[i].t0 - partial.t1
        assert left >= 0 and right >= 0
        margin = left + right
        minmargin = margin
    else:
        minmargin = None
    assert minmargin is not None
    return minmargin


class Track(list):
    def __init__(self):
        super().__init__()
        self.minnote: float = -1
        self.maxnote: float = -1
        self._avgpitch: float = -1

    def addpartial(self, partial:sndtrck.Partial) -> bool:
        """
        Returns True if the partial was added
        """
        for p in self:
            if intersection(p.t0, p.t1, partial.t0, partial.t1) is not None:
                return False
        self.append(partial)
        self.sort(key=lambda p:p.t0)
        # track_addpartial(self, partial)
        minnote = f2m(partial.minfreq)
        maxnote = f2m(partial.maxfreq)
        self.minnote = minnote if self.minnote <= 0 else min(self.minnote, minnote)
        self.maxnote = maxnote if self.maxnote <= 0 else max(self.maxnote, maxnote)
        self._avgpitch = -1
        return True

    def trackrange(self) -> Tup[float, float]:
        return self.minnote, self.maxnote

    def avgpitch(self) -> float:
        if self._avgpitch < 0:
            self._avgpitch = sum(f2m(partial.meanfreq_weighted) for partial in self) / len(self)
        return self._avgpitch


# def track_range(track:Track) -> Tup[float, float]:
#     """
#     Returns the lowest and highest midinote of this track
#
#     :type track: list[sndtrck.Partial]
#     """
#     if not track:
#         raise ValueError("track is empty")
#     minnote = f2m(min(partial.minfreq for partial in track))
#     maxnote = f2m(max(partial.maxfreq for partial in track))
#     assert 0 < minnote <= maxnote < 127, "minnote:{0}, maxnote:{1}".format(
#         minnote, maxnote)
#     return minnote, maxnote
#

def track_ratepartial(track:Track, partial:sndtrck.Partial, minmargin=0.1, maxrange=48.0) -> float:
    """
    Rates how good this partial fits in this track
    """
    if len(track) == 0:
        return 1
    margin = track_margin(track, partial)
    if margin - minmargin < 0:   # does not fit
        return 0
    margin = max(margin, minmargin)
    # Try to pack as tight as possible
    margin_rating = bpf.halfcos(
        max(0, minmargin), 1, 
        max(1, minmargin*4), 0.01, 
        exp=0.6)(margin)
    # trackminnote, trackmaxnote = track_range(track)
    trackminnote, trackmaxnote = track.trackrange()
    minnote = f2m(partial.minfreq)
    maxnote = f2m(partial.maxfreq)
    diff = max(maxnote, trackmaxnote) - min(trackminnote, minnote)
    if diff > maxrange:
        return -1
    range_rating = bpf.expon(0, 1, maxrange, 0.0001, exp=1)(diff)
    avgpitch = track.avgpitch()
    avgdiff = abs(avgpitch - f2m(partial.meanfreq_weighted))
    wrange_rating = bpf.halfcos(0, 1, maxrange, 0.0001, exp=0.5)(avgdiff)
    margin_weight, range_weight, wrange_weight = 3, 1, 1
    total = sqrt((margin_rating*margin_weight)**2 +
                 (range_rating*range_weight)**2 +
                 (wrange_rating*wrange_weight)**2)
    return total


def get_best_track(tracks: List[Track],
                   partial: sndtrck.Partial,
                   maxrange=48.0,
                   minmargin=0.1) -> Opt[Track]:
    """
    Select the best track to append this partial
    """
    results = []  # type: List[Tup[float, Track]]
    for track in tracks:
        rating = track_ratepartial(track, partial, minmargin=minmargin, maxrange=maxrange)
        if rating > 0:
            results.append((rating, track))
    if not results:
        return None
    results.sort(key=lambda rating_track:rating_track[0])
    bestrating, besttrack = results[-1]
    if besttrack:
        # minnote, maxnote = track_range(besttrack)
        minnote, maxnote = besttrack.trackrange()
        partialmaxnote = f2m(partial.maxfreq)
        partialminnote = f2m(partial.minfreq)
        if max(maxnote, partialmaxnote) - min(minnote, partialminnote) > maxrange:
            logger.debug(f"{maxnote} {partialmaxnote} {minnote} {partialminnote} {maxrange}")
            raise AssertionError("pitchrange too big!")
    return besttrack


class Channel:
    """
    A channel is a section of the pitch spectrum used to pack
    similar Partials together. A Channel has a number of tracks,
    each track is a list of non-overlapping Partials
    """
    def __init__(self, freq0:float, freq1:float, weighter:PartialWeighter) -> None:
        self.freq0, self.freq1 = freq0, freq1
        self.partials = []  # type: List[sndtrck.Partial]
        self.tracks = []    # type: List[Track]
        self.rejected = []  # type: List[sndtrck.Partial]
        self.weighter = weighter or _PARTIALWEIGHTER  # type: PartialWeighter

    def append(self, partial:sndtrck.Partial) -> None:
        self.partials.append(partial)
    
    def pack(self, numtracks:int, maxrange=24.0, minmargin=0.1) -> None:

        def _pack_by_weight(partials, numtracks, maxrange, minmargin, weighter):
            # Pack by weight
            partials = sorted(partials, key=weighter.partialweight, reverse=True)
            partials = [p for p in partials if weighter.partialweight(p) > 0]
            tracks = [Track() for _ in range(numtracks)]
            rejected = []
            for partial in partials:
                track = get_best_track(tracks, partial, maxrange=maxrange, minmargin=minmargin)
                if track is not None:
                    track.addpartial(partial)
                else:
                    rejected.append(partial)
            return tracks, rejected
        self.tracks, self.rejected = _pack_by_weight(self.partials, numtracks,
                                                     maxrange, minmargin, self.weighter)

    def weight(self) -> float:
        if not self.partials:
            logger.debug("channel has no partials")
            return 0
        weight = sum(self.weighter.partialweight(p) for p in self.partials)/len(self.partials)
        freqratio = self.weighter.freq2points((self.freq1+self.freq0)/2)/100. ** 0.5
        weight *= freqratio
        return weight


def _pack(spectrum:sndtrck.Spectrum,
          numtracks:int,
          weighter:PartialWeighter,
          maxrange=24.0,
          minmargin=0.0,
          numchannels=-1,
          minfreq=120.0,
          maxfreq=4500.0,
          chanexp=0.7) -> Tup[List[Track], List[sndtrck.Partial]]:
    """
    Pack the partials in spectrum into `numtracks` Tracks.
    
    numchannels: if negative, a sensible default will be chosen
    minfreq, maxfreq: these are used to calculate the channelisation of the spectrum
                      NB: Partials lower than minfreq will still be included in the first channel,
                          Partials higher than maxfreq will still be included in the last channel
    minmargin: time-gap between Partials
    
    Returns (tracks, rejectedpartials)
    """
    if numchannels < 0:
        numchannels = int(numtracks / 2 + 0.5)
    numchannels = min(numtracks, numchannels)
    weighter = weighter or _PARTIALWEIGHTER
    chanFreqCurve = bpf.expon(0, f2m(minfreq*2), 1, f2m(maxfreq), exp=chanexp).m2f()
    splitpoints = [10] + list(chanFreqCurve.map(numchannels))
    channels = [Channel(f0, f1, weighter=weighter)
                for f0, f1 in pairwise(splitpoints)]
    for partial in spectrum:
        for ch in channels:
            if ch.freq0 <= partial.meanfreq_weighted < ch.freq1:
                ch.append(partial)
                break
        else:
            logger.debug("Partial outside defined channels! %dHz"
                   % int(partial.meanfreq_weighted))
    chanWeights = [ch.weight() for ch in channels]
    logger.info("channelsweights: %s" % chanWeights)
    # Each channel should have at least 1 track
    numtracksPerChan = [numtracks+1 for numtracks in dohndt(numtracks-numchannels, chanWeights)]
    tracks = []     # type: List[Track]
    rejected0 = []  # type: List[sndtrck.Partial]
    for ch, numtracks in zip(channels, numtracksPerChan):
        ch.pack(numtracks, maxrange=maxrange, minmargin=minmargin)
        tracks.extend(ch.tracks)
        rejected0.extend(ch.rejected)
    logger.info(str([(len(ch.partials), len(ch.tracks)) for ch in channels]))
    # Try to fit rejected partials
    partials = rejected0
    rejected = []  # type: List[sndtrck.Partial]
    partials.sort(key=lambda par:weighter.partialweight(par), reverse=True)
    for partial in partials:
        track = get_best_track(tracks, partial, maxrange=maxrange*0.7, minmargin=minmargin)
        if track is not None:
            track.addpartial(partial)
        else:
            rejected.append(partial)
    tracks = [track for track in tracks if len(track) > 0]
    def trackweight(tr):
        return (sum(p.meanfreq_weighted*p.duration for p in tr) /
                sum(p.duration for p in track))
    tracks.sort(key=trackweight)
    return tracks, rejected
    

def estimate_minfreq(spectrum:sndtrck.Spectrum) -> float:
    """
    Returns an estimate of the minimum interseting frequency
    """
    f0 = spectrum.estimatef0(20, 3000, 0.1)  # type: sndtrck.io.EstimateF0
    mask = (f0.freq > 0) * (f0.confidence > 0.9)
    t0, t1 = f0.freq.bounds()  # type: Tup[float, float]
    minfreqs = (f0.freq * mask).sample_between(t0, t1, 0.01)
    if len(minfreqs) > 0 and len(minfreqs[minfreqs > 20]) > 0:
        minfreq = minfreqs[minfreqs > 20].min()
    else:
        minfreq = min(p.meanfreq for p in spectrum)
    return minfreq


@returns_tuple("tracks rejected")
def packspectrum(spectrum: sndtrck.Spectrum,
                 numtracks: int,
                 config: RenderConfig,
                 maxrange=36,
                 minfreq: float = -1) -> Tup[List[Track], List[sndtrck.Partial]]:
    """
    Distribute the partials in `spectrum` into `numtracks`.

    * maxrange: maximum range (in midinotes) possible within one track
    * minfreq: value used for heuristics. The lowest value of meaningful partials

    Returns: (tracks: list of tracks, rejected: list of rejected partials)

    NB: a track is a list of non-overlapping partials
    """
    minchannels = min(2, numtracks)
    maxchannels = max(numtracks, minchannels)
    if minfreq < 0:
        minfreq = estimate_minfreq(spectrum)
    assert minfreq > 0
    results = []
    weighter = new_weighter(config, spectrum)
    # the exponential of the freq. curve determining the distr. of channels
    # for packing. An exp<1 results in more resolution for high freq.
    chanexps = [0.7, 1.3]
    for numchannels in range(minchannels, maxchannels+1):
        for chanexp in chanexps:
            tracks, rejected = _pack(spectrum, numtracks,
                                     weighter=weighter, numchannels=numchannels, 
                                     maxrange=maxrange, minfreq=minfreq)
            assignedpartials = reduce(lambda a, b:a+b, tracks, [])
            assignedpoints = sum(weighter.partialweight(p) for p in assignedpartials)
            rejectedpoints = sum(weighter.partialweight(p) for p in rejected)
            rating = assignedpoints/(assignedpoints+rejectedpoints)
            results.append((rating, numchannels, chanexp, tracks, rejected))
    results.sort(key=lambda result:result[0])
    rating, numchannels, exp, tracks, rejected = results[-1]
    logger.info(f"best num. of channels:{numchannels}, freq curve of exp:{exp}")
    for track in tracks:
        track.sort(key=lambda partial: partial.t0)
    for i, track in enumerate(tracks):
        if len(track) > 1:
            assert not _hasoverlap(track), \
                "Partials overlap in track %d: %s" % (i, track)
    return tracks, rejected
    

def _hasoverlap(track):
    """
    Check that no two partials overlap in track

    :type track: list
    :rtype : bool
    """
    # track should always be sorted
    if len(track) < 2:
        return True
    return any(p0.t1 > p1.t0 for p0, p1 in pairwise(track))

