from __future__ import division, print_function
from __future__ import absolute_import
from math import *
import numpy as np

import sndtrck
import bpf4 as bpf
from emlib.pitchtools import *
from emlib.iterlib import pairwise, flatten
from emlib.distribute import dohndt

from .track import Track
from .config import RenderConfig
from .envir import logger
from .typehints import *

from functools import lru_cache




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
            obj = list(flatten(obj, levels=1))
        assert len(obj) % 2 == 0
        Xs = obj[::2]
        Ys = obj[1::2]
        return bpf.core.Linear(Xs, Ys)
    else:
        return bpf.util.asbpf(obj)


Curve = U[float, bpf.BpfInterface, List[Tup[float, float]]]


class PartialWeighter(object):
    def __init__(self,
                 freq2points: Curve = None,
                 amp2points: Curve  = None,
                 dur2points: Curve  = None,
                 freqweight = 2.0,
                 ampweight = 1.0,
                 durweight = 2.0,
                 spectrum: sndtrck.Spectrum = None,
                 f0_gain = 2.0,
                 f0_threshold = 0.1,
                 notf0_gain = 1.0,
                 prefer_overtones = False,
                 harmonicity_gain: Curve = None,
                 overtone_gain: Curve = None,
                 voiced_gain: Curve = None
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

    # @lru_cache(maxsize=10000)
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
        if self.spectrum is None:
            return factor
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
    raise DeprecationWarning("use set_default_weights")


def set_default_weights(**keys):
    """See PartialWeighter for possible keys"""
    global _PARTIALWEIGHTER
    weighter = PartialWeighter(**keys)
    _PARTIALWEIGHTER = weighter


def get_default_weighter():
    global _PARTIALWEIGHTER
    if _PARTIALWEIGHTER is None:
        _PARTIALWEIGHTER = PartialWeighter()
    return _PARTIALWEIGHTER


def get_best_track(tracks: List[Track],
                   partial: sndtrck.Partial,
                   maxrange:int = 48,
                   minmargin=None) -> Opt[Track]:
    """
    Select the best track to append this partial to
    """
    results = []  # type: List[Tup[float, Track]]
    for track in tracks:
        rating = track.rate_partial(partial, minmargin=minmargin, maxrange=maxrange)
        if rating > 0:
            results.append((rating, track))
    if not results:
        return None
    results.sort(key=lambda rating_track:rating_track[0])
    bestrating, besttrack = results[-1]
    if besttrack:
        # minnote, maxnote = track_range(besttrack)
        minnote, maxnote = besttrack.track_range()
        partialmaxnote = f2m(partial.maxfreq)
        partialminnote = f2m(partial.minfreq)
        if max(maxnote, partialmaxnote) - min(minnote, partialminnote) > maxrange:
            logger.debug(f"{maxnote} {partialmaxnote} {minnote} {partialminnote} {maxrange}")
            raise AssertionError("pitchrange too big!")
    return besttrack


def get_nearest_track(tracks: List[Track], partial: sndtrck.Partial, maxrange:int, minmargin: float):
    for track in tracks:
        pass


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

    def pack(self, numtracks:int, maxrange=24.0, minmargin=0.1, method='weight') -> None:
        if method == 'weight':
            return self.pack_by_weight(numtracks, maxrange=maxrange, minmargin=minmargin)
        else:
            return self.pack_by_time(numtracks, maxrange=maxrange, minmargin=minmargin)

    def pack_by_time(self, numtracks:int, maxrange=24.0, minmargin=0.2) -> None:
        partials = sorted(self.partials, key=lambda p:p.t0)
        tracks = [Track(mingap=minmargin) for _ in range(numtracks)]
        rejected = []
        for partial in partials:
            possible_tracks = [track for track in tracks if len(track.partials) == 0 or partial.t0 - track.partials[-1].t1 >= minmargin]
            if possible_tracks:
                possible_tracks.sort(key=lambda track:partial.t0 - track.t1)
                track = possible_tracks[0]
                track.add_partial(partial)
            else:
                rejected.append(partial)
        self.tracks, self.rejected = tracks, rejected


    def pack_by_weight(self, numtracks:int, maxrange=24.0, minmargin=0.1) -> None:

        def _pack_by_weight(partials, numtracks, maxrange, minmargin, weighter):
            # Pack by weight
            partials = sorted(partials, key=weighter.partialweight, reverse=True)
            partials = [p for p in partials if weighter.partialweight(p) > 0]
            tracks = [Track(mingap=minmargin) for _ in range(numtracks)]
            rejected = []
            for partial in partials:
                track = get_best_track(tracks, partial, maxrange=maxrange, minmargin=minmargin)
                if track is not None:
                    track.add_partial(partial)
                else:
                    rejected.append(partial)
            return tracks, rejected
        self.tracks, self.rejected = _pack_by_weight(self.partials, numtracks,
                                                     maxrange, minmargin, self.weighter)

    def weight(self) -> float:
        if not self.partials:
            logger.debug(f"weight: channel [{self.freq0:.0f}Hz - {self.freq1:.0f}Hz] has no partials")
            return 0
        weight = sum(self.weighter.partialweight(p) for p in self.partials)/len(self.partials)
        freqratio = self.weighter.freq2points((self.freq1+self.freq0)/2)/100. ** 0.5
        weight *= freqratio
        return weight


def _pack(spectrum:sndtrck.Spectrum,
          numtracks:int,
          weighter:PartialWeighter,
          maxrange:int,
          minmargin:float,
          chanexp:float,
          method: str,
          numchannels=-1,
          minfreq=120.0,
          maxfreq=4500.0
          ) -> Tup[List[Track], List[sndtrck.Partial]]:
    """
    Pack the partials in spectrum into `numtracks` Tracks.
    
    numchannels: if negative, a sensible default will be chosen
    minfreq, maxfreq: these are used to calculate the channelisation of the spectrum
                      NB: Partials lower than minfreq will still be included in the first channel,
                          Partials higher than maxfreq will still be included in the last channel
    minmargin: time-gap between Partials (should be bigger than 0)
    chanexp: an exponential deterining the distribution of channels across minfreq-maxfreq
    maxrange: the maximum range (in midi notes) a voice can hold
    
    Returns (tracks, rejectedpartials)
    """
    if numchannels < 0:
        numchannels = int(numtracks / 2 + 0.5)
    numchannels = min(numtracks, numchannels)
    weighter = weighter or _PARTIALWEIGHTER
    chanFreqCurve = bpf.expon(0, f2m(minfreq*0.9), 1, f2m(maxfreq), exp=chanexp).m2f()
    # splitpoints = [10] + list(chanFreqCurve.map(numchannels))
    splitpoints = list(chanFreqCurve.map(numchannels+1))
    channels = [Channel(f0, f1, weighter=weighter)
                for f0, f1 in pairwise(splitpoints)]

    logger.debug(f"_pack: Enumerating Channels. (numtracks: {numtracks}, numchannels: {numchannels}, chanexp: {chanexp}")

    for partial in spectrum:
        for ch in channels:
            if ch.freq0 <= partial.meanfreq_weighted < ch.freq1:
                ch.append(partial)
                break

    chanWeights = [ch.weight() for ch in channels]
    # Each channel should have at least 1 track
    numtracksPerChan = [numtracks+1 for numtracks in dohndt(numtracks-numchannels, chanWeights)]
    tracks = []     # type: List[Track]
    rejected0 = []  # type: List[sndtrck.Partial]
    for ch, tracksPerChan in zip(channels, numtracksPerChan):
        ch.pack(tracksPerChan, maxrange=maxrange, minmargin=minmargin, method=method)
        tracks.extend(ch.tracks)
        rejected0.extend(ch.rejected)
    for ch in channels:
        packed_partials = sum(len(track) for track in ch.tracks)
        logger.debug(f"    Channel: {ch.freq0:.0f}-{ch.freq1:.0f}Hz  num. tracks: {len(ch.tracks)}, num. partials: {len(ch.partials)}, packed: {packed_partials}")

    # Try to fit rejected partials
    # rejected0.sort(key=lambda par:weighter.partialweight(par), reverse=True)
    # rejected = []  # type: List[sndtrck.Partial]
    rejected = rejected0
    #logger.debug(f"/////////////////// minmargin: {minmargin}")
    rejected1 = []
    for partial in rejected:
        track = get_best_track(tracks, partial, maxrange=maxrange*0.7, minmargin=minmargin)
        if track is not None:
            # logger.debug(f"Adding previously rejected partial: {partial.meanfreq:.0f}Hz, {partial.t0:.3f}-{partial.t1:.3f}")
            track.add_partial(partial)
        else:
            rejected1.append(partial)
    rejected.extend(rejected1)
    tracks = [track for track in tracks if len(track) > 0]
    logger.debug(f"$$$$$ num. tracks: {len(tracks)}, num partials: {sum(len(track) for track in tracks)}, rejected: {len(rejected)}")

    def trackweight(track):
        return (sum(p.meanfreq_weighted*p.duration for p in track) /
                sum(p.duration for p in track))

    tracks.sort(key=trackweight)
    assert all(isinstance(track, Track) for track in tracks)
    return tracks, rejected
    

def estimate_minfreq(spectrum:sndtrck.Spectrum) -> float:
    """
    Returns an estimate of the minimum interseting frequency
    """
    possfreq0, possfreq1 = 30, 3000
    f0 = spectrum.estimatef0(possfreq0, possfreq1, 0.1)  # type: sndtrck.io.EstimateF0
    mask = (f0.freq > 0) * (f0.confidence > 0.9)
    t0, t1 = f0.freq.bounds()  # type: Tup[float, float]
    minfreqs = (f0.freq * mask).sample_between(t0, t1, 0.01)
    if len(minfreqs) > 0 and len(minfreqs[minfreqs > possfreq0]) > 0:
        minfreq = minfreqs[minfreqs > possfreq0].min()
    else:
        viable_partials = [p for p in spectrum if p.duration > 0.1]
        if viable_partials:
            minfreq = min(p.meanfreq for p in viable_partials)
        else:
            minfreq = min(p.meanfreq for p in spectrum)
    return minfreq


def pack_spectrum(spectrum: sndtrck.Spectrum,
                  config: RenderConfig
                  ) -> Tup[List[Track], List[sndtrck.Partial]]:
    """
    Distribute the partials in `spectrum` into `numtracks`.

    * maxrange: maximum range (in midinotes) possible within one track
    * minfreq: value used for heuristics. The lowest value of meaningful partials

    Returns: (tracks: list of tracks, rejected: list of rejected partials)

    NB: a track is a list of non-overlapping partials
    """
    maxrange = config['staffrange']
    chanexps = config['pack_channel_exponentials']
    numvoices = config['numvoices']
    adaptive_prefilter = config['pack_adaptive_prefilter']
    minfreq = config['minfreq']

    rejected_prefilter = None
    if adaptive_prefilter:
        weighter = sndtrck.reduction.make_weighter('transcription')
        spectrum, rejected_prefilter = sndtrck.adaptive_filter(spectrum, numvoices, weighter=weighter)

    minchannels = min(3, numvoices)
    maxchannels = max(numvoices, minchannels)
    num_possible_channels = min(maxchannels - minchannels, 5)
    possible_channels = [int(x) for x in np.linspace(minchannels, maxchannels, num_possible_channels)]
    if minfreq <= 0:
        minfreq = estimate_minfreq(spectrum)
        logger.debug(f"Estimated minfreq: {minfreq}")
    assert minfreq > 0
    results = []
    weighter = new_weighter(config, spectrum)
    # the exponential of the freq. curve determining the distr. of channels
    # for packing. An exp<1 results in more resolution for high freq.
    # chanexps = [0.7, 1.3]
    for numchannels in possible_channels:
        for chanexp in chanexps:
            logger.debug(f"§§§§§§ Calling _pack numtracks={numvoices} numchannels={numchannels}")
            tracks, rejected = _pack(spectrum,
                                     numtracks=numvoices,
                                     minmargin=config['pack_interpartial_margin'],
                                     weighter=weighter,
                                     numchannels=numchannels,
                                     maxrange=maxrange,
                                     minfreq=minfreq,
                                     chanexp=chanexp,
                                     method=config['pack_criterium'])
            assignedpartials = (partial for track in tracks for partial in track)
            assignedpoints = sum(weighter.partialweight(p) for p in assignedpartials)
            rejectedpoints = sum(weighter.partialweight(p) for p in rejected)
            rating = assignedpoints/(assignedpoints+rejectedpoints)
            n = sum(len(track) for track in tracks)
            results.append((rating, numchannels, chanexp, tracks, rejected))
    for r in results:
        rating, numchannels, chanexp, tracks, rejected = r
        logger.debug(f">>>> Result: numchannels: {numchannels}, chanexp: {chanexp}, rating: {rating:.3f}")
    results.sort(key=lambda result:result[0])
    rating, numchannels, exp, tracks, rejected = results[-1]
    logger.debug(f"~~~ best num. of channels:{numchannels}, freq curve of exp:{exp}")
    for i, track in enumerate(tracks):
        track.partials.sort(key=lambda partial: partial.t0)
        logger.debug(f" ~~~~~~ track {i}: {len(track.partials)} num. partials")
        if len(track) > 1:
            assert not track.has_overlap(), \
                "Partials overlap in track %d: %s" % (i, track)

    if rejected_prefilter:
        rejected.extend(rejected_prefilter)

    assert all(isinstance(track, Track) for track in tracks)
    assert all(isinstance(partial, sndtrck.Partial) for partial in rejected)
    return tracks, rejected

