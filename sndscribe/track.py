from math import sqrt
import bisect

import sndtrck
import bpf4 as bpf

from emlib.lib import intersection, euclidian_distance
from emlib.iterlib import pairwise
from emlib.pitchtools import  f2m

from .definitions import Region
from .typehints import List, Tup, Gen, Opt
from .error import TrackFullError


class Track:
    def __init__(self, partials: List[sndtrck.Partial]=None, mingap:float=0):
        super().__init__()
        self.minnote: float = None
        self.maxnote: float = None
        self.partials = []
        self._avgpitch: float = -1
        self.t0 = 0
        self.t1 = 0
        self.mingap = mingap
        assert mingap > 0
        if partials:
            self.add_partials(partials)

    def add_partials(self, partials: List[sndtrck.Partial]) -> None:
        for p in partials:
            self.add_partial(p)

    def _calculate_range(self):
        self.minnote = f2m(min(p.minfreq for p in self.partials))
        self.maxnote = f2m(max(p.maxfreq for p in self.partials))

    def __len__(self) -> int:
        return len(self.partials)

    def __iter__(self) -> Gen[sndtrck.Partial]:
        return iter(self.partials)

    def set_partials(self, partials:List[sndtrck.Partial]) -> None:
        self.partials = partials
        self.partials.sort(key=lambda p:p.t0)
        assert not self.has_overlap()
        self._calculate_range()
        self._changed()

    def _changed(self):
        self._avgpitch = -1
        self.t0 = self.partials[0].t0
        self.t1 = self.partials[-1].t1

    def add_partial(self, partial:sndtrck.Partial):
        """
        Raises TrackFull if the partial can't be added
        """
        for p in self.partials:
            if p.t0 < partial.t0:
                if p.t1 + self.mingap > partial.t0:
                    if p.t1 > partial.t0:
                        raise TrackFullError(f"Partial doesn't fit. Own partial {p}, t1+mingap={p.t1+self.mingap} > new partial.t0 {partial.t0}")
            else:
                if partial.t1 + self.mingap > p.t0:
                    raise TrackFullError("partial does not fit")
        self.partials.append(partial)
        self.partials.sort(key=lambda p:p.t0)
        assert not self.has_overlap()
        minnote = f2m(partial.minfreq)
        maxnote = f2m(partial.maxfreq)
        if len(self.partials) == 1:
            # track was empty
            self.minnote = minnote
            self.maxnote = maxnote
        else:
            assert self.minnote is not None and self.maxnote is not None
            self.minnote = min(self.minnote, minnote)
            self.maxnote = max(self.maxnote, maxnote)
        self._changed()
        return True

    def isempty(self) -> bool:
        return len(self.partials) == 0

    def track_range(self) -> Tup[float, float]:
        """

        Returns: a tuple (minnote, maxnote)

        """
        if self.isempty():
            raise IndexError("This Track is empty")
        return self.minnote, self.maxnote

    def avgpitch(self) -> float:
        """
        Returns:
            This track's average pitch (midinote)
        """
        if self.isempty():
            raise IndexError("This Track is empty")
        if self._avgpitch < 0:
            self._avgpitch = sum(f2m(partial.meanfreq_weighted) for partial in self.partials) / len(self.partials)
        # assert self.minnote <= self._avgpitch <= self.maxnote, f"minnote={self.minnote}, avg={self._avgpitch}, maxnote={self.maxnote}"
        return self._avgpitch

    def mindistance(self, t0: float, t1: float) -> float:
        """
        Calculate the min. distance between this track and the timerange (t0, t1)

        Args:
            t0: start time
            t1: end time (t0 < t1)

        Returns:
            The min. distance between this track and (t0, t1)

        """
        if self.t0 < t0:
            return t0 - self.t1
        return self.t0 - t1

    def partial_fits(self, t0: float, t1: float) -> bool:
        if self.isempty():
            return True
        t0 -= self.mingap
        t1 += self.mingap
        for partial in self.partials:
            if intersection(partial.t0, partial.t1, t0, t1):
                return False
        return True

    def partial_left_to(self, t: float) -> Opt[int]:
        """
        Find the partial next to (prior to) t, return its index

        Args:
            t: the time t

        Returns:
            The index of the partial ending before t which is
            closest to it, or -1 if no partial ends prior to t
        """
        for irev, p in enumerate(reversed(self.partials)):
            if p.t1 < t:
                return len(self.partials) - 1 - irev
        return None

    def rate_partial(self, partial:sndtrck.Partial, maxrange:int, minmargin:float=None) -> float:
        """
        Rates how good this partial fits in this track. Returns > 0 if partial fits,
        the value returned indicates how good it fits
        """
        if minmargin is None:
            minmargin = self.mingap
        assert minmargin is not None

        isempty = self.isempty()
        if isempty:
            margin = partial.t0
        elif not self.partial_fits(partial.t0, partial.t1):
            return -1
        else:
            partial_left_idx = self.partial_left_to(partial.t0)
            if partial_left_idx is None:
                assert all(p.t0 > partial.t1 for p in self.partials)
                margin = partial.t0

            else:
                # the found partial is either the last partial, or the partial next
                # to it shoudl start AFTER the partial being rated
                assert partial_left_idx == len(self.partials) - 1 or self.partials[partial_left_idx+1].t0 > partial.t1
                p0 = self.partials[partial_left_idx]
                # Partials should be left indented
                margin = partial.t0 - p0.t1
                assert margin >= minmargin

        # Try to pack as tight as possible
        margin_rating = bpf.halfcos(
            minmargin, 1,
            1, 0.01,
            exp=0.6)(margin)

        margin_weight, range_weight, wrange_weight = 3, 1, 1

        if isempty:
            return euclidian_distance([margin_rating, 1, 1], [margin_weight, range_weight, wrange_weight])

        trackminnote, trackmaxnote = self.track_range()
        minnote = f2m(partial.minfreq)
        maxnote = f2m(partial.maxfreq)
        range_with_note = max(trackmaxnote, maxnote) - min(trackminnote, minnote)
        if range_with_note > maxrange:
            return -1
        range_rating = bpf.expon(0, 1, maxrange, 0.0001, exp=1)(range_with_note)
        avgpitch = self.avgpitch()
        avgdiff = abs(avgpitch - f2m(partial.meanfreq_weighted))
        wrange_rating = bpf.halfcos(0, 1, maxrange, 0.0001, exp=0.5)(avgdiff)
        total = euclidian_distance([margin_rating, range_rating, wrange_rating],
                                   [margin_weight, range_weight, wrange_weight])
        return total

    def has_overlap(self) -> bool:
        """
        Check that no two partials overlap in track
        """
        # track should always be sorted
        if len(self.partials) < 2:
            return False
        return any(p0.t1 > p1.t0 for p0, p1 in pairwise(self.partials))

    def remove_short_partials(self, threshold:float) -> List[sndtrck.Partial]:
        """
        Remove partials with a duration less than threshold (inplace),
        returns the removed partials

        Args:
            threshold: the min. duration

        Returns:
            the removed partials (a list)

        """
        partials = [p for p in self if p.duration >= threshold]
        tooshort = [p for p in self if p.duration < threshold]
        self.set_partials(partials)
        return tooshort


def dump_tracks(tracks: List[Track]):
    for i, track in enumerate(tracks):
        assert isinstance(track, Track)
        print(f"Track # {i} -- num. Partials: {len(track.partials)}, Range: {track.track_range()}")
        for p in track:
            print("    ", p)


def track_margin(track:Track, partial:sndtrck.Partial) -> float:
    """
    Returns margin between closest partial in this track and the given partial.
    A negative margin indicates that the partial does not fit

    * If the partial does not fit, it returns a negative margin
    * If the track is empty, returns a margin of 0
    * the margin is the minimal distance to the next partial in track
    """
    if track.isempty():
        return 0

    inf = 9999999999
    regions = [Region(-inf, -inf)] + [Region(p.t0, p.t1) for p in track] + [Region(inf, inf)]
    t0s, t1s = list(zip(*regions))
    i = bisect.bisect(t1s, partial.t0)
    assert all(isinstance(t, (int, float)) for t in (partial.t0, partial.t1, regions[i].t0, regions[i].t1)), \
        f"expected all floats, got {(partial.t0, partial.t1, regions[i].t0, regions[i].t1)}"
    intersect_post = intersection(partial.t0, partial.t1, regions[i].t0, regions[i].t1)
    if intersect_post is not None:
        # They overlap, calculate negative margin
        assert isinstance(intersect_post[0], float) and isinstance(intersect_post[1], float)
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
    return minmargin