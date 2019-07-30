from dataclasses import dataclass

import sndtrck
from emlib.numtheory import nextprime
from emlib.iterlib import pairwise
from . import score as _score
from . import voice as _voice
from . import pack
from . import error as _error
from . import reduction
from .track import Track, dump_tracks
from .dynamics import  DynamicsCurve

from .config import get_default_config, ConfigDict, RenderConfig
from .envir import logger
from . import typehints as t


@dataclass
class ScoreResult:
    score: _score.Score
    spectrum: sndtrck.Spectrum
    tracks: t.List[Track]
    rejected: sndtrck.Spectrum

    def __post_init__(self):
        assert all(isinstance(track, Track) for track in self.tracks), \
            [(track, type(track)) for track in self.tracks if not isinstance(track, Track)]

    def dump_tracks(self):
        return dump_tracks(self.tracks)


def generate_score(spectrum:sndtrck.Spectrum,
                   config:ConfigDict=None,
                   timesig=None,
                   tempo=None,
                   dyncurve: DynamicsCurve=None,
                   ) -> ScoreResult:
    """
    Generate a score from a spectrum

    Args:
        spectrum: a sndtrck.Spectrum.
        config: a configuration as returned by make_config()
        timesig: reserved for future use of a dynamic time signature.
            Right now Use config['timesig'] to set the time signature
        tempo: reserved for future implementation of varying tempo. Use config['tempo']
        dyncurve: a dynamics.DynamicsCurve. Used if given, otherwise the configuration passed
            is used to construct one.

    Returns:
        A ScoreResult(score, spectrum, tracks, rejected_spectrum)

    Example::

    spectrum = sndtrck.analyze("soundfile.wav", resolution=40)
    cfg = make_config()
    cfg['numvoices'] = 8
    result = generate_score(spectrum, cfg)
    result.score.writepdf("score.pdf")
    """

    assert config is None or isinstance(config, ConfigDict)
    assert tempo is None, "Setting tempo here is still not supported. Use config['tempo']"
    assert timesig is None, "Setting timesig here is not supported yet. Use config['timesig']"

    config = config if config is not None else get_default_config()
    render = True

    renderconfig = RenderConfig(config=config, tempo=tempo, timesig=timesig, dyncurve=dyncurve)

    staffsize: int = renderconfig['staffsize']
    pitchres: float = renderconfig['pitch_resolution']
    divisions: t.List[int] = renderconfig['divisions']
    interpartial_margin: float = renderconfig['pack_interpartial_margin']
    partial_mindur: float = renderconfig['partial_mindur']
    assert partial_mindur is None or isinstance(partial_mindur, (int, float))

    if partial_mindur is None:
        partial_mindur = 1.0/max(divisions)

    if dyncurve:
        renderconfig = renderconfig.clone(dyncurve=dyncurve)

    downsample: bool = renderconfig['downsample_spectrum']
    dbs = renderconfig.dyncurve.asdbs()

    SEP = "~~~~~~~~~~~~~~~~~~~"
    logger.info(f"Partial min. dur: {partial_mindur}")

    if partial_mindur > 0:
        spectrum2 = sndtrck.Spectrum([p for p in spectrum if p.duration > partial_mindur])
        numfiltered = len(spectrum) - len(spectrum2)
        logger.info(f"{SEP} filtered short partials (dur < {partial_mindur}: {numfiltered}")
        spectrum = spectrum2
        if len(spectrum) == 0:
            logger.debug("Filtered short partials, but now there are no partials left...")
            raise _error.EmptySpectrum("Spectrum with 0 partials after eliminating short partials")
    spectrum = spectrum.partials_between_freqs(0, renderconfig['maxfreq'])

    logger.info(SEP + "Packing spectrum" + SEP)

    tracks, rejected = pack.pack_spectrum(spectrum, config=renderconfig)

    if not tracks:
        raise _error.ScoreGenerationError("No voices were allocated for the partials given")

    for i, track in enumerate(tracks):
        if track.has_overlap():
            logger.error("Partials should not overlap!")
            logger.error(f"Track #{i}")
            for j, p in enumerate(track):
                logger.error(f"    Partial #{j}: {p}")
            raise _error.ScoreGenerationError("partials inside track overlap")

    if downsample:
        logger.debug(f"Downsampling spectrum ({sum(len(t) for t in tracks)} partials in {len(tracks)} tracks")
        dt = 1/(nextprime(max(divisions)) + 1)
        newtracks = []
        for track in tracks:
            reduced_partials = reduction.reduce_breakpoints(track.partials, pitch_grid=pitchres, db_grid=dbs, time_grid=dt)
            reduced_partials = reduction.fix_quantization_overlap(reduced_partials)
            newtrack = Track(reduced_partials, mingap=interpartial_margin)
            newtracks.append(newtrack)
        logger.debug(f"generate_score: {sum(len(t) for t in newtracks)} partials in {len(newtrack)} tracks after downsampling")
        tracks = newtracks

    # assigned_partials = sum(tracks, [])  # type: t.List[sndtrck.Partial]
    assigned_partials = sum((track.partials for track in tracks), [])
    s = _score.Score(config=renderconfig)
    logger.debug(SEP + "adding partials" + SEP)
    logger.debug(f"Total number of partials: {sum(len(track) for track in tracks)} in {len(tracks)} tracks")
    voices = []

    # check gaps between partials
    for track in tracks:
        for p0, p1 in pairwise(track.partials):
            if p1.t0 - p0.t1 < 0:
                raise ValueError(f"the gap is too small: {p0} {p1}")

    for track in tracks:
        valid_partials = []
        for partial in track:
            if partial.duration < partial_mindur:
                logger.warn(f"short partial found, skipping! {partial.duration} < {partial_mindur}")
            else:
                valid_partials.append(partial)
        track.set_partials(valid_partials)


    for i, track in enumerate(tracks):
        voice = _voice.Voice(lastnote_duration=renderconfig['lastnote_duration'])
        if track.has_overlap():
            raise ValueError("Track has overlapping partials")
        logger.debug(f"Processing Track / Voice # {i} ({len(track)} partials in Track)")
        tooshort = track.remove_short_partials(partial_mindur)
        if tooshort:
            logger.debug(f"Removed short partials: {len(tooshort)}")
        for partial in track:
            if voice.isempty() or partial.t0 >= voice.end:
                voice.addpartial(partial)
            else:
                raise ValueError(f"Partial does not fit in voice: partial.t0 {float(partial.t0):.3f} < voice.end {float(voice.end):.3f}")

        if not voice.isempty():
            voices.append(voice)
        else:
            logger.warn(f"Track #{i} empty")
        assert voice.added_partials == len(track), f"Could not add all partials: partials={len(track)}, added={voice.added_partials}"

    voices.sort(key=lambda x: x.meanpitch(), reverse=True)
    logger.info(SEP + "simplifying notes" + SEP)
    acceptedvoices = []

    for i, voice in enumerate(voices):
        logger.debug(f"simplifying voice #{i}")
        if len(voice.notes) == 0:
            logger.debug(">>>> voice with 0 notes, skipping")
            continue
        if voice.meanpitch() <= 0:
            logger.debug(">>>> voice is empty or has only rests, skipping")
            continue
        simplified_notes = reduction.simplify_notes(voice.notes, pitchres, renderconfig.dyncurve)
        if len(simplified_notes) < len(voice.notes):
            logger.debug("simplified notes: %d --> %d" %
                         (len(voice.notes), len(simplified_notes)))
        voice.notes = simplified_notes
        s.addstaff(_score.Staff(voice, possible_divs=divisions, timesig=renderconfig.timesig,
                               tempo=renderconfig.tempo, size=staffsize))
        acceptedvoices.append(voice)

    logger.debug(f"Accepted voices: {len(acceptedvoices)}")

    if render:
        assert all(voice.meanpitch() > 0 for voice in acceptedvoices)
        logger.info(f"{SEP} rendering... {SEP}")
        s.render()

    assigned_spectrum = sndtrck.Spectrum(assigned_partials)
    rejected_spectrum = sndtrck.Spectrum(rejected)
    return ScoreResult(s, assigned_spectrum, tracks, rejected_spectrum)

