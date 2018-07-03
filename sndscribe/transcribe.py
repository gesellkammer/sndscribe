from __future__ import division
from __future__ import absolute_import
import sndtrck
from emlib.numtheory import nextprime
from emlib.lib import returns_tuple
from . import score
from . import pack 
from .error import EmptySpectrum
from .reduction import reduce_breakpoints

from .config import getdefaultconfig, RenderConfig
from .dynamics import DynamicsCurve
from .envir import logger
from . import typehints as t


@returns_tuple("score spectrum tracks refectedpartials")
def generate_score(spectrum:sndtrck.Spectrum,
                   config:RenderConfig=None,
                   timesig=(4, 4),
                   tempo=60,
                   dyncurve=None,
                   midi=False,
                   render=True
                   ):
    """
    ++ First call config.makeconfig() to generate a config
    
    * spectrum: a sndtrck.Spectrum
    * config: as returned by config.makeconfig
    * dyncurve: as defined in dynamics.DynamicsCurve. It maps dynamics to amplitudes
                If no dynamics curve is given, the config dict is queried for `dyncurvedescr`,
                which is used if present. Otherwise, a default us used.
                NB: use optimize.best_dyncurve to determine the best parameters
    * midi: create a midi file for the resulting score (no microtones)

    Returns: score, associated_spectrum, tracks

    * score: the rendered Score
    * assigned_spectrum: the assigned Spectrum, before downsampling and quantization
    * tracks: a track is a list of non-overlapping Partials which will be
              rendered to a Staff, after downsampling and quantization
    
    The rendered Spectrum can be generated from the tracks:

        rendered_spectrum = sndtrck.Spectrum(sum(tracks, []))

    The Score can be rendered to PDF:
    
        out = generate_score(...)
        out.score.toxml("myfile.xml")
        pdffile = tools.musicxml2pdf("myfile.xml")   # --> myfile.pdf
    """
    assert isinstance(timesig, tuple)
    assert isinstance(tempo, (int, float))

    assert config is None or isinstance(config, dict)

    defaultconfig = getdefaultconfig()
    config = config or getdefaultconfig()

    def get(key:str):
        v = config.get(key)
        if v is not None:
            v = defaultconfig.get(key)
        return v

    numvoices = get('numvoices')
    assert isinstance(numvoices, int)

    pagesize = get('pagesize')
    assert isinstance(pagesize, str)

    pagelayout = get('pagelayout')
    assert isinstance(pagelayout, str)

    staffsize = get('staffsize')
    assert isinstance(staffsize, (int, float))

    minfreq = get('minfreq')
    assert minfreq is None or isinstance(minfreq, int)

    pitchres = get('pitch_resolution')  # type: float
    divisions = get('divisions')        # type: t.List[int]
    partial_mindur = get('partial_mindur')
    assert partial_mindur is None or isinstance(partial_mindur, (int, float))
    if partial_mindur is None:
        partial_mindur = 1.0/max(divisions) * 2
    if dyncurve is not None:
        dynamics_curve = dyncurve
    else:
        dyncurvedescr = get('dyncurvedescr')
        if dyncurvedescr:
            dynamics_curve = DynamicsCurve.fromdescr(**dyncurvedescr)
        else:
            dynamics_curve = score.get_default_dynamicscurve()
    assert isinstance(dynamics_curve, DynamicsCurve)
    downsample: bool = get('downsample_spectrum')
    notesize_follows_dynamic: bool = get('notesize_follows_dynamic')
    include_dynamics: bool = get('show_dynamics')
    dbs = dynamics_curve.asdbs()
    if partial_mindur > 0:
        spectrum = sndtrck.Spectrum([p for p in spectrum if p.duration > partial_mindur])
        if len(spectrum) == 0:
            logger.debug("Filtered short partials, but now there are no partials left...")
            raise EmptySpectrum("Spectrum with 0 partials after eliminating short partials")
    spectrum = spectrum.partials_between_freqs(0, 6000)
    if minfreq is None:
        minfreq = pack.estimate_minfreq(spectrum)
    logger.info("Packing spectrum")
    tracks, rejected = pack.packspectrum(spectrum,
                                         numtracks=numvoices,
                                         config=config,
                                         maxrange=config['staffrange'], 
                                         minfreq=minfreq)
    if not tracks:
        logger.debug("No voices were allocated for the partials")
        return None
    if downsample:
        logger.debug("Downsampling spectrum")
        time_grid = 1/nextprime(max(divisions))
        tracks = [reduce_breakpoints(track, pitch_grid=pitchres, db_grid=dbs, time_grid=time_grid)
                  for track in tracks]
    assigned_partials = sum(tracks, [])  # type: t.List[sndtrck.Partial]
    s = score.Score(timesig=timesig, tempo=tempo, pitch_resolution=pitchres,
                    include_dynamics=include_dynamics,
                    notesize_follows_dynamic=notesize_follows_dynamic,
                    pagesize=pagesize, pagelayout=pagelayout, staffsize=staffsize,
                    possible_divs=divisions, midi_global_instrument=midi,
                    dyncurve=dynamics_curve,
                    # transient_mask=transient_mask,
                    config=config)
    voices = []
    logger.debug("adding partials")
    for track in tracks:
        voice = score.Voice()
        for partial in track:
            logger.debug("voice: %s  --- partial %s" % (voice, partial))
            voice.addpartial(partial)
        voices.append(voice)
    voices.sort(key=lambda x: x.meanpitch(), reverse=True)
    logger.info("~~~~~~~~~~~~~~ simplifying notes ~~~~~~~~~~~~~~~~~~~~~")
    acceptedvoices = []
    for voice in voices:
        logger.debug("simplifying voices")
        if len(voice.notes) == 0:
            logger.debug("voice with 0 notes")
            continue
        if voice.meanpitch() <= 0:
            logger.debug("voice is empty or has only rests")
            continue
        simplified_notes = score.simplify_notes(voice.notes, pitchres, dynamics_curve)
        if len(simplified_notes) < len(voice.notes):
            logger.debug("simplified notes: %d --> %d" %
                         (len(voice.notes), len(simplified_notes)))
        voice.notes = simplified_notes
        s.addstaff(score.Staff(voice, possible_divs=divisions, timesig=timesig, 
                               tempo=tempo, size=staffsize))
        acceptedvoices.append(voice)
    if render:
        assert all(voice.meanpitch() > 0 for voice in acceptedvoices)
        logger.info("rendering...")
        s.render()
    assigned_spectrum = sndtrck.Spectrum(assigned_partials)
    return s, assigned_spectrum, tracks, rejected

