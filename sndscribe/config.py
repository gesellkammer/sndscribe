from __future__ import absolute_import
import warnings
import os
from . import dynamics
from . import envir
import yaml
from typing import Tuple, Any
import logging
from fractions import Fraction

logger = logging.getLogger("sndscribe")


USE_DIFFERENT_NOTEHEADS = False
DEBUG_BEST_DIVISION = False


# See makeconfig for documentation on each key
_defaultconfig = {
    'numvoices': 12,
    'pitch_resolution': 0.5,
    'staffrange': 36,  # the maximum range in semitones in any given staff
    'divisions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
    'dyncurvedescr': {'shape': 'expon(2.5)',
                      'mindb': -75,
                      'maxdb': -6,
                      'dynamics': ['pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff']},
    'debug': False,
    'show_noteshapes': False,
    'remove_silent_breakpoints': True,
    'silence_db': -10,
    'amp_dur_weight': 10.,
    'transient_weight': 0.1,
    'note_weight': 4.0,
    'slur_partials': False,
    'time_accuracy_weight': 1.0,
    'timeshift_penalty': 1.,
    'leftout_penalty': 2,
    'incorrect_duration_penalty': 10,
    'complexity_penalty': 4.0,
    'merged_notes_penalty': 1,
    'notesize_follows_dynamic': True,
    'show_dynamics': False,
    'downsample_spectrum': True,
    'join_small_notes': True,
    'use_css_sizes': False,
    'show_transients': False,
    'partial_mindur': None,
    'staffsize': 12,
    'pagelayout': 'portrait',
    'pagesize': 'A4',
    'minfreq': None,
    'pack_f0_gain': 100,
    'pack_notf0_gain': 1,
    'pack_f0_threshold': 0.1,
    'pack_freqweight': 2,
    'pack_ampweight': 1,
    'pack_durweight': 2,
    'pack_freq2points': [
        # freq -> points (0-100)
        [0, 0.00001],
        [20, 5],
        [50, 100],
        [400, 100],
        [800, 80],
        [2800, 80],
        [3800, 50],
        [4500, 10],
        [5000, 0]
    ],
    'pack_amp2points': {
        'interpolation': 'linear',
        'points': [
            [-90, 0.0001],
            [-60, 2],
            [-35, 40],
            [-12, 95],
            [0, 100]
        ]
    },
    'pack_dur2points': [
        [0, 0.0001],
        [0.1, 5],
        [0.3, 20],
        [0.5, 40],
        [2, 70],
        [5, 100]
    ],
    'pack_prefer_overtones': False,
    'pack_harmonicity_gain': [
        [0.7, 1],
        [1, 2]
    ],
    'pack_overtone_gain': [
        2, 2,
        5, 1
    ],
    # prefer voiced overtones: maps voicedness to gain,
    # where 1 is a harmonic sound, 0 is noise
    'pack_voiced_gain': [
        0, 1,
        1, 1
    ],
    'divcomplexity':  {
        # division: complexity_factor
        1: 2,
        2: 1.7,
        3: 1,
        4: 1,
        5: 1.3,
        6: 1.8,
        7: 4,
        8: 1.5,
        9: 2,
        10: 3.5,
        11: 6,
        12: 3.0,
        13: 6,
        14: 6,
        15: 7,
        16: 3,
        17: 7,
        18: 6,
        19: 7,
        20: 4,
        21: 8,
        22: 8,
        23: 9,
        24: 6,
        25: 6,
        27: 10,
        28: 8,
        29: 12,
        30: 6,
        31: 14}
}


def _editconfig(path):
    apps = envir.get_preferred_applications()
        
    def _edityaml(yaml):
        app = apps.get('yaml-editor', apps['editor'])    
        os.system("{app} {path}".format(app=app, path=yaml))

    def _editjson(json):
        app = apps.get('json-editor', apps['editor'])
        os.system("{app} {path}".format(app=app, path=json))

    base, ext = os.path.splitext(path)
    if ext == ".yaml":
        _edityaml(path)
    elif ext == ".json":
        _editjson(path)
    else:
        raise ValueError("format not supported: %s" % path)


class ConfigDict(dict):
    _allowedkeys = set(_defaultconfig.keys())

    def __init__(self, d, fallback=None, name=None):
        super().__init__()
        self.update(d)
        self.name = name
        self.fallback = fallback

    def __setitem__(self, key, value):
        if key not in self._allowedkeys:
            raise KeyError(f"Key {key} not known")
        super(self.__class__, self).__setitem__(key, value)

    def __getitem__(self, key):
        if key not in self._allowedkeys:
            raise KeyError(f"Key {key} not allowed")
        if key in self:
            return super().__getitem__(key)
        if self.fallback is not None and key in self.fallback:
            return self.fallback[key]
        raise KeyError("Key allowed but not found??")

    def edit(self, fmt="yaml", inplace=True):
        import tempfile
        ext = "." + fmt
        path = tempfile.mktemp(suffix=ext)
        saveconfig(self, path)
        _editconfig(path)
        newconfig = readconfig(path)
        if inplace:
            self.update(newconfig)
            return self
        else:
            return newconfig

    def get(self, key, default=None):
        if key in self:
            return super().__getitem__(key)
        if self.fallback is not None:
            return self.fallback.get(key)
        return default

    def __repr__(self):
        return yaml.dump(self)

    def save(self, outfile):
        return saveconfig(self, outfile)

    def copy(self):
        return self.__class__(super().copy())


yaml.add_representer(ConfigDict, lambda dumper, c: dumper.represent_dict(c))


def saveconfig(config, outfile):
    """
    config: a dict, like defaultconfig, or as returned by makeconfig
    oufile: the path to save config to. Supported formats: yaml, json
    """
    outfile = os.path.abspath(os.path.expanduser(outfile))
    base, ext = os.path.splitext(outfile)
    d = {}
    d.update(config)

    def saveyaml(config, outfile):
        with open(outfile, "w") as f:
            yaml.dump(config, f)

    def savejson(config, outfile):
        import json
        with open(outfile, "w") as f:
            json.dump(config, f)

    if ext == ".yaml":
        saveyaml(d, outfile)
    elif ext == ".json":
        savejson(d, outfile)
    else:
        raise ValueError("only yaml or json formats are supported")


def readconfig(path):
    base, ext = os.path.splitext(path)

    def readyaml(path):
        f = open(path)
        d = yaml.load(f)
        return ConfigDict(d)
    
    def readjson(path):
        import json
        f = open(path)
        d = json.load(f)
        return ConfigDict(d)

    if ext == '.json':
        return readjson(path)
    elif ext == '.yaml':
        return readyaml(path)
    else:
        raise ValueError("only yaml or json formats are supported")


def makeconfig(**kws):
    """
    Make a new configuration, overriding the default config
    
    NB1: Because of the complexity of creating a config, at the REPL
        it is convenient to use 
        
        myconfig = makeconfig().edit()
        
        This will launch a text editor where you can edit your config
        and the changes will be loaded after saving the file and exiting
        the editor

    NB2: for values which are a dict, a dict is expected which
        will override only those values in the original, for example

        makeconfig(divcomplexity={10:100})

        will create a config where the complexity of a 10 subdivision of
        the pulse will be 100, and all over subdivisions will be set
        to the default

    Keys:

    * pitch_resolution: in halftones (0.5=1/4 tones)
    * divisions: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
                 The possible subdivisions of the pulse

    * dyncurve: {'shape': 'expon(2.5)', 'mindb': -75, 'maxdb': -6, 
                dynamics: ['pppp', 'ppp', ..., 'ffff']}
                 A dict of keywords to be passed to DynamicsCurve. This
                 will be the default dyn. curve

    * debug: False            -- Wether to print debug info or not
    * use_different_noteheads: False    -- Use noteheads to show information about noisyness
    * remove_silent_breakpoints: True   -- Interprets Notes with 0 amp as silences
    * silence_db: -40         -- How much should a silence be weighted against notes during quantization
    * amp_dur_weight: 10.     -- Weight of amplitude and duration during quantization
    * transient_weight: 0.1   -- quantization: how important is it that a note has a transient
    * note_weight: 4.0        -- ???
    * time_accuracy_weight: 1.0   quantization: how important is it that a Note is represented
                                  at its exact beginning
    * timeout_penalty: 1.     -- ???
    * leftout_penalty: 8      -- what's the penalty of leaving a note out during quantization
    * incorrect_duration_penalty: 10  
                              -- The penalty of representing a note with a modified duration
    * complexity_penalty: 2      -- Penalty of using complex subdivisions
    * merged_notes_penalty: 0.5  -- Penalty of merging notes together during quantization
    * notesize_folows_dynamic: True  -- Should the size of a note follow the dynamic?
    * show_dynamics: True        -- Should we show the dynamics?
    * downsample_spectrum: True  -- Should the spectrum be downsampled prior to quantization?
    * join_small_notes: True     -- Should we join notes during quantization or pick the best one?
    * use_css_sizes: False       -- musicxml: use css sizes or numeric sizes?
    * divcomplexity              -- A dict mapping complexity to subdivision of the pulse
    * minfreq: None              -- Normally, the freq. resolution of the analysis, or None to 
                                    autodetect it from the Spectrum
    """
    unmatchedkeys = [k for k in kws.keys() if k not in _defaultconfig]
    if unmatchedkeys:
        warnings.warn("Keys not recognised, ignoring: %s" % unmatchedkeys)
    newconfig = _defaultconfig.copy()
    dicts = [(k, v) for k, v in kws.items() if isinstance(v, dict)]
    for k, v in dicts:
        d = newconfig[k].copy()
        d.update(v)
        newconfig[k] = d
        del kws[k]
    newconfig.update(kws)
    out = ConfigDict(newconfig)
    assert out['pack_f0_gain'] == newconfig['pack_f0_gain']
    return out


defaultconfig = ConfigDict(_defaultconfig, name='default')

globalsettings = {
    'dyncurve': dynamics.DynamicsCurve.fromdescr(**_defaultconfig['dyncurvedescr']),
    'config': ConfigDict(_defaultconfig),
    'debug': False
}


def setdefaultconfig(config):
    globalsettings['config'] = config


def getdefaultconfig():
    return globalsettings['config']


def get_default_dynamicscurve():
    """
    Returns a dynamics.DynamicsCurve
    """
    # return dynamics.get_default_curve()
    return globalsettings['dyncurve']


class RenderConfig(object):

    def __init__(self, tempo:float, timesig:Tuple[int, int], config:ConfigDict) -> None:
        assert isinstance(tempo, (int, float))
        assert isinstance(timesig, tuple) and len(timesig) == 2
        assert isinstance(config, ConfigDict), f"Expected a ConfigDict, got {config} of type {type(config)}"
        self.tempo: Fraction = Fraction(tempo)
        self.timesig: Tuple[int, int] = timesig
        self.config = config or getdefaultconfig()
        dyncurvedescr = self.config.get('dyncurvedescr')
        if dyncurvedescr:
            self.dyncurve = dynamics.DynamicsCurve.fromdescr(**dyncurvedescr)
        else:
            logger.warning("No dynamicscurve defined in config! Using default")
            self.dyncurve = get_default_dynamicscurve()

    def __getitem__(self, key):
        # type: (str) -> Any
        return self.config[key]

    def get(self, key:str, default=None) -> Any:
        return self.config.get(key, default)

