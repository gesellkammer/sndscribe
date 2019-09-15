from __future__ import annotations
import subprocess
import os
import tempfile
import logging
from fractions import Fraction
from functools import lru_cache
from typing import Tuple, Any, Dict, Optional as Opt
from ruamel import yaml

from emlib import dialogs
from emlib import conftools

from . import dynamics
from . import envir
from .error import ImmutableError
from . import definitions
from .tools import parse_timesig, as_timesig_tuple


NAME = "sndscribe"


logger = logging.getLogger(NAME)


def _isbool(opt):
    return isinstance(opt, bool)


def _islistof(T):
    return lambda xs: isinstance(xs, list) and all(isinstance(x, T) for x in xs)


def _oneof(*opts):
    return lambda x: x in opts


def _fileexists(path):
    return os.path.exists(os.path.abspath(os.path.expanduser(path)))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
# App Config

_appconfig_default = {
    'debug': False,
    'app.yaml': '',
    'lilypond.path': ''
}

_appconfig_validator = {
    'lilypond.path': _fileexists,
}

appconfig = conftools.ConfigDict(NAME, _appconfig_default, validator=_appconfig_validator)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
# Transcription config
#
# See defaultconfig.yaml for documentation on each key

_validator = {
    'pitch_resolution':lambda r:r in {0.25, 0.5, 1},
    'divisions':lambda divs:all(div in definitions.ALLOWED_DIVISIONS for div in divs),
    'downsample_spectrum': _isbool,
    'dynamics': lambda dynamics: all(dyn in definitions.POSSIBLE_DYNAMICS for dyn in dynamics),
    'dyncurve_mindb': lambda mindb: -120 <= mindb <= 0,
    'dyncurve_maxdb': lambda maxdb: -120 <= maxdb <= 0,
    'dyncurve_shape': lambda shape: shape == 'linear' or shape.startswith('expon'),
    'join_small_notes': _isbool,
    'lastnote_duration':lambda dur:dur>0,
    'lily_dynamics_size': lambda s: isinstance(s, int) and -5 <= s <= 5,
    'maxfreq': lambda freq: isinstance(freq, (int, float)) and freq > 100,
    'minfreq': lambda freq: isinstance(freq, (int, float)) and freq == 0 or freq >= 20,
    'notesize_follows_dynamic': _isbool,
    'numvoices':lambda n:isinstance(n, int) and n>0,
    'pack_adaptive_prefilter': _isbool,
    'pack_channel_exponentials': lambda exps: _islistof((int, float)),
    'pack_criterium': _oneof('weight', 'time'),
    'pack_f0_gain': lambda g: isinstance(g, (int, float)) and g > 0,
    'pack_f0_threshold': lambda th: isinstance(th, float) and 0 < th < 1,
    'pack_interparcial_margin': lambda m: isinstance(m, float) and m >= 0.2,
    'pack_prefer_overtones': _isbool,
    'pagelayout': _oneof('portrait', 'landscape'),
    'pagesize': lambda s: s.lower() in ('a4', 'a3'),
    'partial_mindur': lambda d: d is None or d > 0,
    'remove_silent_breakpoints': _isbool,
    'show_dynamics': _isbool,
    'show_noteshapes': _isbool,
    'show_transients': _isbool,
    'silenct_db': lambda db: isinstance(db, int) and db < 0,
    'slur_partials': _isbool,
    'staffrange': lambda r: isinstance(r, int) and r >= 12,
    'staffsize': lambda s: isinstance(s, int) and 6 <= s <= 40,
    'tempo': lambda t: isinstance(t, (int, float)) and t > 0,
    'timesig': lambda t: isinstance(t, str) and parse_timesig(t) is not None
}

# A list of functions which should return True if the config is coherent
# These are functions which test settings which are interdependent
_post_validator = [
    lambda cfg: cfg['dyncurve_mindb'] < cfg['dyncurve_maxdb'],
]

_help = {
    'pitch_resolution': 'Pitch resolution for transcription (0.5 means 1/4 tone). One of: 1, 0.5, 0.25',
    'divisions': 'Possible divisions of the pulse (a list of int divisions)',
    'downsample_spectrum': 'Downsample spectrum prior to transcription (bool)',
    'dynamics': 'Possible dynamics used in transcription (a list of dynamics)',
    'dyncurve_mindb': 'The amplitude corresponding to the lowest dynamic (dB value)',
    'dyncurve_maxdb': 'The amplitude corresponding to the higher dynamic (dB value)',
    'dyncurve_shape': 'The shape of the curve between mindb and maxdb. Possible values: "linear", "expon(x)"',
    'join_small_notes': 'Join small notes instead of leaving them out (bool)',
    'lastnote_duration': 'Duration of the endnote of a glissando (float)',
    'lily_dynamics_size': 'Relative size of dynamic expressions (int between -5 and 5)',
    'maxfreq': 'Max. freq used in transcription',
    'minfreq': 'Min. freq used in transcription',
}


def default_config_path() -> str:
    """
    Returns:
        The path to the default config.

    Raises:
        FileNotFoundError if the default config is not found
    """
    path = envir.get_datafile("defaultconfig.yaml")
    if path is None:
        raise FileNotFoundError("default config not found")
    return path


def _edit_config(path, wait=True):
    app = appconfig.get('app.yaml')
    if not app:
        apps = envir.get_preferred_applications()
        app = apps.get('yaml-editor', apps['editor'])
    base, ext = os.path.splitext(path)
    if ext != ".yaml":
        raise ValueError("path should have a .yaml extension, got {path}")
    subprocess.call([app, path])
    if wait:
        dialogs.showinfo("Press OK when finished editing")


def _load_config(path: str) -> dict:
    assert os.path.splitext(path)[1] == '.yaml'
    loader = yaml.YAML()
    return loader.load(open(path))


def _purify(obj):
    """
    Convert obj (recursively) to its pure python form, consisting
    only of basic types: int, float, str, bool, dict, list and None

    The use case is to remove subclasses of objects made during
    serialization by ruamel.yaml

    Args:
        obj: the object to purify

    Returns:
        A pure python version of object
    """
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        return float(obj)
    elif isinstance(obj, str):
        return str(obj)
    elif isinstance(obj, int):
        return int(obj)
    elif isinstance(obj, list):
        return [_purify(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _purify(v) for k, v in obj.items()}
    else:
        raise TypeError(f"object {obj} of type {type(obj)} not supported")


@lru_cache(maxsize=1)
def _get_default_yamldict():
    return _load_config(default_config_path())


@lru_cache(maxsize=1)
def _get_allowed_keys() -> set:
    fallback = _get_fallback_dict()
    return set(fallback.keys())


@lru_cache(maxsize=1)
def _get_fallback_dict() -> dict:
    return _purify(_get_default_yamldict())


class ConfigDict(dict):
    _allowedkeys = _get_allowed_keys()

    def __init__(self, d, fallback=None, name=None, frompath=None):
        """
        Don't call this directly as a user: use make_config() or read_config(path_to_config)

        Args:
            d: a dict as read by ruamel.yaml (see _load_config)
            fallback: a python dict with defaults
        """
        self.name: str = name
        self._yamldict = d
        self._fallback: dict = fallback
        self._validator = _validator
        self._immutable = False
        self._help = _help
        self._frompath = frompath
        self._dict = _purify(self._yamldict)
        self.validate()

    def make_immutable(self, state=True):
        self._immutable = state

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def validate(self) -> None:
        v = self._validator
        if v is None:
            return
        for key, value in self._dict.items():
            func = v.get(key)
            if func and not func(value):
                raise ValueError(f"Validation error: key {key} can't be set to {value}")

    def __setitem__(self, key, value) -> None:
        if self._immutable:
            raise ImmutableError("This config is immutable")
        if key not in self._allowedkeys:
            raise KeyError(f"Key {key} not known")
        if self._validator and key in self._validator:
            ok = self._validator[key](value)
            if not ok:
                raise ValueError(f"Can't set {key} to {value}")
        self._yamldict[key] = value
        if isinstance(value, dict):
            v = {}
            v.update(value)
            value = v
        self._dict[key] = value

    def __eq__(self, other):
        if isinstance(other, ConfigDict):
            return self._dict == other._dict
        else:
            raise TypeError("= operation only between instances of ConfigDict")

    def copy(self) -> ConfigDict:
        return ConfigDict(self._yamldict.copy(), name=self.name, fallback=self._fallback)

    def __getitem__(self, key: str):
        if key not in self._allowedkeys:
            raise KeyError(f"Key {key} not allowed")
        if key in self._dict:
            return self._dict[key]
        if self._fallback is not None and key in self._fallback:
            return self._fallback[key]
        raise KeyError("Key allowed but not found??")

    def edit(self) -> None:
        """
        Edit this config inplace. This will open the config file in the default
        application for opening .yaml files. After finishing editing,
        save the file and click OK in the confirmation dialog

        In order to configure a specific application to open the config,
        use:
            appconfig['app.yaml'] = path_to_app
        """
        path = tempfile.mktemp(suffix=".yaml")
        self.save(path)
        _edit_config(path, wait=True)
        newconfig = read_config(path)
        self._replacewith(newconfig)

    def _replacewith(self, replacement: ConfigDict, copy=False):
        yamldict = replacement._yamldict
        if copy:
            yamldict = yamldict.copy()
        self._yamldict = yamldict
        self._dict = _purify(yamldict)

    def update(self, d):
        assert set(d.keys()).issubset(self._allowedkeys)
        self._dict.update(_purify(d))
        self._yamldict.update(d)
        self.validate()

    def get(self, key: str, default=None):
        if key in self._dict:
            return self._dict.get(key)
        if self._fallback is not None:
            return self._fallback.get(key)
        return default

    def __repr__(self) -> str:
        return yaml.dump(self._dict, default_flow_style=None)

    def save(self, outfile=None) -> None:
        """
        Save configuration to a .yaml file

        Example:
            config.save("myconfig.yaml")

        Args:
            outfile: the path to save this config to

        """
        if outfile is None:
            if self._frompath is None:
                raise ValueError("This ConfigDict has no associated filepath. You need to give an outfile")
            outfile = self._frompath
        outfile = os.path.expanduser(outfile)
        ext = os.path.splitext(outfile)[1]
        if ext != ".yaml":
            raise ValueError("outfile should have the extension .yaml")
        with open(outfile, "w") as f:
            y = yaml.YAML()
            y.default_flow_style = None
            y.dump(self._yamldict, f)

    def help(self, key: str, fmt='short') -> Opt[str]:
        """
        Get help for a given key in this config. Returns None
        if no help available

        Example::

            helptxt = config.help("staffrange")

        Args:
            key: the key to get help
            fmt: the format of the help. One of 'short' and 'long'
        """
        if fmt == 'short':
            txt = self._help.get(key)
            return txt or self.help(key, fmt='long')
        attribs = self._yamldict.ca.items.get(key)
        if not attribs:
            return None
        comment: yaml.CommentToken = attribs[2]
        if not comment:
            return None
        txt = comment.value.replace("#", "").replace("\n", "").strip()
        return txt

    def default(self, key: str):
        """
        Get default value for a given key
        """
        return _get_fallback_dict()[key]



def read_config(path: str, name=None) -> ConfigDict:
    """
    Read a configuration file with all needed settings to generate
    a transcription of a spectrum.

    Args:
        path: path to the .yaml file as generated by config.save
        name: give this config a name (optional)

    Returns:
        A ConfigDict. This can be further configured and passed
        to generate_score
    """
    path = os.path.expanduser(path)
    return ConfigDict(_load_config(path), name=name, fallback=_get_fallback_dict(), frompath=path)


def make_config(**kws) -> ConfigDict:
    """
    Make a new configuration, overriding the default config

    Example::
        import sndtrck
        import sndscribe
        cfg = make_config()
        cfg['staffrange'] = 36
        cfg['numvoices'] = 8
        partials = sndtrck.analyze('/my/soundfile.wav', 40)
        partials = partials.filter(mindur=0.02, minbps=2, maxfreq=5000, minfreq=30)
        result = sndscribe.generate_score(partials, cfg)
        result.score.writepdf('myscore.pdf', openpdf=True)

    NB: Because of the complexity of creating a config, at the REPL
        it is convenient to use 
        
        myconfig = make_config().edit()
        
        This saves your config to a temporary file which is open in a text editor.
        Here you can edit the config. The changes will be loaded after saving
        the file and pressing ok in the confirmation dialog.
    """
    out = get_default_config().copy()
    for key, value in kws.items():
        out[key] = value
    return out


def dyncurve_from_config(config: ConfigDict) -> dynamics.DynamicsCurve:
    """
    Create a DynamicsCurve from the description in config. Raises
    KeyError if any of the keys needed are missing
    """
    return dynamics.DynamicsCurve.fromdescr(shape=config['dyncurve_shape'],
                                            mindb=config['dyncurve_mindb'],
                                            maxdb=config['dyncurve_maxdb'],
                                            dynamics=config['dynamics'])



@lru_cache(maxsize=1)
def get_default_config() -> ConfigDict:
    defaultdict = _get_default_yamldict()
    cfg = ConfigDict(defaultdict, fallback=None)
    cfg.make_immutable()
    return cfg


class RenderConfig:

    def __init__(self,
                 config: ConfigDict,
                 tempo:float = None,
                 timesig:Tuple[int, int] = None,
                 dyncurve: dynamics.DynamicsCurve=None
                 ):
        """
        A RenderConfig holds all information needed to render a score.

        Args:
            config: a ConfigDict as returned by read_config or make_config
            tempo: the tempo of the score, or None to use the tempo defined in the config
            timesig: the timesignature of the score, or None to use the timesig definen in the config
            dyncurve: a DynamicsCurve. If not given, the config is used to create one
                by calling DynamicsCurve.fromdescr

        NB: the possibility to override tempo and timesig is here to allow, in the future, to
            use non-static values for these, such as a score structure with changing time signatures and tempi
        """
        assert timesig is None or isinstance(timesig, (str, tuple))
        assert tempo is None or isinstance(tempo, (int, float))
        assert isinstance(config, ConfigDict), f"Expected a ConfigDict, got {config} of type {type(config)}"
        tempo = tempo if tempo is not None else config['tempo']
        timesig = timesig if timesig is not None else config['timesig']
        self.tempo: Fraction = Fraction(tempo)
        self.timesig: Tuple[int, int] = as_timesig_tuple(timesig)
        self.config: ConfigDict = config
        if dyncurve is not None:
            self.dyncurve = dyncurve
        else:
            try:
                self.dyncurve = dyncurve_from_config(self.config)
            except KeyError as e:
                raise KeyError("Missing keys to construct dynamics curve from config")

    def __getitem__(self, key: str):
        return self.config[key]

    def get(self, key:str, default=None):
        return self.config.get(key, default)

    def clone(self, **kws) -> RenderConfig:
        tempo = kws.get('tempo', self.tempo)
        timesig = kws.get('timesig', self.timesig)
        dyncurve = kws.get('dyncurve', self.dyncurve)
        return RenderConfig(config=self.config.copy(), tempo=tempo, timesig=timesig, dyncurve=dyncurve)