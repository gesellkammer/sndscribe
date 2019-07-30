from __future__ import absolute_import
import os
from os import getenv
import subprocess
import json
import appdirs
import logging
from .definitions import APPNAME
from .error import PlatformNotSupported
import shutil
import glob
import pkg_resources as pkg


_scripts_initialized = False

platform_name = os.uname()[0].lower()

logger = logging.getLogger(APPNAME)


def _first(*values):
    for v in values:
        if v is not None:
            return v
    raise ValueError("no valid values in %s" % str(values))


defaultsettings = {
    "linux": {
        "apps": {
            "yaml-editor": _first(getenv("VISUAL"), getenv("EDITOR"), "gedit", "nano"),
            "editor": _first(getenv("VISUAL"), getenv("EDITOR"), "gedit", "nano"),
        }
    },
    "darwin": {
        "apps": {
            "yaml-editor": "nano",
            "editor": "nano"
        }
    }
}


def check_output(args):
    try:
        out = subprocess.check_output(args)
        return out
    except subprocess.CalledProcessError:
        return None


def get_config_dir():
    configdir = appdirs.user_config_dir(APPNAME)
    if not os.path.exists(configdir):
        logger.debug("setting up config dir for the first time: %s" % configdir)
        os.mkdir(configdir)
    return configdir


def get_preferred_applications():
    settings = get_settings()
    apps = settings.get("apps")
    if apps is not None:
        return apps
    else:
        logger.debug("No apps in user settings."
                     "Falling back to apps defined in the default settings")
        return defaultsettings.get(platform_name)['apps']


def get_settings():
    settingspath = get_settings_path()
    if os.path.exists(settingspath):
        # read settings (a json file)
        settings = json.load(open(settingspath))
    else:
        # TODO
        settings = defaultsettings.get(platform_name)
        if not settings:
            raise RuntimeError("platform not supported")
        logger.debug("Saving settings for the first time at: " + settingspath)
        save_settings(settings)
    return settings


def get_settings_path():
    configdir = get_config_dir()
    if not os.path.exists(configdir):
        logger.debug("config dir not found, creating at: " + configdir)
        os.mkdir(configdir)
    settingsbase = APPNAME + ".json"
    settingspath = os.path.join(configdir, settingsbase)
    return settingspath    


def save_settings(settingsdict):
    path = get_settings_path()
    if os.path.exists(path):
        logger.debug("Overwriting settings")
        with open(path, "w") as f:
            json.dump(f, indent=True, sort_keys=True)

            
def detect_lilypond():
    if platform_name == 'linux':
        path = check_output(["which", "lilypond"])
        if path is not None:
            path = path.strip()
            assert os.path.exists(path)
            return path
        paths = ("/usr/bin/lilypond",
                 "/usr/local/bin/lilypond",
                 "~/.local/bin/lilypond")
        paths = [os.path.expanduser(p) for p in paths]
        for path in paths:
            if os.path.exists(path):
                return path
        return None
    elif platform_name == 'darwin':
        paths = ['/Applications/LilyPond.app/Contents/Resources/bin/lilypond']
        paths = [os.path.expanduser(p) for p in paths]
        for path in paths:
            if os.path.exists(path):
                return path
        return None
    else:
        raise PlatformNotSupported("This platform is not supported")


def open_pdf(pdf):
    """Open `pdf` in the default pdf viewer on this system"""
    os.system("open %s" % pdf)


def get_scriptsdir():
    """
    Get the path to the xml2ly scripts directory. 
    """
    datadir = appdirs.user_data_dir(APPNAME)
    scriptsdir = os.path.join(datadir, "xml2ly")
    return scriptsdir


def get_datafile(filename):
    datadir = pkg.resource_filename(pkg.Requirement.parse("sndscribe"), "sndscribe/data")
    target = os.path.join(datadir, filename)
    if not os.path.exists(target):
        return None
    return target

def xml2ly_init():
    global _scripts_initialized
    if _scripts_initialized:
        logger.debug("scripts already initialized")
    # check if this is necessary
    # In the setup script we delete the scripts directory, so
    # it should not exist when we first copy the scripts
    _scripts_initialized = True
    scriptsdir = get_scriptsdir()
    if os.path.exists(scriptsdir):
        logger.debug("xml2ly_init: scripts already copied, skipping")
        return
    logger.debug("xml2ly_init: copying scripts to scripts dir")
    os.makedirs(scriptsdir)
    xml2lydir = pkg.resource_filename(pkg.Requirement.parse("sndscribe"), "sndscribe/xml2ly")
    hackedfiles = glob.glob(os.path.join(xml2lydir, "*.py_"))
    if hackedfiles:
        for f in hackedfiles:
            basename = os.path.split(f)[1]
            f2 = os.path.join(scriptsdir, os.path.splitext(basename)[0] + ".py")
            shutil.copy(f, f2)