#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import setuptools
from setuptools import setup
import shutil


setuptools._dont_write_bytecode = True

if sys.argv[-1] == 'publish':
    os.system('./update-readme')
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()


APPNAME = "sndscribe"


def _get_win_folder_with_pywin32(csidl_name):
    from win32com.shell import shellcon, shell
    dir = shell.SHGetFolderPath(0, getattr(shellcon, csidl_name), 0, 0)
    # Try to make this a unicode path because SHGetFolderPath does
    # not return unicode strings when there is unicode data in the
    # path.
    try:
        # dir = unicode(dir)
        dir = str(dir)

        # Downgrade to short path name if have highbit chars. See
        # <http://bugs.activestate.com/show_bug.cgi?id=85099>.
        has_high_char = False
        for c in dir:
            if ord(c) > 255:
                has_high_char = True
                break
        if has_high_char:
            try:
                import win32api
                dir = win32api.GetShortPathName(dir)
            except ImportError:
                pass
    except UnicodeError:
        pass
    return dir


def user_data_dir(appname=None, appauthor=None, version=None, roaming=False):
    """
    taken from appdirs
    """
    system = sys.platform
    if system == "win32":
        if appauthor is None:
            appauthor = appname
        const = roaming and "CSIDL_APPDATA" or "CSIDL_LOCAL_APPDATA"
        path = os.path.normpath(_get_win_folder_with_pywin32(const))
        if appname:
            if appauthor is not False:
                path = os.path.join(path, appauthor, appname)
            else:
                path = os.path.join(path, appname)
    elif system == 'darwin':
        path = os.path.expanduser('~/Library/Application Support/')
        if appname:
            path = os.path.join(path, appname)
    else:
        path = os.getenv('XDG_DATA_HOME', os.path.expanduser("~/.local/share"))
        if appname:
            path = os.path.join(path, appname)
    if appname and version:
        path = os.path.join(path, version)
    return path


def get_scriptsdir():
    """
    Get the path to the xml2ly scripts directory. 
    """
    datadir = user_data_dir(APPNAME)
    scriptsdir = os.path.join(datadir, "xml2ly")
    return scriptsdir


def clean_previous_install():
    scriptsdir = get_scriptsdir()
    if not scriptsdir:
        print(">>>>> failed to get scriptsdir")
        return 
    if os.path.exists(scriptsdir):
        print(">>>>> cleaning previous installation")
        shutil.rmtree(scriptsdir)
    else:
        print(">>>>> this is a first-time install")


clean_previous_install()


setup(
    name='sndscribe',
    version='0.2.1',
    description='Translate the spectrum of an audio file into musical notation',
    long_description=readme,
    author='Eduardo Moguillansky',
    author_email='eduardo.moguillansky@gmail.com',
    packages=[
        'sndscribe',
    ],
    package_data={'sndscribe':['xml2ly/*.py_']},
    setup_requires=[
        'appdirs'
    ],
    install_requires=[
        "six",
        "decorator",
        "numpy>=1.7",
        "emlib",
        "bpf4",
        "sndtrck",
        "appdirs",
        "pyyaml"],
    license="GPL",
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
