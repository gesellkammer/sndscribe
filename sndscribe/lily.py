"""
Routines to interact with lilypond
"""
import os
import sys
import subprocess
import pkg_resources as pkg
import appdirs
from .error import *
from . import envir
from .typehints import *

logger = envir.logger

def _loggedcall(args:U[str, List[str]], shell=False) -> Tup[Opt[str], str]:
    """
    Call a subprocess with args

    Returns retcode, output
    """
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
    retcode = proc.wait()
    out = proc.stdout.read().decode("utf-8")
    if retcode == 0:
        error = None
    else:
        error = (retcode, proc.stderr.read().decode("utf-8"))
    return error, out
    

def _get_resource(path:str) -> str:
    return pkg.resource_filename(pkg.Requirement.parse("sndscribe"), path)


def _get_script_path(script:str) -> str:
    scriptdir = envir.get_scriptsdir()
    return os.path.join(scriptdir, script)

def _run_musicxml2ly(xmlfile:str, lilyfile:str=None, midi=True, bundled=True) -> str:
    """
    Returns the path to the generated liylpond file

    midi:
        also generate midi output 
    bundled: 
        if True, we use the bundled version of musicxml2ly,
        which enables features like notehead size
    """
    lilyfile = lilyfile or os.path.splitext(xmlfile)[0] + '.ly'
    if os.path.exists(lilyfile):
        logger.debug("Asked to generate a Lilypond file, but the output file "
                     "already exists. Removing")
        os.remove(lilyfile)
    options = ['--no-beaming']
    if midi:
        options.append("--midi")
    if not bundled:
        if sys.platform == 'win32':
            midistr = "--midi" if midi else ""
            s = f'musicxml2ly --no-beaming {midistr} -o "lilyfile" "xmlfile"'
            error, output = _loggedcall(s, shell=True)
        else:
            musicxml2ly = subprocess.check_output(["which", "musicxml2ly"]).decode("utf-8")
            if not os.path.exists(musicxml2ly):
                raise RuntimeError("musicxml2ly not found in PATH. Is lilypond installed")
            args = [musicxml2ly] + options + ['-o', lilyfile, xmlfile]
            error, output = _loggedcall(args, shell=False)
    else:
        # xml2ly = _get_resource("xml2ly/musicxml2ly.py")
        xml2ly = _get_script_path("musicxml2ly.py")
        assert os.path.exists(xml2ly)
        args = ["python2.7", xml2ly] + options + ['-o', lilyfile, xmlfile]
        logger.debug(f"musicxml2ly called with args: {args}")
        error, output = _loggedcall(args, shell=False)
    logger.debug(output)
    if error:
        logger.error(error)
        raise LilypondError("Error while converting to lilypond")
    if not os.path.exists(lilyfile):
        raise LilypondError("No lilypond file was generated!")
    return lilyfile


def xml2lily(xmlfile:str, lilyfile:str=None, midi=True, 
             method="musicxml2ly") -> str:
    """
    Convert a musicxml file to lilypond format.
    Returns the path of the created lilypond file

    Raise LilypondError if unsuccessful
    """
    methods = ("musicxml2ly",)
    if method == "musicxml2ly":
        lilyfile = _run_musicxml2ly(xmlfile, lilyfile=lilyfile, midi=midi, bundled=True)
    else:
        raise ValueError(f"method {method} not supported. It should be one of {methods}")
    return lilyfile


def lily2pdf(lilyfile:str, outfile:Opt[str]=None) -> Opt[str]:
    if outfile is None:
        outfile = lilyfile
    basefile = os.path.splitext(outfile)[0]
    pdffile = basefile + '.pdf'
    if sys.platform == "win32":
        s = f'lilypond -o "{pdffile}" "{lilyfile}"'
        error, output = _loggedcall(s, shell=True)
    else:
        lilybinary = envir.detect_lilypond()
        error, output = _loggedcall([lilybinary, '-o', basefile, lilyfile])
    if not os.path.exists(pdffile):
        logger.error(f"Failed to produce a pdf file: {pdffile}")
        return None
    if error:
        logger.error(f"Error while running lilypond: {output}")
        logger.error(error)
    return pdffile


# remove_type_imports()
# del remove_type_imports

