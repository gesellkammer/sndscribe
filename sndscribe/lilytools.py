"""
Routines to interact with lilypond
"""
import os
import sys
import subprocess
from .error import *
from . import envir
from .typehints import *

from emlib.lib import normalize_path

logger = envir.logger


def _loggedcall(args:U[str, List[str]], shell=False) -> Tup[str, str]:
    """
    Call a subprocess with args

    Returns retcode, output
    """
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
    retcode = proc.wait()
    out = proc.stdout.read().decode("utf-8")
    if retcode == 0:
        error = ""
    else:
        error = (retcode, proc.stderr.read().decode("utf-8"))
    return error, out


def call_lilypond(lilyfile: str, outfile: str):
    lilybin = appconfig.get('lilypond.path')
    if not lilybin:
        lilybin = detect_lilypond()

    if not os.path.exists(lilyfile):
        raise IOError(f"file not found: {lilyfile}")

    if sys.platform == "win32":
        s = f'"{lilybin}" -o "{pdffile}" "{lilyfile}"'
        error, output = _loggedcall(s, shell=True)
    else:
        error, output = _loggedcall([lilybin, '-o', basefile, lilyfile])

    if not os.path.exists(outfile):
        raise LilypondError(f"Failed to produce an output file: {outfile} does not exist")
    
    if error:
        logger.error(f"Error while running lilypond: {output}")
        logger.error(error)


def lily2pdf(lilyfile:str, outfile:str=None) -> str:
    """
    Call lilypond to produce a pdf file

    Args:
        lilyfile: the lilypond file
        outfile: if not given, a filename is autogenerated

    Returns:
        The generated pdffile

    Raises:
        LilypondError if failed to produce a pdf file
    """
    lilyfile = normalize_path(lilyfile)
    if outfile is None:
        outfile = os.path.splitext(source)[0] + '.pdf'
    outfile = normalize_path(outfile)
    call_lilypond(lilyfile, outfile)
    return outfile

