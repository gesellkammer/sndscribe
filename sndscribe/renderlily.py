from __future__ import annotations
import os
import warnings
from string import Template

from emlib import lib

from .score import Score
from .staff import Staff
from .conversions import lily_clef, lily_pitch, lily_duration
from .tools import previous_power_of_two
from .scorefuncs import notated_duration
from .note import Note, Rest
from . import envir
from .definitions import DYNAMIC_TO_RELATIVESIZE

from .typehints import List


# Templates


# The header of a .ly file
_toplevel = Template(r"""
\version "$version"

$includes

% inline code
$inline

\pointAndClickOff

% Comment or adapt next line as needed (default is 20)
#(set-global-staff-size $staffsize)

\header {
    $headerinsert
    
    % Do not display the default LilyPond footer
    tagline = ##f
}

\paper {
  paper-width          = $paperwidth\cm
  paper-height         = $paperheight\cm
  indent               = 2\cm
  %short-indent        = 0\cm
  
  %page-count          = 1
  %system-count        = 1
}

\layout {
  \context {
    \Score
    autoBeaming = ##$autobeaming % to display tuplets brackets
    $globaloverrides
  }
}

""")

# This is used for each staff, goes inside the score section ($staffs)



def _make_toplevel(version:str,
                   papersize='a4',
                   orientation='portrait',
                   staffsize=12,
                   autobeaming=True,
                   includes:List[str] = None,
                   inlinecode:str = None,
                   title:str = None,
                   global_overrides=''
                   ) -> str:
    """

    Args:
        version: the version to use in the \version tag
        papersize: a DIN paper size (a4, a3)
        orientation: landscape or portrait
        staffsize: the staff size in points
        autobeaming: should lilypond do the beaming itself?
        includes: a list of include files
        inlinecode: code to be included as part of the top-level definitions
        title: the title of the score
        global_overrides: code to be included at the score context, for global overrides

    Returns:
        The lilypond code
    """
    heightmm, widthmm = lib.page_dinsize_to_mm(papersize, orientation)

    if not includes:
        includestr = ""
    else:
        includelines = []
        for inc in includes:
            if not os.path.exists(inc):
                warnings.warn("include file {inc} not found")
            line = f'\\include "{inc}"'
            includelines.append(line)
        includestr = "\n".join(includelines)

    inlinestr = inlinecode or ''

    print("******** includestr: ", includestr)

    headerlines = []
    if title is not None:
        headerlines.append(f'    title = "{title}"')

    headerinsert = "\n".join(headerlines)

    return _toplevel.substitute({
        'version': version,
        'paperwidth' : f"{widthmm/10:.1f}",
        'paperheight': f"{heightmm/10:.1f}",
        'staffsize': str(staffsize),
        'autobeaming': "t" if autobeaming else "f",
        'includes': includestr,
        'inline': inlinestr,
        'headerinsert': headerinsert,
        'globaloverrides': global_overrides
    })


def _generate_part_name(partidx:int) -> str:
    """
    A lilypond part can't have digits. Here we translate an index (e.g 0)
    to a name liek PartAAA

     0 -> 000 -> AAA
    34 -> 034 -> ADE

    """
    preffix = "Part"
    orig = f"{partidx:04d}"
    zero = 48  # ord("0")
    A = 65     # ord("A")
    letters = "".join(chr(A + ord(digit) - zero) for digit in orig)
    return preffix + letters


_part_template = Template(r"""
    $part_name = \absolute {
        \numericTimeSignature \time $timesig
        \clef "$clef"
        \tempo 4=$tempo
        $music
    }
""")


def make_part(staff: Staff, part_name: str, microtones=True) -> str:

    assert len(staff.voices) == 1

    timesig = staff.renderconfig.timesig
    voice = staff.voices[0]

    include_dynamics = staff.renderconfig['show_dynamics']
    notesize_follows_dynamic = staff.renderconfig['notesize_follows_dynamic']
    last_dynamic = current_dynamic = None

    vs = []

    if not microtones:
        raise ValueError("only microtones mode supported")

    for m in voice.measures:
        for p in m:
            p.join_tied_notes()
            p.verify()
    voice.check_ties()

    for measnum, measure in enumerate(voice.measures):
        for pulse in measure.pulses:
            actual_notes = pulse.subdivision
            normal_notes = previous_power_of_two(actual_notes)
            tuplet_open = False
            if actual_notes != normal_notes:
                vs.append(rf"\tuplet {actual_notes}/{normal_notes} {{")
                tuplet_open = True

            pulsedur = pulse.pulse_dur

            for note in pulse.notes:
                durtype, nbeams, dots = notated_duration(note.dur, pulse.subdivision, pulsedur, timesig)
                lilydur = lily_duration(durtype, dots)
                if isinstance(note, Rest):
                    lilynote = "r" + lilydur
                    last_dynamic = None
                elif isinstance(note, Note):
                    if notesize_follows_dynamic:
                        fontsize = DYNAMIC_TO_RELATIVESIZE.get(note.dynamic, 0)
                        if fontsize != 0:
                            vs.append(rf"\once \override NoteHead #'font-size = #{fontsize}")
                    lilynote = lily_pitch(note.pitch) + lilydur
                    if note.tied:
                        lilynote += "~"
                    current_dynamic = note.dynamic
                    assert current_dynamic is not None
                    if current_dynamic != last_dynamic:
                        last_dynamic = current_dynamic
                        if include_dynamics:
                            lilynote += f"\{current_dynamic}"
                else:
                    raise TypeError(f"Expected a Note or a Rest, got {type(note)}")
                vs.append(lilynote)

            if tuplet_open:
                vs.append("}")

            vs.append("   ")
        vs.append(f" | % {measnum + 1} \n")


    # ~~~~~~~~~~ finished notes ~~~~~~~~~~~~~

    timesigstr = "%d/%d" % staff.renderconfig.timesig
    clefstr = lily_clef(staff.clef)
    musicstr = " ".join(vs)
    s = _part_template.substitute(part_name=part_name,
                                  music=musicstr,
                                  timesig=timesigstr,
                                  clef=clefstr,
                                  tempo=str(staff.renderconfig.tempo))
    return s


def _indent_lily(lycode: str, indentsize=2, level=0) -> List[str]:
    """
    indent the given code, returns an iterator to the lines

    indented = "\n".join(_indent_lily(unindented_code)
    """
    outlines = []
    bracket_level = 0
    shift_level = 0
    initial_level = level
    spaces = "                                                                 "
    for line in lycode.splitlines():
        indented_line = spaces[:level * indentsize] + line.strip()
        yield indented_line
        bracket_balance = line.count("{")-line.count("}")
        bracket_level += bracket_balance
        shift_balance = line.count("<<")-line.count(">>")
        shift_level += shift_balance
        level = initial_level + bracket_level + shift_level


_staffdef_template = Template(r"""
\new Staff
<<
    \set Staff.instrumentName = "$instrname"
    \context Staff <<
        \context Voice = "$partname" { \$partname }
    >>
>>      
""")

# The score section
_score_section = Template(r"""
% The score definition
\score {
    <<
        $staffs
    >>
}
""")


def _microtonal_definitions_path() -> str:
    return envir.get_datafile("microtonal.ily")
    # return '/home/em/dev/lilypond/microtonal.ily'


def score_to_lily(score: Score, outfile: str=None, inline=True) -> str:
    """
    Render score as a lilypond file

    inline: if True, include files are inlined in the .ly file, otherwise
            they are included and must be present for rendering
    """
    cfg = score.renderconfig

    version = "2.19"

    title = score.title

    microtonal_defs = _microtonal_definitions_path()
    if microtonal_defs is None:
        raise IOError("Could not find microtonal.ily file")

    if inline:
        inlinecode = open(microtonal_defs).read()
        includes = None
    else:
        inlinecode = None
        includes = [microtonal_defs]

    overrides = []
    dynamics_size = score.renderconfig.get('lily_dynamics_size', 0)
    if dynamics_size != 0:
        overrides.append(rf"\override DynamicText #'font-size = #{dynamics_size}")

    header = _make_toplevel(version=version,
                            papersize=cfg['pagesize'],
                            orientation=cfg['pagelayout'],
                            includes=includes,
                            inlinecode=inlinecode,
                            title=title,
                            global_overrides="\n".join(overrides))
    sections = [header]
    part_names = []
    parts = []

    for i, staff in enumerate(score.staffs):
        part_name = _generate_part_name(i)
        part_names.append(part_name)
        lypart = make_part(staff, part_name=part_name)
        parts.append(lypart)

    sections.extend(parts)

    staffdefs = [_staffdef_template.substitute(partname=part_name, instrname=str(i))
                 for i, part_name in enumerate(part_names)]

    score_section = _score_section.substitute(staffs="\n".join(staffdefs))
    sections.append(score_section)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    lilyfile = "\n".join(sections)

    if outfile:
        outfile = lib.normalize_path(outfile)
        with open(outfile, "w") as f:
            for line in _indent_lily(lilyfile):
                f.write(line)
                f.write("\n")
    else:
        for line in _indent_lily(lilyfile):
            print(line)




