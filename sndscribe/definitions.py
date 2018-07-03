from __future__ import absolute_import
import bpf4 as bpf
import os
from .typehints import *

APPNAME = "sndscribe"

platform_name = os.uname()[0]

STR2CLASS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 8, 'B': 10}

# Noteheadsize as defined by lilypond
DYNAMIC_TO_RELATIVESIZE = {
    'pppp': -6,
    'ppp' : -5,
    'pp'  : -3,
    'p'   : -2,
    'mp'  : -1,
    'mf'  : 0,
    'f'   : 2,
    'ff'  : 3,
    'fff' : 4,
    'ffff': 6
}

# musicxml font-size attribute
DYNAMIC_TO_FONTSIZE = {
    'pppp': 10,
    'ppp':  12,
    'pp':   16,
    'p':    18,
    'mp':   20,
    'mf':   24,
    'f':    28,
    'ff':   32,
    'fff':  36,
    'ffff': 40   
}

# musicxml font-size attribute as css-font-size
DYNAMIC_TO_CSSFONTSIZE = {
    'pppp': 'xx-small',
    'ppp':  'xx-small',
    'pp':   'x-small',
    'p':    'small',
    'mp':   'medium',
    'mf':   'medium',
    'f':    'large',
    'ff':   'x-large',
    'fff':  'xx-large',
    'ffff': 'xx-large'
}


REGULAR_NOTETYPES = {
    4:   'quarter',
    8:   'eighth',
    16:  '16th',
    32:  '32nd',
    64:  '64th',
    128: '128th'
}


IRREGULAR_NOTETYPES = {
    # note_type, dots
    (7,4): ('quarter', 2),
    (3,2): ('quarter', 1),
    (5,4): ('quarter', 0),  #
    (1,1): ('quarter', 0),
    (7,8): ('eighth', 2),
    (3,4): ('eighth', 1),
    (5,8): ('eighth', 0),    #
    (1,2): ('eighth', 0),
    (7,16):('16th', 2),
    (3,8): ('16th', 1),
    (5,16):('16th', 0),
    (1,4): ('16th', 0),
    (7,32):('32nd', 2),
    (3,16):('32nd', 1),
    (1, 8):('32nd', 0),
    (7,64):('64th', 2),
    (3,32):('64th', 1),
    (1,16):('64th', 0),

    (1,3):('eighth', 0),
    (2,3):('quarter', 0),

    (1,5):('16th', 0),
    (2,5):('eighth', 0),
    (3,5):('eighth', 1),
    (4,5):('quarter', 0),

    (1,6):('16th', 0),
    (5,6):('quarter', 0),   #

    (1,7):('16th', 0),
    (2,7):('eighth', 0),
    (3,7):('eighth', 1),
    (4,7):('quarter', 0),
    (5,7):('quarter', 0),   #
    (6,7):('quarter', 1),

    (1,9):('32nd', 0),
    (2,9):('16th', 0),
    (3,9):('16th', 1),
    (4,9):('eighth', 0),
    (5,9):('eighth', 0),    #
    (6,9):('eighth', 1),
    (7,9):('eighth', 2),
    (8,9):('quarter', 0),

    (1,10):('32nd', 0),
    (3,10):('16th', 1),
    (7,10):('eighth', 2),

    (1,11):('32nd', 0),
    (2,11):('16th', 0),
    (3,11):('16th', 1),
    (4,11):('eight', 0),
    (5,11):('eigth', 0),
    (6,11):('eigth', 1),
    (7,11):('eigth', 2),
    (8,11):('quarter', 0),
    (9,11):('quarter', 0),
    (10,11):('quarter', 1),


    (1,12):('32nd', 0),
    (2,12):('16th', 0),
    (3,12):('16th', 1),
    (4,12):('eighth', 0),
    (5,12):('eighth', 0),    #
    (6,12):('eighth', 1),
    (7,12):('eighth', 2),    #
    (8,12):('quarter', 0),
    (9,12):('quarter', 0),   #

    (11,12):('quarter', 1),  #

    (1, 15):('32nd', 0),
    (2, 15):('16th', 0),
    (3, 15):('16th', 1),
    (4, 15):('eighth', 0),
    (5, 15):('eighth', 0),
    (6, 15):('eighth', 1),
    (7, 15):('eighth', 2),
    (8, 15):('quarter', 0),
    (9, 15):('quarter', 0),
    (10, 15):('quarter', 0),
    (11, 15):('quarter', 0),
    (12, 15):('quarter', 1),
    (13, 15):('quarter', 1),
    (14, 15):('quarter', 1),

    (1, 16):('64th', 0),
    (2, 16):('32nd', 0),
    (3, 16):('32nd', 1),
    (4, 16):('16th', 0),
    (6, 16):('16th', 1),
    (7, 16):('16th', 2),
    (8,16):('eigth', 0),
    (11,16):('eigth', 0),
    (12,16):('eigth', 1),
    (15,16):('eighth', 3),

    (1, 32):('128th', 0),
    (2, 32):('64th', 0),
    (3, 32):('64th', 1),
    (7, 32):('32th', 2),
    (11,32):('16th', 0),
    (15, 32):('16th', 3),
    (31, 32):('eigth', 4),
}

NOTETYPES = list(REGULAR_NOTETYPES.values())

MUSICXML_ACCIDENTALS = {
   -1.00: 'flat',
   -0.75: 'quarter-flat',
   -0.50: 'quarter-flat',
   -0.25: 'natural',
   0.00: 'natural',
   0.25: 'natural',
   0.50: 'quarter-sharp',
   0.75: 'quarter-sharp',
   1.00: 'sharp',
   1.25: 'sharp',
   1.50: 'three-quarters-sharp',
   1.75: 'three-quarters-sharp'}


class XmlNotehead(NamedTuple):
    shape: str
    filled: bool


MUSICXML_NOTEHEADS = [
    XmlNotehead("normal", filled=False),
    XmlNotehead("square", filled=False),
    XmlNotehead("diamond", filled=False),
    XmlNotehead("harmonic", filled=False),
    XmlNotehead("x", filled=False),
    XmlNotehead("circle-x", filled=False)
]  # type: List[XmlNotehead]


bw_to_noteheadindex = bpf.linear(0, 0, 1, len(MUSICXML_NOTEHEADS)-1)


LILYPATH = {
    'Darwin':'/Applications/LilyPond.app/Contents/Resources/bin/lilypond',
    'Linux':'/usr/bin/lilypond'
}[platform_name]


MUSICXML_DIVISIONS = 55440    # exactitud de la grid en la traduccion a musicxml
MUSICXML_TENTHS = 40
DEFAULT_POSSIBLE_DIVISIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)


IRREGULAR_TUPLES = (
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 9),
    (9, 10),
    (9, 11),
    (9, 13),
    (9, 14),
    (9, 16),
    (10, 11),
    (10, 12),
    (10, 13),
    (10, 15),
    (10, 16),
    (11, 12),
    (11, 13),
    (11, 14),
    (11, 15),
    (11, 16),
)


NOTE_DICT = dict(((0, 'C0'), (1, 'C1'), (2, 'D0'), (3, 'D1'), (4, 'E0'),
                  (5, 'F0'), (6, 'F1'), (7, 'G0'), (8, 'G1'), (9, 'A0'),
                  (10, 'A1'), (11, 'B0')))

