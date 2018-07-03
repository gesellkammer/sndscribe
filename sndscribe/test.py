from __future__ import division
import sndtrck
import os
from .score import *
from .dynamics import *


def test1():
    notes = [
        Note(60.1, 0, 0.05, 1),
        Note(50, 0, 0.02, .1),
        Note(51, 0, 0.02, .1),
        Note(52, 0, 0.02, .2),
        Note(53, 0, 0.02, .45),
        Note(54, 0, 0.02, .12),
        Note(55, 0, 0.02, .04),
        Note(56, 0, 0.02, .01),
        Note(57, 0, 0.02, .15),
        Note(58, 0, 0.02, .5),
        Note(60.2, 0, 0.05, 1),
        Note(60.5, 0, 0.05, 1),
        Note(61.2, 0, 0.05, 1),
        Note(66.2, 0, 0.1, 1),
        Note(66.0, 0, 0.1, 1),
        Note(67.0, 0, 0.2, 1),
        Note(68.5, 0, 0.35, 1),
        Note(60.0, 0, 0.05, 1),
        Note(64.2, 0, 0.05, 1),
        Note(63.2, 0, 0.05, 1),
        Note(64.2, 0, 0.64, 1),
    ]
    t = 0
    for n in notes:
        n.start = t
        t += n.dur
    voice = Voice(notes)
    staff = Staff([voice], clef="G", possible_divisions=(1,2,3,4,5,6,7,8,9,10,12))
    score = Score()
    score.append(staff)
    score.render()
    return score

possible_divisions = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16)
    
def test2():
    notes = [
        Note(60, 0, 0.5, 1),
        #Note(61, 0.25, 0.25, 1),
        Note(62, 0.52, 0.20, 1),
        Note(63, 0.75, 0.25, 1),
    ]
    division, assigned_notes, info = find_best_division(notes, (1,2,3,4,5,6,7,8,9, 10, 12, 16), pulsestart=0, debug=True)
    assert division == 4
    # return division, assigned_notes


def test2b():
    notes = [
        Note(60, 0.2, 0.2, 1),
        Note(61, 0.4, 0.1, 1),
        Note(62, 0.8, 0.1, 1),
        Note(63, 0.9, 0.1, 1),
    ]
    division, assigned_notes, info = find_best_division(notes, possible_divisions, pulsestart=0,
                                                       join_small=False, debug=True)
    assert division == 10
    division, assigned_notes, info = find_best_division(notes, possible_divisions, pulsestart=0,
                                                       join_small=True, debug=True)
    assert division == 10
    return division, assigned_notes


def test3():
    notes = [
        Note(60, 0, 0.333, 1),
        Note(61, 0.34, 0.433, 1),
        # Note(62, 0.67, 0.233, 1),
        Note(63, 0.75, 0.25, 1),
    ]
    division, assigned_notes, info = find_best_division(notes, possible_divisions, pulsestart=0, debug=False)
    # assert division == 3
    division, assigned_notes, info = find_best_division(notes, possible_divisions, pulsestart=0, join_small=True)
    # assert division == 3
    return division, assigned_notes


def test4():
    notes = [
        Note(60.1,  0, 0.02, 0.1),
        Note(50,    0, 0.02, 1),
        Note(51,    0, 0.02, 0.1),
        Note(52,    0, 0.02, 0.2),
        Note(53,    0, 0.02, 0.45),
        Note(54,    0, 0.02, 0.12),   # 0.1
        Note(55,    0, 0.02, 0.04),
        Note(56,    0, 0.02, 0.01),
        Note(57,    0, 0.02, 0.15),
        Note(58,    0, 0.02, 0.5),
        Note(60.2,  0, 1/5., 0.3),    # 0.2
        Note(60.5,  0, 1/3., 1),      # 0.5
        Note(61.2,  0, 1/10, 0.5),    # 0.6
        Note(66.2,  0, 7/10, 0.2),    # 0.7
    ]
    pulsestart = 2.1

    t = pulsestart
    for n in notes:
        n.start = t
        t += n.dur
    #division, assigned_notes, info = find_best_division(notes, (1,2,3,4,5,6,7,8,9, 10, 12, 16))
    voice = Voice(notes)
    possible_divisions = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16)
    staff = Staff([voice], clef="G", possible_divisions=possible_divisions)
    score = Score(join_small_notes=True, possible_divisions=possible_divisions)
    score.addstaff(staff)
    score.render()
    
    return score


def test5():
    notes = [
        Note(111, 1, 0.6, 0.005087),
        Note(111, 1.6, 0.08, 0.000634),
        Note(111, 1.68, 0.32, 0.001167)
    ]
    voice = Voice(notes)
    staff = Staff([voice], clef="G", possible_divisions=(1,2,3,4,5,6,7,8,9,10,12,16))
    score = Score()
    score.append(staff)
    score.render()
    return score


def test6():
    notes = [
        Note(60, 0, 0.333, 1),
    ]
    division, assigned_notes, info = find_best_division(notes, (1,2,3,4,5,6,7,8,9, 10, 12, 16), pulsestart=0, debug=True)
    assert division == 3
    division, assigned_notes, info = find_best_division(notes, (1,2,3,4,5,6,7,8,9, 10, 12, 16), pulsestart=0, join_small=True, debug=True)
    assert division == 3
    return division, assigned_notes


def test7():
    notes = [
        Note(60, 0, 0.25, 1),
    ]
    division, assigned_notes, info = find_best_division(notes, (1,2,3,4,5,6,7,8,9, 10, 12, 16), pulsestart=0, join_small=False, debug=True)
    assert division == 4
    division, assigned_notes, info = find_best_division(notes, (1,2,3,4,5,6,7,8,9, 10, 12, 16), pulsestart=0, join_small=True, debug=True)
    assert division == 4
    return division, assigned_notes


def test8():
    notes = [
        Note(60, 0, 0.125, 1),
        Note(61, 0.75, 0.5, 0.5)
    ]
    division, assigned_notes, info = find_best_division(notes, (1,2,3,4,5,6,7,8,9, 10, 12, 16), pulsestart=0, join_small=False, debug=True)
    print(info)
    #assert division == 8
    division, assigned_notes, info = find_best_division(notes, (1,2,3,4,5,6,7,8,9, 10, 12, 16), pulsestart=0, join_small=True, debug=True)
    #assert division == 8
    print(info)
    #return division, assigned_notes



def test9():
    notes = [
        Note(60, 0, 0.125, 1),
        Note(70, 0.125, 0.5, 1),
        Note(61, 0.75, 0.125, 1),
        Note(62, 0.875, 0.125, 1),
        Note(63, 1.5, 0.75, 1)
    ]
    #division, assigned_notes = find_best_division(notes[:-1], range(1, 10), pulse_start=0, debug=True)
    voice = Voice(notes)
    staff = Staff([voice], clef="G", possible_divisions=(1,2,3,4,5,6,7,8,9,10,12,16))
    score = Score()
    score.addstaff(staff)
    score.render()
    return score


def test_spectrum1():
    p = [
        (0, 60, "ppp"),
        (0.2, 61.5, "f"),
        (0.4, 62, "mf"),
        (0.9, 60.4, "p"),
        (1.1, 60.4, "p")
    ]
    times, midis, dyns = zip(*p)
    freqs = map(m2f, midis)
    amps = map(dyn2amp, dyns)
    partial = sndtrck.Partial(times=times, freqs=freqs, amps=amps)
    sp = sndtrck.Spectrum([partial])
    score = partials2score(sp, render=True, join_small_notes=False, include_dynamics=False, notesize_follows_dynamic=True)
    return score


def test_render1(outfile="test.xml"):
    score = test_spectrum1()
    score.toxml(outfile)
    base = os.path.splitext(outfile)[0]
    os.system("musicxml2ly %s" % outfile)
    os.system("lilypond %s.ly" % base)
    os.system("open %s.pdf" % base)
