defaultconfig = {
    'numvoices':12,
    'pitch_resolution':0.5,
    'staffrange':36,  # the maximum range in semitones in any given staff
    'divisions':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
    'dynamics':['pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff'],
    'dyncurve_shape':'expon(2.5)',
    'dyncurve_mindb':-75,
    'dyncurve_maxdb':-6,
    'debug':False,

    'show_noteshapes':False,
    'show_dynamics':True,
    'show_transients':False,

    'remove_silent_breakpoints':True,
    'silence_db':-10,

    'weight_ampdur':10.,
    'weight_transient':0.1,
    'weight_note':4.0,
    'weight_time_accuracy':1.0,

    'penalty_timeshift':1.,
    'penalty_incorrect_duration':10,
    'penalty_leftout':2,
    'penalty_complexity':4.0,
    'penalty_merged_notes':1,

    'notesize_follows_dynamic':False,
    'lastnote_duration':1/16,
    'slur_partials':False,
    'downsample_spectrum':True,
    'join_small_notes':False,
    'partial_mindur':None,
    'staffsize':12,
    'pagelayout':'portrait',
    'pagesize':'A4',
    'minfreq':None,
    'maxfreq':6000,

    # how important is the fundamental (for packing). A high value will make sure that a partial identified
    # as the fundamental at that point will be included
    'pack_f0_gain':100,

    # how much margin to leave between two partials in a voice
    'pack_interpartial_margin':0.2,

    # the packing gain to apply to partials not being the fundamental
    'pack_notf0_gain':1,

    # the threshold used to identify the funcdamental
    'pack_f0_threshold':0.1,

    'pack_freqweight':2,
    'pack_ampweight':1,
    'pack_durweight':2,
    'pack_channel_exponentials':[0.7, 1.3],
    # min. gap between two overtones
    'pack_freq2points':[
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
    'pack_amp2points':[
        [-90, 0.0001],
        [-60, 2],
        [-35, 40],
        [-12, 95],
        [0, 100]
    ],
    'pack_dur2points':[
        [0, 0.0001],
        [0.1, 5],
        [0.3, 20],
        [0.5, 40],
        [2, 70],
        [5, 100]
    ],
    # one of weight / time
    'pack_criterium':'weight',
    'pack_prefer_overtones':False,
    'pack_harmonicity_gain':[
        [0.7, 1],
        [1, 2]
    ],
    # Apply an adaptive_prefilter before packing to reduce partials (see sndtrck.adaptive_prefilter)
    'pack_adaptive_prefilter':True,
    'pack_overtone_gain':[
        2, 2,
        5, 1
    ],
    # prefer voiced overtones: maps voicedness to gain,
    # where 1 is a harmonic sound, 0 is noise
    'pack_voiced_gain':[
        0, 1,
        1, 1
    ],
    'divcomplexity':{
        # division: complexity_factor
        1:2,
        2:1.7,
        3:1,
        4:1,
        5:1.2,
        6:1.2,
        7:4,
        8:1.2,
        9:2,
        10:3.5,
        11:999,
        12:3.0,
        16:3,
    },
    # override for dynamics fontsize (0=default, 1=12% bigger, -1=12% smaller)
    'lily_dynamics_size':-1,

}
