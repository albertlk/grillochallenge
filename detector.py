"""
This is a sample detector that serves as a placeholder for 
your awesome sophisticated ML detector.
"""

from obspy.core import Trace, UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def predict(tr):

    cft = recursive_sta_lta(tr.data, int(1 * tr.stats.sampling_rate), int(10 * tr.stats.sampling_rate))
    trigs = trigger_onset(cft, 3, 1)

    if len(trigs)>0:
        # This is how you can turn detection given in sample number into time!
        prediction = tr.stats.starttime + trigs[0][0]/tr.stats.sampling_rate
    else:
        prediction = None

    return prediction
