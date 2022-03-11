import os
import obspy
import numpy as np
import pandas as pd
import random as random
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

signalmodel = load_model('signalmodel')
locmodel = load_model('locmodel')

def predict(trace):
    
    ## Things to Specify -- Ensure that they align with trained model
    # For rolling averages
    roll_short = 25
    roll_long = 50

    # For location segments
    window_step = 30
    window_size = 30
    
    # Probability cut-offs
    p_prob = 0.25 # for p-wave
    s_prob = 0.25 # for signal
    
    ## Reading in Trace and Splitting Channels
    sig_trace = trace.normalize()
    
    samp_rate = sig_trace.stats.sampling_rate
    start_time = sig_trace.stats.starttime
    
    sigs = sig_trace.data
    
    sig_df = pd.DataFrame(sigs, columns = ["trace"])
    
    sigfeatures = []
    
    ## Calculating various features needed for the signal model
    tr = sig_df["trace"]
    mag = abs(tr)
    d = {"trace":tr, "magnitude":mag}
    temp_df = pd.DataFrame(data = d)
    
    temp_df["STA"] = temp_df["magnitude"].rolling(roll_short).mean()
    temp_df["LTA"] = temp_df["magnitude"].rolling(roll_long).mean()
    temp_df["RAV"] = temp_df["STA"]/temp_df["LTA"]
    temp_df["STV"] = temp_df["magnitude"].rolling(roll_short).var()
    temp_df["LTV"] = temp_df["magnitude"].rolling(roll_long).var()

    temp_df.dropna(inplace = True)
    sigfeatures.append(temp_df.values)
    
    ## Predicting whether or not there is a pwave in the trace
    is_sig = signalmodel.predict(np.array(sigfeatures))    

    locfeatures = []

    ## If there is a pwave in the trace, continue to find location of pwave
    start_ind = 0
    end_ind = start_ind + window_size

    while end_ind < (1000 - roll_long):
        trwindow = temp_df["trace"].iloc[start_ind:end_ind]
        magwindow = temp_df["magnitude"].iloc[start_ind:end_ind]
        ravwindow = temp_df["RAV"].iloc[start_ind:end_ind]
        stvwindow = temp_df["STV"].iloc[start_ind:end_ind]
        ltvwindow = temp_df["LTV"].iloc[start_ind:end_ind]

        window_data = {"trace": trwindow, "magnitude": magwindow,
                            "RAV": ravwindow, "STV": stvwindow, "LTV": ltvwindow}

        window_df = pd.DataFrame(data = window_data)

        locfeatures.append(window_df.values)

        start_ind += window_step
        end_ind = start_ind + window_size

    prob_vec = locmodel.predict(np.array(locfeatures))
    
    # Since we know if there is a p-wave, it will be in the last 3 seconds:
    end_ind = len(prob_vec)
    beg_ind = len(prob_vec) - int(3*window_size/samp_rate + 1)
    p_segment = beg_ind + np.where(prob_vec[beg_ind:end_ind] == max(prob_vec[beg_ind:end_ind]))[0][0]
        
    tick_delta = (p_segment + 0.5)*window_size + (roll_long - 1)
        
    time_delta = (tick_delta)/samp_rate
    p_time = start_time + time_delta
        
    if (is_sig >= s_prob)|(max(prob_vec[beg_ind:end_ind]) >= p_prob):
        return p_time
    else:
        return None