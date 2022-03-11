import os
import obspy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from pathlib import Path
os.getcwd()

import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import scipy
import sklearn.metrics
from sklearn.model_selection import train_test_split
## Functions for generating rotation matrices

# Rotates matrix theta radians about the X axis
def rot_x (mat, theta):
    sint = math.sin(theta)
    cost = math.cos(theta)
    
    Rx = [[1, 0, 0],
      [0, cost, -sint],
      [0, sint, cost]]
    
    rot_mat = np.matmul(Rx, mat)
    
    return rot_mat

# Rotates matrix theta radians about the Y axis
def rot_y (mat, theta):
    sint = math.sin(theta)
    cost = math.cos(theta)
    
    Ry = [[cost, 0, sint],
      [0, 1, 0],
      [-sint, 0, cost]]
    
    rot_mat = np.matmul(Ry, mat)
    
    return rot_mat

# Rotates matrix theta radians about the Z axis 
def rot_z (mat, theta):
    sint = math.sin(theta)
    cost = math.cos(theta)
    
    Rz = [[cost, -sint, 0],
      [sint, cost, 0],
      [0, 0, 1]]
    
    rot_mat = np.matmul(Rz, mat)
    
    return rot_mat

# General rotation with yaw of a radians, pitch of b radians, and roll of c radians
def rot_xyz(mat, a, b, c):
    rz = rot_z(np.identity(3), a)
    ry = rot_y(np.identity(3), b)
    rx = rot_x(np.identity(3), c)
    
    rot_mat = np.matmul(rx, mat)
    rot_mat = np.matmul(ry, rot_mat)
    rot_mat = np.matmul(rz, rot_mat)
    
    return rot_mat
# Plots all three channels from a given sample from the signals df
def plot_sig (df, index):
    x_channel = df['trmatrix_cut'][index][0]
    y_channel = df['trmatrix_cut'][index][1]
    z_channel = df['trmatrix_cut'][index][2]
    
    fig, axs = plt.subplots(3)
    
    axs[0].plot(x_channel)
    axs[1].plot(y_channel)
    axs[2].plot(z_channel)
    
    return None

# convlstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot


# Signal List
####################!!!!!!!!!!!!!!!!!!!!!!
os.chdir('C:/Users/Lenovo/Desktop/challenge/lse_challenge/lse_challenge/data/signal')
signals = []
signal_files = os.listdir()
signal_files = random.sample(signal_files, 500) #Need to expand for final model
for sigfile in signal_files: 
    tmp_trace = obspy.read(sigfile)
    tmp_trace = tmp_trace.normalize()
    cutlength = np.random.randint(16, 150) # Aligning with the [0.5 sec, 3 sec] range
    trace_mat = [tmp_trace[0].data[cutlength:1000+cutlength], tmp_trace[1].data[cutlength:1000+cutlength], tmp_trace[2].data[cutlength:1000+cutlength]]
    #mag = np.sqrt(trace_mat[0]**2 + trace_mat[1]**2 + trace_mat[2]**2)
    #signals.append([trace_mat[0], trace_mat[1], trace_mat[2], mag, cutlength, 1])
    signals.append([trace_mat[0], cutlength, 1])
    signals.append([trace_mat[1], cutlength, 1])
    signals.append([trace_mat[2], cutlength, 1])
    
    # Negative Transformations
    neg_mat = rot_xyz(trace_mat, 0, -np.pi, 0)
    #signals.append([neg_mat[0], neg_mat[1], neg_mat[2], mag, cutlength, 1])
    signals.append([neg_mat[0], cutlength, 1])
    signals.append([neg_mat[1], cutlength, 1])
    signals.append([neg_mat[2], cutlength, 1])
    
    # X, Y, and Z Rotations by Theta
    theta = np.random.uniform()*2*np.pi
    x_mat = rot_x(trace_mat, theta)
    #signals.append([x_mat[0], x_mat[1], x_mat[2], mag, cutlength, 1])
    signals.append([x_mat[0], cutlength, 1])
    signals.append([x_mat[1], cutlength, 1])
    signals.append([x_mat[2], cutlength, 1])
    y_mat = rot_y(trace_mat, theta)
    #signals.append([y_mat[0], y_mat[1], y_mat[2], mag, cutlength, 1])
    signals.append([y_mat[0], cutlength, 1])
    signals.append([y_mat[1], cutlength, 1])
    signals.append([y_mat[2], cutlength, 1])
    z_mat = rot_z(trace_mat, theta)
    #signals.append([z_mat[0], z_mat[1], z_mat[2], mag, cutlength, 1])
    signals.append([z_mat[0], cutlength, 1])
    signals.append([z_mat[1], cutlength, 1])
    signals.append([z_mat[2], cutlength, 1])    
    
    # General Rotation by alpha, beta, gamma
    alpha = np.random.uniform()*2*np.pi
    beta = np.random.uniform()*2*np.pi
    gamma = np.random.uniform()*2*np.pi
    rot_mat = rot_xyz(trace_mat, alpha, beta, gamma)
    #signals.append([rot_mat[0], rot_mat[1], rot_mat[2], mag, cutlength, 1])
    signals.append([rot_mat[0], cutlength, 1])
    signals.append([rot_mat[1], cutlength, 1])
    signals.append([rot_mat[2], cutlength, 1])
    

# Noise List
####################!!!!!!!!!!!!!!!!!!!!!!
os.chdir('C:/Users/Lenovo/Desktop/challenge/lse_challenge/lse_challenge/data/noise')
noise_files = os.listdir()
noise_files = random.sample(noise_files, 1500) #Need to expand for final model
for noisefile in noise_files: 
    tmp_trace = obspy.read(noisefile)
    tmp_trace = tmp_trace.normalize()
    cutlength = np.random.randint(16, 100) # Aligning with the [0.5 sec, 3 sec] range
    trace_mat = [tmp_trace[0].data[cutlength:1000+cutlength], tmp_trace[1].data[cutlength:1000+cutlength], tmp_trace[2].data[cutlength:1000+cutlength]]
    #mag = np.sqrt(trace_mat[0]**2 + trace_mat[1]**2 + trace_mat[2]**2)
    #signals.append([trace_mat[0], trace_mat[1], trace_mat[2], mag, cutlength, 0])
    signals.append([trace_mat[0], cutlength, 0])
    signals.append([trace_mat[1], cutlength, 0])
    signals.append([trace_mat[2], cutlength, 0])
    
    # Negative Transformations
    neg_mat = rot_xyz(trace_mat, 0, -np.pi, 0)
    #signals.append([neg_mat[0], neg_mat[1], neg_mat[2], mag, cutlength, 0])
    signals.append([neg_mat[0], cutlength, 0])
    signals.append([neg_mat[1], cutlength, 0])
    signals.append([neg_mat[2], cutlength, 0])
    
    # X, Y, and Z Rotations by Theta
    theta = np.random.uniform()*2*np.pi
    x_mat = rot_x(trace_mat, theta)
    #signals.append([x_mat[0], x_mat[1], x_mat[2], mag, cutlength, 0])
    signals.append([x_mat[0], cutlength, 0])
    signals.append([x_mat[1], cutlength, 0])
    signals.append([x_mat[2], cutlength, 0])
    y_mat = rot_y(trace_mat, theta)
    #signals.append([y_mat[0], y_mat[1], y_mat[2], mag, cutlength, 0])
    signals.append([y_mat[0], cutlength, 0])
    signals.append([y_mat[1], cutlength, 0])
    signals.append([y_mat[2], cutlength, 0])
    z_mat = rot_z(trace_mat, theta)
    #signals.append([z_mat[0], z_mat[1], z_mat[2], mag, cutlength, 0])
    signals.append([z_mat[0], cutlength, 0])
    signals.append([z_mat[1], cutlength, 0])
    signals.append([z_mat[2], cutlength, 0])
    
    # General Rotation by alpha, beta, gamma
    alpha = np.random.uniform()*2*np.pi
    beta = np.random.uniform()*2*np.pi
    gamma = np.random.uniform()*2*np.pi
    rot_mat = rot_xyz(trace_mat, alpha, beta, gamma)
    #signals.append([rot_mat[0], rot_mat[1], rot_mat[2], mag, cutlength, 0])
    signals.append([rot_mat[0], cutlength, 0])
    signals.append([rot_mat[1], cutlength, 0])
    signals.append([rot_mat[2], cutlength, 0])
    
    
os.chdir('../')

sig_df = pd.DataFrame(signals, columns = ["trace", "cutlength", "signal"])
sig_df['p_arrival'] = 1000 - sig_df['cutlength']





# For rolling averages
roll_short = 25
roll_long = 50

# For location segments
window_step = 30
window_size = 30


# Features and Targets for Identifying the Entire Trace

sigfeatures = []
sigtargets = []

for index, slice_df in sig_df.iterrows():
    # x = slice_df["tr_x"]
    # y = slice_df["tr_y"]
    # z = slice_df["tr_z"]
    tr = slice_df["trace"]
    mag = abs(tr)
    # mag = slice_df["magnitude"]
    signal = slice_df["signal"]
    p_arrival = slice_df["p_arrival"]
    #d = {"x": x, "y":y, "z":z, "magnitude":mag}
    d = {"trace": tr, "magnitude":mag}
    
    
    temp_df = pd.DataFrame(data = d)
    temp_df["STA"] = temp_df["magnitude"].rolling(roll_short).mean()
    temp_df["LTA"] = temp_df["magnitude"].rolling(roll_long).mean()
    temp_df["RAV"] = temp_df["STA"]/temp_df["LTA"]
    temp_df["STV"] = temp_df["magnitude"].rolling(roll_short).var()
    temp_df["LTV"] = temp_df["magnitude"].rolling(roll_long).var()
    
    temp_df.dropna(inplace = True)
    
    sigfeatures.append(temp_df.values)
    sigtargets.append(signal)






X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(sigfeatures, sigtargets, test_size = 0.3)

X_train_sig = np.array(X_train_sig)
X_test_sig = np.array(X_test_sig)
y_train_sig = np.array(y_train_sig)
y_test_sig = np.array(y_test_sig)

####################!!!!!!!!!!!!!!!!!!!!!!
n_timesteps, n_features, n_outputs = X_train_sig.shape[1], X_train_sig.shape[2], 2
n_steps, n_length = 3, 317
signalmodel = Sequential()
signalmodel.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, 317, 7)))
signalmodel.add(Dropout(0.5))
signalmodel.add(Flatten())
signalmodel.add(Dense(100, activation='relu'))
signalmodel.add(Dense(2, activation='softmax'))
signalmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

signalmodel.summary()

####################!!!!!!!!!!!!!!!!!!!!!!
X_train_sig1 = X_train_sig.reshape((25200, 3, 1, 317, 7))
X_test_sig1 = X_test_sig.reshape((10800, 3, 1, 317, 7))
y_train_sig1 = to_categorical(y_train_sig)
y_test_sig1 = to_categorical(y_test_sig)
signalmodel.fit(X_train_sig1, y_train_sig1, epochs = 30)






model_loss_sig, model_accuracy_sig = signalmodel.evaluate(X_test_sig1, y_test_sig1, verbose = 0)
print(f"Loss: {model_loss_sig}, Accuracy: {model_accuracy_sig}")


predictions_test_sig = signalmodel.predict(X_test_sig1)
predictions_train_sig = signalmodel.predict(X_train_sig1)



class_test_sig = predictions_test_sig > 0.2
f1_score_test_sig = sklearn.metrics.f1_score(y_test_sig1, class_test_sig,average = 'micro')
print(f"F1 Score: {f1_score_test_sig}")



class_train_sig = predictions_train_sig > 0.2
f1_score_train_sig = sklearn.metrics.f1_score(y_train_sig1, class_train_sig,average = 'micro')
print(f"F1 Score: {f1_score_train_sig}")

####################!!!!!!!!!!!!!!!!!!!!!!
######### I can't run this
tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test_sig1, class_test_sig).ravel()
(tn, fp, fn, tp)

# Exporting signal model
signalmodel.save('signalmodel')


#  part 2


# Features and Targets for Identifying Windows of Trace
locfeatures = []
loctargets = []

for index, slice_df in sig_df.iterrows():
    
    signal = slice_df["signal"]
    
    if signal == 1:
        # x = slice_df["tr_x"]
        # y = slice_df["tr_y"]
        # z = slice_df["tr_z"]
        tr = slice_df["trace"]
        mag = abs(tr)
        # mag = slice_df["magnitude"]

        p_arrival = slice_df["p_arrival"]
        d = {"trace":tr, "magnitude":mag}

        temp_df = pd.DataFrame(data = d)
        temp_df["STA"] = temp_df["magnitude"].rolling(roll_short).mean()
        temp_df["LTA"] = temp_df["magnitude"].rolling(roll_long).mean()
        temp_df["RAV"] = temp_df["STA"]/temp_df["LTA"]
        temp_df["STV"] = temp_df["magnitude"].rolling(roll_short).var()
        temp_df["LTV"] = temp_df["magnitude"].rolling(roll_long).var()

        temp_df.dropna(inplace = True)

        start_ind = 0
        end_ind = start_ind + window_size



        while end_ind < (1000 - roll_long):
            # xwindow = temp_df["x"].iloc[start_ind:end_ind]
            # ywindow = temp_df["y"].iloc[start_ind:end_ind]
            # zwindow = temp_df["z"].iloc[start_ind:end_ind]
            trwindow = temp_df["trace"].iloc[start_ind:end_ind]
            magwindow = temp_df["magnitude"].iloc[start_ind:end_ind]
            # stawindow = temp_df["STA"].iloc[start_ind:end_ind]
            # ltawindow = temp_df["LTA"].iloc[start_ind:end_ind]
            ravwindow = temp_df["RAV"].iloc[start_ind:end_ind]
            stvwindow = temp_df["STV"].iloc[start_ind:end_ind]
            ltvwindow = temp_df["LTV"].iloc[start_ind:end_ind]

            window_data = {"trace": trwindow, "magnitude": magwindow,
                        "RAV": ravwindow, "STV": stvwindow, "LTV": ltvwindow}
            window_df = pd.DataFrame(data = window_data)

            locfeatures.append(window_df.values)

            if ((p_arrival-roll_long) >= start_ind) and ((p_arrival-roll_long) <= end_ind):
                loctargets.append(1)
            else:
                loctargets.append(0)

            start_ind += window_step
            end_ind = start_ind + window_size



X_train_loc, X_test_loc, y_train_loc, y_test_loc = train_test_split(locfeatures, loctargets, test_size = 0.3)

X_train_loc = np.array(X_train_loc)
X_test_loc = np.array(X_test_loc)
y_train_loc = np.array(y_train_loc)
y_test_loc = np.array(y_test_loc)
# feature_shape = X_train_loc[0].shape
# feature_shape


# verbose, epochs, batch_size = 0, 25, 64
# n_timesteps, n_features, n_outputs = X_train_loc.shape[1], X_train_loc.shape[2], 2
# n_steps, n_length = 3, 10

####################!!!!!!!!!!!!!!!!!!!!!!
locmodel = Sequential()
locmodel.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, 10, 5)))
locmodel.add(Dropout(0.5))
locmodel.add(Flatten())
locmodel.add(Dense(100, activation='relu'))
locmodel.add(Dense(2, activation='softmax'))
locmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

signalmodel.summary()

####################!!!!!!!!!!!!!!!!!!!!!!
X_train_loc1 = X_train_loc.reshape((195300, 3, 1, 10, 5))
X_test_loc1 = X_test_loc.reshape((83700, 3, 1, 10, 5))
y_train_loc1 = to_categorical(y_train_loc)
y_test_loc1 = to_categorical(y_test_loc)
locmodel.fit(X_train_loc1, y_train_loc1, epochs = 30)

model_loss_loc, model_accuracy_loc = locmodel.evaluate(X_test_loc1, y_test_loc1, verbose = 0)
print(f"Loss: {model_loss_loc}, Accuracy: {model_accuracy_loc}")

predictions_test_loc = locmodel.predict(X_test_loc1)
predictions_train_loc = locmodel.predict(X_train_loc1)

class_test_loc = predictions_test_loc > 0.25
f1_score_test_loc = sklearn.metrics.f1_score(y_test_loc1, class_test_loc,average='micro')
print(f"F1 Score: {f1_score_test_loc}")

class_train_loc = predictions_train_loc > 0.25
f1_score_train_loc = sklearn.metrics.f1_score(y_train_loc1, class_train_loc,average='micro')
print(f"F1 Score: {f1_score_train_loc}")

####################!!!!!!!!!!!!!!!!!!!!!!
############# can't run it either
tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test_loc, class_test_loc).ravel()
(tn, fp, fn, tp)

# Exporting location model
locmodel.save('locmodel')



#####################
loaded_signalmodel = load_model('signalmodel')

loaded_locmodel = load_model('locmodel')

slice_df = sig_df.iloc[29]

sigfeatures = []
sigtargets = []
locfeatures = []
loctargets = []

# x = slice_df["tr_x"]
# y = slice_df["tr_y"]
# z = slice_df["tr_z"]
tr = slice_df["trace"]
mag = abs(tr)
# mag = slice_df["magnitude"]
signal = slice_df["signal"]
p_arrival = slice_df["p_arrival"]
#d = {"x": x, "y":y, "z":z, "magnitude":mag}
d = {"trace": tr, "magnitude":mag}
temp_df = pd.DataFrame(data = d)
temp_df["STA"] = temp_df["magnitude"].rolling(roll_short).mean()
temp_df["LTA"] = temp_df["magnitude"].rolling(roll_long).mean()
temp_df["RAV"] = temp_df["STA"]/temp_df["LTA"]
temp_df["STV"] = temp_df["magnitude"].rolling(roll_short).var()
temp_df["LTV"] = temp_df["magnitude"].rolling(roll_long).var()
    
temp_df.dropna(inplace = True)
sigfeatures.append(temp_df.values)
sigtargets.append(signal)

sigfeatures = np.array(sigfeatures)
sigfeatures1 = sigfeatures.reshape((1, 3, 1, 317, 7))
loaded_signalmodel.predict(sigfeatures1)

sigtargets


start_ind = 0
end_ind = start_ind + window_size
    
while end_ind < (1000 - roll_long):
    # xwindow = temp_df["x"].iloc[start_ind:end_ind]
    # ywindow = temp_df["y"].iloc[start_ind:end_ind]
    # zwindow = temp_df["z"].iloc[start_ind:end_ind]
    magwindow = temp_df["magnitude"].iloc[start_ind:end_ind]
    trwindow = temp_df["trace"].iloc[start_ind:end_ind]
    ravwindow = temp_df["RAV"].iloc[start_ind:end_ind]
    stvwindow = temp_df["STV"].iloc[start_ind:end_ind]
    ltvwindow = temp_df["LTV"].iloc[start_ind:end_ind]

    window_data = {"trace":trwindow, "magnitude": magwindow,
                    "RAV": ravwindow, "STV": stvwindow, "LTV": ltvwindow}
        
    window_df = pd.DataFrame(data = window_data)

    locfeatures.append(window_df.values)

    if ((p_arrival-roll_long) >= start_ind) and ((p_arrival-roll_long) <= end_ind):
        loctargets.append(1)
    else:
        loctargets.append(0)

    start_ind += window_step
    end_ind = start_ind + window_size
    

locfeatures = np.array(locfeatures)
locfeatures1 = locfeatures.reshape((31, 3, 1, 10, 5))

loaded_locmodel.predict(locfeatures1)

loctargets