"""
This is a validation script for the OpenEEW challenge.
Please change the script so it incorporates your model.
"""

"""
Place your imports and setup between here...
"""
from detectorCNN import predict
"""
...and here
"""

# imports
import glob
import random
import numpy as np
import detector
from obspy.core import read
import matplotlib.pyplot as plt

# set variables
## random.seed(0)
accuracy = .5 # max time difference in seconds
p_win = (.5, 3) # setting limits for P-wave 
sampling_rate = 31.25
min_snr = 1.5
datapath = "validation_data"
hit = 0; miss = 0; false_alarm = 0; correct_rejection = 0
hit_accuracy = []

# get all filenames
valid_filenames = [f for f in glob.glob(datapath + "/*/*")]
number_of_files = len(valid_filenames)
number_of_noise = len([f for f in valid_filenames if "_N" in f])
number_of_pwave = number_of_files-number_of_noise

# iterate over all files in validation set
for f in valid_filenames:

    print("File: {}".format(f))

    # read in the file
    st = read(f)
    cut = int((random.random()*p_win[1]+p_win[0])*sampling_rate)

    # iterate over traces in mseed files
    for tr in st:

        # there was a comment raised during one of the sessions that some p-waves have very low signal-to-noise ratio
        # here I require all p-waves to have snr 1.5 or greater
        # snr is calcuated as a ratio of sums of abs values of waveforms 1s before and 1s after p-wave arrival
        snr = sum(abs(tr.data[1000:1032]))/sum(abs(tr.data[1000-32:1000]))

        # in case of small snr, just print a message
        if "_P" in f and snr<=min_snr:
            print("  Channel: {}, passing, not high-enough P wave S/N ratio ({:4.2f})".format(tr.stats.channel, snr))      
        
        # otherwise do everything
        else:
            tr.data = tr.data[cut:1000+cut]

            # set correct answers
            if "_N" in f:
                # if it is a noise segment, the correct answer is None
                correct_answer = None
            else:
                # if it is a p-wave segment, the correct answer is the p-wave arrival time (UTCDateTime format)
                correct_answer = tr.stats.starttime + (1000-cut)/tr.stats.sampling_rate

            """
            Your magic starts here.
            Here you are given an obspy trace. Your should determine whether it contains a p-wave.
            If so, return time of the p-wave arrival. If not, return None.
            Swap my function below with yours.

            :param: tr (``obspy Trace``) --
                1000-samples-long obspy Trace that eiter contains p-wave or not.
        
            :return: prediction (``obspy UTCDateTime`` or None) --
                Time of the P-wave detection or None if none detected.
            """
            prediction = predict(tr)
            """
            Your magic ends here.
            prediction variable needs to be either UTCDateTime object or None.
            """

            # Evaluation logic
            # noise segment, no p-wave detected
            if prediction is None and correct_answer is None:
                correct_rejection += 1
                print("  Channel: {}, correct rejection!".format(tr.stats.channel))
            
            # p-wave segment but no detection
            elif prediction is None and correct_answer is not None:
                miss += 1
                print("  Channel: {}, miss!".format(tr.stats.channel))

            # p-wave falsely detected in noise segment
            elif prediction is not None and correct_answer is None:
                false_alarm += 1
                print("  Channel: {}, false alarm!".format(tr.stats.channel))

            # p-wave detected in p-wave segment
            else:
                # evaluate if the detection is accurate enough, if so it is a hit...
                if abs(correct_answer-prediction)<=accuracy:
                    hit += 1
                    hit_accuracy.append(correct_answer-prediction)
                    print("  Channel: {}, hit, accuracy {:4.2f}s !".format(tr.stats.channel, correct_answer-prediction))
                
                # ...else it is unfortunately a miss.
                else:
                    miss += 1
                    hit_accuracy.append(correct_answer-prediction)
                    print("  Channel: {}, miss, error {:4.2f}s !".format(tr.stats.channel, correct_answer-prediction))

# calculate the stats
precision = hit/(hit+false_alarm)
recall = hit/(hit+miss)
f1score = (2*precision*recall)/(precision+recall)

# and print them
print(' ')
print('EVALUATING THE MODEL BY:')
print(' ')
print('{:5.0f} Validation segments'.format(number_of_files))
print('{:5.0f} contains a P-wave'.format(number_of_pwave))
print('{:5.0f} is noise'.format(number_of_noise))
print(' ')
print('MODEL PREDICTION EVALUATION')
print('---------------------------')
print(' ')
print('         |         |         |')
print('         |    P    |  noise  |')
print('---------|---------|---------|')
print('   Tag   |  {:5.0f}  |  {:5.0f}  |'.format(hit, false_alarm))
print('---------|---------|---------|')
print(' No tag  |  {:5.0f}  |  {:5.0f}  |'.format(miss, correct_rejection))
print('---------|---------|---------|')
print(' ')
print('Precision: {:4.2f}'.format(precision))
print('Recall: {:4.2f}'.format(recall))
print('Hit misfit: mean {:4.2f}, std {:4.2f}'.format(np.array(hit_accuracy).mean(), np.array(hit_accuracy).std()))
print(' ')
print('////////////////////')
print('// F1 score: {:4.2f} //'.format(f1score))
print('////////////////////')
print(' ')

# plot misfits
out = plt.hist(hit_accuracy, bins=np.arange(-2, 2, .05))
plt.vlines(-accuracy, 0, max(out[0]), color=[.2, .2, .2])
plt.vlines(accuracy, 0, max(out[0]), color=[.2, .2, .2])
plt.xlabel("Misfit [s]")
plt.ylabel("Number of segments")
plt.savefig("misfit.png")
plt.show()