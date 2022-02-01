# import modules
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt 

# 1. Read mseed file
st = read("test_data/my_sensor.mseed")
st.plot(outfile="mysensor.png")

# 2. Cut out a segment
# set the desired central time
pick_time = UTCDateTime(2021,11,6,14,15,0) # year, month, day, hour, minute, second

# set the desired number of data points before and after the central time
# (to match data given you for the challenge, choose 1000) 
padding_samp = 1000 

# set the output MSEED name
mseed_name = "test"

crop_s = padding_samp/31.25
# trim the trace
st.trim(starttime=pick_time-crop_s, endtime=pick_time+crop_s+1/30)
# merge trace if needed
st.merge(method=0, fill_value='interpolate', interpolation_samples=0)

# make sure that the trace length is 2*padding_samp, if not, force trim
if len(st[0].data) > 2*padding_samp:
    for tr in st:
        tr.data = tr.data[0:2*padding_samp]

if len(st[0].data) < 2*padding_samp:
    print("Warning: Your segment is shorter than desired. Your central time might be at the edge of the trace.")

# save MSEED
st.write(mseed_name + '.MSEED')

