# import modules
from obspy import read
import matplotlib.pyplot as plt 

# 1. Read mseed file
st = read("test_data/examp_eq1_sta021.mseed")
print(len(st))

# 2. Plot mseed file and save output in a file
st.plot(outfile="my_plot.png")

# 3. Access Trace data
tr = st[0]
trace_data = tr.data
print(type(trace_data))
print(trace_data)