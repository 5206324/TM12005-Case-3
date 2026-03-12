#%% Using fourier to analyse the RR intervals
# Fourier over de moving average function heen halen, hier al gefilterd 

from read_telemetry_ecg import ecgmai
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
import datetime
import matplotlib.dates as mdates
import pandas as pd

# %%

plt(t_plot, ecgmai[:lim])


# %%
