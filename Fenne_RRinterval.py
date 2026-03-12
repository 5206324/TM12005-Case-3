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

plt.plot(t_plot, ecgmai[:lim]) 

# %%
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

plt.plot(t_plot, ecgmai[:lim]) 

# %%
#%% Threshold

threshold_short = mean_RR_interval * 0.85  # De "vroege" grens
threshold_long = mean_RR_interval * 1.15   # De "pauze" grens
# %% Zoek de index van intervallen die korter zijn dan de drempel
pacs_kort = np.where(RR_intervals_sec < threshold_short)[0]
pacs_lang = np.where(RR_intervals_sec > threshold_long)[0]

print(f"Aantal vroege slagen: {len(pacs_kort)}")
print(f"Aantal pauzes: {len(pacs_lang)}")
# %%
