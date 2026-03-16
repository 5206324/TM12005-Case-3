# Read Telemetry
# Advanced Signal Processing (TM12005)
# Made by: M.S. van Schie (m.vanschie@erasmusmc.nl) & M.M. de Boer (m.m.deboer@erasmusmc.nl)
#%% Import
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
import datetime
import matplotlib.dates as mdates
import pandas as pd
from Stap1_data_loader import laad_ecg_bestand #in dit bestand staat een functie die automatisch de data map vindt en het bestand inlaadt, je hoeft alleen de naam van het bestand aan te passen als je een ander bestand wilt inladen (dus het bestand met alleen de PACs bv)

# %%
#hier kan je de naam van het bestand aanpassen als je een ander bestand wilt inladen (dus het bestand met alleen de PACs bv)
ecg, fs, t = laad_ecg_bestand("004_Groenewoud_PACs+PVCs.mat", plotresult=True)



# %% Pan TOMKINS
#%% PAN TOMKINS 


# Pan Tomkins datasetje bouwen
def ecg_PT(ecg, fs,t):
    # Stap 1: bandpass filter
    b,a = signal.butter(2, [5, 15], btype="band", fs=fs)
    band_passed = signal.filtfilt(b, a, ecg)

    # Stap 2: derivative filter
    T = 1 / fs          # Bereken de tijd per sample in seconden
    a = 8 * T           # Gebruik T in plaats van de timestamp t[1]
    b = [-1, -2, 0, 2, 1]
    deriv_filtered = signal.lfilter(b, a, band_passed)

    # Stap 3: squaring
    squared = deriv_filtered**2

    # Stap 4: moving average integration

    N = 30
    a = 1
    b = np.ones(N)/N
    ecgmai = signal.lfilter(b, a, squared)
    
    return ecgmai, band_passed, deriv_filtered, squared, fs, t

#%%
ecgmai, band_passed, deriv_filtered, squared, fs, t = ecg_PT(ecg, fs, t)
# peak detection
locs, prop = signal.find_peaks(ecgmai, height=1e6, distance=int(.3*fs))


# rr intervals
RR_intervals_sec = np.diff(locs) / fs
mean_RR_interval = np.mean(RR_intervals_sec)
mean_heartrate = 60 / mean_RR_interval



#%%
print(t)
#%%

# plot results
# --- PLOT RESULTS ---
# Define how many samples to plot to avoid MemoryError.
# 200 Hz * 10 seconds = 2000 samples
lim = 3000

# Slice the time array and find which peaks belong in this 10-second window
t_plot = t[:lim]
locs_plot = locs[locs < lim]

fig, ax = plt.subplots(5, 1, figsize=(9, 11.7))

ax[0].plot(t_plot, ecg[:lim])
ax[0].set_ylabel("ECG (mV)")
ax[0].set_title("Raw ECG signal (First 10s)")
ax[0].set_xlim(t_plot[0], t_plot[-1])

ax[1].plot(t_plot, band_passed[:lim])
ax[1].set_ylabel("ECG (mV)")
ax[1].set_title("Bandpass filter")
ax[1].set_xlim(t_plot[0], t_plot[-1])

ax[2].plot(t_plot, deriv_filtered[:lim])
ax[2].set_ylabel("ECG (mV)")
ax[2].set_title("Derivative filter")
ax[2].set_xlim(t_plot[0], t_plot[-1])

ax[3].plot(t_plot, squared[:lim])
ax[3].set_ylabel("ECG 2 (mV)")
ax[3].set_title("Squaring function")
ax[3].set_xlim(t_plot[0], t_plot[-1])

ax[4].plot(t_plot, ecgmai[:lim])
ax[4].plot(t_plot[locs_plot], ecgmai[locs_plot], 'r*') # Plot red stars on peaks
ax[4].text(t_plot[-1], np.max(ecgmai[:lim])*0.8, f'mean RR = {mean_RR_interval:.3f} s', ha='right')
ax[4].text(t_plot[-1], np.max(ecgmai[:lim])*0.5, f'HR = {mean_heartrate:.1f} bpm', ha='right')
ax[4].set_xlabel("Time")
ax[4].set_ylabel("ECG 2 (mV)")
ax[4].set_title("Moving average integration")
ax[4].set_xlim(t_plot[0], t_plot[-1])

plt.tight_layout()
plt.show()


    # %%
