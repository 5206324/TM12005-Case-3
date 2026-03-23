#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
import datetime
import matplotlib.dates as mdates
import pandas as pd
from Stap1_data_loader import laad_ecg_bestand #in dit bestand staat een functie die automatisch de data map vindt en het bestand inlaadt, je hoeft alleen de naam van het bestand aan te passen als je een ander bestand wilt inladen (dus het bestand met alleen de PACs bv)
from Stap3_Ventriculaire_activiteit import ecg_PT_both

#%%
# Inladen van de data
ecg, fs, t = laad_ecg_bestand("004_Groenewoud_PACs+PVCs.mat")
ecgmai, squared, band_passed, deriv_filtered = ecg_PT_both(ecg, fs)


# %%
plt.figure(figsize=(15, 5))
#plt.plot(t, deriv_filtered, label="derivative filtered")
locs, prop = signal.find_peaks(deriv_filtered, height=(500, 1400), distance=int(.3*fs))
# %%
lim = 3000

# Slice the time array and find which peaks belong in this 10-second window
t_plot = t[:lim]
locs_plot = locs[locs < lim]

fig, ax = plt.subplots(3, 1, figsize=(9, 11.7))

ax[0].plot(t_plot, ecg[:lim])
ax[0].set_ylabel("ECG (mV)")
ax[0].set_title("Raw ECG signal (First 10s)")
ax[0].set_xlim(t_plot[0], t_plot[-1])

ax[1].plot(t_plot, band_passed[:lim])
ax[1].set_ylabel("ECG (mV)")
ax[1].set_title("Bandpass filter")
ax[1].set_xlim(t_plot[0], t_plot[-1])

ax[2].plot(t_plot, deriv_filtered[:lim])
ax[2].plot(t_plot[locs_plot], deriv_filtered[locs_plot], 'r*')
ax[2].set_ylabel("ECG (mV)")
ax[2].set_title("Derivative filter")
ax[2].set_xlim(t_plot[0], t_plot[-1])



plt.tight_layout()
plt.show()
# %%
