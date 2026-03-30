#%% === Stap 2: filteren en visualisatie === 
# --- Data % pakketen inladen ---

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
import datetime
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path
from Stap1_data_loader import laad_ecg_bestand 

ecg, fs, t = laad_ecg_bestand("004_Groenewoud_PACs+PVCs.mat", plotresult=True)

#%% --- PAN TOMKINS --- 
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

ecgmai, band_passed, deriv_filtered, squared, fs, t = ecg_PT(ecg, fs, t)
# peak detection
locs, prop = signal.find_peaks(ecgmai, height=1e6, distance=int(.3*fs))

# rr intervals
RR_intervals_sec = np.diff(locs) / fs
mean_RR_interval = np.mean(RR_intervals_sec)
mean_heartrate = 60 / mean_RR_interval

#%% --- PLOT RESULTS PAN TOMKINS---
lim = 3000                          #x-as limitatie om het leesbaar te houden
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
ax[4].plot(t_plot[locs_plot], ecgmai[locs_plot], 'r*') 
ax[4].text(t_plot[-1], np.max(ecgmai[:lim])*0.8, f'mean RR = {mean_RR_interval:.3f} s', ha='right')
ax[4].text(t_plot[-1], np.max(ecgmai[:lim])*0.5, f'HR = {mean_heartrate:.1f} bpm', ha='right')
ax[4].set_xlabel("Time")
ax[4].set_ylabel("ECG 2 (mV)")
ax[4].set_title("Moving average integration")
ax[4].set_xlim(t_plot[0], t_plot[-1])

plt.tight_layout()
plt.show()

#%% --- Analyse van Ventriculaire Activiteit en Frequentie ---

# 1. Gat in meting vinden en verwijderen
rr_ruw = np.diff(locs) / fs
gap_index = np.argmax(rr_ruw)           # De index met het allergrootste RR-interval
locs_deel_1 = locs[:gap_index + 1]
locs_deel_2 = locs[gap_index + 1:]

print(f"- Deel 1 (voor pauze): {len(locs_deel_1)} slagen")
print(f"- Deel 2 (na pauze): {len(locs_deel_2)} slagen")

locs_analyse = locs_deel_1              # Deel 1 zo gebruiken voor de analyse 
locs_analyse2 = locs_deel_2

# 2. RR-intervallen en de Hartslag (BPM) berekenen voor beide analyses
rr_intervallen = np.diff(locs_analyse) / fs
t_slagen = t[locs_analyse[1:]] 
hartslag_bpm = 60 / rr_intervallen

rr_intervallen2 = np.diff(locs_analyse2) / fs
t_slagen2 = t[locs_analyse2[1:]] 
hartslag_bpm2 = 60 / rr_intervallen2

# --- PLOT 1: Hartslag (Frequentie) over tijd ---
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# --- fig 1: Hartslag freq ---
ax[0, 0].plot(t_slagen, hartslag_bpm, marker='.', linestyle='', color='b', markersize=2)
ax[0, 0].axhline(y=np.median(hartslag_bpm), color='g', linestyle='--', label=f'Mediaan: {np.median(hartslag_bpm):.1f} BPM')
ax[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax[0, 0].set_title("1. Ventriculaire Frequentie over Tijd- voor gat")
ax[0, 0].set_xlabel("Tijd")
ax[0, 0].set_ylabel("Hartslag (BPM)")
ax[0, 0].set_ylim(40, 180)
ax[0, 0].legend()
ax[0, 0].grid(True)

# --- fig 2: Tachogram ---
ax[0, 1].plot(t_slagen, rr_intervallen, marker='.', linestyle='', color='purple', markersize=2, label='RR-intervallen')
ax[0, 1].axhline(y=np.median(rr_intervallen), color='g', linestyle='--', label=f'Mediaan: {np.median(rr_intervallen):.3f}s')
ax[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) # Formatteer als tijd
ax[0, 1].set_title("2. Tachogram: Variatie in RR - voor gat")
ax[0, 1].set_xlabel("Slag nummer")
ax[0, 1].set_ylabel("RR-interval (seconden)")
ax[0, 1].legend(loc='upper right', fontsize='small')
ax[0, 1].grid(True)

# --- fig 3: Hartslag na het gat ---
ax[1, 0].set_title("3. Ventriculaire Frequentie over Tijd - na gat")
ax[1, 0].plot(t_slagen2, hartslag_bpm2, marker='.', linestyle='', color='b', markersize=2)
ax[1, 0].axhline(y=np.median(hartslag_bpm2), color='g', linestyle='--', label=f'Mediaan: {np.median(hartslag_bpm2):.1f} BPM')
ax[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) # Tijdformaat toevoegen
ax[1, 0].set_xlabel("Tijd")
ax[1, 0].set_ylabel("Hartslag (BPM)")
ax[1, 0].set_ylim(40, 180)
ax[1, 0].grid(True)
ax[1, 0].legend()

# --- fig 4: Tachogram na het gat ---
ax[1, 1].plot(t_slagen2, rr_intervallen2, marker='.', linestyle='', color='purple', markersize=2, label='RR-intervallen')
ax[1, 1].axhline(y=np.median(rr_intervallen2), color='g', linestyle='--', label=f'Mediaan: {np.median(rr_intervallen2):.3f}s')
ax[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) # Formatteer als tijd
ax[1, 1].set_title("4. Tachogram: Variatie in RR - na gat")
ax[1, 1].set_xlabel("Slag nummer")
ax[1, 1].set_ylabel("RR-interval (seconden)")
ax[1, 1].legend(loc='upper right', fontsize='small')
ax[1, 1].grid(True)

plt.tight_layout() 
plt.show()



# %%
