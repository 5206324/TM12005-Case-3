#%%Voor de detectie van AF ben je benieuwd of het
#entriculaire ritme regulair is. Je besluit daarom om de ventriculaire activiteit te detecteren en de frequentie te bepalen. Zorg ervoor dat je script de ventriculaire activiteit kan detecteren en bepaal de frequentie.
#Laat zien hoe de frequentie verandert gedurende de opnames. Zijn er momenten met een hoge of lage hartslag?
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
import datetime
import matplotlib.dates as mdates
import pandas as pd


#%% Path, leads and def read_ecg_mat 
leads = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
path = r"C:\Users\vmoba\OneDrive\Bureaublad\Kt\Ms1\TM12005\EMC opdrachten\TM12005-Case-3\TM12005 Advanced Signal Processing (202526 Q3) - 322026 - 159 PM 2\004_Groenewoud_PACs+PVCs.mat"


def read_ecg_mat(path, plotresult=True):
    # open datafile
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    ecg = data['ecg'].sig[:,leads.index('II')]

    fs = data['ecg'].header.Sampling_Rate
    t0 = datetime.datetime(*data['ecg'].start_vec)

    nSamples = data['ecg'].sig.shape[0]
    t = pd.date_range(
        start=t0,
        periods=nSamples,
        freq=pd.Timedelta(seconds=1/fs)
    )
        
    # Plot signal in time domain
    if plotresult:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(t, ecg)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ECG (mV)")
        ax.set_title("Raw ECG Signal")
        plt.show()
    
    return ecg, fs, t
# %%
ecg, fs, t = read_ecg_mat(path, plotresult=True)


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

#%%
#%% Analyse van Ventriculaire Activiteit en Frequentie (Inclusief opschoning)

# 1. Zoek het enorme gat in de meting om de data te splitsen
rr_ruw = np.diff(locs) / fs
gap_index = np.argmax(rr_ruw) # De index met het allergrootste RR-interval

# Splits de pieken in vóór het gat en ná het gat
locs_deel_1 = locs[:gap_index + 1]
locs_deel_2 = locs[gap_index + 1:]

print(f"Dataset gesplitst vanwege groot gat in de meting.")
print(f"- Deel 1 (voor pauze): {len(locs_deel_1)} slagen")
print(f"- Deel 2 (na pauze): {len(locs_deel_2)} slagen")

# We gebruiken Deel 1 voor de analyse (pas dit gerust aan naar Deel 2 als je die wilt bekijken)
locs_analyse = locs_deel_1
locs_analyse2 = locs_deel_2

# 2. Bereken de RR-intervallen en de Hartslag (BPM) voor de analyse-set
rr_intervallen = np.diff(locs_analyse) / fs
t_slagen = t[locs_analyse[1:]] # Haal de echte tijdstippen (timestamps) op
hartslag_bpm = 60 / rr_intervallen

rr_intervallen2 = np.diff(locs_analyse2) / fs
t_slagen2 = t[locs_analyse2[1:]] # Haal de echte tijdstippen (timestamps) op
hartslag_bpm2 = 60 / rr_intervallen2

# --- PLOT 1: Hartslag (Frequentie) over tijd ---
# Dit beantwoordt je opdracht: "Laat zien hoe de frequentie verandert gedurende de opnames"
# Maak een 2x2 grid aan. 
# Tip: Maak de figsize wat hoger (bijv 14, 10) zodat de 4 grafieken niet op elkaar gepropt zitten.
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# --- PLAATJE 1: Linksboven (ax[0, 0]) - Jouw Hartslag frequentie ---
ax[0, 0].plot(t_slagen, hartslag_bpm, marker='.', linestyle='', color='b', markersize=2)
ax[0, 0].axhline(y=np.median(hartslag_bpm), color='g', linestyle='--', label=f'Mediaan: {np.median(hartslag_bpm):.1f} BPM')

ax[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax[0, 0].set_title("1. Ventriculaire Frequentie over Tijd")
ax[0, 0].set_xlabel("Tijd")
ax[0, 0].set_ylabel("Hartslag (BPM)")
ax[0, 0].set_ylim(40, 180)
ax[0, 0].legend()
ax[0, 0].grid(True)


# --- PLAATJE 2: Rechtsboven (ax[0, 1]) - Bijv. het Tachogram ---
# Hier kun je mooi je RR-intervallen plotten
ax[0, 1].plot(rr_intervallen, marker='.', linestyle='', color='purple', markersize=2)
ax[0, 1].axhline(y=np.median(rr_intervallen), color='g', linestyle='--')
ax[0, 1].set_title("2. Tachogram: Variatie in RR")
ax[0, 1].set_xlabel("Slag nummer")
ax[0, 1].set_ylabel("RR-interval (seconden)")
ax[0, 1].grid(True)


# --- PLAATJE 3: Linksonder (ax[1, 0]) - Bijv. Poincaré of AF detectie ---
# ax[1, 0].plot(...) 
ax[1, 0].set_title("3. deel 2")
ax[1, 0].plot(t_slagen2, hartslag_bpm2, marker='.', linestyle='', color='b', markersize=2)
ax[1, 0].axhline(y=np.median(hartslag_bpm2), color='g', linestyle='--', label=f'Mediaan: {np.median(hartslag_bpm2):.1f} BPM')

ax[1, 0].grid(True)


# --- PLAATJE 4: Rechtsonder (ax[1, 1]) - Bijv. Ruw ECG met PAC/PVC stippen ---
# ax[1, 1].plot(...)
ax[0, 1].plot(rr_intervallen, marker='.', linestyle='', color='purple', markersize=2)
ax[0, 1].axhline(y=np.median(rr_intervallen), color='g', linestyle='--')
ax[0, 1].set_title("2. Tachogram: Variatie in RR")
ax[0, 1].set_xlabel("Slag nummer")
ax[0, 1].set_ylabel("RR-interval (seconden)")
ax[0, 1].grid(True)


# Zorgt ervoor dat de titels en as-labels niet over elkaar heen vallen
plt.tight_layout() 

# Toon het hele dashboard
# %%
