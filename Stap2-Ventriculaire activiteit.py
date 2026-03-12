#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
import datetime
import matplotlib.dates as mdates
import pandas as pd

# --- CONFIG & DATA LOAD ---
leads = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
path_both = r"/Users/fenne/Documents/Technical Medicine/TM12005 Advanced Signal Processing/Case 3/TM12005-Case-3/TM12005 Advanced Signal Processing (202526 Q3) - 322026 - 159 PM 2/004_Groenewoud_PACs+PVCs.mat"

def read_ecg_both(path_both):
    data_both = loadmat(path_both, squeeze_me=True, struct_as_record=False)
    ecg_both = data_both['ecg'].sig[:,leads.index('II')]
    fs = data_both['ecg'].header.Sampling_Rate
    t0 = datetime.datetime(*data_both['ecg'].start_vec)
    nSamples = len(ecg_both)
    t = pd.date_range(start=t0, periods=nSamples, freq=pd.Timedelta(seconds=1/fs))
    return ecg_both, fs, t

def ecg_PT_both(ecg_both, fs):
    b, a = signal.butter(2, [5, 15], btype="band", fs=fs)
    band_passed = signal.filtfilt(b, a, ecg_both)
    T = 1 / fs
    deriv_filtered = signal.lfilter([-1, -2, 0, 2, 1], 8 * T, band_passed)
    squared = deriv_filtered**2
    N = int(0.150 * fs)
    ecgmai = signal.lfilter(np.ones(N)/N, 1, squared)
    return ecgmai
#%%
# --- EXECUTION ---
ecg_raw, fs, t_raw = read_ecg_both(path_both)
ecgmai = ecg_PT_both(ecg_raw, fs)

# Peak detection (Ventriculaire activiteit)
locs, _ = signal.find_peaks(ecgmai, height=1e6, distance=int(0.3*fs))

# Bereken RR-intervallen
rr_intervals_sec = np.diff(locs) / fs
t_peaks = t_raw[locs[1:]]

# --- GAP FILTERING (De 6 uur verwijderen) ---
# We definiëren een gap als alles groter dan 60 seconden
gap_threshold = 60 
valid_mask = rr_intervals_sec < gap_threshold

clean_rr = rr_intervals_sec[valid_mask]
clean_t = t_peaks[valid_mask]

# Om de 6 uur echt uit de x-as te halen, maken we een "relatieve" tijd-as
# die de sprong negeert voor de plot-index
x_axis = np.arange(len(clean_rr)) 

#%%
# --- PLOTTING (Zoals je voorbeeldafbeelding) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Paneel A: ECG (Zoom op eerste 10 seconden van bruikbare data)
zoom_samples = int(10 * fs)
ax1.plot(np.arange(zoom_samples)/fs, ecg_raw[:zoom_samples])
ax1.set_title("ECG Signaal (Eerste 10 seconden)")
ax1.set_ylabel("Amplitude (mV)")
ax1.set_xlabel("Tijd (s)")

# Paneel C: RR-interval over tijd (ZONDER de 6 uur gap)
# We plotten hier tegen de index om de tijd-gap fysiek te verwijderen
ax2.plot(x_axis, clean_rr, 'o-', markersize=2, linewidth=0.5, color='tab:blue')

# We zetten de x-ticks terug naar leesbare tijden, maar zonder de gaten
# We tonen elke 500 slagen een tijdstip
tick_spacing = max(1, len(x_axis)//10)
ax2.set_xticks(x_axis[::tick_spacing])
ax2.set_xticklabels([clean_t[i].strftime('%H:%M:%S') for i in range(0, len(x_axis), tick_spacing)], rotation=45)

ax2.set_title("Ventriculaire Activiteit (RR-intervallen zonder dode tijd)")
ax2.set_ylabel("RR-interval (sec)")
ax2.set_xlabel("Tijdstip in opname")
ax2.set_ylim(0.3, 1.5) # Focus op fysiologische range
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
#%%
# --- ANALYSE ---
mean_hr = 60 / np.mean(clean_rr)
print(f"Gemiddelde ventriculaire frequentie: {mean_hr:.1f} BPM")
print(f"Aantal gedetecteerde ventriculaire activaties: {len(locs)}")
# %%
# --- ZOEK EEN SCHOON SEGMENT VAN 10 SECONDEN ---

# We zoeken vanaf een startpunt naar het eerste stukje 'continue' data
search_start_time = 60  # Start met zoeken vanaf 60 seconden
window_size = 100        # We willen 10 seconden zien

# Filter locs die na de start_time vallen
future_locs = locs[locs > (search_start_time * fs)]

# Zoek naar het eerste segment waar de intervallen allemaal klein zijn (< 10s)
found_start_idx = None
for i in range(len(future_locs) - 1):
    # Check of de volgende 10 seconden aan hartslagen geen gat bevatten
    current_time = future_locs[i] / fs
    # Kijk naar alle slagen in de komende 10 seconden
    mask_10s = (future_locs >= future_locs[i]) & (future_locs <= future_locs[i] + (window_size * fs))
    potential_intervals = np.diff(future_locs[mask_10s]) / fs
    
    if len(potential_intervals) > 5 and np.all(potential_intervals < 10):
        found_start_idx = future_locs[i]
        break

if found_start_idx is not None:
    # Definieer de grenzen voor de plot
    zoom_start = found_start_idx
    zoom_end = zoom_start + int(window_size * fs)

    # Slice de data
    ecg_segment = ecg_raw[zoom_start:zoom_end]
    t_segment = t_raw[zoom_start:zoom_end]
    
    # Vind pieken in dit segment
    segment_locs = locs[(locs >= zoom_start) & (locs < zoom_end)]
    t_peaks_rel = (segment_locs - zoom_start) / fs
    
    # RR-intervallen berekenen
    segment_rr = np.diff(t_peaks_rel)
    t_rr_rel = t_peaks_rel[1:]

    # --- PLOTTEN ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Paneel A: ECG met sterretjes op de R-toppen
    ax1.plot(np.linspace(0, window_size, len(ecg_segment)), ecg_segment, color='tab:blue')
    ax1.plot(t_peaks_rel, ecg_segment[(segment_locs - zoom_start)], 'r*', label='R-top')
    ax1.set_ylabel("ECG (mV)")
    ax1.set_title(f"Ingezoomd ECG (Start op {zoom_start/fs:.1f}s - Geen gaten gevonden)")
    ax1.grid(alpha=0.3)

    # Paneel C: RR-intervallen (Ventriculaire Activiteit)
    # We gebruiken de 'o-' stijl uit jouw voorbeeld
    ax2.plot(t_rr_rel, segment_rr, 'o-', color='tab:red', markersize=4, linewidth=1)
    ax2.set_ylabel("RR-interval (sec)")
    ax2.set_xlabel("Tijd in segment (s)")
    ax2.set_ylim(0.3, 1.3) # Fysiologische focus
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    print("Geen segment van 10 seconden zonder gaten gevonden.")
# %%
