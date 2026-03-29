
#%%import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from Stap1_data_loader import laad_ecg_bestand 

# --- 1. DATA INLADEN ---
# We laden het bestand in. De loader geeft ecg_raw en fs terug.
ecg_raw, fs, t_raw = laad_ecg_bestand("004_Groenewoud_PACs+PVCs.mat", plotresult=False)

def ecg_PT_filters(ecg, fs):
    """Pan-Tompkins filters om QRS-energie te isoleren"""
    b, a = signal.butter(2, [5, 15], btype="band", fs=fs)
    band_passed = signal.filtfilt(b, a, ecg)
    T = 1 / fs
    deriv = signal.lfilter([-1, -2, 0, 2, 1], 8 * T, band_passed)
    squared = deriv**2
    N = int(0.150 * fs)
    ecg_mai = signal.lfilter(np.ones(N)/N, 1, squared)
    return ecg_mai

# --- 2. SEGMENT SELECTIE (Na de 6-uur sprong) ---
zoom_start_sec =150
zoom_end_sec = 162  # We bekijken 20 seconden
start_samp = int(zoom_start_sec * fs)
end_samp = int(zoom_end_sec * fs)

# Pak de data en centreer deze (essentieel voor Phasor Transform)
ecg_seg = ecg_raw[start_samp:end_samp]
ecg_seg_clean = ecg_seg - np.median(ecg_seg)
t_seg = np.linspace(zoom_start_sec, zoom_end_sec, len(ecg_seg_clean))

# --- 3. DETECTIE LOOP ---
# QRS detectie op het segment
seg_mai = ecg_PT_filters(ecg_seg_clean, fs)
# Detecteer pieken in de energie-curve
seg_locs, _ = signal.find_peaks(seg_mai, height=np.mean(seg_mai)*3, distance=int(0.3*fs))

r_x_seg, r_y_seg = [], []
p_x_seg, p_y_seg = [], []
pr_ms_seg = []

# Dynamische Phasor gevoeligheid (Rv) gebaseerd op de lokale variatie
Rv_p = np.std(ecg_seg_clean) * 0.1 

for loc in seg_locs:
    # 1. VERFIJN R-TOP (De allergrootste uitslag)
    win = int(0.06 * fs)
    s, e = max(0, loc-win), min(len(ecg_seg_clean), loc+win)
    
    # We zoeken de index van de grootste absolute waarde
    # Dit zorgt dat de ster ALTIJD op de scherpste piek komt
    abs_fragment = np.abs(ecg_seg_clean[s:e])
    r_idx = s + np.argmax(abs_fragment)
    
    # 2. DEFINIEER HET P-VENSTER (Vrij streng)
    # Zoek tussen 250ms en 100ms vóór de R-top
    p_s = max(0, r_idx - int(0.250 * fs))
    p_e = max(0, r_idx - int(0.100 * fs)) 
    
    if p_s < p_e:
        p_zone = ecg_seg_clean[p_s:p_e]
        
        # Omdat de P-top in jouw signaal omhoog lijkt te wijzen:
        phi = np.arctan(p_zone / 10.0) # Rv iets hoger voor stabiliteit
        
        # Middelen om ruis-spikes te negeren
        phi_smooth = np.convolve(phi, np.ones(5)/5, mode='same')
        p_idx = p_s + np.argmax(phi_smooth)
        
        # 3. OPSLAAN
        r_x_seg.append(zoom_start_sec + (r_idx / fs))
        r_y_seg.append(ecg_seg_clean[r_idx])
        
        p_x_seg.append(zoom_start_sec + (p_idx / fs))
        p_y_seg.append(ecg_seg_clean[p_idx])
        
        # PR-interval berekenen
        pr_ms_seg.append(((r_idx - p_idx) / fs) * 1000)
# --- 4. VISUALISATIE ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Paneel 1: ECG met R- en P-markers
ax1.plot(t_seg, ecg_seg_clean, label='Gecentreerd ECG', color='tab:blue', alpha=0.8, linewidth=1)
if len(r_x_seg) > 0:
    ax1.scatter(r_x_seg, r_y_seg, color='red', marker='*', s=150, label='Gedetecteerde R-top', zorder=5)
    ax1.scatter(p_x_seg, p_y_seg, color='green', marker='o', s=80, label='P-top (Phasor)', zorder=5)

ax1.set_title(f"ECG Analyse met Phasor Transform (Seconde {zoom_start_sec} - {zoom_end_sec})")
ax1.set_ylabel("Amplitude (gecentreerd)")
# Inzoomen op de hartslagen, negeer eventuele uitschieters
ax1.set_ylim(np.percentile(ecg_seg_clean, 1)*2, np.percentile(ecg_seg_clean, 99)*2)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Paneel 2: PR-interval trend
ax2.plot(r_x_seg[:len(pr_ms_seg)], pr_ms_seg, 'go-', markersize=6, linewidth=1.5, label='PR-interval')
ax2.axhline(200, color='red', linestyle='--', alpha=0.7, label='Grens AV-blok (200ms)')
ax2.axhline(120, color='orange', linestyle='--', alpha=0.7, label='Kort PR (120ms)')

ax2.set_title("PR-Interval Variabiliteit (Analyse van atriale geleiding)")
ax2.set_ylabel("PR-interval (ms)")
ax2.set_xlabel("Tijd in opname (s)")
ax2.set_ylim(50, 300) # Fysiologische focus
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- RESULTATEN ---
if len(pr_ms_seg) > 0:
    print(f"Gemiddeld PR-interval in dit segment: {np.mean(pr_ms_seg):.1f} ms")
    print(f"Hartslag in dit segment: {60 / (np.mean(np.diff(r_x_seg))):.1f} BPM")