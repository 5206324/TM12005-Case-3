#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from Stap1_data_loader import laad_ecg_bestand 

# --- 1. DATA INLADEN ---
# We laden het bestand in. De loader geeft ecg_raw en fs terug.
ecg_raw, fs, t_raw = laad_ecg_bestand("004_Groenewoud_PACs+PVCs.mat", plotresult=False)

# Inladen van de vooraf gedetecteerde R-toppen (indices en tijden)
r_tijden = np.load("r_toppen_tijden.npy")
r_indices = np.load("r_toppen_indices.npy")

print(f"Aantal ingeladen R-toppen: {len(r_tijden)}")

# --- 2. ANALYSE VAN ALLE R-TOPPEN (Voor het lange overzicht) ---
# --- 2. ANALYSE VAN ALLE R-TOPPEN (Eerst verfijnen, dan P-top zoeken) ---
r_x_all, pr_ms_all = [], []

for r_idx_raw in r_indices:
    # --- STAP A: VERFIJN DE R-TOP (Snap-to-Peak) ---
    # We kijken in een venster van 100ms rondom de Pan-Tompkins detectie
    win_r = int(0.100 * fs)
    s_r, e_r = max(0, r_idx_raw - win_r), min(len(ecg_raw), r_idx_raw + win_r)
    
    # Zoek de ECHTE piek (hoogste absolute uitslag) in het ruwe signaal
    r_idx_refined = s_r + np.argmax(np.abs(ecg_raw[s_r:e_r]))
    
    # --- STAP B: IDENTIFICEER DE P-TOP (Relatief aan de verfijnde R-top) ---
    # Nu we de echte piek hebben, bepalen we het P-venster (80ms tot 220ms ervoor)
    p_s = max(0, r_idx_refined - int(0.220 * fs))
    p_e = max(0, r_idx_refined - int(0.100 * fs))
    
    if p_s < p_e and p_s > 0:
        p_zone = ecg_raw[p_s:p_e]
        
        # Phasor Transform met Rv=20 voor stabiliteit
        phi = np.arctan(p_zone / 20.0) 
        phi_smooth = np.convolve(phi, np.ones(5)/5, mode='same')
        p_idx_abs = p_s + np.argmax(phi_smooth)
        
        # Sla de gegevens op gebaseerd op de VERFIJNDE R-top
        r_x_all.append(r_idx_refined / fs)
        pr_ms_all.append(((r_idx_refined - p_idx_abs) / fs) * 1000)

r_x_all = np.array(r_x_all)
pr_ms_all = np.array(pr_ms_all)
# --- 3. HET GAT VERWIJDEREN (Tijd-shift logica) ---
r_x_shifted = np.array(r_x_all)
pr_ms_all = np.array(pr_ms_all)

# We zoeken naar sprongen in de tijd groter dan 1 uur (3600 sec) om het 6-uurs gat te dichten
diffs = np.diff(r_x_all)
gap_indices = np.where(diffs > 3600)[0]

if len(gap_indices) > 0:
    for idx in gap_indices:
        gap_duration = r_x_all[idx+1] - r_x_all[idx]
        # Verschuif alle tijdstippen na het gat naar voren
        r_x_shifted[idx+1:] -= gap_duration
    print(f"Gat succesvol gedicht in de overzichtsgrafiek.")

# --- 4. SEGMENT SELECTIE & DETAIL FILTERING ---
# We nemen een segment van 12 seconden vanaf seconde 150 voor de detailplots.
zoom_start_sec = 150
zoom_end_sec = 162
start_samp = int(zoom_start_sec * fs)
end_samp = int(zoom_end_sec * fs)

# Pak de data en centreer deze om baseline wander te verwijderen
ecg_seg = ecg_raw[start_samp:end_samp]
ecg_seg_clean = ecg_seg - np.median(ecg_seg)
t_seg = np.linspace(zoom_start_sec, zoom_end_sec, len(ecg_seg_clean))

# Filter de reeds berekende data voor de detail-vensters
mask_seg = (np.array(r_x_all) >= zoom_start_sec) & (np.array(r_x_all) <= zoom_end_sec)
r_x_seg_filtered = np.array(r_x_all)[mask_seg]
pr_ms_seg_filtered = np.array(pr_ms_all)[mask_seg]

r_y_seg, p_x_seg, p_y_seg = [], [], []

for i, r_time in enumerate(r_x_seg_filtered):
    # Vind de relatieve index binnen het 12-seconde segment
    r_idx_rel = int((r_time - zoom_start_sec) * fs)
    
    # VERFIJN R-TOP (Snap-to-Peak logica)
    # Zoek de index van de hoogste absolute amplitude in een window van 60ms
    win = int(0.06 * fs)
    s, e = max(0, r_idx_rel-win), min(len(ecg_seg_clean), r_idx_rel+win)
    r_idx_refined = s + np.argmax(np.abs(ecg_seg_clean[s:e]))
    
    # P-top locatie herleiden uit het PR-interval
    p_time = r_time - (pr_ms_seg_filtered[i] / 1000)
    p_idx_rel = int((p_time - zoom_start_sec) * fs)
    
    r_y_seg.append(ecg_seg_clean[r_idx_refined])
    p_x_seg.append(p_time)
    p_y_seg.append(ecg_seg_clean[p_idx_rel] if 0 <= p_idx_rel < len(ecg_seg_clean) else 0)

# --- 5. VISUALISATIE (3 Panelen) ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))

# Paneel 1: ECG Detail met Markers
ax1.plot(t_seg, ecg_seg_clean, label='Gecentreerd ECG', color='tab:blue', alpha=0.8)
ax1.scatter(r_x_seg_filtered, r_y_seg, color='red', marker='*', s=150, label='R-top', zorder=5)
ax1.scatter(p_x_seg, p_y_seg, color='green', marker='o', s=80, label='P-top', zorder=5)
ax1.set_title(f"Detail: ECG Analyse (Seconde {zoom_start_sec} - {zoom_end_sec})")
ax1.set_ylim(np.percentile(ecg_seg_clean, 1)*2.5, np.percentile(ecg_seg_clean, 99)*2.5)
ax1.legend(loc='upper right'); ax1.grid(True, alpha=0.3)

# Paneel 2: PR-interval Detail Trend
ax2.plot(r_x_seg_filtered, pr_ms_seg_filtered, 'go-', markersize=6, label='PR-interval (Detail)')
ax2.axhline(200, color='red', linestyle='--', alpha=0.7, label='Grens AV-blok (200ms)')
ax2.set_title("Detail: PR-Interval Variabiliteit")
ax2.set_ylabel("PR (ms)"); ax2.set_ylim(50, 350); ax2.grid(True, alpha=0.3)

# Paneel 3: Lang Venster (Zonder het 6-uurs gat)
ax3.plot(r_x_shifted, pr_ms_all, color='darkgreen', linewidth=0.5, alpha=0.5, label='PR-verloop')
if len(pr_ms_all) > 50:
    trend = pd.Series(pr_ms_all).rolling(window=50, center=True).mean()
    ax3.plot(r_x_shifted, trend, color='orange', linewidth=2, label='Trend (50 slagen)')

ax3.axhline(200, color='red', linestyle='--', alpha=0.8)
ax3.set_title("Overzicht: PR-Interval Verloop (Gat verwijderd)")
ax3.set_xlabel("Tijd (s) - Aaneengesloten as"); ax3.set_ylabel("PR (ms)")
ax3.set_ylim(50, 400); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()