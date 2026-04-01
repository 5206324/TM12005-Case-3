#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
# Zorg dat Stap1_data_loader in dezelfde map staat
from Stap1_data_loader import laad_ecg_bestand 

# --- HULPFUNCTIE: SUB-SAMPLE INTERPOLATIE ---
def vind_subsample_piek(segment):
    """
    Vindt de piek van een segment met kwadratische interpolatie (parabool fit).
    Voorkomt 'trapjes' (kwantiseringsruis) in de PR-data.
    """
    if len(segment) < 3:
        return float(np.argmax(np.abs(segment)))
    
    idx = np.argmax(np.abs(segment))
    if idx <= 0 or idx >= len(segment) - 1:
        return float(idx)
    
    y1, y2, y3 = segment[idx-1], segment[idx], segment[idx+1]
    noemer = 2 * (y1 - 2*y2 + y3)
    if noemer == 0:
        return float(idx)
    
    d = (y1 - y3) / noemer # De verschuiving t.o.v. de integer index
    return float(idx + d)

# --- 1. DATA INLADEN ---
ecg_raw, fs, t_raw = laad_ecg_bestand("004_Groenewoud_PACs+PVCs.mat", plotresult=False)

# Inladen van de vooraf gedetecteerde R-toppen (indices en tijden)
r_indices = np.load("r_toppen_indices.npy")

print(f"Aantal ingeladen R-toppen: {len(r_indices)}")

# --- 2. ANALYSE VAN ALLE R-TOPPEN (Met sub-sample precisie) ---
r_x_all, pr_ms_all = [], []

for r_idx_raw in r_indices:
    # --- STAP A: VERFIJN DE R-TOP (Snap-to-Peak) ---
    win_r = int(0.100 * fs)
    s_r, e_r = max(0, r_idx_raw - win_r), min(len(ecg_raw), r_idx_raw + win_r)
    
    r_rel_sub = vind_subsample_piek(ecg_raw[s_r:e_r])
    r_idx_refined = s_r + r_rel_sub
    
    # --- STAP B: IDENTIFICEER DE P-TOP ---
    p_s = max(0, int(r_idx_refined) - int(0.220 * fs))
    p_e = max(0, int(r_idx_refined) - int(0.100 * fs))
    
    if p_s < p_e and p_s > 0:
        p_zone = ecg_raw[p_s:p_e]
        
        # Phasor Transform voor P-top stabiliteit
        phi = np.arctan(p_zone / 20.0) 
        phi_smooth = np.convolve(phi, np.ones(5)/5, mode='same')
        
        p_rel_sub = vind_subsample_piek(phi_smooth)
        p_idx_abs = p_s + p_rel_sub
        
        r_x_all.append(r_idx_refined / fs)
        pr_ms_all.append(((r_idx_refined - p_idx_abs) / fs) * 1000)

r_x_all = np.array(r_x_all)
pr_ms_all = np.array(pr_ms_all)

# --- 3. HET GAT VERWIJDEREN (Tijd-shift logica) ---
r_x_shifted = np.copy(r_x_all)
diffs = np.diff(r_x_all)
gap_indices = np.where(diffs > 3600)[0]

if len(gap_indices) > 0:
    cumulative_gap = 0
    for idx in gap_indices:
        gap_duration = r_x_all[idx+1] - r_x_all[idx]
        cumulative_gap += gap_duration
        r_x_shifted[idx+1:] = r_x_all[idx+1:] - cumulative_gap
    print(f"Gat succesvol gedicht.")

# --- 4. PAC DETECTIE (PR-variabiliteit) ---
pac_indices = []
window_size = 15 
threshold = 20 # ms drempel voor sprong

for i in range(window_size, len(pr_ms_all) - 1):
    local_ref = np.median(pr_ms_all[i-window_size:i])
    
    pr_curr = pr_ms_all[i]
    pr_prev = pr_ms_all[i-1]
    pr_next = pr_ms_all[i+1]
    
    # Kenmerk: Korter dan historie, korter dan vorige, herstel bij volgende
    if (pr_curr < local_ref - threshold) and \
       (pr_curr < pr_prev - 15) and \
       (pr_next > pr_curr + 15):
        pac_indices.append(i)

pac_times = r_x_shifted[pac_indices]
pac_values = pr_ms_all[pac_indices]

# --- 5. VISUALISATIE ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Paneel 1: PR-Interval over de tijd (Gat verwijderd)
ax1.plot(r_x_shifted, pr_ms_all, color='tab:green', linewidth=0.5, alpha=0.6, label='PR-verloop (Sub-sample)')
if len(pr_ms_all) > 50:
    trend = pd.Series(pr_ms_all).rolling(window=50, center=True).mean()
    ax1.plot(r_x_shifted, trend, color='orange', linewidth=2, label='Trend (50 slagen)')

# Markeer PACs
ax1.scatter(pac_times, pac_values, color='red', marker='x', s=40, label='PAC Detectie', zorder=10)

ax1.axhline(200, color='red', linestyle='--', alpha=0.8, label='Grens AV-blok (200ms)')
ax1.set_title("Overzicht: PR-Interval Verloop & PAC Detectie")
ax1.set_xlabel("Tijd (s) - Aaneengesloten"); ax1.set_ylabel("PR (ms)")
ax1.set_ylim(50, 350); ax1.grid(True, alpha=0.3)
ax1.legend()

# Paneel 2: Histogram van PR-intervallen
ax2.hist(pr_ms_all[~np.isnan(pr_ms_all)], bins=100, color='skyblue', edgecolor='black')
ax2.set_title("Distributie van PR-intervallen")
ax2.set_xlabel("PR (ms)"); ax2.set_ylabel("Aantal")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- 6. STATISTIEKEN ---
print("-" * 30)
print(f"RESULTATEN ANALYSE")
print("-" * 30)
print(f"Gemiddelde PR: {np.nanmean(pr_ms_all):.2f} ms")
print(f"Mediaan PR:    {np.nanmedian(pr_ms_all):.2f} ms")
print(f"Aantal PACs:   {len(pac_indices)}")
print("-" * 30)