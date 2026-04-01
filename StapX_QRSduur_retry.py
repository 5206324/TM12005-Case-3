#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import peak_widths
from scipy.io import loadmat
from scipy import signal
from scipy.fft import rfft, rfftfreq
import datetime
import matplotlib.dates as mdates
import pandas as pd
from read_telemetry_ecg import ecg_PT, t, fs,ecg, locs
from Stap3_Ventriculaire_activiteit import ecg_PT_both

#%%
# --- 1. Voorbereiding & Schoonmaken ---
# We begrenzen extreme uitschieters (clipping) om de grafiek leesbaar te houden.
# Alles boven 5x de mediaan van de pieken wordt als ruis/uitschieter beschouwd.
ecgmai, squared, band_passed, deriv_filtered= ecg_PT_both(ecg, fs)

limit = np.median(ecgmai[locs]) * 10
ecgmai_clean = np.clip(ecgmai, 0, limit)

# Bereken de breedte van elke gevonden piek (locs) in het moving average signaal.
# We meten op 50% van de hoogte (Full Width at Half Maximum).
widths_samples = peak_widths(ecgmai_clean, locs, rel_height=0.5)[0]
widths_seconds = widths_samples / fs

# Lijsten voor de resultaten
normale_slagen = []
pac_slagen = []
pvc_slagen = []
ruis_momenten = []

window_size = 5 

# --- 2. De Analyse Loop ---
for i in range(window_size, len(locs)):
    
    # A. TIMING (RR-intervallen)
    # t is een Pandas Series met Timestamps, dus diff() geeft Timedeltas.
    # We zetten dit om naar seconden via .total_seconds()
    rr_huidig = (t[locs[i]] - t[locs[i-1]]).total_seconds()
    
    # Historie voor baseline (laatste 5 intervallen)
    rr_historie = []
    for j in range(i-window_size, i):
        diff = (t[locs[j+1]] - t[locs[j]]).total_seconds()
        rr_historie.append(diff)
    rr_baseline = np.median(rr_historie)
    
    # B. MORFOLOGIE (Hoogte en Breedte)
    hoogte_huidig = ecgmai_clean[locs[i]]
    hoogte_baseline = np.median([ecgmai_clean[locs[j]] for j in range(i-window_size, i)])
    
    breedte_huidig = widths_seconds[i]
    breedte_baseline = np.median(widths_seconds[i-window_size : i])

    # --- 3. CLASSIFICATIE LOGICA ---
    
    # Criterium 1: Is het een hartslag of ruis?
    # Een QRS complex is fysiologisch zelden breder dan 250ms.
    is_ruis = breedte_huidig > 0.25 or hoogte_huidig > (limit * 0.95)
    
    # Criterium 2: Is de slag te vroeg (Prematuur)?
    # 0.80 is een veilige grens voor PAC detectie.
    is_prematuur = rr_huidig < 0.80 * rr_baseline
    
    # Criterium 3: Is de vorm afwijkend?
    # Een PVC is typisch breder (>20%) EN/OF hoger (>50%) dan normaal.
    is_afwijkend = (breedte_huidig > 1.20 * breedte_baseline) or (hoogte_huidig > 1.50 * hoogte_baseline)

    if is_ruis:
        ruis_momenten.append(locs[i])
    elif is_prematuur:
        if is_afwijkend:
            pvc_slagen.append(locs[i])
        else:
            pac_slagen.append(locs[i])
    else:
        normale_slagen.append(locs[i])

# --- 4. Resultaten & Visualisatie ---
print(f"--- Analyse Resultaten ---")
print(f"Normale slagen: {len(normale_slagen)}")
print(f"PAC's gevonden: {len(pac_slagen)}")
print(f"PVC's gevonden: {len(pvc_slagen)}")
print(f"Gefilterde ruis: {len(ruis_momenten)}")

# Visualisatie
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(t, ecgmai_clean, color='lightgray', label='ECG Moving Average', zorder=1)

# Plot de categorieën
ax.scatter(t[normale_slagen], ecgmai_clean[normale_slagen], color='green', s=15, label='Normaal', alpha=0.7)
ax.scatter(t[pac_slagen], ecgmai_clean[pac_slagen], color='orange', s=25, label='PAC', alpha=0.8)
ax.scatter(t[pvc_slagen], ecgmai_clean[pvc_slagen], color='red', marker='x', s=60, label='PVC', zorder=5)

# X-as instellen (Eerste 30 seconden vanaf de eerste gevonden slag)
# start_t = t[locs[window_size]]
# eind_t = start_t + pd.Timedelta(seconds=30)
# ax.set_xlim(start_t, eind_t)

# Y-as instellen op een zinvol bereik
#

# Tijd-formattering verbeteren
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()

plt.title("Gefilterde ECG Classificatie (Breedte + Timing)")
plt.xlabel("Tijd")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.2)
plt.show()