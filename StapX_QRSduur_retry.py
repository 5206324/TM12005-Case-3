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
af_momenten = []
window_size = 5 
# --- Nieuwe variabelen voor AF detectie ---
#%%w
# Een drempelwaarde voor de Coefficient of Variation (CV). 
# Meestal duidt een CV > 0.12 - 0.15 op AF bij een rustig ritme.
af_threshold = 0.20

# --- 2. De Analyse Loop ---
for i in range(window_size, len(locs)):
        # Sla deze slag over en reset de historie
    # A. TIMING (RR-intervallen)
    rr_huidig = (t[locs[i]] - t[locs[i-1]]).total_seconds()
     # Bovenin de loop:

    # Historie voor baseline en variabiliteit (bijv. laatste 10 slagen voor AF stabiliteit)
    rr_historie = []
    for j in range(i-window_size, i):
        diff = (t[locs[j+1]] - t[locs[j]]).total_seconds()
        rr_historie.append(diff)
    
    rr_baseline = np.median(rr_historie)
    
    # --- AF CALCULATIE ---
    # Bereken de Coefficient of Variation (Standaarddeviatie / Gemiddelde)
    rr_std = np.std(rr_historie)
    rr_mean = np.mean(rr_historie)
    cv = rr_std / rr_mean if rr_mean > 0 else 0
    
    is_af = cv > af_threshold
    # ---------------------

    # B. MORFOLOGIE (Hoogte en Breedte)
    # ... (houd de bestaande code voor hoogte_huidig, breedte_huidig, etc.) ...
    hoogte_huidig = ecgmai_clean[locs[i]]
    hoogte_baseline = np.median([ecgmai_clean[locs[j]] for j in range(i-window_size, i)])
    breedte_huidig = widths_seconds[i]
    breedte_baseline = np.median(widths_seconds[i-window_size : i])

    # --- 3. CLASSIFICATIE LOGICA ---
    is_ruis = breedte_huidig > 0.25 or hoogte_huidig > (limit * 0.95)
    is_prematuur = rr_huidig < 0.80 * rr_baseline
    is_afwijkend = (breedte_huidig > 1.20 * breedte_baseline) or (hoogte_huidig > 1.50 * hoogte_baseline)

    if is_ruis:
        ruis_momenten.append(locs[i])
    elif is_af:
        # Als het ritme totaal onregelmatig is, labelen we het als AF
        af_momenten.append(locs[i])
    elif is_prematuur:
        if is_afwijkend:
            pvc_slagen.append(locs[i])
        else:
            pac_slagen.append(locs[i])
    else:
        normale_slagen.append(locs[i])
#%%
# --- 4. Resultaten Update ---


# Voeg dit toe aan de plot:

# --- 4. Resultaten & Visualisatie ---
print(f"--- Analyse Resultaten ---")
print(f"Momenten van AF: {len(af_momenten)}")
print(f"Normale slagen: {len(normale_slagen)}")
print(f"PAC's gevonden: {len(pac_slagen)}")
print(f"PVC's gevonden: {len(pvc_slagen)}")
print(f"Gefilterde ruis: {len(ruis_momenten)}")

# --- 4. Visualisatie zonder tijdsgat ---

fig, ax = plt.subplots(figsize=(15, 7))

# We maken een nieuwe index-as voor de locs die we gebruiken
# i_normale_slagen bevat de indices van de normale slagen in de 'locs' lijst
# We zoeken de positie van de momenten in de 'locs' array
indices_normaal = [np.where(locs == m)[0][0] for m in normale_slagen]
indices_pac = [np.where(locs == m)[0][0] for m in pac_slagen]
indices_pvc = [np.where(locs == m)[0][0] for m in pvc_slagen]
indices_af = [np.where(locs == m)[0][0] for m in af_momenten]

# Plot de categorieën op basis van hun volgorde (index i)
ax.scatter(indices_normaal, ecgmai_clean[normale_slagen], color='green', s=15, label='Normaal', alpha=0.7)
# ax.scatter(indices_pac, ecgmai_clean[pac_slagen], color='orange', s=25, label='PAC', alpha=0.8)
ax.scatter(indices_pvc, ecgmai_clean[pvc_slagen], color='red', marker='x', s=60, label='PVC', zorder=5)

#ax.vlines(indices_normaal, ymin=0, ymax=limit, color='green', alpha=0.002, label='Normaal')
# AF Zone als verticale lijnen op de index-as
if indices_af:
    ax.vlines(indices_af, ymin=0, ymax=limit, color='purple', alpha=0.02, label='AF Zone')
ax.vlines(indices_pac, ymin=0, ymax=limit, color='orange', alpha=0.02, label='PAC')
# --- De X-as labels terugzetten naar Tijd ---
# We kiezen bijv. 10 mooie plekken op de as om de tijd te tonen
num_ticks = 10
tick_indices = np.linspace(0, len(locs)-1, num_ticks, dtype=int)
tick_labels = [t[locs[i]].strftime('%H:%M') for i in tick_indices]

ax.set_xticks(tick_indices)
ax.set_xticklabels(tick_labels)

# Y-as begrenzen voor betere zoom op de slagen
ax.set_ylim(-100, limit * 1.2)

plt.title("ECG Classificatie: Gat van 6 uur verwijderd (Gecomprimeerde tijd)")
plt.xlabel("Tijd (sprong tussen sessies)")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.2)
plt.show()
