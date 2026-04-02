#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from Stap1_data_loader import laad_ecg_bestand 

#%%
# --- 1. DATA INLADEN ---
# We laden het bestand in. De loader geeft ecg_raw en fs terug.
ecg_raw, fs, t_raw = laad_ecg_bestand("004_Groenewoud_PACs+PVCs.mat", plotresult=False)

# Inladen van de vooraf gedetecteerde R-toppen (indices en tijden)
r_tijden = np.load("r_toppen_tijden.npy")
r_indices = np.load("r_toppen_indices.npy")

print(f"Aantal ingeladen R-toppen: {len(r_tijden)}")

#%%
# --- 2. ANALYSE VAN ALLE R-TOPPEN (Eerst verfijnen, dan P-top zoeken) ---
r_x_all, pr_ms_all = [], []
# --- 0. INITIALISEER LIJSTEN ---
r_x_all, pr_ms_all = [], []
p_duur_all = []    # Lijst om alle duren in op te slaan
p_polariteit = []  # Om te kijken of de P-top positief of negatief is

for r_idx_raw in r_indices:
    # --- STAP A: VERFIJN DE R-TOP ---
    win_r = int(0.100 * fs)
    s_r, e_r = max(0, r_idx_raw - win_r), min(len(ecg_raw), r_idx_raw + win_r)
    r_idx_refined = s_r + np.argmax(np.abs(ecg_raw[s_r:e_r]))
    
    # --- STAP B: P-TOP VENSTER ---
    p_s = max(0, r_idx_refined - int(0.250 * fs)) # lets ruimer venster voor onset/offset
    p_e = max(0, r_idx_refined - int(0.080 * fs))
    
    if p_s < p_e and p_s > 0:
        p_zone = ecg_raw[p_s:p_e]
        
        # Phasor Transform
        phi = np.arctan(p_zone / 20.0) 
        phi_smooth = np.convolve(phi, np.ones(5)/5, mode='same')
        
        # Zoek de piek (hoogste absolute waarde voor morfologie-onafhankelijkheid)
        p_peak_idx = np.argmax(np.abs(phi_smooth))
        p_peak_val = phi_smooth[p_peak_idx]
        
        # 1. POLARITEIT (Richting)
        # Is de P-top omhoog (+) of omlaag (-)?
        p_polariteit.append(1 if p_peak_val > 0 else -1)

        # 2. DUUR (Onset en Offset)
        # We gebruiken 10% van de piekhoogte als drempel voor het begin/einde
        threshold = 0.1 * np.abs(p_peak_val)
        
        # Zoek Onset (terugwaarts)
        onset_idx = p_peak_idx
        while onset_idx > 0 and np.abs(phi_smooth[onset_idx]) > threshold:
            onset_idx -= 1

        # Zoek Offset (voorwaarts)
        offset_idx = p_peak_idx
        while offset_idx < len(phi_smooth)-1 and np.abs(phi_smooth[offset_idx]) > threshold:
            offset_idx += 1

        # Bereken duur en sla op in de LIJST
        duur_ms = ((offset_idx - onset_idx) / fs) * 1000
        p_duur_all.append(duur_ms)
        
        # PR-interval opslaan
        p_idx_abs = p_s + p_peak_idx
        r_x_all.append(r_idx_refined / fs)
        pr_ms_all.append(((r_idx_refined - p_idx_abs) / fs) * 1000)

# --- 3. PRINT STATISTIEKEN (Buiten de loop) ---
p_duur_all = np.array(p_duur_all)

print("\n--- P-TOP MORFOLOGIE ANALYSE ---")
print(f"Gemiddelde duur:  {np.mean(p_duur_all):.1f} ms")
print(f"Mediaan duur:     {np.median(p_duur_all):.1f} ms")
print(f"Standaarddeviatie: {np.std(p_duur_all):.1f} ms")




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



#%%
# --- 3. PAC DETECTIE (Gebaseerd op PR-variabiliteit t.o.v. historie) ---
pac_indices = []
window_size = 10  # Aantal voorgaande slagen om de "normaal" te bepalen

# We loopen door de berekende PR-intervallen
for i in range(window_size, len(pr_ms_all) - 1):
    # Pak de voorgaande PR-intervallen als referentie
    local_ref = np.median(pr_ms_all[i-window_size:i])
    # Een PAC heeft vaak een PR-interval dat afwijkt van het ritme
    # We kijken naar de 'sprong' (huidige slag vs vorige)
    current_diff = pr_ms_all[i] - pr_ms_all[i-1]
    recovery_diff = pr_ms_all[i+1] - pr_ms_all[i]
    
    # PAC Kenmerk: 
    # 1. De huidige PR wijkt significant af van de mediane historie (> 20ms)
    # 2. Er is een duidelijke 'dip' gevolgd door een 'herstel' (V-vorm in de grafiek)
    threshold = 20 # milliseconden
    
    if (current_diff < -threshold) and (recovery_diff > 2*threshold):
        pac_indices.append(i)

pac_times = r_x_shifted[pac_indices]
pac_values = pr_ms_all[pac_indices]

print(f"Aantal potentiële PACs gedetecteerd: {len(pac_indices)}")
#%% 
# --- 4. SEGMENT SELECTIE & DETAIL FILTERING ---
# We nemen een segment van 12 seconden vanaf seconde 150 voor de detailplots.
zoom_start_sec = 2000
zoom_end_sec = 2030
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


#%%
# --- VISUALISATIE P-DUUR OVER TIJD ---
plt.figure(figsize=(15, 6))

# Plot de individuele metingen
plt.plot(r_x_shifted, p_duur_all, color='tab:purple', linewidth=0.5, alpha=0.4, label='P-duur (per slag)')

# Voeg een trendlijn toe voor leesbaarheid (rolling mean van 50 slagen)
if len(p_duur_all) > 50:
    p_trend = pd.Series(p_duur_all).rolling(window=50, center=True).mean()
    plt.plot(r_x_shifted, p_trend, color='darkviolet', linewidth=2, label='Trend (50 slagen)')

# Voeg klinische grenzen toe (bijv. 120ms als bovengrens voor normaal)
plt.axhline(120, color='red', linestyle='--', alpha=0.7, label='120ms')
plt.axhline(80, color='blue', linestyle='--', alpha=0.5, label='80ms')

plt.title("Verloop van de P-top Duur (Morfologie Analyse)")
plt.xlabel("Tijd (s)")
plt.ylabel("P-duur (ms)")
plt.ylim(40, 200) # Focus op het relevante gebied
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
# --- 5. VISUALISATIE (3 Panelen) ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 24))

# Paneel 1: ECG Detail met Markers
ax1.plot(t_seg, ecg_seg_clean, label='Gecentreerd ECG', color='tab:blue', alpha=0.8)
ax1.scatter(r_x_seg_filtered, r_y_seg, color='red', marker='*', s=150, label='R-top', zorder=5)
ax1.scatter(p_x_seg, p_y_seg, color='green', marker='o', s=80, label='P-top', zorder=5)
ax1.set_title(f"Detail: ECG Analyse (Seconde {zoom_start_sec} - {zoom_end_sec})")
ax1.set_ylim(np.percentile(ecg_seg_clean, 1)*2.5, np.percentile(ecg_seg_clean, 99)*2.5)
ax1.set_xlabel("Tijd (s)"); ax1.set_ylabel("Amplitude (mV)")
ax1.axhline(25, color='red', linestyle='--', alpha=0.5, label='normale P-top amplitude')
ax1.axhline(0, color='black', linestyle='-', alpha=0.5, label= '0 mV')
ax1.axhline(-25, color='red', linestyle='--', alpha=0.5, label='normale P-top amplitude')
ax1.legend(loc='upper right'); ax1.grid(True, alpha=0.3)

# Paneel 2: PR-interval Detail Trend
ax2.plot(r_x_seg_filtered, pr_ms_seg_filtered, 'go-', markersize=6, label='PR-interval (Detail)')
ax2.axhline(200, color='red', linestyle='--', alpha=0.7, label='200ms')
ax2.set_title("Detail: PR-Interval Variabiliteit")
ax2.set_ylabel("PR (ms)"); ax2.set_ylim(50, 350); ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
# Paneel 3: Lang Venster (Zonder het 6-uurs gat)
ax3.plot(r_x_shifted, pr_ms_all, color='darkgreen', linewidth=0.5, alpha=0.5, label='PR-verloop')
if len(pr_ms_all) > 50:
    trend = pd.Series(pr_ms_all).rolling(window=50, center=True).mean()
    ax3.plot(r_x_shifted, trend, color='orange', linewidth=2, label='Trend (50 slagen)')

ax3.axhline(200, color='red', linestyle='--', alpha=0.8)
# ax3.scatter(pac_times, pac_values, color='red', marker='x', label='PAC (PR-jump)', zorder=10)
# ax3.legend()
ax3.set_title("Overzicht: PR-Interval Verloop volledige meting")
ax3.set_xlabel("Tijd (s)"); ax3.set_ylabel("PR (ms)")
ax3.set_ylim(50, 400); ax3.grid(True, alpha=0.3)

# Paneel 2: Histogram van PR-intervallen
ax4.hist(pr_ms_all[~np.isnan(pr_ms_all)], bins=100, color='skyblue', edgecolor='black')
ax4.set_title("Distributie van PR-intervallen")
ax4.set_xlabel("PR (ms)"); ax4.set_ylabel("Aantal")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# --- 6. RESULTATEN PRINTEN ---
gemiddelde_pr = np.nanmean(pr_ms_all)
mediaan_pr = np.nanmedian(pr_ms_all)
std_pr = np.nanstd(pr_ms_all)

print("-" * 30)
print(f"STATISTIEKEN PR-INTERVAL")
print("-" * 30)
print(f"Gemiddelde PR-interval: {gemiddelde_pr:.2f} ms")
print(f"Mediaan PR-interval:    {mediaan_pr:.2f} ms")
print(f"Standaarddeviatie:      {std_pr:.2f} ms")
print("-" * 30)

# Optioneel: Interpretatie
if gemiddelde_pr > 200:
    print("Interpretatie: Het gemiddelde PR-interval is verlengd (>200ms), wat kan duiden op een 1e-graads AV-blok.")
else:
    print("Interpretatie: Het gemiddelde PR-interval valt binnen de normale grenzen (120-200ms).")

# %%
import numpy as np
import matplotlib.pyplot as plt

# --- 1. FILTERING EN TIJDSVERSCHUIVING ---
# Definieer het gat (ongeveer tussen 17.500 en 39.500 seconden)
mask = ~((np.array(r_x_all) > 17500) & (np.array(r_x_all) < 39500))

t_filt = np.array(r_x_all)[mask]
duur_filt = np.array(p_duur_all)[mask]
pol_filt = np.array(p_polariteit)[mask]
pr_filt = np.array(pr_ms_all)[mask]

# Verschuif de tijd om het gat fysiek te sluiten
gap_offset = 39500 - 17500
t_shifted = np.array([t - gap_offset if t > 39500 else t for t in t_filt])

# --- 2. BEREKENINGEN (P-P en Trends) ---
# Bereken absolute P-top tijden: P_tijd = R_tijd - (PR_interval / 1000)
p_times = t_shifted - (pr_filt / 1000)
pp_intervals = np.diff(p_times) * 1000  # in ms

# Trends berekenen (rolling average over 50 slagen)
def get_trend(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='same')

trend_pp = get_trend(pp_intervals)
trend_duur = get_trend(duur_filt)

# --- 4. RITME CLASSIFICATIE ---
def classify_rhythm(pp_values, window_size=10):
    n = len(pp_values)
    labels = np.zeros(n, dtype=int) # 0 = Sinus, 1 = PAC, 2 = AF
    
    # Bereken lokale statistieken voor AF detectie
    # We gebruiken een verschuivend venster om de variabiliteit te bepalen
    for i in range(window_size, n):
        local_pp = pp_values[i-window_size:i]
        cv = np.std(local_pp) / np.mean(local_pp)
        
        # 1. Check voor AF (hoge variabiliteit over langere tijd)
        if cv > 0.20: # Drempelwaarde voor AF (15% variatie)
            labels[i] = 2
        
        # 2. Check voor PAC (individuele slag is prematuur)
        # Vergelijking met het gemiddelde van de vorige 5 slagen
        local_avg = np.mean(pp_values[i-5:i])
        if pp_values[i] < 0.80 * local_avg: # 30% vroeger dan gemiddeld
            labels[i] = 1
            
    return labels

# --- 1. RMSSD BEREKENING ---
def get_rhythm_classification(pp_intervals, window=20):
    # 1. Bereken opeenvolgende verschillen (successive differences)
    diffs = np.diff(pp_intervals)
    
    # 2. Rollende RMSSD berekening
    # We gebruiken een squared diff en dan een convolution voor het gemiddelde
    squared_diffs = diffs**2
    rolling_ms = np.convolve(squared_diffs, np.ones(window)/window, mode='same')
    rmssd = np.sqrt(rolling_ms)
    
    # 3. Classificatie logica
    labels = []
    for i in range(len(rmssd)):
        val = rmssd[i]
        curr_pp = pp_intervals[i]
        
        # Gemiddelde van de omgeving voor prematuriteit-check
        start = max(0, i-5)
        local_avg = np.mean(pp_intervals[start:i]) if i > 0 else curr_pp
        
        if val > 150: 
            # Continu hoge variabiliteit wijst op AF
            labels.append("AF")
        elif curr_pp < 0.75 * local_avg:
            # Een plotselinge korte slag wijst op een PAC
            labels.append("PAC")
        else:
            # Stabiel ritme
            labels.append("Sinus")
            
    return rmssd, labels

# Uitvoeren op jouw data
rmssd_values, rhythm_labels = get_rhythm_classification(pp_intervals)

# Voer classificatie uit
ritme_labels = classify_rhythm(pp_intervals)

# Mapping voor visualisatie
# Sinus = Groen, PAC = Oranje/Geel, AF = Rood
colors_ritme = np.array(['#2ECC71', '#F1C40F', '#E74C3C'])
current_colors = colors_ritme[ritme_labels]
# --- 3. PLOTTEN (Subplots) ---
#%%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18), sharex=True)

# Kleuren-thema
c_light = '#D1B9E1' # Lichtpaars
c_dark = '#9400D3'  # Donkerpaars


# --- Plot 1: P-P Interval ---
ax1.plot(p_times[1:], pp_intervals, color=c_light, linewidth=0.5, alpha=0.5, label='P-P (per slag)')
ax1.fill_between(p_times[1:], 0, pp_intervals, color=c_light, alpha=0.3)
ax1.plot(p_times[1:], trend_pp, color=c_dark, linewidth=2, label='Trend (50 slagen)')
ax1.set_ylabel('P-P Interval (ms)')
ax1.set_ylim(400, 1500) # Fysiologisch bereik voor betere details
ax1.set_title('Atriale Analyse: Ritmiek en Morfologie')
ax1.legend(loc='upper right')
# --- Update Plot 1 (P-P Interval met Classificatie) ---
# ax2.plot(p_times[1:], pp_intervals, color=c_light, linewidth=0.5, alpha=0.5)

# # Voeg scatterpunten toe voor de classificatie
# ax2.scatter(p_times[1:], pp_intervals, c=current_colors, s=20, zorder=3, label='Gedetecteerd Ritme')

# # Legenda handmatig verduidelijken
# from matplotlib.lines import Line2D
# legend_elements = [
#     Line2D([0], [0], color='#2ECC71', marker='o', linestyle='None', label='Sinus'),
#     Line2D([0], [0], color='#F1C40F', marker='o', linestyle='None', label='PAC'),
#     Line2D([0], [0], color='#E74C3C', marker='o', linestyle='None', label='AF/Onregelmatig'),
#     Line2D([0], [0], color=c_dark, label='Trend')
# ]
# ax2.legend(handles=legend_elements, loc='upper right')

# Voeg een extra subplot toe voor RMSSD
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

# ... (jouw bestaande ax1 en ax2 code) ...
# --- Stap 1: Definieer de drempelwaarden ---
af_threshold = 130    # RMSSD boven 130ms = waarschijnlijk AF
pac_threshold = 0.85  # P-P interval < 85% van gemiddelde = PAC

# --- Stap 2: Loop door de data en bepaal de kleur per segment ---
# We maken 'zones' voor de achtergrond
zones = [] 

for i in range(1, len(p_times)-1):
    curr_pp = pp_intervals[i-1]
    curr_rmssd = rmssd_values[i-1]
    
    # Bepaal kleur
    if curr_rmssd > af_threshold:
        color = 'red'      # AF
        label = 'AF'
    elif curr_pp < pac_threshold * np.mean(pp_intervals[max(0, i-10):i]):
        color = 'orange'   # PAC
        label = 'PAC'
    else:
        color = 'green'    # Sinus
        label = 'Sinus'
    
    zones.append((p_times[i], p_times[i+1], color))

ax2.plot(p_times[2:], rmssd_values, color='purple', linewidth=1)
ax2.set_ylim(0, 300)  # Beperk de as tot fysiologische waarden (0-300ms)
ax2.axhline(y=100, color='red', linestyle='--', label='AF grens')
ax2.set_ylabel('RMSSD (ms)')
ax2.set_title('Variabiliteit Analyse (RMSSD)')# --- Stap 3: Teken de zones in de plot (ax2) ---
for start, end, kol in zones:
    ax2.axvspan(start, end, color=kol, alpha=0.1, lw=0)

# Handmatige legenda voor de kleuren
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='green', alpha=1, lw=4),
                Line2D([0], [0], color='orange', alpha=1, lw=4),
                Line2D([0], [0], color='red', alpha=0.3, lw=4)]
ax2.legend(custom_lines, ['Sinus', 'PAC', 'AF'], loc='upper right')
# Plot RMSSD op ax3
# In je plot sectie voor RMSSD:




# --- Plot 2: P-duur (Precies zoals je voorbeeld) ---
ax3.plot(t_shifted, duur_filt, color=c_light, linewidth=0.5, alpha=0.9, label='P-duur (per slag)')
ax3.fill_between(t_shifted, 40, duur_filt, color=c_light, alpha=0.3)
ax3.plot(t_shifted, trend_duur, color=c_dark, linewidth=2, label='Trend (50 slagen)')
ax3.axhline(y=120, color='red', linestyle='--', linewidth=1, label='120ms')
ax3.axhline(y=80, color='blue', linestyle='--', linewidth=1, label='80ms')
ax3.set_ylabel('P-duur (ms)')
ax3.set_ylim(40, 180)
ax3.legend(loc='upper right')

# --- Plot 3: P-polariteit ---
# Scatter plot met kleurcodering voor polariteit
fig = plt.figure(figsize=(15, 4))
colors = ['#E74C3C' if p == -1 else '#3498DB' for p in pol_filt]
plt.scatter(t_shifted, pol_filt, c=colors, s=15, alpha=0.6)
# plt.set_yticks([1, -1])
plt.xlim(20000, 20050)
plt.yticks([1, -1], ['Positief (+)', 'Negatief (-)'])
plt.ylabel('Polariteit')
plt.xlabel('Gecorrigeerde Tijd (s)')
plt.ylim(-1.5, 1.5)

# Algemene styling
for ax in [ax1, ax2, ax3]:
    ax.grid(True, linestyle='-', alpha=0.2)

plt.tight_layout()
plt.show()
# %%
