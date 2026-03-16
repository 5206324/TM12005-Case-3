#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
import datetime
import matplotlib.dates as mdates
import pandas as pd
from Stap1_data_loader import laad_ecg_bestand #in dit bestand staat een functie die automatisch de data map vindt en het bestand inlaadt, je hoeft alleen de naam van het bestand aan te passen als je een ander bestand wilt inladen (dus het bestand met alleen de PACs bv)


# --- CONFIG & DATA LOAD ---
# leads = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
# path_both = r"/Users/fenne/Documents/Technical Medicine/TM12005 Advanced Signal Processing/Case 3/TM12005-Case-3/TM12005 Advanced Signal Processing (202526 Q3) - 322026 - 159 PM 2/004_Groenewoud_PACs+PVCs.mat"

# def read_ecg_both(path_both):
#     data_both = loadmat(path_both, squeeze_me=True, struct_as_record=False)
#     ecg_both = data_both['ecg'].sig[:,leads.index('II')]
#     fs = data_both['ecg'].header.Sampling_Rate
#     t0 = datetime.datetime(*data_both['ecg'].start_vec)
#     nSamples = len(ecg_both)
#     t = pd.date_range(start=t0, periods=nSamples, freq=pd.Timedelta(seconds=1/fs))
#     return ecg_both, fs, t


# %%
#hier kan je de naam van het bestand aanpassen als je een ander bestand wilt inladen (dus het bestand met alleen de PACs bv)
ecg, fs, t = laad_ecg_bestand("004_Groenewoud_PACs+PVCs.mat", plotresult=True)


def ecg_PT_both(ecg, fs):
    b, a = signal.butter(2, [5, 15], btype="band", fs=fs)
    band_passed = signal.filtfilt(b, a, ecg)
    T = 1 / fs
    deriv_filtered = signal.lfilter([-1, -2, 0, 2, 1], 8 * T, band_passed)
    squared = deriv_filtered**2
    N = int(0.150 * fs)
    ecgmai = signal.lfilter(np.ones(N)/N, 1, squared)
    return ecgmai
#%%
# --- EXECUTION ---
# ecg_raw, fs, t_raw = read_ecg_both(path_both)
ecgmai = ecg_PT_both(ecg, fs)

# Peak detection (Ventriculaire activiteit)
locs, _ = signal.find_peaks(ecgmai, height=1e6, distance=int(0.3*fs))

# Bereken RR-intervallen
rr_intervals_sec = np.diff(locs) / fs
t_peaks = t[locs[1:]]

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
ax1.plot(np.arange(zoom_samples)/fs, ecg[:zoom_samples])
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
print(f"Gemiddelde ventriculaire frequentie voor slicen en outlier verwijdering: {mean_hr:.1f} BPM")
print(f"Aantal gedetecteerde ventriculaire activaties voor slicen en outlier verwijdering: {len(locs)}")
# %%
# --- ZOEK EEN SCHOON SEGMENT VAN 10 SECONDEN ---

# We zoeken vanaf een startpunt naar het eerste stukje 'continue' data
search_start_time = 60  # Start met zoeken vanaf 60 seconden
window_size = 100        # We willen 100 seconden zien

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
    
    # --- NIEUW: Verschuif de start met 2 seconden om het initiële artefact over te slaan ---
    zoom_start = found_start_idx + int(2 * fs) 
    zoom_end = zoom_start + int(window_size * fs)

    # Slice de data
    ecg_segment = ecg[zoom_start:zoom_end]
    t_segment = t[zoom_start:zoom_end]

    # Slice de data
    ecg_segment = ecg[zoom_start:zoom_end]
    t_segment = t[zoom_start:zoom_end]
    
    # Vind pieken in dit segment
    segment_locs = locs[(locs >= zoom_start) & (locs < zoom_end)]
    t_peaks_rel = (segment_locs - zoom_start) / fs
    
    # RR-intervallen berekenen
    segment_rr = np.diff(t_peaks_rel)
    t_rr_rel = t_peaks_rel[1:]

    # --- PLOTTEN ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Paneel A: ECG met sterretjes op de R-toppen
  # Paneel A: ECG met sterretjes op de R-toppen
    ax1.plot(np.linspace(0, window_size, len(ecg_segment)), ecg_segment, color='tab:blue')
   # --- NIEUW: Corrigeer filtervertraging door de echte R-top te zoeken ---
    zoek_window = int(0.1 * fs) # Zoek in een window van 100ms rondom de gevonden piek
    echte_x_waardes = []
    echte_y_waardes = []

    for rel_idx in (segment_locs - zoom_start):
        # Bepaal zoekgrenzen binnen het segment
        start_idx = max(0, rel_idx - zoek_window)
        end_idx = min(len(ecg_segment), rel_idx + zoek_window)
        
        # Vind de index van de allerhoogste waarde in dit kleine stukje
        lokale_piek_rel_idx = start_idx + np.argmax(ecg_segment[start_idx:end_idx])
        
        echte_x_waardes.append(lokale_piek_rel_idx / fs)
        echte_y_waardes.append(ecg_segment[lokale_piek_rel_idx])

    # Plot de gecorrigeerde sterretjes
    ax1.plot(echte_x_waardes, echte_y_waardes, 'r*', label='Gecorrigeerde R-top')
    ax1.set_title(f"Ingezoomd ECG (Start op {zoom_start/fs:.1f}s - Geen gaten gevonden)")
    
    # --- NIEUW: Verwijder de extreme outlier uit de weergave ---
    # We berekenen het 1e en 99e percentiel van het signaal om de QRS toppen wel te zien, 
    # maar de gigantische uitschieters buiten beeld te laten.
    p1, p99 = np.percentile(ecg_segment, [1, 99])
    marge = (p99 - p1) * 1.5  # Extra marge boven en onder
    ax1.set_ylim(p1 - marge, p99 + marge)
    
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
# --- STATISTIEKEN VAN HET SEGMENT ---
gemiddelde_hr_segment = 60 / np.mean(segment_rr)
aantal_pieken_segment = len(segment_locs)
    
print("\n--- Resultaten voor het schone 100s segment ---")
print(f"Aantal ventriculaire activaties in segment: {aantal_pieken_segment}")
print(f"Gemiddelde hartslag in segment: {gemiddelde_hr_segment:.1f} BPM")

# %%
# --- RR VARIABILITEIT OVER TIJD (Gemiddelde RR per minuut) ---

# 1. Filter extreme waarden eruit VOORDAT we gemiddelden berekenen
# We gebruiken alleen RR-intervallen tussen 0.1s (600 BPM) en 8.0s (75 BPM)
fysiologisch_mask = (clean_rr >= 0.1) & (clean_rr <= 8.0)
betrouwbare_rr = clean_rr[fysiologisch_mask]
betrouwbare_t = clean_t[fysiologisch_mask]

# 2. Zet de schone data in een Pandas DataFrame
df_rr = pd.DataFrame({'RR_interval': betrouwbare_rr}, index=betrouwbare_t)

# 3. Bereken het gemiddelde per minuut. 
# LET OP: We gebruiken expres GEEN .dropna() meer!
# De lege uren krijgen nu de waarde NaN, waardoor de lijn in de plot netjes breekt.
df_rr_per_minuut = df_rr.resample('1Min').mean()

# 4. Maak de plot
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_rr_per_minuut.index, df_rr_per_minuut['RR_interval'], 
        marker='o', linestyle='-', markersize=3, color='tab:purple')

ax.set_title("Gemiddeld RR-interval per minuut (Gefilterd)")
ax.set_ylabel("Gemiddeld RR (sec)")
ax.set_xlabel("Tijdstip in opname")

# Zet de y-as vast op een logische fysiologische range zodat we goed detail zien
ax.set_ylim(0.5, 1.5) 

# Zorg dat de x-as netjes uren en minuten toont
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.autofmt_xdate(rotation=45)

ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()


# %%
# --- RR VARIABILITEIT EN HARTSLAG OVER TIJD (Gemiddelde + Variatie per minuut) ---

# 1. Filter extreme waarden eruit (fysiologische grenzen)
fysiologisch_mask = (clean_rr >= 0.4) & (clean_rr <= 2.0)
betrouwbare_rr = clean_rr[fysiologisch_mask]
betrouwbare_t = clean_t[fysiologisch_mask]

# 2. Zet in een DataFrame en bereken direct de hartslag (BPM) voor elke slag
df = pd.DataFrame({
    'RR_interval': betrouwbare_rr,
    'HR_bpm': 60 / betrouwbare_rr
}, index=betrouwbare_t)

# 3. Bereken het gemiddelde EN de variatie (standaarddeviatie) per minuut
df_per_minuut = df.resample('1Min').agg(['mean', 'std'])

# Haal de specifieke lijnen uit onze nieuwe tabel voor overzichtelijkheid
rr_mean = df_per_minuut['RR_interval']['mean']
rr_std  = df_per_minuut['RR_interval']['std']

hr_mean = df_per_minuut['HR_bpm']['mean']
hr_std  = df_per_minuut['HR_bpm']['std']

# 4. Maak een plot met twee panelen
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --- Bovenste plot: Gemiddeld RR-interval + Variatie ---
ax1.plot(rr_mean.index, rr_mean, 
         marker='o', linestyle='-', markersize=3, color='tab:purple', label='Gemiddelde')

# Teken een halfdoorzichtige schaduw voor de variatie (+ en - 1 standaarddeviatie)
ax1.fill_between(rr_mean.index, 
                 rr_mean - rr_std, 
                 rr_mean + rr_std, 
                 color='tab:purple', alpha=0.3, label='Variatie (±1 SD)')

ax1.set_title("RR-interval per minuut (met variatie binnen de minuut)")
ax1.set_ylabel("RR (sec)")
ax1.set_ylim(0.3, 1.7) # Iets bredere range gelaten voor de schaduwband
ax1.grid(True, alpha=0.4)
ax1.legend(loc='upper right')

# --- Onderste plot: Gemiddelde Hartslag + Variatie ---
ax2.plot(hr_mean.index, hr_mean, 
         marker='o', linestyle='-', markersize=3, color='tab:red', label='Gemiddelde')

# Teken ook de schaduw voor hartslag variatie
ax2.fill_between(hr_mean.index, 
                 hr_mean - hr_std, 
                 hr_mean + hr_std, 
                 color='tab:red', alpha=0.3, label='Variatie (±1 SD)')

ax2.set_title("Hartslag per minuut (met variatie binnen de minuut)")
ax2.set_ylabel("Hartslag (BPM)")
ax2.set_xlabel("Tijdstip in opname")
ax2.set_ylim(30, 130) # Iets bredere range voor de schaduwband
ax2.grid(True, alpha=0.4)
ax2.legend(loc='upper right')

# Zorg dat de x-as netjes uren en minuten toont
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.autofmt_xdate(rotation=45)

plt.tight_layout()
plt.show()
# %%
# --- ALGEMENE STATISTIEKEN VAN DE OPNAME ---

# 'betrouwbare_rr' bevat al géén gaten (want we filterden eerder alles > 60s weg)
# en géén outliers (want we filterden tussen 0.4s en 2.0s).

gemiddelde_rr_totaal = np.mean(betrouwbare_rr)
gemiddelde_hr_totaal = 60 / gemiddelde_rr_totaal

print("\n" + "="*40)
print("--- STATISTIEKEN HELE METING (Opgeschoond) ---")
print(f"Totaal aantal bruikbare slagen: {len(betrouwbare_rr) + 1}")
print(f"Gemiddelde hartslag over de hele meting: {gemiddelde_hr_totaal:.1f} BPM")
print("="*40 + "\n")
# %%
