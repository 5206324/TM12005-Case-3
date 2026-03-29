#%%
import numpy as np
import pandas as pd  # <-- NIEUW!
import matplotlib.pyplot as plt
from scipy import signal

from Stap1_data_loader import laad_ecg_bestand 
from Stap3_Ventriculaire_activiteit import ecg_PT_both

# ==============================================================================
# 1. FUNCTIES DEFINIËREN
# ==============================================================================

def bereken_qrs_duur_verbeterd(squared_signaal, locs, fs, drempel_fractie=0.10):
    onsets = []
    offsets = []
    duren = []
    
    max_zoek_samples = int(0.25 * fs) 
    geduld_max_samples = int(0.02 * fs) 
    
    for piek_idx in locs:
        raampje = int(0.05 * fs) 
        start_raam = max(0, piek_idx - raampje)
        eind_raam = min(len(squared_signaal), piek_idx + raampje)
        
        echte_piek_idx = start_raam + np.argmax(squared_signaal[start_raam:eind_raam])
        
        piek_hoogte = squared_signaal[echte_piek_idx]
        drempel = piek_hoogte * drempel_fractie
        
        # --- Zoek ONSET ---
        onset_idx = echte_piek_idx
        geduld_teller = 0
        zoek_limiet_links = max(0, echte_piek_idx - max_zoek_samples)
        
        while onset_idx > zoek_limiet_links:
            if squared_signaal[onset_idx] > drempel:
                geduld_teller = 0 
            else:
                geduld_teller += 1 
                
            if geduld_teller > geduld_max_samples:
                onset_idx += geduld_max_samples 
                break
            onset_idx -= 1
            
        # --- Zoek OFFSET ---
        offset_idx = echte_piek_idx
        geduld_teller = 0
        zoek_limiet_rechts = min(len(squared_signaal) - 1, echte_piek_idx + max_zoek_samples)
        
        while offset_idx < zoek_limiet_rechts:
            if squared_signaal[offset_idx] > drempel:
                geduld_teller = 0 
            else:
                geduld_teller += 1 
                
            if geduld_teller > geduld_max_samples:
                offset_idx -= geduld_max_samples 
                break
            offset_idx += 1
            
        duur_ms = ((offset_idx - onset_idx) / fs) * 1000.0
        
        onsets.append(onset_idx)
        offsets.append(offset_idx)
        duren.append(duur_ms)
        
    return np.array(onsets), np.array(offsets), np.array(duren)

# ==============================================================================
# NIEUW: Functie om overlappende PVC's aan elkaar te lijmen
# ==============================================================================
def voeg_overlappende_pvcs_samen(onsets, offsets, lange_qrs_indexen, fs, marge_ms=50):
    """
    Kijkt naar alle gevonden brede slagen. Als twee slagen overlappen of heel 
    dicht bij elkaar liggen, worden ze samengevoegd tot 1 groot event.
    """
    if len(lange_qrs_indexen) == 0:
        return []

    # Haal alle onsets en offsets op van de verbrede slagen
    brede_events = [(onsets[idx], offsets[idx]) for idx in lange_qrs_indexen]
    # Sorteer ze netjes op tijd
    brede_events.sort(key=lambda x: x[0])
    
    samengevoegd = []
    huidige_onset, huidige_offset = brede_events[0]
    marge_samples = int((marge_ms / 1000.0) * fs)
    
    for volgende_onset, volgende_offset in brede_events[1:]:
        # Als de volgende slag begint vóórdat de huidige is afgelopen (+ een kleine marge)
        if volgende_onset <= (huidige_offset + marge_samples):
            # Rek de offset op tot het einde van de volgende slag
            huidige_offset = max(huidige_offset, volgende_offset)
        else:
            # Ze overlappen niet, dus sla de huidige op en begin een nieuwe
            samengevoegd.append((huidige_onset, huidige_offset))
            huidige_onset = volgende_onset
            huidige_offset = volgende_offset
            
    # Vergeet de allerlaatste niet op te slaan!
    samengevoegd.append((huidige_onset, huidige_offset))
    
    return samengevoegd

# ==============================================================================
# PLOT FUNCTIE (Aangepast voor de samengevoegde PVC's)
# ==============================================================================
def plot_verbrede_qrs_complexen_slim(ecg_signaal, samengevoegde_pvcs, fs, drempel_ms, start_sec=0, eind_sec=10):
    start_sample = int(start_sec * fs)
    eind_sample = int(eind_sec * fs)
    eind_sample = min(eind_sample, len(ecg_signaal))
    
    tijd_sec_stukje = np.arange(start_sample, eind_sample) / fs
    ecg_stukje = ecg_signaal[start_sample:eind_sample]
    
    plt.figure(figsize=(15, 6))
    plt.plot(tijd_sec_stukje, ecg_stukje, color='blue', alpha=0.6, label='ECG Signaal')
    
    eerste_arceer = True 
    for onset_sample, offset_sample in samengevoegde_pvcs:
        # Check of hij in onze plot valt
        if onset_sample >= start_sample and offset_sample <= eind_sample:
            tijd_start = onset_sample / fs
            tijd_eind = offset_sample / fs
            totale_duur_ms = (tijd_eind - tijd_start) * 1000.0
            
            label_naam = f'Definitieve PVC (>{drempel_ms}ms)' if eerste_arceer else ""
            plt.axvspan(tijd_start, tijd_eind, color='red', alpha=0.3, label=label_naam)
            eerste_arceer = False 
            
            midden_tijd = (tijd_start + tijd_eind) / 2
            plt.text(midden_tijd, np.max(ecg_stukje)*0.8, f'{totale_duur_ms:.0f}ms', 
                     color='red', fontsize=10, fontweight='bold', ha='center')

    plt.title(f'ECG: Definitieve PVC-detectie | Tijd: {start_sec}s tot {eind_sec}s')
    plt.xlabel('Tijd (seconden)')
    plt.ylabel('Amplitude')
    plt.xlim(start_sec, eind_sec)
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)


# ==============================================================================
# 2. HOOFDSCRIPT 
# ==============================================================================

bestandsnaam = "004_Groenewoud_PACs+PVCs.mat"
grens_verbreed_ms = 160.0

# A. Data inladen
ecg, fs, t = laad_ecg_bestand(bestandsnaam)
ecgmai, squared, band_passed, deriv_filtered = ecg_PT_both(ecg, fs)
locs, prop = signal.find_peaks(deriv_filtered, height=(500, 1400), distance=int(.3*fs))

# B. Basis statistieken berekenen
onsets, offsets, duren = bereken_qrs_duur_verbeterd(squared, locs, fs, drempel_fractie=0.10)

# C. Vind alle slagen die over de grens heen gaan
lange_qrs_indexen = np.where(duren > grens_verbreed_ms)[0]

# D. Voeg overlappende / dubbel getelde pieken samen tot 1 solide PVC event!
definitieve_pvcs = voeg_overlappende_pvcs_samen(onsets, offsets, lange_qrs_indexen, fs, marge_ms=300)

print("-" * 40)
print("PVC ANALYSE RESULTATEN:")
print(f"Totaal aantal hartslagen gemeten: {len(locs)}")
print(f"Aantal definitieve PVC's gevonden: {len(definitieve_pvcs)}")
print("-" * 40)

# ==============================================================================
# NIEUW: 3. SLA OP IN EEN PANDAS DATAFRAME
# ==============================================================================
df_pvcs = pd.DataFrame(columns=["Bestand", "PVC_Starttijd_sec", "PVC_Eindtijd_sec", "Totale_Duur_ms"])

for onset, offset in definitieve_pvcs:
    start_s = onset / fs
    eind_s = offset / fs
    duur_msec = (offset - onset) / fs * 1000.0
    
    # Voeg een nieuwe rij toe aan de DataFrame
    df_pvcs.loc[len(df_pvcs)] = [bestandsnaam, start_s, eind_s, duur_msec]

# Print de eerste 5 rijen van je nieuwe tabel om te checken
print("\nDit is de resulterende DataFrame:")
print(df_pvcs.head())

# Optioneel: Sla de tabel op als Excel of CSV bestand
df_pvcs.to_csv('Gevonden_PVCs_Patient004.csv', index=False)
print("\n--> DataFrame is opgeslagen als 'Gevonden_PVCs_Patient004.csv'")

# ==============================================================================
# 4. PLOTTEN
# ==============================================================================
if len(definitieve_pvcs) > 0:
    eerste_onset, eerste_offset = definitieve_pvcs[0]
    tijd_eerste_pvc_sec = eerste_onset / fs
    
    plot_start = max(0, tijd_eerste_pvc_sec - 2)
    plot_eind = plot_start + 10
    
    plot_verbrede_qrs_complexen_slim(band_passed, definitieve_pvcs, fs, 
                                     drempel_ms=grens_verbreed_ms, 
                                     start_sec=plot_start, eind_sec=plot_eind)
    plt.show()
#%%
# Pak bijvoorbeeld de eerste 4 PVC's uit je nieuwe DataFrame
# --- STAP 1: Zorg dat de PVC's echt samengevoegd zijn ---
# We gebruiken een marge van 350ms om die dubbele pieken te vangen
definitieve_pvcs = voeg_overlappende_pvcs_samen(onsets, offsets, lange_qrs_indexen, fs, marge_ms=350)

# --- STAP 2: Update de DataFrame met de samengevoegde data ---
pvc_data = []
for onset, offset in definitieve_pvcs:
    pvc_data.append({
        "Bestand": bestandsnaam,
        "Start_sec": onset / fs,
        "Eind_sec": offset / fs,
        "Duur_ms": ((offset - onset) / fs) * 1000.0
    })
df_pvcs = pd.DataFrame(pvc_data)

# ==============================================================================
# 4. VISUALISATIE VAN DE CONTEXT (20 SECONDEN WINDOW)
# ==============================================================================

# We pakken de eerste 4 gevonden PVC's uit de tabel
aantal_om_te_tonen = min(10, len(df_pvcs))

if aantal_om_te_tonen > 0:
    # Maak de figure aan
    fig, axes = plt.subplots(aantal_om_te_tonen, 1, figsize=(15, 4 * aantal_om_te_tonen))
    
    # Forceer naar lijst als er maar 1 resultaat is
    if aantal_om_te_tonen == 1: axes = [axes]
    
    t_vector = np.arange(len(band_passed)) / fs
    
    for i in range(aantal_om_te_tonen):
        row = df_pvcs.iloc[i]
        t_start = row['Start_sec']
        t_eind = row['Eind_sec']
        
        # Bepaal het midden en zet het venster op 20 seconden (10s links, 10s rechts)
        midden = (t_start + t_eind) / 2
        window_start = max(0, midden - 10.0)
        window_eind = min(t_vector[-1], midden + 10.0)
        
        ax = axes[i]
        
        # Alleen de data binnen het venster plotten voor snelheid en geheugen
        mask = (t_vector >= window_start) & (t_vector <= window_eind)
        ax.plot(t_vector[mask], band_passed[mask], color='blue', alpha=0.7, lw=0.8, label='Gefilterd ECG')
        
        # Teken de rode balk over de gehele gedetecteerde PVC
        ax.axvspan(t_start, t_eind, color='red', alpha=0.3, label=f"PVC ({row['Duur_ms']:.0f}ms)")
        
        # Extra opmaak
        ax.set_title(f"PVC #{i+1} op {t_start:.2f}s | Context: 20 seconden")
        ax.set_xlim(window_start, window_eind)
        ax.set_xlabel("Tijd (seconden)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
else:
    print("Geen PVC's gevonden met de huidige instellingen (>160ms).")
# %%
