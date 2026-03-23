#%%
import numpy as np
import matplotlib.pyplot as plt
from Stap1_data_loader import laad_ecg_bestand #in dit bestand staat een functie die automatisch de data map vindt en het bestand inlaadt, je hoeft alleen de naam van het bestand aan te passen als je een ander bestand wilt inladen (dus het bestand met alleen de PACs bv)
from Stap3_Ventriculaire_activiteit import ecg_PT_both
from scipy import signal
#%%
# Inladen van de data
ecg, fs, t = laad_ecg_bestand("004_Groenewoud_PACs+PVCs.mat")
ecgmai, squared, band_passed, deriv_filtered = ecg_PT_both(ecg, fs)
locs, prop = signal.find_peaks(deriv_filtered, height=(500, 1400), distance=int(.3*fs))
#%%
def plot_verbrede_qrs_complexen_slim(ecg_signaal, onsets, offsets, duren, fs, start_sec=0, eind_sec=10):
    """
    Knipt eerst het signaal af op de gewenste tijd, en plot pas daarna. 
    Dit voorkomt dat je werkgeheugen crasht bij miljoenen datapunten.
    """
    drempel_ms = 120.0
    
    # 1. Reken seconden om naar de juiste indexen in je data
    start_sample = int(start_sec * fs)
    eind_sample = int(eind_sec * fs)
    
    # Zorg dat we niet voorbij het einde van het bestand kunnen knippen
    eind_sample = min(eind_sample, len(ecg_signaal))
    
    # 2. Knip de data af
    tijd_sec_stukje = np.arange(start_sample, eind_sample) / fs
    ecg_stukje = ecg_signaal[start_sample:eind_sample]
    
    # Vind alle complexe slagen
    lange_qrs_indexen = np.where(duren > drempel_ms)[0]
    
    # 3. Start de plot
    plt.figure(figsize=(15, 6))
    plt.plot(tijd_sec_stukje, ecg_stukje, color='blue', alpha=0.6, label='ECG Signaal')
    
    eerste_arceer = True 
    for idx in lange_qrs_indexen:
        onset_sample = onsets[idx]
        offset_sample = offsets[idx]
        
        # 4. CRUCIAAL: Teken het rode vlakje ALLEEN als hij in ons 10-seconden raam valt
        if onset_sample >= start_sample and offset_sample <= eind_sample:
            tijd_start = onset_sample / fs
            tijd_eind = offset_sample / fs
            
            label_naam = 'Verbreed QRS (>120ms)' if eerste_arceer else ""
            plt.axvspan(tijd_start, tijd_eind, color='red', alpha=0.3, label=label_naam)
            eerste_arceer = False 
            
            R_piek_sample = int((onset_sample + offset_sample) / 2)
            # Y-hoogte voor de tekst afstemmen op de max van dít specifieke stukje
            plt.text(R_piek_sample / fs, np.max(ecg_stukje)*0.8, f'{duren[idx]:.0f}ms', 
                     color='red', fontsize=10, fontweight='bold', ha='center')

    plt.title(f'ECG: Verbrede QRS-complexen (> {drempel_ms}ms) | Tijd: {start_sec}s tot {eind_sec}s')
    plt.xlabel('Tijd (seconden)')
    plt.ylabel('Amplitude')
    plt.xlim(start_sec, eind_sec)
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)


# ==============================================================================
# HOOFDSCRIPT (Zoekt automatisch de eerste verbrede slag op!)
# ==============================================================================

# A: Bereken de statistieken
onsets, offsets, duren = bereken_qrs_duur_verbeterd(squared, locs, fs, drempel_fractie=0.005)

# B: Vind de indexen van de verbrede slagen
lange_qrs_indexen = np.where(duren > 120.0)[0]

aantal_totaal = len(duren)
aantal_verbreed = len(lange_qrs_indexen)

print("-" * 40)
print("QRS ANALYSE RESULTATEN:")
print(f"Totaal aantal slagen:        {aantal_totaal}")
print(f"Aantal verbrede complexen:   {aantal_verbreed}")
print("-" * 40)

# C: Bepaal waar we gaan plotten
if aantal_verbreed > 0:
    # Pak de allereerste verbrede slag die het algoritme heeft gevonden
    eerste_lange_idx = lange_qrs_indexen[0]
    
    # Bereken op welke seconde deze slag valt
    tijd_eerste_lange_sec = onsets[eerste_lange_idx] / fs
    print(f"\n--> Succes! De eerste verbrede slag valt rond seconde {tijd_eerste_lange_sec:.1f}.")
    print("Ik teken nu een grafiek van 5 seconden rondom deze specifieke slag...")
    
    # Maak een window van ~2 seconden vóór de slag, tot 3 seconden erna
    plot_start = max(0, tijd_eerste_lange_sec - 2)
    plot_eind = plot_start + 60
    
    # Run de plot functie voor dit specifieke tijdvak
    plot_verbrede_qrs_complexen_slim(band_passed, onsets, offsets, duren, fs, 
                                     start_sec=plot_start, eind_sec=plot_eind)
    plt.show()

else:
    print("\nEr zijn in het hele signaal geen QRS complexen > 120ms gevonden.")
    print("Misschien is de drempelwaarde (drempel_fractie=0.05) te streng, of is het een compleet gezond ECG.")
# %%
