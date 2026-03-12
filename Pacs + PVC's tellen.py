#%%
from read_telemetry_ecg import ecgmai
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
from scipy.fft import rfft, rfftfreq
import datetime
import matplotlib.dates as mdates
import pandas as pd
from read_telemetry_ecg import ecg_PT, t, fs,ecg, locs

# %%
ecgmai, band_passed, deriv_filtered, squared, fs, t = ecg_PT(ecg, fs, t)
lim = 3000


# %%
import numpy as np

# Zorg dat je 'locs' (je gevonden pieken uit find_peaks), 'ecgmai' en 'fs' hebt ingeladen.
normale_slagen = []
pac_slagen = []
pvc_slagen = []

window_size = 5  # We kijken steeds naar de 5 voorgaande slagen als 'baseline'

# We beginnen bij index 'window_size' zodat we een historie hebben
for i in range(window_size, len(locs)):
    
    # --- 1. RITME (Timing) ---
    # Bereken het huidige RR-interval in seconden
    rr_huidig = (locs[i] - locs[i-1]) / fs
    
    # Bereken de baseline van de afgelopen 5 RR-intervallen
    # We gebruiken de mediaan zodat eerdere afwijkingen het gemiddelde niet verpesten
    rr_historie = np.diff(locs[i-window_size : i]) / fs
    rr_baseline = np.median(rr_historie)
    
    # --- 2. VORM (Morfologie via ecgmai) ---
    # Wat is de amplitude van deze specifieke piek in je moving average signaal?
    hoogte_huidig = ecgmai[locs[i]]
    
    # Bereken de baseline hoogte van de afgelopen 5 pieken
    hoogte_historie = [ecgmai[locs[j]] for j in range(i-window_size, i)]
    hoogte_baseline = np.median(hoogte_historie)
    
    # --- 3. CLASSIFICATIE LOGICA ---
    # Is de slag prematuur? (Hier: korter dan 80% van wat normaal is)
    if rr_huidig < 0.80 * rr_baseline or rr_huidig > 1.15 * rr_baseline:
        
        # Is de piek ook significant 'dikker' of hoger? (Hier: 50% groter dan normaal)
        # Omdat ecgmai de oppervlakte meet, schiet deze bij een brede PVC flink omhoog.
        if hoogte_huidig > 1.50 * hoogte_baseline:
            pvc_slagen.append(locs[i])  # Te vroeg EN afwijkende vorm -> PVC
        else:
            pac_slagen.append(locs[i])  # Wel te vroeg, maar normale vorm -> PAC
            
    else:
        normale_slagen.append(locs[i])  # Niet te vroeg -> Normale slag

# Print het resultaat
print(f"Resultaat: {len(normale_slagen)} Normaal, {len(pac_slagen)} PAC's, {len(pvc_slagen)} PVC's")
# %%
