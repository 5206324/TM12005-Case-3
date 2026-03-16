# Dit is het bestand: data_loader.py

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.io import loadmat
from pathlib import Path

def laad_ecg_bestand(bestandsnaam, plotresult=False):
    """
    Zoekt automatisch de datamap op (Mac/Windows compatible) en 
    laadt het gespecificeerde ECG .mat bestand in.
    """
    # 1. Vind de projectmap
    huidige_map = Path.cwd()
    
    # 2. Construeer het pad naar de map waar al jullie data in staat
    # (Pas de naam van deze map aan als jullie hem ooit hernoemen)
    data_map = huidige_map / "TM12005 Advanced Signal Processing (202526 Q3) - 322026 - 159 PM 2"
    
    # 3. Plak de specifieke bestandsnaam erachter
    data_path = data_map / bestandsnaam
    
    # Foutafhandeling voor als je een typfout maakt in de bestandsnaam
    if not data_path.exists():
        raise FileNotFoundError(f"Bestand niet gevonden! Ik zocht hier: {data_path}")

    # 4. Data inladen
    data = loadmat(data_path, squeeze_me=True, struct_as_record=False)
    
    # We pakken standaard Lead II
    leads = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
    ecg = data['ecg'].sig[:, leads.index('II')]

    fs = data['ecg'].header.Sampling_Rate
    t0 = datetime.datetime(*data['ecg'].start_vec)

    # Tijds-as aanmaken
    nSamples = data['ecg'].sig.shape[0]
    t = pd.date_range(
        start=t0,
        periods=nSamples,
        freq=pd.Timedelta(seconds=1/fs)
    )
        
    # Optioneel plotten
    if plotresult:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(t, ecg)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ECG (mV)")
        ax.set_title(f"Raw ECG Signal - {bestandsnaam}")
        plt.show()
    
    # Geef de bruikbare data terug aan de gebruiker
    return ecg, fs, t