#%% === Inladen data ===
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# --- PARAMETER FUNCTIE ---
def bereken_rolling_rmssd(rr_intervals, window_size=20):
    rmssd_values = []
    for i in range(len(rr_intervals) - window_size):
        window = rr_intervals[i : i + window_size]
        diff_sq = np.diff(window)**2
        rmssd_values.append(np.sqrt(np.mean(diff_sq)))
    return np.array(rmssd_values)

# --- BEREKENING ---
window_n = 20
rmssd_trend = bereken_rolling_rmssd(clean_rr, window_size=window_n)

# 6 uur gap verwijderen
x_index = np.arange(len(rmssd_trend))

# De bijbehorende tijdstippen voor de labels
t_trend = clean_t[window_n:]
t_labels = t_trend
#%%
def analyseer_episodes(status_lijst):
    pacs = 0
    af_runs = 0
    in_af_run = False
    
    # We tellen hoe vaak de status verspringt
    for s in status_lijst:
        if s == 1: # Losse PAC
            pacs += 1
        elif s == 2: # AF
            if not in_af_run:
                af_runs += 1
                in_af_run = True
        else: # Normaal
            in_af_run = False
            
    return pacs, af_runs

#%% --- PLOTTING RMS---
fig, ax = plt.subplots(figsize=(12, 6))

# Plot tegen de index, niet tegen de tijdwaarde
ax.plot(x_index, rmssd_trend, color='firebrick', linewidth=0.8, label='RMSSD (Onregelmatigheid)')

# Referentielijn
ax.axhline(y=0.1, color='black', linestyle='--', alpha=0.6, label='Drempelwaarde AF')

# Nu "faken" we de x-as labels:
# We kiezen bv. 10 plekken op de as om een tijdstip te laten zien
num_ticks = 10
tick_indices = np.linspace(0, len(x_index) - 1, num_ticks, dtype=int)
ax.set_xticks(tick_indices)
ax.set_xticklabels([t_labels[i].strftime('%H:%M:%S') for i in tick_indices], rotation=45)

ax.set_title("Ventriculaire Onregelmatigheid (Tijd-gap verwijderd)")
ax.set_ylabel("RMSSD (seconden)")
ax.set_xlabel("Tijdstip (gecomprimeerde as)")
ax.set_ylim(0, 0.4)
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
# %% --- Interval visualisatie ---

start_tijd_str = '14:00:00' 
half_uur_later = (pd.to_datetime(start_tijd_str) + pd.Timedelta(minutes=60)).strftime('%H:%M:%S')

# --- FILTER ---
mask_30min = (clean_t.strftime('%H:%M:%S') >= start_tijd_str) & \
             (clean_t.strftime('%H:%M:%S') <= half_uur_later)

segment_t = clean_t[mask_30min]
segment_rr = clean_rr[mask_30min]

# Bijbehorende RMSSD waarden
mask_rmssd = (t_trend.strftime('%H:%M:%S') >= start_tijd_str) & \
              (t_trend.strftime('%H:%M:%S') <= half_uur_later)
segment_rmssd = rmssd_trend[mask_rmssd]
t_rmssd_seg = t_trend[mask_rmssd]

# --- PLOTTEN ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# RR-intervallen
ax1.plot(segment_t, segment_rr, 'o-', color='tab:blue', markersize=3, linewidth=0.5, alpha=0.7)
ax1.set_title(f"Ingezoomd: RR-intervallen tussen {start_tijd_str} en {half_uur_later}")
ax1.set_ylabel("RR-interval (s)")
ax1.set_ylim(0.3, 1.5)
ax1.grid(alpha=0.3)

# De berekende Regulariteit (RMSSD)
ax2.plot(t_rmssd_seg, segment_rmssd, color='firebrick', linewidth=1)
ax2.axhline(y=0.1, color='black', linestyle='--', label='AF Drempelwaarde')
ax2.set_title("Regulariteit (RMSSD) in dit interval")
ax2.set_ylabel("RMSSD (s)")
ax2.set_xlabel("Tijd")
ax2.set_ylim(0, 0.4)
ax2.legend()
ax2.grid(alpha=0.3)

# X-as formatting
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
#%% --- AUTOMATISCHE BASELINE DEFINITIE ---
# We zoeken een venster van ongeveer 300 slagen (5 minuten bij 64.8 bpm)
window_size_baseline = 324

# We zoeken het punt waar de gemiddelde RMSSD het laagst is (meest stabiele ritme)
min_avg_rmssd = np.inf
best_start_idx = 0

for i in range(len(rmssd_trend) - window_size_baseline):
    current_avg = np.mean(rmssd_trend[i : i + window_size_baseline])
    if current_avg < min_avg_rmssd:
        min_avg_rmssd = current_avg
        best_start_idx = i

# Nu definiëren we het segment dat je in de berekening gebruikt
baseline_segment = rmssd_trend[best_start_idx : best_start_idx + window_size_baseline]

print(f"Baseline segment gevonden vanaf index: {best_start_idx}")
#%% Baseline drempelwaarde berekenen voor deze pt
mu_baseline = np.mean(baseline_segment)
sigma_baseline = np.std(baseline_segment)

# 2. Stel een fysiologische ondergrens in
# Voor een 50-jarige is alles onder de 0.05 - 0.06s bijna altijd 'normaal'
# We zetten de absolute ondergrens op 0.08s (80ms) om ademhalings-variabiliteit te negeren.
fysiologische_grens = 0.08 

# 3. Kies de hoogste van de twee
berekende_drempel = max(fysiologische_grens, mu_baseline + 3 * sigma_baseline)

print(f"Statistische drempel was: {mu_baseline + 3 * sigma_baseline:.3f} s")
print(f"Toegepaste (robuuste) drempel: {berekende_drempel:.3f} s")
# %% --- Scheiden normaal // PAC // AF ---
drempel = berekende_drempel
af_window = 3  # Aantal opeenvolgende slagen om het "AF" te noemen

# Maak een status-lijst
status = [] # 0 = Normaal, 1 = PAC, 2 = AF

# Simpele rolling check
for i in range(len(segment_rmssd)):
    if segment_rmssd[i] > drempel:
        # Check of de buren ook hoog zijn (duidt op AF)
        start = max(0, i-15)
        end = min(len(segment_rmssd), i+15)
        if np.mean(segment_rmssd[start:end] > drempel) > 0.7:
            status.append(2) # AF
        else:
            status.append(1) # PAC
    else:
        status.append(0) # Normaal

# --- STATUS SMOOTHING  ---
status_clean = np.array(status).copy() # We maken een kopie om mee te werken
window_smooth = 3 

for i in range(window_smooth, len(status_clean) - window_smooth):
    # Als de huidige slag GEEN AF (2) is...
    if status_clean[i] != 2:
        # Kijk naar de buren (bijv. 5 slagen ervoor en 5 erna)
        omgeving = status_clean[i - window_smooth : i + window_smooth]
        
        # Tel hoe vaak AF (2) voorkomt in die omgeving
        aantal_af = np.sum(omgeving == 2)
        
        # Als meer dan de helft van de buren AF is, vul dan het gat op
        if aantal_af > window_smooth: # Bij window 5 kijken we naar 10 buren
            status_clean[i] = 2

# BELANGRIJK: Overschrijf je oude status met de schone versie voor het plotten
status = status_clean.tolist()

# --- PLOTTEN MET 3 KLEUREN ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax1.plot(segment_t, segment_rr, 'o-', color='black', markersize=2, linewidth=0.5, alpha=0.3)
ax1.set_title(f"Differentiële Analyse: Normaal vs PAC vs AF ({start_tijd_str})")

# Kleur de achtergrond op basis van status
for i in range(len(t_rmssd_seg) - 1):
    curr_status = status[i]
    if curr_status == 2:
        color, alpha, label = 'red', 0.3, 'AF'
    elif curr_status == 1:
        color, alpha, label = 'orange', 0.5, 'PAC'
    else:
        color, alpha, label = 'green', 0.1, 'Normaal'
    
    ax1.axvspan(t_rmssd_seg[i], t_rmssd_seg[i+1], color=color, alpha=alpha)
    ax2.axvspan(t_rmssd_seg[i], t_rmssd_seg[i+1], color=color, alpha=alpha)

ax2.plot(t_rmssd_seg, segment_rmssd, color='firebrick', linewidth=1)
ax2.axhline(y=drempel, color='black', linestyle='--')
ax2.set_ylabel("RMSSD (s)")
ax2.set_ylim(0, 0.4)

# Legenda
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.2, label='Normaal Ritme'),
    Patch(facecolor='orange', alpha=0.5, label='PAC (Ectopie)'),
    Patch(facecolor='red', alpha=0.3, label='AF (Aanhoudend)')
]
ax2.legend(handles=legend_elements)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
# 1. BEREKEN SDRR VOOR HET HALF UUR
sdrr_30min = np.std(segment_rr)

# 2. POINCARÉ PLOT MET KLEUREN OP BASIS VAN RMSSD
plt.figure(figsize=(8, 8))
# We kleuren de punten: rood is hoge RMSSD (onregelmatig), blauw is laag.
scatter = plt.scatter(segment_rr[:-1], segment_rr[1:], 
                      c=segment_rmssd[:-1], cmap='jet', s=15, alpha=0.6)
plt.colorbar(scatter, label='RMSSD (s)')

# Identiteitslijn (Diagonal)
plt.plot([0.4, 1.4], [0.4, 1.4], 'k--', alpha=0.5, label='Identiteitslijn')

plt.title(f"Poincaré Plot (SDRR = {sdrr_30min:.3f}s)")
plt.xlabel('$RR_n$ (s)')
plt.ylabel('$RR_{n+1}$ (s)')
plt.xlim(0.4, 1.4)
plt.ylim(0.4, 1.4)
plt.grid(alpha=0.2)
plt.show()

print(f"SDRR van dit segment: {sdrr_30min:.3f} s")
# %%
# --- ANALYSE PER UUR ---
# We maken een lijstje van alle unieke uren in de dataset
unieke_uren = pd.Series(clean_t).dt.hour.unique()

print(f"{'Uur':<10} | {'PACs (<3)':<12} | {'AF-Runs (>=3)':<15} | {'AF-Burden (%)':<12}")
print("-" * 55)

voor_tabel = []

for uur in unieke_uren:
    mask_uur = (clean_t.hour == uur)
    uur_rmssd = rmssd_trend[(t_labels.hour == uur)]
    uur_status = []

    for i in range(len(uur_rmssd)):
        if uur_rmssd[i] > berekende_drempel:
            start = max(0, i-15)
            end = min(len(uur_rmssd), i+15)
            if np.mean(uur_rmssd[start:end] > berekende_drempel) > 0.7:
                uur_status.append(2) # AF
            else:
                uur_status.append(1) # PAC
        else:
            uur_status.append(0)
    # --- SMOOTHING STAP  ---
    uur_status = np.array(uur_status) # Even omzetten naar array voor berekening
    ws = 5 
    for i in range(ws, len(uur_status) - ws):
        if uur_status[i] != 2:
            # Als meer dan de helft van de buren AF (2) is, maak deze ook AF
            if np.sum(uur_status[i-ws : i+ws] == 2) > ws:
                uur_status[i] = 2

    # --- Tel de episodes ---
    pacs, af_runs = analyseer_episodes(uur_status)
    
    # --- Bereken burden ---
    burden = (sum(1 for s in uur_status if s == 2) / len(uur_status) * 100) if len(uur_status) > 0 else 0
    
    print(f"{uur:02d}:00 uur  | {pacs:<12} | {af_runs:<15} | {burden:<12.1f}%")
    voor_tabel.append([uur, pacs, af_runs, burden])

# --- BEREKEN TOTALEN ---
totaal_pacs = sum(rij[1] for rij in voor_tabel)
totaal_af_runs = sum(rij[2] for rij in voor_tabel)

# Totale burden over de gehele opname berekenen
totaal_slagen_af = sum( (rij[3]/100) * (len(rmssd_trend[t_labels.hour == rij[0]])) for rij in voor_tabel )
totaal_slagen_gemeten = len(rmssd_trend)
totaal_burden_procent = (totaal_slagen_af / totaal_slagen_gemeten) * 100 if totaal_slagen_gemeten > 0 else 0

# --- PRINT DE TOTAALREGEL ---
print("-" * 55)
print(f"{'TOTAAL':<10} | {totaal_pacs:<12} | {totaal_af_runs:<15} | {totaal_burden_procent:<12.1f}%")
print("-" * 55)

# --- BONUS: OPSLAAN ALS CSV (voor Excel) ---
# df_resultaten = pd.DataFrame(voor_tabel, columns=['Uur', 'PACs', 'AF_Runs', 'Burden_Procent'])
# df_resultaten.to_csv("analyse_resultaten.csv", index=False)
# %%
#%% --- PLOTTING RMS---
fig, ax = plt.subplots(figsize=(12, 6))

# Plot tegen de index, niet tegen de tijdwaarde
ax.plot(x_index, rmssd_trend, color='firebrick', linewidth=0.8, label='RMSSD (Onregelmatigheid)')

# Referentielijn
ax.axhline(y=berekende_drempel, color='black', linestyle='--', alpha=0.6, label='Drempelwaarde AF')

# Nu "faken" we de x-as labels:
# We kiezen bv. 10 plekken op de as om een tijdstip te laten zien
num_ticks = 10
tick_indices = np.linspace(0, len(x_index) - 1, num_ticks, dtype=int)
ax.set_xticks(tick_indices)
ax.set_xticklabels([t_labels[i].strftime('%H:%M:%S') for i in tick_indices], rotation=45)

ax.set_title("Ventriculaire Onregelmatigheid (Tijd-gap verwijderd)")
ax.set_ylabel("RMSSD (seconden)")
ax.set_xlabel("Tijdstip (gecomprimeerde as)")
ax.set_ylim(0, 0.4)
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
# %%
