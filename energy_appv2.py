import streamlit as st
import pandas as pd
import numpy as np
import holidays
from io import BytesIO
from pathlib import Path

# --- 1. UI & PRE-CHECKS ---
st.set_page_config(page_title="Расчет оптимального потребления", layout="wide")
st.title("🔋 Расчет оптимального потребления с помощью батареи")

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Введите пароль для доступа к системе", type="password", on_change=password_entered, key="password")
        return False
    return st.session_state["password_correct"]

def password_entered():
    if st.session_state["password"] == "Secretb4t4re1!":
        st.session_state["password_correct"] = True
        del st.session_state["password"]
    else:
        st.session_state["password_correct"] = False

if not check_password():
    st.stop()

# --- 2. PARAMETERS & CONFIG ---
st.sidebar.header("Параметры объекта")
region_choice = st.sidebar.radio("Выберите регион:", ["Samara", "Ulyanovsk", "Kaliningrad"])
month_choice = st.sidebar.selectbox("Выберите месяц:", ["nov25", "dec25"])

REGION_PATH = region_choice.lower()
try:
    df_reg_config = pd.read_excel(f"reference_data/{REGION_PATH}/tariffs/regional_config.xlsx")
    match = df_reg_config[df_reg_config['month'].astype(str).str.lower() == month_choice.lower()]
    default_gen, default_admin, default_net = match.iloc[0][['Генераторная (покупная) мощность', 'Ставка за управление', 'Ставка за содержание сетей']]
except:
    default_gen, default_admin, default_net = 0.0, 0.0, 0.0

st.sidebar.header("Параметры тарифов")
gen_pwr = st.sidebar.number_input("Генераторная мощность, руб/Мвт", value=float(default_gen))
gen_adm = st.sidebar.number_input("Ставка за управление, руб/Мвт", value=float(default_admin))
net_rate = st.sidebar.number_input("Ставка за содержание сетей, руб/Мвт", value=float(default_net))

TOTAL_RATE_MWH = gen_pwr + gen_adm
NETWORK_RATE_MWH = net_rate
KW_TO_MWH = 1 / 1000
MODULE_KWH = 14.6
LOSS_FACTOR = 1.10 
HR_COLS = [f"{h}.00-{h+1}.00" for h in range(24)]

# --- 3. CORE LOGIC FUNCTIONS ---
def is_biz_day(dt):
    if dt.month == 11 and dt.day == 1: return True
    return not (dt.weekday() >= 5 or dt in holidays.Russia(years=[dt.year]))

def optimize_discharge_aggressive(row_data, target_map, capacity, active_window):
    discharge = np.zeros(24)
    rem = capacity
    # Priority 1: Zero out Green Hours
    for h in active_window:
        if target_map.get(h, False):
            val = min(row_data[h], rem)
            discharge[h] += val
            rem -= val
    # Priority 2: Aggressive Leveling (Minimize Variation)
    if rem > 0.01:
        while rem > 0.001:
            current_loads = row_data - discharge
            window_loads = {h: current_loads[h] for h in active_window if current_loads[h] > 0.1}
            if not window_loads: break
            peak_h = max(window_loads, key=window_loads.get)
            step = min(0.1, rem, window_loads[peak_h])
            discharge[peak_h] += step
            rem -= step
    return discharge

def get_assessment_hours():
    try:
        df = pd.read_excel(f"reference_data/{REGION_PATH}/hours/assessment_hours.xlsx")
        raw = df[month_choice].dropna().tolist()
        return sorted([int(str(h).split(':')[0]) if ':' in str(h) else int(float(h)) for h in raw])
    except: return [7, 8, 9, 10, 15, 16, 17, 18, 19, 20]

def get_green_mask(df_base):
    try:
        df_ref = pd.read_excel(f"reference_data/{REGION_PATH}/hours/generating_hours_{month_choice.lower()}.xlsx")
        df_ref.iloc[:, 0] = pd.to_datetime(df_ref.iloc[:, 0], dayfirst=True).dt.date
        masks = []
        for _, row in df_base.iterrows():
            d = row.iloc[0].date()
            match = df_ref[df_ref.iloc[:, 0] == d]
            h_mask = {h: False for h in range(24)}
            if not match.empty:
                h_idx = int(match.iloc[0, 1]) - 1
                if 0 <= h_idx <= 23: h_mask[h_idx] = True
            masks.append(h_mask)
        return masks
    except: return [{h: False for h in range(24)} for _ in range(len(df_base))]

# --- 4. EXECUTION ---
u_input = st.file_uploader("Загрузить данные (xlsx)", type=["xlsx"])

if u_input:
    df_raw = pd.read_excel(u_input)
    df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True)
    df_raw[HR_COLS] = df_raw[HR_COLS].astype(float)
    
    ALL_ASSESS = get_assessment_hours()
    gaps = [ALL_ASSESS[i+1] - ALL_ASSESS[i] for i in range(len(ALL_ASSESS)-1)]
    split_idx = gaps.index(max(gaps)) + 1 if gaps else 0
    morn_assess, eve_assess = ALL_ASSESS[:split_idx], ALL_ASSESS[split_idx:]
    
    night_charge_win = list(range(0, min(ALL_ASSESS)))
    gap_charge_win = list(range(max(morn_assess)+1, min(eve_assess))) if eve_assess else []

    df_p = pd.read_excel(f"reference_data/{REGION_PATH}/tariffs/hourly_tariffs_{month_choice.lower()}.xlsx")
    price_map = df_p.set_index(df_p.columns[0]).to_dict('index')
    price_cols = df_p.columns[1:]

    if st.button("🚀 Симулировать"):
        biz_mask = df_raw.iloc[:, 0].apply(is_biz_day)
        green_masks = get_green_mask(df_raw)
        results = []
        excel_sheets = {"Baseline": df_raw}

        # FACT Baseline calculation logic
        base_kwh = df_raw[HR_COLS].sum().sum()
        net_peak_fact = df_raw[biz_mask][[HR_COLS[h] for h in ALL_ASSESS]].max(axis=1).mean()
        gen_peaks = [df_raw.loc[i, [HR_COLS[h] for h, a in green_masks[i].items() if a]].max() for i in range(len(df_raw)) if biz_mask[i]]
        gen_peak_fact = np.mean([p for p in gen_peaks if not np.isnan(p)]) if gen_peaks else 0
        en_cost_fact = sum(row[HR_COLS[h]] * (price_map[row.iloc[0].day][price_cols[h]]/1000) for i, row in df_raw.iterrows() if row.iloc[0].day in price_map for h in range(24))

        results.append({"Setup": "ФАКТ", "Total Monthly kWh": round(base_kwh, 2), "Generating Peak (kW)": round(gen_peak_fact, 4), "Avg Assessment Peak (MW)": round(net_peak_fact/1000, 4), "Generating cost": round(gen_peak_fact*KW_TO_MWH*TOTAL_RATE_MWH, 2), "Max network charge": round((net_peak_fact/1000)*NETWORK_RATE_MWH, 2), "Total Consumption Cost": round(en_cost_fact, 2), "GRAND TOTAL COST": round(en_cost_fact + (gen_peak_fact*KW_TO_MWH*TOTAL_RATE_MWH) + ((net_peak_fact/1000)*NETWORK_RATE_MWH), 2)})

        for m in [5, 6, 7, 8]:
            cap = m * MODULE_KWH
            df_sim = df_raw.copy(); df_sch = df_raw.copy(); df_sch[HR_COLS] = 0.0
            
            for i, row in df_raw.iterrows():
                if not biz_mask[i]: continue
                day = row.iloc[0].day
                if day not in price_map: continue
                
                # --- AGGRESSIVE LEVELING ---
                morn_d = optimize_discharge_aggressive(row[HR_COLS].values, green_masks[i], cap, morn_assess)
                remaining_after_morn = row[HR_COLS].values - morn_d
                eve_d = optimize_discharge_aggressive(remaining_after_morn, green_masks[i], cap, eve_assess)
                
                final_discharge = morn_d + eve_d
                
                # --- CHARGING WITH LOSS ---
                final_charge = np.zeros(24)
                # Night Charge
                n_hrs = sorted(night_charge_win, key=lambda h: price_map[day][price_cols[h]])[:2]
                amt_n = (sum(morn_d) * LOSS_FACTOR) / len(n_hrs) if n_hrs else 0
                for h in n_hrs: final_charge[h] = amt_n
                # Gap Charge
                g_hrs = sorted(gap_charge_win, key=lambda h: price_map[day][price_cols[h]])[:2]
                amt_g = (sum(eve_d) * LOSS_FACTOR) / len(g_hrs) if g_hrs else 0
                for h in g_hrs: final_charge[h] = amt_g

                for h in range(24):
                    net_h = max(0, row[HR_COLS[h]] - final_discharge[h] + final_charge[h])
                    df_sim.at[i, HR_COLS[h]] = net_h
                    df_sch.at[i, HR_COLS[h]] = final_discharge[h] - final_charge[h]

            # Re-calculate metrics for scenario
            sim_kwh = df_sim[HR_COLS].sum().sum()
            sim_net_p = df_sim[biz_mask][[HR_COLS[h] for h in ALL_ASSESS]].max(axis=1).mean()
            sim_gen_peaks = [df_sim.loc[idx, [HR_COLS[h] for h, a in green_masks[idx].items() if a]].max() for idx in range(len(df_sim)) if biz_mask[idx]]
            sim_gen_p = np.mean([p for p in sim_gen_peaks if not np.isnan(p)]) if sim_gen_peaks else 0
            sim_en_c = sum(df_sim.iloc[idx][HR_COLS[h]] * (price_map[df_sim.iloc[idx,0].day][price_cols[h]]/1000) for idx in range(len(df_sim)) for h in range(24) if df_sim.iloc[idx,0].day in price_map)
            
            total_c = round(sim_en_c + (sim_gen_p*KW_TO_MWH*TOTAL_RATE_MWH) + ((sim_net_p/1000)*NETWORK_RATE_MWH), 2)
            results.append({"Setup": f"{m}_Modules {round(cap,1)}kW", "Total Monthly kWh": round(sim_kwh, 2), "Generating Peak (kW)": round(sim_gen_p, 4), "Avg Assessment Peak (MW)": round(sim_net_p/1000, 4), "Generating cost": round(sim_gen_p*KW_TO_MWH*TOTAL_RATE_MWH, 2), "Max network charge": round((sim_net_p/1000)*NETWORK_RATE_MWH, 2), "Total Consumption Cost": round(sim_en_c, 2), "GRAND TOTAL COST": total_c})
            excel_sheets[f"{m}_Modules_Load"] = df_sim; excel_sheets[f"{m}_Schedule"] = df_sch

        # --- FINAL V_REPORT ---
        v_cols = [r['Setup'] for r in results]
        v_report = [
            {"": "Потребление", **{c: "" for c in v_cols}},
            {"": "Объем потребления, кВт×ч", **{r['Setup']: r['Total Monthly kWh'] for r in results}},
            {"": "Генераторная мощность, кВт", **{r['Setup']: r['Generating Peak (kW)'] for r in results}},
            {"": "Сетевая мощность, кВт", **{r['Setup']: round(r['Avg Assessment Peak (MW)']*1000, 2) for r in results}},
            {"": "Тарифы", **{c: "" for c in v_cols}},
            {"": "Средняя стоимость электроэнергии, руб/кВтч", **{r['Setup']: round(r['Total Consumption Cost']/r['Total Monthly kWh'], 2) for r in results}},
            {"": "ИТОГО:", **{c: "" for c in v_cols}},
            {"": "Стоимость электроэнергии, руб", **{r['Setup']: r['Total Consumption Cost'] for r in results}},
            {"": "Стоимость генераторной, руб", **{r['Setup']: r['Generating cost'] for r in results}},
            {"": "Стоимость сетевой, руб", **{r['Setup']: r['Max network charge'] for r in results}},
            {"": "Стоимость без НДС 20%, руб", **{r['Setup']: r['GRAND TOTAL COST'] for r in results}},
            {"": "Стоимость с НДС 20%, руб", **{r['Setup']: round(r['GRAND TOTAL COST']*1.2, 2) for r in results}}
        ]

        orig_name = Path(u_input.name).stem
        final_fn = f"{orig_name}_{region_choice}_{month_choice}.xlsx"
        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            pd.DataFrame(results).to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame(v_report).to_excel(writer, sheet_name="Executive_Financial_Report", index=False)
            for sn, df_s in excel_sheets.items(): df_s.to_excel(writer, sheet_name=sn, index=False)
        st.success(f"✅ Готово: {final_fn}")
        st.download_button("📥 Скачать результат", out.getvalue(), file_name=final_fn)
