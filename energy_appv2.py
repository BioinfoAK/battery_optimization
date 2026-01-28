import streamlit as st
import pandas as pd
import numpy as np
import holidays
from io import BytesIO
from pathlib import Path

# --- 1. UI & ACCESS ---
st.set_page_config(page_title="Расчет оптимального потребления", layout="wide")
st.title("🔋 Расчет: 13 модулей и Сравнение стратегий")

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Введите пароль", type="password", on_change=password_entered, key="password")
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

# --- 2. SIDEBAR CONFIG ---
st.sidebar.header("Параметры объекта")
region_choice = st.sidebar.radio("Регион:", ["Samara", "Ulyanovsk", "Kaliningrad"])
month_choice = st.sidebar.selectbox("Месяц:", ["jan25", "feb25", "mar25", "apr25", "may25", "jun25", "jul25", "aug25", "sep25", "oct25", "nov25", "dec25", "jan26"])

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
LOSS_FACTOR = 1.10 # 10% loss during charging
HR_COLS = [f"{h}.00-{h+1}.00" for h in range(24)]

# --- 3. CORE LOGIC ---
def is_biz_day(dt):
    if dt.month == 11 and dt.day == 1: return True
    return not (dt.weekday() >= 5 or dt in holidays.Russia(years=[dt.year]))

def optimize_discharge_aggressive(row_data, target_map, capacity, active_window):
    discharge = np.zeros(24)
    rem = capacity
    window_indices = [h for h in active_window if row_data[h] > 0]
    if not window_indices: return discharge

    # Priority 1: Target (Generating) Hours
    for h in window_indices:
        if target_map.get(h, False):
            val = min(row_data[h], rem)
            discharge[h] += val
            rem -= val

    # Priority 2: General Peak Shaving
    while rem > 0.0001:
        current_net = [row_data[h] - discharge[h] for h in range(24)]
        loads_in_window = {h: current_net[h] for h in window_indices if current_net[h] > 0.0001}
        if not loads_in_window: break
        max_val = max(loads_in_window.values())
        peak_hours = [h for h, val in loads_in_window.items() if val >= max_val - 0.0001]
        remaining_loads = sorted(list(set(loads_in_window.values())), reverse=True)
        next_val = remaining_loads[1] if len(remaining_loads) > 1 else 0
        target_shave_depth = max_val - next_val
        total_energy_needed = target_shave_depth * len(peak_hours)
        if rem >= total_energy_needed:
            for h in peak_hours: discharge[h] += target_shave_depth
            rem -= total_energy_needed
        else:
            shave_per_hour = rem / len(peak_hours)
            for h in peak_hours: discharge[h] += shave_per_hour
            rem = 0
    return discharge
    
def distribute_charge(energy_to_store, charge_window, price_map, day, price_cols, max_pwr):
    """
    energy_to_store: net kWh to be put into battery
    Returns: array of 24h grid consumption (includes losses)
    """
    charge_profile = np.zeros(24)
    rem_grid = energy_to_store * LOSS_FACTOR # We pull 10% more from grid
    if not charge_window: return charge_profile
    sorted_hrs = sorted(charge_window, key=lambda h: price_map[day][price_cols[h]])
    for h in sorted_hrs:
        if rem_grid <= 0: break
        can_take = min(rem_grid, max_pwr * LOSS_FACTOR)
        charge_profile[h] = can_take
        rem_grid -= can_take
    return charge_profile

# --- 4. EXECUTION ---
u_input = st.file_uploader("Загрузить файл потребления (xlsx)", type=["xlsx"])

if u_input:
    df_raw = pd.read_excel(u_input)
    df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True)
    df_raw[HR_COLS] = df_raw[HR_COLS].astype(float)
    
    try:
        df_h = pd.read_excel(f"reference_data/{REGION_PATH}/hours/assessment_hours.xlsx")
        raw_h = df_h[month_choice].dropna().tolist()
        ALL_ASSESS = sorted([int(str(h).split(':')[0]) if ':' in str(h) else int(float(h)) for h in raw_h])
    except:
        ALL_ASSESS = [7, 8, 9, 10, 15, 16, 17, 18, 19, 20]

    gaps = [ALL_ASSESS[i+1] - ALL_ASSESS[i] for i in range(len(ALL_ASSESS)-1)]
    split_idx = gaps.index(max(gaps)) + 1 if gaps else 0
    morn_assess, eve_assess = ALL_ASSESS[:split_idx], ALL_ASSESS[split_idx:]
    night_charge_win = list(range(0, min(ALL_ASSESS)))
    gap_charge_win = list(range(max(morn_assess)+1, min(eve_assess))) if eve_assess else []

    df_p = pd.read_excel(f"reference_data/{REGION_PATH}/tariffs/hourly_tariffs_{month_choice.lower()}.xlsx")
    price_map = df_p.set_index(df_p.columns[0]).to_dict('index')
    price_cols = df_p.columns[1:]

    if st.button("🚀 Начать расчет"):
        biz_mask = df_raw.iloc[:, 0].apply(is_biz_day)
        df_ref = pd.read_excel(f"reference_data/{REGION_PATH}/hours/generating_hours_{month_choice.lower()}.xlsx")
        df_ref.iloc[:, 0] = pd.to_datetime(df_ref.iloc[:, 0], dayfirst=True).dt.date
        
        green_masks = []
        for _, row in df_raw.iterrows():
            d = row.iloc[0].date()
            match = df_ref[df_ref.iloc[:, 0] == d]
            h_m = {h: False for h in range(24)}
            if not match.empty:
                # FIX: 20 in file -> 19 in index (19.00-20.00)
                h_val = int(float(match.iloc[0, 1])) 
                h_idx = h_val - 1 
                if 0 <= h_idx <= 23: h_m[h_idx] = True
            green_masks.append(h_m)

        results = []
        excel_sheets = {"Baseline": df_raw}

        # --- FACT ---
        base_kwh = df_raw[HR_COLS].sum().sum()
        base_net_p = df_raw[biz_mask][[HR_COLS[h] for h in ALL_ASSESS]].max(axis=1).mean()
        base_gen_peaks = [df_raw.loc[i, [HR_COLS[h] for h, a in green_masks[i].items() if a]].max() for i in range(len(df_raw)) if biz_mask[i]]
        base_gen_p = np.mean([p for p in base_gen_peaks if not np.isnan(p)]) if base_gen_peaks else 0
        base_en_c = sum(row[HR_COLS[h]] * (price_map[row.iloc[0].day][price_cols[h]]/1000) for i, row in df_raw.iterrows() if row.iloc[0].day in price_map for h in range(24))

        results.append({
            "Setup": "ФАКТ", "Total Monthly kWh": round(base_kwh, 2), "Generating Peak (kW)": round(base_gen_p, 4), 
            "Avg Assessment Peak (MW)": round(base_net_p/1000, 4), "Generating cost": round(base_gen_p*KW_TO_MWH*TOTAL_RATE_MWH, 2), 
            "Max network charge": round((base_net_p/1000)*NETWORK_RATE_MWH, 2), "Total Consumption Cost": round(base_en_c, 2), 
            "GRAND TOTAL COST": round(base_en_c + (base_gen_p*KW_TO_MWH*TOTAL_RATE_MWH) + ((base_net_p/1000)*NETWORK_RATE_MWH), 2)
        })

        # --- CONFIGS ---
        # 13 modules with Current Approach vs disregarding generating hour
        configs = [13, "13_LevelingOnly"]
        for config in configs:
            is_no_gen = (config == "13_LevelingOnly")
            m = 13
            cap_limit = m * MODULE_KWH
            max_pwr = cap_limit * 0.5
            
            df_sim = df_raw.copy(); df_sch = df_raw.copy(); df_sch[HR_COLS] = 0.0

            for i, row in df_raw.iterrows():
                if not biz_mask[i] or row.iloc[0].day not in price_map: continue
                day_d = row.iloc[0].day
                active_mask = {} if is_no_gen else green_masks[i]
                
                # Cycles
                cur_usable = cap_limit
                m_d = optimize_discharge_aggressive(row[HR_COLS].values, active_mask, cur_usable, morn_assess)
                c_gap = distribute_charge(sum(m_d), gap_charge_win, price_map, day_d, price_cols, max_pwr)
                
                # Update usable capacity for evening (Spent morning, added back charge)
                cur_usable = (cap_limit - sum(m_d)) + (sum(c_gap)/LOSS_FACTOR)
                cur_usable = min(cur_usable, cap_limit)
                
                l_after = row[HR_COLS].values - m_d + c_gap
                e_d = optimize_discharge_aggressive(l_after, active_mask, cur_usable, eve_assess)
                c_night = distribute_charge(sum(e_d), night_charge_win, price_map, day_d, price_cols, max_pwr)

                f_dis = m_d + e_d; f_chg = c_gap + c_night
                for h in range(24):
                    df_sim.at[i, HR_COLS[h]] = max(0, row[HR_COLS[h]] - f_dis[h] + f_chg[h])
                    df_sch.at[i, HR_COLS[h]] = f_dis[h] - f_chg[h]
            
            # Metrics
            sim_kwh = df_sim[HR_COLS].sum().sum()
            sim_net_p = df_sim[biz_mask][[HR_COLS[h] for h in ALL_ASSESS]].max(axis=1).mean()
            sim_gen_peaks = [df_sim.loc[idx, [HR_COLS[h] for h, a in green_masks[idx].items() if a]].max() for idx in df_sim.index[biz_mask]]
            sim_gen_p = np.mean([p for p in sim_gen_peaks if not np.isnan(p)]) if sim_gen_peaks else 0
            sim_en_c = sum(row[HR_COLS[h]] * (price_map[row.iloc[0].day][price_cols[h]]/1000) for _, row in df_sim.iterrows() if row.iloc[0].day in price_map for h in range(24))
            
            label = f"{m} Модулей" + (" (Без ген. часа)" if is_no_gen else " (Стандарт)")
            results.append({
                "Setup": label, "Total Monthly kWh": round(sim_kwh, 2), "Generating Peak (kW)": round(sim_gen_p, 4), 
                "Avg Assessment Peak (MW)": round(sim_net_p/1000, 4), "Generating cost": round(sim_gen_p*KW_TO_MWH*TOTAL_RATE_MWH, 2), 
                "Max network charge": round((sim_net_p/1000)*NETWORK_RATE_MWH, 2), "Total Consumption Cost": round(sim_en_c, 2), 
                "GRAND TOTAL COST": round(sim_en_c + (sim_gen_p*KW_TO_MWH*TOTAL_RATE_MWH) + ((sim_net_p/1000)*NETWORK_RATE_MWH), 2)
            })
            excel_sheets[f"{label}_Load"] = df_sim

        # --- REPORTING ---
        v_cols = [r['Setup'] for r in results]
        v_report = [
            {"": "Потребление", **{c: "" for c in v_cols}},
            {"": "Объем потребления, кВт×ч", **{r['Setup']: r['Total Monthly kWh'] for r in results}},
            {"": "Генераторная мощность, кВт", **{r['Setup']: r['Generating Peak (kW)'] for r in results}},
            {"": "Сетевая мощность, кВт", **{r['Setup']: round(r['Avg Assessment Peak (MW)']*1000, 2) for r in results}},
            {"": "", **{c: "" for c in v_cols}},
            {"": "Тарифы", **{c: "" for c in v_cols}},
            {"": "Средняя стоимость электроэнергии, руб/кВтч", **{r['Setup']: round(r['Total Consumption Cost']/r['Total Monthly kWh'], 2) if r['Total Monthly kWh'] > 0 else 0 for r in results}},
            {"": "Генераторная мощность, руб/МВт", **{c: round(TOTAL_RATE_MWH, 2) for c in v_cols}},
            {"": "Ставка за содержание сетей, руб/МВт", **{c: round(NETWORK_RATE_MWH, 2) for c in v_cols}},
            {"": "", **{c: "" for c in v_cols}},
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
            for sn, df_s in excel_sheets.items(): df_s.to_excel(writer, sheet_name=sn[:31], index=False)
        st.success("✅ Сравнение 13 модулей готово")
        st.download_button("📥 Скачать результат", out.getvalue(), file_name=f"Comparison_13_Modules_{month_choice}.xlsx")
