import streamlit as st
import pandas as pd
import numpy as np
import holidays
from io import BytesIO

# --- 1. UI & ACCESS ---
st.set_page_config(page_title="Расчет оптимального потребления", layout="wide")
st.title("🔋 Финальный расчет: 13 модулей (Сверка баланса)")

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

# --- 2. SIDEBAR ---
st.sidebar.header("Параметры")
region_choice = st.sidebar.radio("Регион:", ["Samara", "Ulyanovsk", "Kaliningrad"])
month_choice = st.sidebar.selectbox("Месяц:", ["jan25", "feb25", "mar25", "apr25", "may25", "jun25", "jul25", "aug25", "sep25", "oct25", "nov25", "dec25", "jan26"])

REGION_PATH = region_choice.lower()
try:
    df_reg_config = pd.read_excel(f"reference_data/{REGION_PATH}/tariffs/regional_config.xlsx")
    match = df_reg_config[df_reg_config['month'].astype(str).str.lower() == month_choice.lower()]
    default_gen, default_admin, default_net = match.iloc[0][['Генераторная (покупная) мощность', 'Ставка за управление', 'Ставка за содержание сетей']]
except:
    default_gen, default_admin, default_net = 0.0, 0.0, 0.0

gen_pwr = st.sidebar.number_input("Генераторная мощность", value=float(default_gen))
gen_adm = st.sidebar.number_input("Ставка за управление", value=float(default_admin))
net_rate = st.sidebar.number_input("Ставка за содержание сетей", value=float(default_net))

TOTAL_RATE_MWH = gen_pwr + gen_adm
NETWORK_RATE_MWH = net_rate
KW_TO_MWH = 1 / 1000
MODULE_KWH = 14.6
LOSS_FACTOR = 1.10 
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
    for h in window_indices:
        if target_map.get(h, False):
            val = min(row_data[h], rem)
            discharge[h] = val
            rem -= val
    while rem > 0.0001:
        current_net = [row_data[h] - discharge[h] for h in range(24)]
        loads_in_window = {h: current_net[h] for h in window_indices if current_net[h] > 0.0001}
        if not loads_in_window: break
        max_val = max(loads_in_window.values())
        peak_hours = [h for h, val in loads_in_window.items() if val >= max_val - 0.0001]
        remaining_loads = sorted(list(set(loads_in_window.values())), reverse=True)
        next_val = remaining_loads[1] if len(remaining_loads) > 1 else 0
        target_shave = (max_val - next_val) * len(peak_hours)
        if rem >= target_shave:
            for h in peak_hours: discharge[h] += (max_val - next_val)
            rem -= target_shave
        else:
            shave_per_h = rem / len(peak_hours)
            for h in peak_hours: discharge[h] += shave_per_h
            rem = 0
    return discharge

def distribute_charge(energy_to_store, charge_window, price_map, day, price_cols, max_pwr):
    charge_profile = np.zeros(24)
    total_to_pull = energy_to_store * LOSS_FACTOR
    rem = total_to_pull
    if not charge_window: return charge_profile
    sorted_hrs = sorted(charge_window, key=lambda h: price_map[day][price_cols[h]])
    for h in sorted_hrs:
        if rem <= 0: break
        can_take = min(rem, max_pwr * LOSS_FACTOR)
        charge_profile[h] = can_take
        rem -= can_take
    return charge_profile

# --- 4. EXECUTION ---
u_input = st.file_uploader("Загрузить потребление (xlsx)", type=["xlsx"])

if u_input:
    df_raw = pd.read_excel(u_input)
    df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True)
    df_raw[HR_COLS] = df_raw[HR_COLS].astype(float)
    
    df_h = pd.read_excel(f"reference_data/{REGION_PATH}/hours/assessment_hours.xlsx")
    raw_h = df_h[month_choice].dropna().tolist()
    ALL_ASSESS = sorted([int(str(h).split(':')[0]) if ':' in str(h) else int(float(h)) for h in raw_h])

    morn_assess = [h for h in ALL_ASSESS if h < 13]
    eve_assess = [h for h in ALL_ASSESS if h >= 13]
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
                h_idx = int(float(match.iloc[0, 1])) - 1
                if 0 <= h_idx <= 23: h_m[h_idx] = True
            green_masks.append(h_m)

        results = []; excel_sheets = {"Baseline": df_raw}

        # FACT calculation
        base_kwh = df_raw[HR_COLS].sum().sum()
        base_net_p = df_raw[biz_mask][[HR_COLS[h] for h in ALL_ASSESS]].max(axis=1).mean()
        base_gen_peaks = [df_raw.loc[i, [HR_COLS[h] for h, a in green_masks[i].items() if a]].max() for i in range(len(df_raw)) if biz_mask[i]]
        base_gen_p = np.mean([p for p in base_gen_peaks if not np.isnan(p)]) if base_gen_peaks else 0
        base_en_c = sum(row[HR_COLS[h]] * (price_map[row.iloc[0].day][price_cols[h]]/1000) for i, row in df_raw.iterrows() if row.iloc[0].day in price_map for h in range(24))

        results.append({"Setup": "ФАКТ", "Total Monthly kWh": round(base_kwh, 2), "Gen Peak": base_gen_p, "Net Peak": base_net_p/1000, "Gen Cost": base_gen_p*KW_TO_MWH*TOTAL_RATE_MWH, "Net Cost": (base_net_p/1000)*NETWORK_RATE_MWH, "Energy Cost": base_en_c})

        # Test cases
        for config in [13, "13_NoGen"]:
            m = 13
            cap = m * MODULE_KWH
            max_p = cap * 0.5
            df_sim = df_raw.copy()
            
            for i, row in df_raw.iterrows():
                if not biz_mask[i] or row.iloc[0].day not in price_map: continue
                mask = {} if config == "13_NoGen" else green_masks[i]
                
                d1 = optimize_discharge_aggressive(row[HR_COLS].values, mask, cap, morn_assess)
                c1 = distribute_charge(sum(d1), gap_charge_win, price_map, row.iloc[0].day, price_cols, max_p)
                
                usable_e2 = min(cap, (cap - sum(d1)) + (sum(c1)/LOSS_FACTOR))
                load_e2 = row[HR_COLS].values - d1 + c1
                d2 = optimize_discharge_aggressive(load_e2, mask, usable_e2, eve_assess)
                c2 = distribute_charge(sum(d2), night_charge_win, price_map, row.iloc[0].day, price_cols, max_p)
                
                for h in range(24):
                    # THE CRITICAL MATH: Load - Out + In
                    df_sim.at[i, HR_COLS[h]] = max(0, row[HR_COLS[h]] - (d1[h]+d2[h]) + (c1[h]+c2[h]))

            # Metrics
            s_kwh = df_sim[HR_COLS].sum().sum()
            s_net_p = df_sim[biz_mask][[HR_COLS[h] for h in ALL_ASSESS]].max(axis=1).mean()
            s_gen_peaks = [df_sim.loc[idx, [HR_COLS[h] for h, a in green_masks[idx].items() if a]].max() for idx in df_sim.index[biz_mask]]
            s_gen_p = np.mean([p for p in s_gen_peaks if not np.isnan(p)]) if s_gen_peaks else 0
            s_en_c = sum(row[HR_COLS[h]] * (price_map[row.iloc[0].day][price_cols[h]]/1000) for _, row in df_sim.iterrows() if row.iloc[0].day in price_map for h in range(24))
            
            name = f"13_Modules_{config}"
            results.append({"Setup": name, "Total Monthly kWh": round(s_kwh, 2), "Gen Peak": s_gen_p, "Net Peak": s_net_p/1000, "Gen Cost": s_gen_p*KW_TO_MWH*TOTAL_RATE_MWH, "Net Cost": (s_net_p/1000)*NETWORK_RATE_MWH, "Energy Cost": s_en_c})
            excel_sheets[name] = df_sim

        # --- FINANCIAL REPORT ---
        v_cols = [r['Setup'] for r in results]
        v_report = [
            {"": "Объем потребления, кВт×ч", **{r['Setup']: r['Total Monthly kWh'] for r in results}},
            {"": "Генераторная мощность, кВт", **{r['Setup']: round(r['Gen Peak'], 2) for r in results}},
            {"": "Сетевая мощность, кВт", **{r['Setup']: round(r['Net Peak']*1000, 2) for r in results}},
            {"": "", **{c: "" for c in v_cols}},
            {"": "Средняя стоимость электроэнергии, руб/кВтч", **{r['Setup']: round(r['Energy Cost']/r['Total Monthly kWh'], 2) for r in results}},
            {"": "Генераторная мощность, руб/МВт", **{c: round(TOTAL_RATE_MWH, 2) for c in v_cols}},
            {"": "Ставка за содержание сетей, руб/МВт", **{c: round(NETWORK_RATE_MWH, 2) for c in v_cols}},
            {"": "", **{c: "" for c in v_cols}},
            {"": "Стоимость электроэнергии, руб", **{r['Setup']: round(r['Energy Cost'], 2) for r in results}},
            {"": "Стоимость генераторной, руб", **{r['Setup']: round(r['Gen Cost'], 2) for r in results}},
            {"": "Стоимость сетевой, руб", **{r['Setup']: round(r['Net Cost'], 2) for r in results}},
            {"": "ИТОГО без НДС 20%, руб", **{r['Setup']: round(r['Energy Cost']+r['Gen Cost']+r['Net Cost'], 2) for r in results}},
            {"": "ИТОГО с НДС 20%, руб", **{r['Setup']: round((r['Energy Cost']+r['Gen Cost']+r['Net Cost'])*1.2, 2) for r in results}}
        ]

        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            pd.DataFrame(v_report).to_excel(writer, sheet_name="Financial_Report", index=False)
            for sn, df_s in excel_sheets.items(): df_s.to_excel(writer, sheet_name=sn[:31], index=False)
        st.download_button("📥 Скачать Excel", out.getvalue(), "Report_13_Modules.xlsx")
