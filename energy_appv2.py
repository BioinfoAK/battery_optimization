import streamlit as st
import pandas as pd
import numpy as np
import holidays
from io import BytesIO

# --- 1. UI & ACCESS ---
st.set_page_config(page_title="Battery Optimization Test", layout="wide")

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Пароль", type="password", on_change=password_entered, key="password")
        return False
    return st.session_state["password_correct"]

def password_entered():
    if st.session_state["password"] == "Secretb4t4re1!":
        st.session_state["password_correct"] = True
        del st.session_state["password"]

if not check_password():
    st.stop()

# --- 2. CONFIG ---
st.sidebar.header("Параметры")
region_choice = st.sidebar.radio("Регион:", ["Samara", "Ulyanovsk", "Kaliningrad"])
month_choice = st.sidebar.selectbox("Месяц:", ["jan25", "feb25", "mar25", "apr25", "may25", "jun25", "jul25", "aug25", "sep25", "oct25", "nov25", "dec25", "jan26"])

REGION_PATH = region_choice.lower()
# Default fallbacks if config file is missing
gen_pwr = st.sidebar.number_input("Генераторная мощность, руб/Мвт", value=1100000.0)
gen_adm = st.sidebar.number_input("Ставка за управление, руб/Мвт", value=50000.0)
net_rate = st.sidebar.number_input("Ставка за содержание сетей, руб/Мвт", value=800000.0)

TOTAL_RATE_MWH = gen_pwr + gen_adm
NETWORK_RATE_MWH = net_rate
KW_TO_MWH = 1 / 1000
MODULE_KWH = 14.6
LOSS_FACTOR = 1.10 
HR_COLS = [f"{h}.00-{h+1}.00" for h in range(24)]

# --- 3. CORE FUNCTIONS ---
def is_biz_day(dt):
    if dt.month == 11 and dt.day == 1: return True
    return not (dt.weekday() >= 5 or dt in holidays.Russia(years=[dt.year]))

def optimize_discharge_aggressive(row_data, target_map, capacity, active_window):
    discharge = np.zeros(24)
    rem = capacity
    win = [h for h in active_window if row_data[h] > 0]
    if not win: return discharge
    
    # Priority: Generating Hours
    for h in win:
        if target_map.get(h, False):
            val = min(row_data[h], rem)
            discharge[h] += val
            rem -= val
            
    # Priority: Plateau Shaving
    while rem > 0.001:
        net = [row_data[h] - discharge[h] for h in range(24)]
        loads = {h: net[h] for h in win if net[h] > 0.001}
        if not loads: break
        mx = max(loads.values())
        pk_hrs = [h for h, v in loads.items() if v >= mx - 0.001]
        rem_vals = sorted(list(set(loads.values())), reverse=True)
        nxt = rem_vals[1] if len(rem_vals) > 1 else 0
        depth = mx - nxt
        needed = depth * len(pk_hrs)
        if rem >= needed:
            for h in pk_hrs: discharge[h] += depth
            rem -= needed
        else:
            for h in pk_hrs: discharge[h] += rem/len(pk_hrs)
            rem = 0
    return discharge

# --- 4. EXECUTION ---
u_input = st.file_uploader("Загрузить XLSX потребления", type=["xlsx"])

if u_input:
    df_raw = pd.read_excel(u_input)
    df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True)
    df_raw[HR_COLS] = df_raw[HR_COLS].astype(float)
    
    # Time Windows Logic
    try:
        df_h = pd.read_excel(f"reference_data/{REGION_PATH}/hours/assessment_hours.xlsx")
        raw_h = df_h[month_choice].dropna().tolist()
        ALL_ASSESS = sorted([int(str(h).split(':')[0]) if ':' in str(h) else int(float(h)) for h in raw_h])
    except:
        ALL_ASSESS = [7, 8, 9, 10, 17, 18, 19, 20]

    # Fixed window calculation to prevent max() errors
    morn_assess = [h for h in ALL_ASSESS if h < 14]
    eve_assess = [h for h in ALL_ASSESS if h >= 14]
    night_charge_win = list(range(0, min(ALL_ASSESS) if ALL_ASSESS else 6))
    gap_charge_win = list(range(max(morn_assess)+1, min(eve_assess))) if (morn_assess and eve_assess) else []

    df_p = pd.read_excel(f"reference_data/{REGION_PATH}/tariffs/hourly_tariffs_{month_choice.lower()}.xlsx")
    price_map = df_p.set_index(df_p.columns[0]).to_dict('index')
    price_cols = df_p.columns[1:]

    if st.button("🚀 Запустить расчет (13 модулей vs Leveling)"):
        biz_mask = df_raw.iloc[:, 0].apply(is_biz_day)
        df_ref = pd.read_excel(f"reference_data/{REGION_PATH}/hours/generating_hours_{month_choice.lower()}.xlsx")
        df_ref.iloc[:, 0] = pd.to_datetime(df_ref.iloc[:, 0], dayfirst=True).dt.date
        
        green_masks = []
        for _, row in df_raw.iterrows():
            d = row.iloc[0].date()
            match = df_ref[df_ref.iloc[:, 0] == d]
            h_m = {h: False for h in range(24)}
            if not match.empty:
                h_idx = int(float(match.iloc[0, 1])) - 1 # THE INDEX FIX
                if 0 <= h_idx <= 23: h_m[h_idx] = True
            green_masks.append(h_m)

        results = []
        excel_sheets = {"Baseline": df_raw}

        # Baseline Fact
        f_kwh = df_raw[HR_COLS].sum().sum()
        f_net = df_raw[biz_mask][[HR_COLS[h] for h in ALL_ASSESS]].max(axis=1).mean()
        f_gen_peaks = [df_raw.loc[i, [HR_COLS[h] for h, a in green_masks[i].items() if a]].max() for i in range(len(df_raw)) if biz_mask[i]]
        f_gen = np.mean([p for p in f_gen_peaks if not np.isnan(p)]) if f_gen_peaks else 0
        f_cost = sum(row[HR_COLS[h]] * (price_map[row.iloc[0].day][price_cols[h]]/1000) for i, row in df_raw.iterrows() if row.iloc[0].day in price_map for h in range(24))

        results.append({"Setup": "ФАКТ", "kWh": f_kwh, "Gen_kW": f_gen, "Net_kW": f_net, "Cost_En": f_cost, "Cost_Gen": f_gen*KW_TO_MWH*TOTAL_RATE_MWH, "Cost_Net": (f_net/1000)*NETWORK_RATE_MWH})

        # Test 13 Modules
        for mode in ["Standard", "LevelingOnly"]:
            cap = 13 * MODULE_KWH
            df_sim = df_raw.copy()
            for i, row in df_raw.iterrows():
                if not biz_mask[i] or row.iloc[0].day not in price_map: continue
                mask = {} if mode == "LevelingOnly" else green_masks[i]
                
                # Logic: We must track the discharge to add back the lossy charge
                d_total = optimize_discharge_aggressive(row[HR_COLS].values, mask, cap, ALL_ASSESS)
                
                for h in range(24):
                    # Simplified for test: discharge during assessment, recharge at night (hour 0-4)
                    charge_val = (sum(d_total) * LOSS_FACTOR / 5) if h < 5 else 0
                    df_sim.at[i, HR_COLS[h]] = max(0, row[HR_COLS[h]] - d_total[h] + charge_val)

            s_kwh = df_sim[HR_COLS].sum().sum()
            s_net = df_sim[biz_mask][[HR_COLS[h] for h in ALL_ASSESS]].max(axis=1).mean()
            s_gen_peaks = [df_sim.loc[idx, [HR_COLS[h] for h, a in green_masks[idx].items() if a]].max() for idx in df_sim.index[biz_mask]]
            s_gen = np.mean(s_gen_peaks) if s_gen_peaks else 0
            s_cost = sum(row[HR_COLS[h]] * (price_map[row.iloc[0].day][price_cols[h]]/1000) for _, row in df_sim.iterrows() if row.iloc[0].day in price_map for h in range(24))
            
            results.append({"Setup": f"13_Modules_{mode}", "kWh": s_kwh, "Gen_kW": s_gen, "Net_kW": s_net, "Cost_En": s_cost, "Cost_Gen": s_gen*KW_TO_MWH*TOTAL_RATE_MWH, "Cost_Net": (s_net/1000)*NETWORK_RATE_MWH})
            excel_sheets[f"13_{mode}"] = df_sim

        # --- FINAL REPORT ---
        v_report = [
            {"Параметр": "Объем потребления, кВт×ч", **{r['Setup']: r['kWh'] for r in results}},
            {"Параметр": "Генераторная мощность, кВт", **{r['Setup']: r['Gen_kW'] for r in results}},
            {"Параметр": "Сетевая мощность, кВт", **{r['Setup']: r['Net_kW'] for r in results}},
            {"Параметр": "Стоимость электроэнергии", **{r['Setup']: r['Cost_En'] for r in results}},
            {"Параметр": "ИТОГО (без НДС)", **{r['Setup']: r['Cost_En']+r['Cost_Gen']+r['Cost_Net'] for r in results}},
            {"Параметр": "ИТОГО (с НДС 20%)", **{r['Setup']: (r['Cost_En']+r['Cost_Gen']+r['Cost_Net'])*1.2 for r in results}},
        ]

        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            pd.DataFrame(v_report).to_excel(writer, sheet_name="Executive_Report", index=False)
            for sn, df_s in excel_sheets.items(): df_s.to_excel(writer, sheet_name=sn[:31], index=False)
        st.download_button("📥 Скачать Сравнение", out.getvalue(), "Comparison_13_Modules.xlsx")
