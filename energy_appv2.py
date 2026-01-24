import streamlit as st
import pandas as pd
import numpy as np
import holidays
from io import BytesIO
from pathlib import Path

# --- 1. CORE CONFIG ---
st.set_page_config(page_title="Battery Optimization Pro", layout="wide")

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Введите пароль для доступа", type="password", on_change=password_entered, key="password")
        return False
    return st.session_state["password_correct"]

def password_entered():
    if st.session_state["password"] == "Secretb4t4re1!":
        st.session_state["password_correct"] = True
        del st.session_state["password"]

if not check_password():
    st.stop()

# --- 2. INTERFACE & PARAMETERS ---
st.sidebar.header("Параметры объекта")
region_choice = st.sidebar.radio("Регион:", ["Samara", "Ulyanovsk", "Kaliningrad"])
month_choice = st.sidebar.selectbox("Месяц:", ["nov25", "dec25"])

REGION_PATH = region_choice.lower()
try:
    df_reg = pd.read_excel(f"reference_data/{REGION_PATH}/tariffs/regional_config.xlsx")
    match = df_reg[df_reg['month'].astype(str).str.lower() == month_choice.lower()]
    d_gen, d_admin, d_net = match.iloc[0][['Генераторная (покупная) мощность', 'Ставка за управление', 'Ставка за содержание сетей']]
except:
    d_gen, d_admin, d_net = 0.0, 0.0, 0.0

gen_power_input = st.sidebar.number_input("Генераторная мощность, руб/Мвт", value=float(d_gen))
gen_change_input = st.sidebar.number_input("Ставка за управление, руб/Мвт", value=float(d_admin))
network_rate_input = st.sidebar.number_input("Ставка за содержание сетей, руб/Мвт", value=float(d_net))

TOTAL_RATE_MW = gen_power_input + gen_change_input
NET_RATE_MW = network_rate_input
KW_TO_MWH = 1 / 1000
MODULE_KWH = 14.6
LOSS_FACTOR = 1.10
HR_COLS = [f"{h}.00-{h+1}.00" for h in range(24)]

# --- 3. HELPER LOGIC ---
def is_biz_day(dt):
    if dt.month == 11 and dt.day == 1: return True
    return not (dt.weekday() >= 5 or dt in holidays.Russia(years=[dt.year]))

def get_assessment_hours():
    try:
        df = pd.read_excel(f"reference_data/{REGION_PATH}/hours/assessment_hours.xlsx")
        raw = df[month_choice].dropna().tolist()
        return sorted([int(str(h).split(':')[0]) if ':' in str(h) else int(float(h)) for h in raw])
    except: return [7, 8, 9, 10, 15, 16, 17, 18, 19, 20]

def get_green_mask(df_base):
    df_ref = pd.read_excel(f"reference_data/{REGION_PATH}/hours/generating_hours_{month_choice.lower()}.xlsx")
    df_ref.iloc[:, 0] = pd.to_datetime(df_ref.iloc[:, 0], dayfirst=True).dt.date
    mask = []
    for _, row in df_base.iterrows():
        d = row.iloc[0].date()
        match = df_ref[df_ref.iloc[:, 0] == d]
        h_mask = {h: False for h in range(24)}
        if not match.empty:
            h_idx = int(match.iloc[0, 1]) - 1
            if 0 <= h_idx <= 23: h_mask[h_idx] = True
        mask.append(h_mask)
    return mask

# --- 4. MAIN EXECUTION ---
u_input = st.file_uploader("Загрузить файл потребления", type=["xlsx"])

if u_input:
    df_raw = pd.read_excel(u_input)
    df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True)
    df_raw[HR_COLS] = df_raw[HR_COLS].astype(float)
    
    ALL_ASSESS = get_assessment_hours()
    gaps = [ALL_ASSESS[i+1] - ALL_ASSESS[i] for i in range(len(ALL_ASSESS)-1)]
    split_idx = gaps.index(max(gaps)) + 1 if gaps else 0
    morn_assess = ALL_ASSESS[:split_idx]
    eve_assess = ALL_ASSESS[split_idx:]
    
    # Windows for Charge
    night_charge_win = list(range(0, min(ALL_ASSESS)))
    gap_charge_win = list(range(max(morn_assess)+1, min(eve_assess))) if eve_assess else []

    df_p = pd.read_excel(f"reference_data/{REGION_PATH}/tariffs/hourly_tariffs_{month_choice.lower()}.xlsx")
    price_map = df_p.set_index(df_p.columns[0]).to_dict('index')

    if st.button("🚀 Начать расчет"):
        biz_mask = df_raw.iloc[:, 0].apply(is_biz_day)
        green_masks = get_green_mask(df_raw)
        
        results = []
        excel_sheets = {"Baseline": df_raw}
        
        # --- FACT (BASELINE) CALCULATIONS ---
        base_total_kwh = df_raw[HR_COLS].sum().sum()
        biz_df = df_raw[biz_mask]
        net_peak_fact_kw = biz_df[[HR_COLS[h] for h in ALL_ASSESS]].max(axis=1).mean()
        
        gen_peaks_fact = []
        for i, row in df_raw.iterrows():
            if biz_mask[i]:
                g_hrs = [HR_COLS[h] for h, active in green_masks[i].items() if active]
                if g_hrs: gen_peaks_fact.append(row[g_hrs].max())
        gen_peak_fact = np.mean(gen_peaks_fact) if gen_peaks_fact else 0
        
        en_cost_fact = 0
        for i, row in df_raw.iterrows():
            d = row.iloc[0].day
            if d in price_map:
                en_cost_fact += sum(row[HR_COLS[h]] * (price_map[d][h+1]/1000) for h in range(24))

        res_fact = {
            "Setup": "ФАКТ", 
            "Total Monthly kWh": round(base_total_kwh, 2),
            "Generating Peak (kW)": round(gen_peak_fact, 4), 
            "Avg Assessment Peak (MW)": round(net_peak_fact_kw / 1000, 4),
            "Generating cost": round(gen_peak_fact * KW_TO_MWH * TOTAL_RATE_MW, 2),
            "Max network charge": round((net_peak_fact_kw / 1000) * NET_RATE_MW, 2),
            "Total Consumption Cost": round(en_cost_fact, 2)
        }
        res_fact["GRAND TOTAL COST"] = res_fact["Generating cost"] + res_fact["Max network charge"] + res_fact["Total Consumption Cost"]
        results.append(res_fact)

        # --- MODULE SIMULATION ---
        module_list = [5, 6, 7, 8]
        for m in module_list:
            cap = m * MODULE_KWH
            df_sim = df_raw.copy()
            df_sch = df_raw.copy(); df_sch[HR_COLS] = 0.0
            
            for i, row in df_raw.iterrows():
                if not biz_mask[i]: continue
                day_idx = row.iloc[0].day
                if day_idx not in price_map: continue
                
                # DISCHARGE LOGIC
                d_flow = np.zeros(24)
                
                # Cycle 1: Morning
                rem_morn = cap
                for h in morn_assess:
                    if green_masks[i][h]:
                        val = min(row[HR_COLS[h]], rem_morn)
                        d_flow[h] += val; rem_morn -= val
                if rem_morn > 0:
                    for h in sorted(morn_assess, key=lambda x: row[HR_COLS[x]], reverse=True):
                        val = min(row[HR_COLS[h]] - d_flow[h], rem_morn)
                        d_flow[h] += val; rem_morn -= val
                
                # Cycle 2: Evening
                rem_eve = cap
                for h in eve_assess:
                    if green_masks[i][h]:
                        val = min(row[HR_COLS[h]], rem_eve)
                        d_flow[h] += val; rem_eve -= val
                if rem_eve > 0:
                    for h in sorted(eve_assess, key=lambda x: row[HR_COLS[x]], reverse=True):
                        val = min(row[HR_COLS[h]] - d_flow[h], rem_eve)
                        d_flow[h] += val; rem_eve -= val

                # CHARGE LOGIC (Including Efficiency Loss)
                # Charge 1: Night
                night_hrs = sorted(night_charge_win, key=lambda h: price_map[day_idx][h+1])[:2]
                to_chg_night = cap * LOSS_FACTOR
                for h in night_hrs:
                    amt = min(to_chg_night, (cap * 0.5) * LOSS_FACTOR)
                    d_flow[h] -= amt; to_chg_night -= amt
                
                # Charge 2: Gap
                if gap_charge_win:
                    gap_hrs = sorted(gap_charge_win, key=lambda h: price_map[day_idx][h+1])[:2]
                    to_chg_gap = cap * LOSS_FACTOR
                    for h in gap_hrs:
                        amt = min(to_chg_gap, (cap * 0.5) * LOSS_FACTOR)
                        d_flow[h] -= amt; to_chg_gap -= amt

                for h in range(24):
                    df_sim.at[i, HR_COLS[h]] = max(0, row[HR_COLS[h]] - d_flow[h])
                    df_sch.at[i, HR_COLS[h]] = d_flow[h]

            # Scenario Metrics
            sim_total_kwh = df_sim[HR_COLS].sum().sum()
            sim_biz = df_sim[biz_mask]
            sim_net_peak = sim_biz[[HR_COLS[h] for h in ALL_ASSESS]].max(axis=1).mean()
            
            sim_gen_peaks = []
            for idx, r in df_sim.iterrows():
                if biz_mask[idx]:
                    g_hrs = [HR_COLS[h] for h, act in green_masks[idx].items() if act]
                    if g_hrs: sim_gen_peaks.append(r[g_hrs].max())
            sim_gen_peak = np.mean(sim_gen_peaks) if sim_gen_peaks else 0
            
            sim_en_cost = 0
            for idx, r in df_sim.iterrows():
                d = r.iloc[0].day
                if d in price_map:
                    sim_en_cost += sum(r[HR_COLS[h]] * (price_map[d][h+1]/1000) for h in range(24))

            setup_key = f"{m}_Modules {round(cap,1)}kW"
            res_m = {
                "Setup": setup_key, 
                "Total Monthly kWh": round(sim_total_kwh, 2),
                "Generating Peak (kW)": round(sim_gen_peak, 4), 
                "Avg Assessment Peak (MW)": round(sim_net_peak / 1000, 4),
                "Generating cost": round(sim_gen_peak * KW_TO_MWH * TOTAL_RATE_MW, 2),
                "Max network charge": round((sim_net_peak / 1000) * NET_RATE_MW, 2),
                "Total Consumption Cost": round(sim_en_cost, 2),
                "System Energy Loss": round(sim_total_kwh - base_total_kwh, 2),
                "Energy Cycle Efficiency": f"{round((base_total_kwh/sim_total_kwh)*100, 1)}%"
            }
            res_m["GRAND TOTAL COST"] = res_m["Generating cost"] + res_m["Max network charge"] + res_m["Total Consumption Cost"]
            results.append(res_m)
            excel_sheets[f"{m}_Modules_Load"] = df_sim
            excel_sheets[f"{m}_Schedule"] = df_sch

        # --- 5. THE COMPLETE EXECUTIVE REPORT ---
        cols_v = ["ФАКТ"] + [f"{m}_Modules {round(m*MODULE_KWH,1)}kW" for m in module_list]
        
        def get_row(label, key, multiplier=1):
            row = {"": label}
            for r in results:
                row[r['Setup']] = round(r.get(key, 0) * multiplier, 2) if isinstance(r.get(key), (int, float)) else r.get(key)
            return row

        v_report = [
            {"": "Потребление", **{c: "" for c in cols_v}},
            get_row("Объем потребления, кВт×ч", "Total Monthly kWh"),
            get_row("Генераторная (покупная) мощность, кВт", "Generating Peak (kW)"),
            get_row("Сетевая мощность, кВт", "Avg Assessment Peak (MW)", 1000),
            {"": "", **{c: "" for c in cols_v}},
            {"": "Тарифы", **{c: "" for c in cols_v}},
            {"": "Средняя стоимость электроэнергии, руб/MВтч", "ФАКТ": round(res_fact['Total Consumption Cost']/(res_fact['Total Monthly kWh']*KW_TO_MWH), 2)},
            {"": "Генераторная мощность, руб/МВт", **{c: round(TOTAL_RATE_MW, 2) for c in cols_v}},
            {"": "Ставка за содержание сетей, руб/МВт", **{c: round(NET_RATE_MW, 2) for c in cols_v}},
            {"": "", **{c: "" for c in cols_v}},
            {"": "ИТОГО:", **{c: "" for c in cols_v}},
            get_row("Стоимость электроэнергии, руб", "Total Consumption Cost"),
            get_row("Стоимость генераторной, руб", "Generating cost"),
            get_row("Стоимость сетевой, руб", "Max network charge"),
            get_row("Стоимость без НДС 20%, руб", "GRAND TOTAL COST"),
            {"": "Стоимость с НДС 20%, руб", **{r['Setup']: round(r['GRAND TOTAL COST']*1.2, 2) for r in results}}
        ]
        
        # Calculate Weighted Avg for scenarios in Tariffs section
        for i, setup_name in enumerate(cols_v):
            if i > 0: # Scenarios only
                res_obj = results[i]
                v_report[6][setup_name] = round(res_obj['Total Consumption Cost']/(res_obj['Total Monthly kWh']*KW_TO_MWH), 2)

        # --- FILENAME & EXPORT ---
        orig_name = Path(u_input.name).stem
        final_filename = f"{orig_name}_{region_choice}_{month_choice}.xlsx"
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(results).to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame(v_report).to_excel(writer, sheet_name="Executive_Financial_Report", index=False)
            for sn, df_s in excel_sheets.items():
                df_s.to_excel(writer, sheet_name=sn, index=False)
        
        st.success(f"✅ Расчет завершен для файла: {final_filename}")
        st.download_button("📥 Скачать Excel результат", output.getvalue(), file_name=final_filename)
