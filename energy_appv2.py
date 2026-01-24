import streamlit as st
import pandas as pd
import numpy as np
import holidays
from io import BytesIO
from pathlib import Path

# --- 1. ORIGINAL UI & PASSWORD ---
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

# --- 2. SIDEBAR PARAMETERS ---
st.sidebar.header("Параметры объекта")
region_choice = st.sidebar.radio("Выберите регион:", ["Samara", "Ulyanovsk", "Kaliningrad"])
month_choice = st.sidebar.selectbox("Выберите месяц:", ["nov25", "dec25"])

REGION_PATH = region_choice.lower()
try:
    df_reg_config = pd.read_excel(f"reference_data/{REGION_PATH}/tariffs/regional_config.xlsx")
    match = df_reg_config[df_reg_config['month'].astype(str).str.lower() == month_choice.lower()]
    if not match.empty:
        default_gen = float(match.iloc[0]['Генераторная (покупная) мощность'])
        default_admin = float(match.iloc[0]['Ставка за управление'])
        default_net = float(match.iloc[0]['Ставка за содержание сетей'])
    else:
        default_gen, default_admin, default_net = 0.0, 0.0, 0.0
except:
    default_gen, default_admin, default_net = 0.0, 0.0, 0.0

st.sidebar.header("Параметры тарифов (Автозагрузка)")
gen_pwr = st.sidebar.number_input("Генераторная мощность, руб/Мвт", value=default_gen)
gen_adm = st.sidebar.number_input("Ставка за управление, руб/Мвт", value=default_admin)
net_rate = st.sidebar.number_input("Ставка за содержание сетей, руб/Мвт", value=default_net)

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

# --- 4. EXECUTION ---
u_input = st.file_uploader("Выбрать файл с данными потребления объекта (xlsx)", type=["xlsx"])

if u_input:
    df_raw = pd.read_excel(u_input)
    df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True)
    df_raw[HR_COLS] = df_raw[HR_COLS].astype(float)
    
    ALL_ASSESS = get_assessment_hours()
    gaps = [ALL_ASSESS[i+1] - ALL_ASSESS[i] for i in range(len(ALL_ASSESS)-1)]
    split_idx = gaps.index(max(gaps)) + 1 if gaps else 0
    morn_assess = ALL_ASSESS[:split_idx]
    eve_assess = ALL_ASSESS[split_idx:]
    
    night_charge_win = list(range(0, min(ALL_ASSESS)))
    gap_charge_win = list(range(max(morn_assess)+1, min(eve_assess))) if eve_assess else []

    df_p = pd.read_excel(f"reference_data/{REGION_PATH}/tariffs/hourly_tariffs_{month_choice.lower()}.xlsx")
    # Fix potential key issues: map by day and then by column index
    price_map = df_p.set_index(df_p.columns[0]).to_dict('index')
    price_cols = df_p.columns[1:] # The 24 hour columns

    if st.button("🚀 Симулируем"):
        biz_mask = df_raw.iloc[:, 0].apply(is_biz_day)
        green_masks = get_green_mask(df_raw)
        results = []
        excel_sheets = {"Baseline": df_raw}
        
        # --- FACT ---
        base_kwh = df_raw[HR_COLS].sum().sum()
        net_peak_fact = df_raw[biz_mask][[HR_COLS[h] for h in ALL_ASSESS]].max(axis=1).mean()
        
        gen_peaks = []
        for i, row in df_raw.iterrows():
            if biz_mask[i]:
                g_hrs = [HR_COLS[h] for h, active in green_masks[i].items() if active]
                if g_hrs: gen_peaks.append(row[g_hrs].max())
        gen_peak_fact = np.mean(gen_peaks) if gen_peaks else 0
        
        en_cost_fact = 0
        for i, row in df_raw.iterrows():
            day = row.iloc[0].day
            if day in price_map:
                for h in range(24):
                    en_cost_fact += row[HR_COLS[h]] * (price_map[day][price_cols[h]] / 1000)

        res_fact = {"Setup": "ФАКТ", "Total Monthly kWh": round(base_kwh, 2), "Generating Peak (kW)": round(gen_peak_fact, 4), "Avg Assessment Peak (MW)": round(net_peak_fact/1000, 4), "Generating cost": round(gen_peak_fact*KW_TO_MWH*TOTAL_RATE_MWH, 2), "Max network charge": round((net_peak_fact/1000)*NETWORK_RATE_MWH, 2), "Total Consumption Cost": round(en_cost_fact, 2)}
        res_fact["GRAND TOTAL COST"] = res_fact["Generating cost"] + res_fact["Max network charge"] + res_fact["Total Consumption Cost"]
        results.append(res_fact)

        # --- MODULES ---
        for m in [5, 6, 7, 8]:
            cap = m * MODULE_KWH
            df_sim = df_raw.copy(); df_sch = df_raw.copy(); df_sch[HR_COLS] = 0.0
            
            for i, row in df_raw.iterrows():
                if not biz_mask[i]: continue
                day = row.iloc[0].day
                if day not in price_map: continue
                
                d_flow = np.zeros(24)
                # Discharge (Assessment only)
                for win in [morn_assess, eve_assess]:
                    rem = cap
                    for h in win: # Green first
                        if green_masks[i][h]:
                            v = min(row[HR_COLS[h]], rem); d_flow[h] += v; rem -= v
                    if rem > 0: # Peak shave second
                        for h in sorted(win, key=lambda x: row[HR_COLS[x]], reverse=True):
                            v = min(row[HR_COLS[h]] - d_flow[h], rem); d_flow[h] += v; rem -= v
                
                # Charge (Night + Gap)
                for win in [night_charge_win, gap_charge_win]:
                    if not win: continue
                    ch_hrs = sorted(win, key=lambda h: price_map[day][price_cols[h]])[:2]
                    to_chg = cap * LOSS_FACTOR
                    for h in ch_hrs:
                        amt = min(to_chg, (cap * 0.5) * LOSS_FACTOR)
                        d_flow[h] -= amt; to_chg -= amt

                for h in range(24):
                    df_sim.at[i, HR_COLS[h]] = max(0, row[HR_COLS[h]] - d_flow[h])
                    df_sch.at[i, HR_COLS[h]] = d_flow[h]

            # Metrics
            sim_kwh = df_sim[HR_COLS].sum().sum()
            sim_net_p = df_sim[biz_mask][[HR_COLS[h] for h in ALL_ASSESS]].max(axis=1).mean()
            sim_gen_p = np.mean([df_sim.iloc[idx][[HR_COLS[h] for h, a in green_masks[idx].items() if a]].max() for idx in range(len(df_sim)) if biz_mask[idx]])
            sim_en_c = sum(df_sim.iloc[idx][HR_COLS[h]] * (price_map[df_sim.iloc[idx,0].day][price_cols[h]]/1000) for idx in range(len(df_sim)) for h in range(24) if df_sim.iloc[idx,0].day in price_map)
            
            s_key = f"{m}_Modules {round(cap,1)}kW"
            res_m = {"Setup": s_key, "Total Monthly kWh": round(sim_kwh, 2), "Generating Peak (kW)": round(sim_gen_p, 4), "Avg Assessment Peak (MW)": round(sim_net_p/1000, 4), "Generating cost": round(sim_gen_p*KW_TO_MWH*TOTAL_RATE_MWH, 2), "Max network charge": round((sim_net_p/1000)*NETWORK_RATE_MWH, 2), "Total Consumption Cost": round(sim_en_c, 2), "System Energy Loss": round(sim_kwh - base_kwh, 2), "Energy Cycle Efficiency": f"{round((base_kwh/sim_kwh)*100, 1)}%"}
            res_m["GRAND TOTAL COST"] = res_m["Generating cost"] + res_m["Max network charge"] + res_m["Total Consumption Cost"]
            results.append(res_m)
            excel_sheets[f"{m}_Modules_Load"] = df_sim; excel_sheets[f"{m}_Schedule"] = df_sch

        # --- V_REPORT ---
        v_cols = [r['Setup'] for r in results]
        v_report = [
            {"": "Потребление", **{c: "" for c in v_cols}},
            {"": "Объем потребления, кВт×ч", **{r['Setup']: r['Total Monthly kWh'] for r in results}},
            {"": "Генераторная (покупная) мощность, кВт", **{r['Setup']: r['Generating Peak (kW)'] for r in results}},
            {"": "Сетевая мощность, кВт", **{r['Setup']: round(r['Avg Assessment Peak (MW)']*1000, 2) for r in results}},
            {"": "", **{c: "" for c in v_cols}},
            {"": "Тарифы", **{c: "" for c in v_cols}},
            {"": "Средняя стоимость электроэнергии, руб/кВтч", **{r['Setup']: round(r['Total Consumption Cost']/(r['Total Monthly kWh']), 2) for r in results}},
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

        # --- SAVE ---
        orig_name = Path(u_input.name).stem
        final_fn = f"{orig_name}_{region_choice}_{month_choice}.xlsx"
        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            pd.DataFrame(results).to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame(v_report).to_excel(writer, sheet_name="Executive_Financial_Report", index=False)
            for sn, df_s in excel_sheets.items(): df_s.to_excel(writer, sheet_name=sn, index=False)
        
        st.success(f"✅ Готово: {final_fn}")
        st.download_button("📥 Скачать результат", out.getvalue(), file_name=final_fn)
