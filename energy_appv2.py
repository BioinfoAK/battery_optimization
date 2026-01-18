import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
import holidays
from io import BytesIO

import streamlit as st

def check_password():
    """Returns True if the user had the correct password."""
    if "password_correct" not in st.session_state:
        st.text_input("Введите пароль для доступа к системе", type="password", on_change=password_entered, key="password")
        return False
    return st.session_state["password_correct"]

def password_entered():
    """Checks whether a password entered by the user is correct."""
    if st.session_state["password"] == "Secretb4t4re1!": # Set your password here
        st.session_state["password_correct"] = True
        del st.session_state["password"]  # don't store password
    else:
        st.session_state["password_correct"] = False

if not check_password():
    st.stop()  # Do not run the rest of the app



# --- 1. INTERFACE SETUP ---
st.set_page_config(page_title="Расчет оптимального потребления", layout="wide")
st.title("🔋 Расчет оптимального потребления с помощью батареи")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Финансовые показатели")

# We set your current hardcoded values as the 'value' (the default)
gen_power_input = st.sidebar.number_input(
    "Generating Power Rate (Ставка за мощность)", 
    value=1132614.35, 
    format="%.2f"
)

gen_change_input = st.sidebar.number_input(
    "Generating Change Rate (Ставка за энергию)", 
    value=1317.3, 
    format="%.2f"
)

network_rate_input = st.sidebar.number_input(
    "Network Capacity Rate (Сетевой тариф)", 
    value=2487916.6, 
    format="%.2f"
)

# Now, we use these inputs to calculate our constants
TOTAL_RATE_RUB_M_WH = gen_power_input + gen_change_input
NETWORK_CAPACITY_RATE = network_rate_input
KW_TO_MWH = 1 / 1000
# Internal Constants
TOTAL_RATE_RUB_M_WH = generating_power + generating_change
NETWORK_CAPACITY_RATE = network_rate
KW_TO_MWH = 1 / 1000

MODULE_COUNTS = [5, 6, 7, 8]
MODULE_KWH = 14.6
LOSS_FACTOR = 1.10
ALL_ASSESS = [7, 8, 9, 10, 15, 16, 17, 18, 19, 20]
NIGHT_WINDOW = [0, 1, 2, 3, 4, 5, 6]
GAP_WINDOW = [11, 12, 13, 14]

COLUMN_NAMES_RU = {
    "Setup": "Конфигурация",
    "Total Monthly kWh": "Объем потребления, кВт×ч",
    "Generating Peak (kW)": "Генераторная (покупная) мощность, кВт",
    "Avg Assessment Peak (MW)": "Сетевой пик, кВт",
    "Generating cost": "Затраты на генераторную мощность, руб",
    "Max network charge": "Затраты на сетевую мощность, руб",
    "Total Consumption Cost": "Общие затраты на энергию, руб",
    "GRAND TOTAL COST": "Итог без НДС, руб",
    "Success Rate (%)": "Эффективность минимизации генераторной (%)"
}

# --- 3. CORE FUNCTIONS (Copied Exactly) ---
def is_biz_day(dt, year):
    if dt.month == 11 and dt.day == 1: return True
    ru_holidays = holidays.Russia(years=[year])
    return not (dt.weekday() >= 5 or dt in ru_holidays)

def get_target_mask_from_upload(uploaded_file):
    wb = openpyxl.load_workbook(uploaded_file, data_only=True)
    ws = wb.active
    mask = []
    for row in ws.iter_rows(min_row=2):
        row_mask = {hr: (row[hr + 2].fill and hasattr(row[hr + 2].fill, 'start_color') and 
                    row[hr + 2].fill.start_color.index not in ['00000000', 'FFFFFFFF', 0]) 
                    for hr in range(24)}
        mask.append(row_mask)
    return mask

def get_gen_peak_mean(df, mask_list, current_biz_mask):
    daily_peaks = []
    for i, (idx, row) in enumerate(df.iterrows()):
        if current_biz_mask[idx]:
            green_loads = [row.iloc[hr + 2] for hr, is_target in mask_list[i].items() if is_target]
            if green_loads: daily_peaks.append(max(green_loads))
    return np.mean(daily_peaks) if daily_peaks else 0

def optimize_discharge(row_data, target_map, capacity):
    discharge = {hr: 0 for hr in range(24)}
    rem = capacity
    for hr in range(24):
        if target_map.get(hr, False):
            val = row_data[hr]
            actual = min(val, rem)
            discharge[hr] = actual
            rem -= actual
    if rem > 0.1:
        while rem > 0.1:
            curr_loads = {h: (row_data[h] - discharge[h]) for h in ALL_ASSESS}
            peak_hr = max(curr_loads, key=curr_loads.get)
            if curr_loads[peak_hr] <= 0: break
            shave = min(0.5, rem, curr_loads[peak_hr])
            discharge[peak_hr] += shave
            rem -= shave
    return discharge

def calculate_success_rate(df_scenario, mask):
    total_green_hours = 0
    zeros_achieved = 0
    for i, row_mask in enumerate(mask):
        for hr, is_green in row_mask.items():
            if is_green:
                total_green_hours += 1
                if df_scenario.iloc[i, hr + 2] <= 0.05: zeros_achieved += 1
    return round((zeros_achieved / total_green_hours) * 100, 2) if total_green_hours > 0 else 100.0

def calculate_network_charge_average(df_scenario, current_biz_mask, hr_cols):
    biz_data = df_scenario[current_biz_mask]
    assessment_cols = [hr_cols[h] for h in ALL_ASSESS]
    avg_peak_kw = biz_data[assessment_cols].max(axis=1).mean()
    avg_peak_mw = avg_peak_kw * KW_TO_MWH
    return round(avg_peak_mw * NETWORK_CAPACITY_RATE, 2), round(avg_peak_mw, 4)

def calculate_total_energy_cost(df_scenario, df_prices, hour_columns):
    total_rubles = 0
    for idx, row in df_scenario.iterrows():
        day_num = row.iloc[0].day
        try:
            price_row = df_prices[df_prices.iloc[:, 0] == day_num].iloc[0, 1:].values
            total_rubles += np.sum((row[hour_columns].values / 1000) * price_row)
        except: continue
    return round(total_rubles, 2)

# --- 4. STREAMLIT UPLOAD LOGIC ---
u_input = st.file_uploader("Выбрать файл с данными потребления объекта (xlsx)", type=["xlsx"])
u_price = st.file_uploader("Выбрать файл с почасовыми тарифами (xlsx)", type=["xlsx"])

if u_input and u_price:
    if st.button("🚀 Симуляция в процессе"):
        with st.spinner("Рассчитываем сценарии..."):
            # Load Data
            df_raw = pd.read_excel(u_input)
            df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True)
            year = df_raw.iloc[0, 0].year
            biz_mask = df_raw.iloc[:, 0].apply(lambda x: is_biz_day(x, year))
            target_mask_list = get_target_mask_from_upload(u_input)
            df_prices = pd.read_excel(u_price)
            hr_cols = df_raw.columns[2:26]
            
            summary_results = []
            excel_data = {} # Store DFs for sheets

            # --- 1. BASELINE ---
            net_charge_base, peak_mw_base = calculate_network_charge_average(df_raw, biz_mask, hr_cols)
            gen_peak_base = get_gen_peak_mean(df_raw, target_mask_list, biz_mask)
            gen_cost_base = gen_peak_base * KW_TO_MWH * TOTAL_RATE_RUB_M_WH
            energy_cost_base = calculate_total_energy_cost(df_raw, df_prices, hr_cols)
            
            summary_results.append({
                "Setup": "ФАКТ", "Total Monthly kWh": round(df_raw[hr_cols].sum().sum(), 2),
                "Generating Peak (kW)": round(gen_peak_base, 4), "Avg Assessment Peak (MW)": peak_mw_base*1000,
                "Generating cost": round(gen_cost_base, 2), "Max network charge": net_charge_base,
                "Total Consumption Cost": energy_cost_base, "GRAND TOTAL COST": round(gen_cost_base + net_charge_base + energy_cost_base, 2),
                "Success Rate (%)": calculate_success_rate(df_raw, target_mask_list)
            })
            excel_data["Baseline"] = df_raw

            # --- 2. MODULES LOOP ---
            module_names = {5: "5_Modules 73kW", 6: "6_Modules 87,6kW", 7: "7_Modules 102,2kW", 8: "8_Modules 116,8kW"}
            for m in MODULE_COUNTS:
                cap = m * MODULE_KWH
                df_sim = df_raw.copy()
                df_schedule = df_raw.copy()
                df_schedule[hr_cols] = 0.0
                
                for idx, row in df_sim.iterrows():
                    if not biz_mask[idx]: continue
                    price_row = df_prices[df_prices.iloc[:,0] == row.iloc[0].day].iloc[0, 1:].values
                    d_map = optimize_discharge(row[hr_cols].values, target_mask_list[idx], cap)
                    
                    c_night = sorted(NIGHT_WINDOW, key=lambda h: price_row[h])[:2]
                    c_gap = sorted(GAP_WINDOW, key=lambda h: price_row[h])[:2]
                    v_night = sum(d_map[h] for h in range(7, 11)) * LOSS_FACTOR
                    v_gap = sum(d_map[h] for h in range(15, 21)) * LOSS_FACTOR
                    
                    for h in range(24):
                        charge_val = (v_night/2 if h in c_night else 0) + (v_gap/2 if h in c_gap else 0)
                        net_flow = d_map[h] - charge_val
                        df_schedule.iloc[idx, h + 2] = round(net_flow, 4)
                        df_sim.iloc[idx, h + 2] = max(0, row.iloc[h + 2] - net_flow)

                m_net, m_peak_mw = calculate_network_charge_average(df_sim, biz_mask, hr_cols)
                m_gen_p = get_gen_peak_mean(df_sim, target_mask_list, biz_mask)
                m_gen_c = m_gen_p * KW_TO_MWH * TOTAL_RATE_RUB_M_WH
                m_en_c = calculate_total_energy_cost(df_sim, df_prices, hr_cols)

                summary_results.append({
                    "Setup": module_names[m], "Total Monthly kWh": round(df_sim[hr_cols].sum().sum(), 2),
                    "Generating Peak (kW)": round(m_gen_p, 4), "Avg Assessment Peak (MW)": m_peak_mw*1000,
                    "Generating cost": round(m_gen_c, 2), "Max network charge": m_net,
                    "Total Consumption Cost": m_en_c, "GRAND TOTAL COST": round(m_gen_c + m_net + m_en_c, 2),
                    "Success Rate (%)": calculate_success_rate(df_sim, target_mask_list)
                })
                excel_data[f"{m}_Modules_Load"] = df_sim
                excel_data[f"{m}_Schedule"] = df_schedule

            # --- 3. EXECUTIVE REPORT (Copied Exactly) ---
            def get_weighted_avg_price(res_entry):
                if res_entry['Total Monthly kWh'] == 0: return 0
                return res_entry['Total Consumption Cost'] / (res_entry['Total Monthly kWh'] * KW_TO_MWH)

            v_report = [
                {"": "Потребление", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                {"": "Объем потребления, кВт×ч", **{res['Setup']: res['Total Monthly kWh'] for res in summary_results}},
                {"": "Генераторная (покупная) мощность, кВт", **{res['Setup']: res['Generating Peak (kW)'] for res in summary_results}},
                {"": "Сетевая мощность, кВт", **{res['Setup']: round(res['Avg Assessment Peak (MW)'], 2) for res in summary_results}},
                {"": "", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                {"": "Тарифы", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                {"": "Средняя стоимость электроэнергии, руб/MВтч", **{res['Setup']: round(get_weighted_avg_price(res)/1000, 2) for res in summary_results}},
                {"": "Генераторная (покупная) мощность", **{res['Setup']: round(TOTAL_RATE_RUB_M_WH/1000, 2) for res in summary_results}},
                {"": "Ставка за содержание сетей", **{res['Setup']: round(NETWORK_CAPACITY_RATE/1000, 2) for res in summary_results}},
                {"": "", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                {"": "ИТОГО:", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                {"": "Стоимость электроэнергии", **{res['Setup']: res['Total Consumption Cost'] for res in summary_results}},
                {"": "Стоимость генераторной", **{res['Setup']: res['Generating cost'] for res in summary_results}},
                {"": "Стоимость сетевой", **{res['Setup']: res['Max network charge'] for res in summary_results}},
                {"": "Стоимость без НДС 20%", **{res['Setup']: res['GRAND TOTAL COST'] for res in summary_results}},
                {"": "Стоимость с НДС 20%", **{res['Setup']: round(res['GRAND TOTAL COST'] * 1.20, 2) for res in summary_results}}
            ]

            # --- 4. EXCEL EXPORT ---
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Summary Sheet
                pd.DataFrame(summary_results).rename(columns=COLUMN_NAMES_RU).to_excel(writer, sheet_name="Summary", index=False)
                # Executive Report Sheet
                pd.DataFrame(v_report).to_excel(writer, sheet_name="Executive_Financial_Report", index=False)
                # All calculation sheets
                for sheet_name, df in excel_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            st.success("✅ Готово!")
            st.download_button(
                label="📥 Скачать результаты Excel",
                data=output.getvalue(),
                file_name="Battery_Final_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("Для начала расчета, выберите файлы.")
