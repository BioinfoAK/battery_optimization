import streamlit as st
import pandas as pd
import numpy as np
import holidays
from io import BytesIO

def check_password():
    """Returns True if the user had the correct password."""
    if "password_correct" not in st.session_state:
        st.text_input("Введите пароль для доступа к системе", type="password", on_change=password_entered, key="password")
        return False
    return st.session_state["password_correct"]

def password_entered():
    """Checks whether a password entered by the user is correct."""
    if st.session_state["password"] == "Secretb4t4re1!": 
        st.session_state["password_correct"] = True
        del st.session_state["password"]
    else:
        st.session_state["password_correct"] = False

if not check_password():
    st.stop()

# --- 1. INTERFACE SETUP ---
st.set_page_config(page_title="Расчет оптимального потребления", layout="wide")
st.title("🔋 Расчет оптимального потребления с помощью батареи")

st.sidebar.header("Параметры объекта")
region_choice = st.sidebar.radio("Выберите регион:", ["Samara", "Ulyanovsk", "Kaliningrad"])
month_choice = st.sidebar.selectbox("Выберите месяц:", ["nov25", "dec25"])

REGION_PATH = region_choice.lower()
REGIONAL_CONFIG_PATH = f"reference_data/{REGION_PATH}/tariffs/regional_config.xlsx"

try:
    df_reg_config = pd.read_excel(REGIONAL_CONFIG_PATH)
    match = df_reg_config[df_reg_config['month'].astype(str).str.lower() == month_choice.lower()]
    
    if not match.empty:
        default_gen = float(match.iloc[0]['Генераторная (покупная) мощность'])
        default_admin = float(match.iloc[0]['Ставка за управление'])
        default_net = float(match.iloc[0]['Ставка за содержание сетей'])
    else:
        st.sidebar.warning(f"⚠️ Месяц {month_choice} не найден")
        default_gen, default_admin, default_net = 0.0, 0.0, 0.0
except Exception as e:
    st.sidebar.error(f"❌ Не удалось загрузить конфиг: {e}")
    default_gen, default_admin, default_net = 0.0, 0.0, 0.0

st.sidebar.header("Параметры тарифов (Автозагрузка)")
gen_power_input = st.sidebar.number_input("Генераторная мощность, руб/Мвт", value=default_gen, format="%.2f")
gen_change_input = st.sidebar.number_input("Ставка за управление, руб/Мвт", value=default_admin, format="%.2f")
network_rate_input = st.sidebar.number_input("Ставка за содержание сетей, руб/Мвт", value=default_net, format="%.2f")

TOTAL_RATE_RUB_M_WH = gen_power_input + gen_change_input
NETWORK_CAPACITY_RATE = network_rate_input
KW_TO_MWH = 1 / 1000

MONTH_FILE = f"generating_hours_{month_choice.lower()}.xlsx"
REF_HOURS_PATH = f"reference_data/{REGION_PATH}/hours/{MONTH_FILE}"
ASSESS_FILE_PATH = f"reference_data/{REGION_PATH}/hours/assessment_hours.xlsx"
PRICE_FILE_NAME = f"hourly_tariffs_{month_choice.lower()}.xlsx"
REF_PRICE_PATH = f"reference_data/{REGION_PATH}/tariffs/{PRICE_FILE_NAME}"
MODULE_COUNTS = [5, 6, 7, 8] 
MODULE_KWH = 14.6
LOSS_FACTOR = 1.10
HR_COLS = [f"{h}.00-{h+1}.00" for h in range(24)]

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

# --- 2. CORE FUNCTIONS ---
def is_biz_day(dt, year):
    if dt.month == 11 and dt.day == 1: return True
    ru_holidays = holidays.Russia(years=[year])
    return not (dt.weekday() >= 5 or dt in ru_holidays)

def get_target_mask_from_ref(df_baseline, ref_path):
    try:
        df_ref = pd.read_excel(ref_path)
        df_ref.iloc[:, 0] = pd.to_datetime(df_ref.iloc[:, 0], dayfirst=True).dt.strftime('%Y-%m-%d')
        mask = []
        for _, row in df_baseline.iterrows():
            current_date_str = pd.to_datetime(row.iloc[0]).strftime('%Y-%m-%d')
            match = df_ref[df_ref.iloc[:, 0] == current_date_str]
            row_mask = {hr: False for hr in range(24)}
            if not match.empty:
                try:
                    hour_val = int(match.iloc[0, 1])
                    hour_idx = hour_val - 1
                    if 0 <= hour_idx <= 23: row_mask[hour_idx] = True
                except: pass
            mask.append(row_mask)
        return mask
    except Exception as e:
        st.error(f"Ошибка загрузки маски: {e}")
        return None

def get_assessment_hours(ref_path, month_col):
    try:
        df_assess = pd.read_excel(ref_path)
        if month_col in df_assess.columns:
            raw_hours = df_assess[month_col].dropna().unique().tolist()
            active_hours = []
            for h in raw_hours:
                h_str = str(h).strip()
                hour_val = int(h_str.split(":")[0]) if ":" in h_str else int(float(h_str))
                if 0 <= hour_val <= 23: active_hours.append(hour_val)
            return sorted(active_hours)
    except: pass
    return [7, 8, 9, 10, 15, 16, 17, 18, 19, 20]

def get_gen_peak_mean(df, mask_list, current_biz_mask):
    daily_peaks = []
    for i, (idx, row) in enumerate(df.iterrows()):
        if current_biz_mask[idx]:
            green_loads = [row[HR_COLS[hr]] for hr, is_target in mask_list[i].items() if is_target]
            if green_loads: daily_peaks.append(max(green_loads))
    return np.mean(daily_peaks) if daily_peaks else 0

def optimize_discharge(row_data, target_map, capacity, active_window):
    discharge = {hr: 0 for hr in range(24)}
    rem = capacity
    # 1. Discharge GREEN hours first
    for hr in range(24):
        if hr in active_window and target_map.get(hr, False):
            val = row_data[hr]
            actual = min(val, rem)
            discharge[hr] = actual
            rem -= actual
    # 2. Shave peaks in the rest of the window if battery has juice left
    if rem > 0.01 and active_window:
        shave_window = [h for h in active_window if discharge[h] < row_data[h]]
        while rem > 0.01 and shave_window:
            current_loads = {h: (row_data[h] - discharge[h]) for h in shave_window}
            if not current_loads or max(current_loads.values()) <= 0.01: break
            peak_hr = max(current_loads, key=current_loads.get)
            shave_amount = min(0.5, rem, current_loads[peak_hr])
            discharge[peak_hr] += shave_amount
            rem -= shave_amount
    return discharge

def calculate_success_rate(df_scenario, mask):
    total_green_hours = 0
    zeros_achieved = 0
    for i, row_mask in enumerate(mask):
        for hr, is_green in row_mask.items():
            if is_green:
                total_green_hours += 1
                if df_scenario.iloc[i][HR_COLS[hr]] <= 0.05: zeros_achieved += 1
    return round((zeros_achieved / total_green_hours) * 100, 2) if total_green_hours > 0 else 100.0

def calculate_network_charge_average(df_scenario, current_biz_mask, hr_cols, assess_hours):
    biz_data = df_scenario[current_biz_mask].copy()
    if biz_data.empty or not assess_hours: return 0.0, 0.0
    assessment_col_names = [hr_cols[h] for h in assess_hours]
    daily_max_series = biz_data[assessment_col_names].max(axis=1)
    avg_peak_kw = daily_max_series.mean()
    avg_peak_mw = avg_peak_kw / 1000 
    return round(avg_peak_mw * NETWORK_CAPACITY_RATE, 2), round(avg_peak_mw, 4)

def calculate_total_energy_cost(df_scenario, price_map, hour_columns):
    total_rubles = 0
    for idx, row in df_scenario.iterrows():
        day_num = pd.to_datetime(row.iloc[0]).day
        if day_num in price_map:
            day_data = price_map[day_num]
            day_cost = sum(row[hr] * (day_data[hr] / 1000) for hr in hour_columns)
            total_rubles += day_cost
    return round(total_rubles, 2)

# --- 3. EXECUTION ---
u_input = st.file_uploader("Выбрать файл с данными потребления объекта (xlsx)", type=["xlsx"])

try:
    df_prices = pd.read_excel(REF_PRICE_PATH)
    df_prices.iloc[:, 0] = df_prices.iloc[:, 0].astype(int)
    price_map = df_prices.set_index(df_prices.columns[0]).to_dict(orient='index')
except Exception as e:
    st.error(f"❌ Ошибка тарифов: {e}")
    st.stop()

if u_input:
    if st.button("🚀 Симулируем"):
        with st.spinner("Рассчитываем сценарии..."):
            df_raw = pd.read_excel(u_input)
            df_raw[HR_COLS] = df_raw[HR_COLS].astype(float)
            df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True)
            year = df_raw.iloc[0, 0].year
            total_kwh_baseline = df_raw[HR_COLS].sum().sum()
            
            biz_mask = df_raw.iloc[:, 0].apply(lambda x: is_biz_day(x, year))
            target_mask_list = get_target_mask_from_ref(df_raw, REF_HOURS_PATH)
            ALL_ASSESS = get_assessment_hours(ASSESS_FILE_PATH, month_choice)
            
            st.caption(f"📊 Часы оценки для {month_choice}: {ALL_ASSESS}")

            # Define windows for 2-cycle charge/discharge
            first_peak = min(ALL_ASSESS) if ALL_ASSESS else 7
            DYN_NIGHT_WINDOW = [h for h in range(0, first_peak)]
            morning_peaks = [h for h in ALL_ASSESS if h < 13]
            evening_peaks = [h for h in ALL_ASSESS if h >= 13]
            if morning_peaks and evening_peaks:
                DYN_GAP_WINDOW = [h for h in range(max(morning_peaks) + 1, min(evening_peaks))]
            else:
                DYN_GAP_WINDOW = [12, 13, 14]

            summary_results = []
            excel_data = {}

            # --- BASELINE ---
            net_charge_base, peak_mw_base = calculate_network_charge_average(df_raw, biz_mask, HR_COLS, ALL_ASSESS)
            gen_peak_base = get_gen_peak_mean(df_raw, target_mask_list, biz_mask)
            gen_cost_base = gen_peak_base * KW_TO_MWH * TOTAL_RATE_RUB_M_WH
            energy_cost_base = calculate_total_energy_cost(df_raw, price_map, HR_COLS)

            summary_results.append({
                "Setup": "ФАКТ", "Total Monthly kWh": round(total_kwh_baseline, 2),
                "Generating Peak (kW)": round(gen_peak_base, 4), "Avg Assessment Peak (MW)": peak_mw_base,
                "Generating cost": round(gen_cost_base, 2), "Max network charge": net_charge_base,
                "Total Consumption Cost": energy_cost_base, "GRAND TOTAL COST": round(gen_cost_base + net_charge_base + energy_cost_base, 2),
                "Success Rate (%)": calculate_success_rate(df_raw, target_mask_list)
            })
            excel_data["Baseline"] = df_raw

            # --- MODULES LOOP ---
            module_names = {5: "5_Modules 73kW", 6: "6_Modules 87,6kW", 7: "7_Modules 102,2kW", 8: "8_Modules 116,8kW"}
            
            for m in MODULE_COUNTS:
                cap = m * MODULE_KWH
                max_charge_pwr = cap * 0.5 
                df_sim = df_raw.copy()
                df_schedule = df_raw.copy()
                df_schedule[HR_COLS] = 0.0
                
                total_night_charge_vol = 0; total_gap_charge_vol = 0; days_count = 0 

                for idx, row in df_raw.iterrows():
                    if not biz_mask[idx]: continue
                    day_dt = pd.to_datetime(row.iloc[0])
                    day_num = day_dt.day
                    if day_num not in price_map: continue
                    days_count += 1
                    day_prices = price_map[day_num]
                    
                    # 1. Morning Discharge
                    morn_window = [h for h in range(0, 14)]
                    d_morn = optimize_discharge(row[HR_COLS].values, target_mask_list[idx], cap, morn_window)
                    
                    # 2. Evening Discharge
                    eve_window = [h for h in range(14, 24)]
                    row_after_morn = row[HR_COLS].values - np.array([d_morn[h] for h in range(24)])
                    d_eve = optimize_discharge(row_after_morn, target_mask_list[idx], cap, eve_window)
                    
                    net_flow = {h: (d_morn[h] + d_eve[h]) for h in range(24)}
                    
                    # 3. Charging Logic (Refill for morning and Refill for evening)
                    for window, vol_tracker in [(DYN_NIGHT_WINDOW, "night"), (DYN_GAP_WINDOW, "gap")]:
                        subset = {h: day_prices[HR_COLS[h]] for h in window}
                        cheapest_hours = sorted(subset, key=subset.get)[:2]
                        needed = cap * LOSS_FACTOR
                        for h in cheapest_hours:
                            charge = min(needed, max_charge_pwr)
                            net_flow[h] -= charge # Negative = Buying
                            needed -= charge
                            if vol_tracker == "night": total_night_charge_vol += charge
                            else: total_gap_charge_vol += charge

                    # Apply to DataFrames
                    for h in range(24):
                        flow = net_flow[h]
                        df_sim.at[idx, HR_COLS[h]] = max(0, row[HR_COLS[h]] - flow)
                        df_schedule.at[idx, HR_COLS[h]] = flow

                # --- SUMMARY METRICS ---
                m_net, m_peak_mw = calculate_network_charge_average(df_sim, biz_mask, HR_COLS, ALL_ASSESS)
                m_gen_p = get_gen_peak_mean(df_sim, target_mask_list, biz_mask)
                m_gen_c = m_gen_p * KW_TO_MWH * TOTAL_RATE_RUB_M_WH
                m_en_c = calculate_total_energy_cost(df_sim, price_map, HR_COLS)
                total_kwh_sim = df_sim[HR_COLS].sum().sum()
                
                summary_results.append({
                    "Setup": module_names[m], 
                    "Total Monthly kWh": round(total_kwh_sim, 2),
                    "Added from Battery": round(total_kwh_sim - total_kwh_baseline, 2),
                    "Generating Peak (kW)": round(m_gen_p, 4), 
                    "Avg Assessment Peak (MW)": m_peak_mw, 
                    "Night Charge (Daily Avg kWh)": round(total_night_charge_vol/max(1, days_count), 1),
                    "Gap Charge (Daily Avg kWh)": round(total_gap_charge_vol/max(1, days_count), 1),
                    "Generating cost": round(m_gen_c, 2), 
                    "Max network charge": m_net,
                    "Total Consumption Cost": m_en_c, 
                    "GRAND TOTAL COST": round(m_gen_c + m_net + m_en_c, 2),
                    "Success Rate (%)": calculate_success_rate(df_sim, target_mask_list),
                    "Total Energy Bought for Battery": round(total_night_charge_vol + total_gap_charge_vol, 2),
                    "Baseline kWh": round(total_kwh_baseline, 2),
                    "System Energy Loss": round(total_kwh_sim - total_kwh_baseline, 2), 
                    "Energy Cycle Efficiency": f"{round((total_kwh_baseline / total_kwh_sim) * 100, 1)}%" 
                })
                excel_data[f"{m}_Modules_Load"] = df_sim
                excel_data[f"{m}_Schedule"] = df_schedule

            # --- EXECUTIVE REPORT ---
            def get_weighted_avg_price(res_entry):
                if res_entry['Total Monthly kWh'] == 0: return 0
                return res_entry['Total Consumption Cost'] / (res_entry['Total Monthly kWh'] * KW_TO_MWH)

            v_report = [
                {"": "Потребление", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                {"": "Объем потребления, кВт×ч", **{res['Setup']: res['Total Monthly kWh'] for res in summary_results}},
                {"": "Генераторная (покупная) мощность, кВт", **{res['Setup']: res['Generating Peak (kW)'] for res in summary_results}},
                {"": "Сетевая мощность, кВт", **{res['Setup']: round(res['Avg Assessment Peak (MW)'] * 1000, 2) for res in summary_results}},
                {"": "", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                {"": "Тарифы", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                {"": "Средняя стоимость электроэнергии, руб/MВтч", **{res['Setup']: round(get_weighted_avg_price(res), 2) for res in summary_results}},
                {"": "Генераторная (покупная) мощность, руб/МВт", **{res['Setup']: round(TOTAL_RATE_RUB_M_WH, 2) for res in summary_results}},
                {"": "Ставка за содержание сетей, руб/МВт", **{res['Setup']: round(NETWORK_CAPACITY_RATE, 2) for res in summary_results}},
                {"": "", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                {"": "ИТОГО:", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                {"": "Стоимость электроэнергии, руб", **{res['Setup']: res['Total Consumption Cost'] for res in summary_results}},
                {"": "Стоимость генераторной, руб", **{res['Setup']: res['Generating cost'] for res in summary_results}},
                {"": "Стоимость сетевой, руб", **{res['Setup']: res['Max network charge'] for res in summary_results}},
                {"": "Стоимость без НДС 20%, руб", **{res['Setup']: res['GRAND TOTAL COST'] for res in summary_results}},
                {"": "Стоимость с НДС 20%, руб", **{res['Setup']: round(res['GRAND TOTAL COST'] * 1.20, 2) for res in summary_results}}
            ]

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                pd.DataFrame(summary_results).rename(columns=COLUMN_NAMES_RU).to_excel(writer, sheet_name="Summary", index=False)
                pd.DataFrame(v_report).to_excel(writer, sheet_name="Executive_Financial_Report", index=False)
                for sheet_name, df in excel_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            st.success("✅ Расчет завершен!")
            st.download_button("📥 Скачать результаты", output.getvalue(), file_name=f"results_{region_choice}_{month_choice}.xlsx")
