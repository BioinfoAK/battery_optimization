import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
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
# --- SIDEBAR INPUTS ---
st.sidebar.header("Параметры объекта")
region_choice = st.sidebar.radio("Выберите регион:", ["Samara", "Ulyanovsk", "Kaliningrad"])
month_choice = st.sidebar.selectbox("Выберите месяц:", ["nov25", "dec25"])

# 1. Path logic setup
REGION_PATH = region_choice.lower()
REGIONAL_CONFIG_PATH = f"reference_data/{REGION_PATH}/tariffs/regional_config.xlsx"

# 2. LOAD REGIONAL TARIFFS (Dynamic Loading)
try:
    df_reg_config = pd.read_excel(REGIONAL_CONFIG_PATH)
    match = df_reg_config[df_reg_config['month'].astype(str).str.lower() == month_choice.lower()]
    
    if not match.empty:
        default_gen = float(match.iloc[0]['Генераторная (покупная) мощность'])
        default_admin = float(match.iloc[0]['Ставка за управление'])
        default_net = float(match.iloc[0]['Ставка за содержание сетей'])
    else:
        st.sidebar.warning(f"⚠️ Месяц {month_choice} не найден в {REGIONAL_CONFIG_PATH}")
        default_gen, default_admin, default_net = 0.0, 0.0, 0.0
except Exception as e:
    st.sidebar.error(f"❌ Не удалось загрузить конфиг региона: {e}")
    default_gen, default_admin, default_net = 0.0, 0.0, 0.0

# 3. INTERACTIVE INPUTS (Using values from Excel as defaults)
st.sidebar.header("Параметры тарифов (Автозагрузка)")

gen_power_input = st.sidebar.number_input(
    "Генераторная (покупная) мощность, руб/Мвт", 
    value=default_gen, format="%.2f"
)
gen_change_input = st.sidebar.number_input(
    "Ставка за управление, руб/Мвт", 
    value=default_admin, format="%.2f"
)
network_rate_input = st.sidebar.number_input(
    "Ставка за содержание сетей, руб/Мвт", 
    value=default_net, format="%.2f"
)

# 4. FINAL CONSTANTS
TOTAL_RATE_RUB_M_WH = gen_power_input + gen_change_input
NETWORK_CAPACITY_RATE = network_rate_input
KW_TO_MWH = 1 / 1000

# Path logic for hours and prices
MONTH_FILE = f"generating_hours_{month_choice.lower()}.xlsx"
REF_HOURS_PATH = f"reference_data/{REGION_PATH}/hours/{MONTH_FILE}"
ASSESS_FILE_PATH = f"reference_data/{REGION_PATH}/hours/assessment_hours.xlsx"
PRICE_FILE_NAME = f"hourly_tariffs_{month_choice.lower()}.xlsx"
REF_PRICE_PATH = f"reference_data/{REGION_PATH}/tariffs/{PRICE_FILE_NAME}"
MODULE_COUNTS = [5, 6, 7, 8] 
MODULE_KWH = 14.6
LOSS_FACTOR = 1.10
# Column headers format
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

# --- 3. CORE FUNCTIONS (Copied Exactly) ---
def is_biz_day(dt, year):
    if dt.month == 11 and dt.day == 1: return True
    if df.month == 12 and dt.day == 31: return True
    ru_holidays = holidays.Russia(years=[year])
    return not (dt.weekday() >= 5 or dt in ru_holidays)

def get_target_mask_from_ref(df_baseline, ref_path):
    try:
        df_ref = pd.read_excel(ref_path)
        # Convert both to date-only to avoid timestamp mismatches
        df_ref.iloc[:, 0] = pd.to_datetime(df_ref.iloc[:, 0], dayfirst=True).dt.date
        
        mask = []
        for _, row in df_baseline.iterrows():
            current_date = pd.to_datetime(row.iloc[0]).date()
            match = df_ref[df_ref.iloc[:, 0] == current_date]
            
            row_mask = {hr: False for hr in range(24)}
            if not match.empty:
                hour_idx = int(match.iloc[0, 1]) - 1
                if 0 <= hour_idx <= 23:
                    row_mask[hour_idx] = True
            mask.append(row_mask)
        return mask
    except Exception as e:
        st.error(f"Ошибка загрузки маски: {e}")
        return None

def get_assessment_hours(ref_path, month_col):
    """
    Читает assessment_hours.xlsx и возвращает список активных часов.
    Обрабатывает форматы '7', 7, '07:00' или '07:00-08:00'.
    """
    try:
        df_assess = pd.read_excel(ref_path)
        
        if month_col in df_assess.columns:
            raw_hours = df_assess[month_col].dropna().unique().tolist()
            active_hours = []
            
            for h in raw_hours:
                h_str = str(h).strip()
                # Если формат '07:00-08:00', берем первые два символа '07'
                if ":" in h_str:
                    hour_val = int(h_str.split(":")[0])
                # Если это просто число или строка с числом
                else:
                    hour_val = int(float(h_str))
                
                if 0 <= hour_val <= 23:
                    active_hours.append(hour_val)
            
            return sorted(active_hours)
        else:
            st.error(f"Колонку '{month_col}' не нашли в assessment_hours.xlsx")
            return [7, 8, 9, 10, 15, 16, 17, 18, 19, 20] # Фоллбэк
    except Exception as e:
        st.error(f"Ошибка загрузки часов оценки: {e}")
        return [7, 8, 9, 10, 15, 16, 17, 18, 19, 20]
        
def get_gen_peak_mean(df, mask_list, current_biz_mask):
    daily_peaks = []
    # Use the HR_COLS list we defined at the top
    for i, (idx, row) in enumerate(df.iterrows()):
        if current_biz_mask[idx]:
            # Look up the specific hour identified in the mask
            green_loads = [row[HR_COLS[hr]] for hr, is_target in mask_list[i].items() if is_target]
            if green_loads: 
                daily_peaks.append(max(green_loads))
    return np.mean(daily_peaks) if daily_peaks else 0

def optimize_discharge(row_data, target_map, capacity, active_window):
    discharge = {hr: 0 for hr in range(24)}
    rem = capacity
    max_out = capacity 
    
    # 1. Mandatory Target Hour (ONLY if it falls within the current window)
    for hr in range(24):
        if hr in active_window and target_map.get(hr, False):
            val = row_data[hr]
            actual = min(val, rem, max_out)
            discharge[hr] = actual
            rem -= actual
            
    # 2. Aggressive Shaving (ONLY for hours in the active window)
    if rem > 0.1 and active_window:
        while rem > 0.01:
            current_loads = {h: (row_data[h] - discharge[h]) for h in active_window}
            if not current_loads or max(current_loads.values()) <= 0:
                break
            
            peak_hr = max(current_loads, key=current_loads.get)
            can_add = max_out - discharge[peak_hr]
            
            if can_add <= 0:
                # This hour is at physical limit, temporarily ignore it
                active_window = [h for h in active_window if h != peak_hr]
                continue

            shave_amount = min(0.1, rem, current_loads[peak_hr], can_add)
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
                # Use string header to check if battery covered the load
                if df_scenario.iloc[i][HR_COLS[hr]] <= 0.05: 
                    zeros_achieved += 1
    return round((zeros_achieved / total_green_hours) * 100, 2) if total_green_hours > 0 else 100.0

def calculate_network_charge_average(df_scenario, current_biz_mask, hr_cols):
    """
    Calculates the average of daily maximums during assessment hours.
    """
    # 1. Filter only for Business Days
    biz_data = df_scenario[current_biz_mask].copy()
    
    if biz_data.empty or not ALL_ASSESS:
        return 0.0, 0.0

    # 2. Map ALL_ASSESS (indices) to your string column names (e.g., '7.00-8.00')
    assessment_col_names = [hr_cols[h] for h in ALL_ASSESS]
    
    # 3. Calculate the MAX value for EACH day within those specific columns
    # axis=1 tells pandas to find the max across the columns for every row
    daily_max_series = biz_data[assessment_col_names].max(axis=1)
    
    # 4. Calculate the MEAN (average) of those daily peaks
    avg_peak_kw = daily_max_series.mean()
    
    # 5. Convert to MW for the financial calculation
    avg_peak_mw = avg_peak_kw / 1000 
    
    # 6. Calculate the final RUB charge
    total_network_charge = round(avg_peak_mw * NETWORK_CAPACITY_RATE, 2)
    
    return total_network_charge, round(avg_peak_mw, 4)
    
def calculate_total_energy_cost(df_scenario, price_map, hour_columns):
    total_rubles = 0
    for idx, row in df_scenario.iterrows():
        day_num = row.iloc[0].day
        if day_num in price_map:
            day_data = price_map[day_num]
            # Sum up (kWh * (Price_per_MWh / 1000)) for all 24 hours
            day_cost = sum(row[hr] * (day_data[hr] / 1000) for hr in hour_columns)
            total_rubles += day_cost
    return round(total_rubles, 2)
    
# --- 4. STREAMLIT UPLOAD LOGIC ---
u_input = st.file_uploader("Выбрать файл с данными потребления объекта (xlsx)", type=["xlsx"])
try:
    df_prices = pd.read_excel(REF_PRICE_PATH)
    # Ensure the first column (Days) is integer type
    df_prices.iloc[:, 0] = df_prices.iloc[:, 0].astype(int)
    # Map the day column to headers for easy lookup later
    price_map = df_prices.set_index(df_prices.columns[0]).to_dict(orient='index')
except Exception as e:
    st.error(f"❌ Ошибка загрузки тарифов: {e}")
    st.stop()
# Note: We are using u_input for your local data, 
# but we pull the mask automatically from GitHub based on your sidebar buttons.
if u_input:
    if st.button("🚀 Симулируем"):
        with st.spinner("Рассчитываем сценарии..."):
            # 1. Load Data
            df_raw = pd.read_excel(u_input)
            # Ensure the first column is datetime
            df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True)
            year = df_raw.iloc[0, 0].year
            
            # 2. Setup Masks and Columns
            biz_mask = df_raw.iloc[:, 0].apply(lambda x: is_biz_day(x, year))
        
            # Using the path constructed from your Region/Month buttons
            target_mask_list = get_target_mask_from_ref(df_raw, REF_HOURS_PATH)
            
            if target_mask_list is None:
                st.stop() 
            # --- 1. PRE-CALCULATION SETUP ---
            hr_cols = HR_COLS 
            ALL_ASSESS = get_assessment_hours(ASSESS_FILE_PATH, month_choice)
            st.caption(f"📊 Часы оценки (Peak Shaving) для {month_choice}: {ALL_ASSESS}")

            # Define windows
            first_peak = min(ALL_ASSESS) if ALL_ASSESS else 7
            DYN_NIGHT_WINDOW = [h for h in range(0, first_peak)]
            
            morning_peaks = [h for h in ALL_ASSESS if h < 13]
            evening_peaks = [h for h in ALL_ASSESS if h >= 13]
            if morning_peaks and evening_peaks:
                DYN_GAP_WINDOW = [h for h in range(max(morning_peaks) + 1, min(evening_peaks)) if h not in ALL_ASSESS]
            else:
                DYN_GAP_WINDOW = [h for h in range(12, 15) if h not in ALL_ASSESS]

            summary_results = []
            excel_data = {}
            # --- 2. BASELINE (ФАКТ) ---

            net_charge_base, peak_mw_base = calculate_network_charge_average(df_raw, biz_mask, hr_cols)
            gen_peak_base = get_gen_peak_mean(df_raw, target_mask_list, biz_mask)
            gen_cost_base = gen_peak_base * KW_TO_MWH * TOTAL_RATE_RUB_M_WH
            
            # FIX: Calculate the cost FIRST, then assign it to the dictionary
            energy_cost_base = calculate_total_energy_cost(df_raw, price_map, hr_cols)
            
            summary_results.append({
                "Setup": "ФАКТ", 
                "Total Monthly kWh": round(df_raw[hr_cols].sum().sum(), 2),
                "Generating Peak (kW)": round(gen_peak_base, 4), 
                "Avg Assessment Peak (MW)": peak_mw_base*1000,
                "Night Charge (Daily Avg kWh)": 0,
                "Gap Charge (Daily Avg kWh)": 0,
                "Generating cost": round(gen_cost_base, 2), 
                "Max network charge": net_charge_base,
                "Total Consumption Cost": energy_cost_base, 
                "GRAND TOTAL COST": round(gen_cost_base + net_charge_base + energy_cost_base, 2),
                "Success Rate (%)": calculate_success_rate(df_raw, target_mask_list)
            })
            excel_data["Baseline"] = df_raw
# --- 2. MODULES LOOP ---
            module_names = {5: "5_Modules 73kW", 6: "6_Modules 87,6kW", 7: "7_Modules 102,2kW", 8: "8_Modules 116,8kW"}
            
            for m in MODULE_COUNTS:
                cap = m * MODULE_KWH
                max_charge_pwr = cap * 0.5 
                df_sim = df_raw.copy()
                df_schedule = df_raw.copy()
                df_schedule[hr_cols] = 0.0
                
                # ADD THESE INITIALIZERS HERE:
                total_night_charge_vol = 0
                total_gap_charge_vol = 0
                days_count = 0 

                # CORRECTED INDENTATION: This loop must be inside the module loop
                for idx, row in df_sim.iterrows():
                    if not biz_mask[idx]: 
                        continue
                    
                    days_count += 1
                    day_dt = pd.to_datetime(row.iloc[0])
                    day_num = day_dt.day
                    
                    if day_num not in price_map: 
                        continue
                        
                    day_prices = price_map[day_num]
                    
                    # 1. DISCHARGE (Peak Shaving)
                    morning_hrs = [h for h in ALL_ASSESS if h < (min(DYN_GAP_WINDOW) if DYN_GAP_WINDOW else 24)]
                    evening_hrs = [h for h in ALL_ASSESS if h not in morning_hrs]
                    
                    d_morn = optimize_discharge(row[hr_cols].values, target_mask_list[idx], cap, morning_hrs)
                    d_eve = optimize_discharge(row[hr_cols].values, target_mask_list[idx], cap, evening_hrs)
                    d_total = {h: d_morn[h] + d_eve[h] for h in range(24)}

                    # 2. CHARGE: PICK 2 CHEAPEST HOURS
                    net_flow = {h: d_total[h] for h in range(24)}
                    
                    # Night: Cheapest 2 in DYN_NIGHT_WINDOW
                    night_price_subset = {h: day_prices[HR_COLS[h]] for h in DYN_NIGHT_WINDOW}
                    cheapest_night = sorted(night_price_subset, key=night_price_subset.get)[:2]
                    needed_night = cap * LOSS_FACTOR
                    for h in cheapest_night:
                        charge = min(needed_night, max_charge_pwr)
                        net_flow[h] -= charge
                        needed_night -= charge
                        total_night_charge_vol += charge

                    # Gap: Cheapest 2 in DYN_GAP_WINDOW
                    gap_price_subset = {h: day_prices[HR_COLS[h]] for h in DYN_GAP_WINDOW}
                    cheapest_gap = sorted(gap_price_subset, key=gap_price_subset.get)[:2]
                    needed_gap = cap * LOSS_FACTOR
                    for h in cheapest_gap:
                        charge = min(needed_gap, max_charge_pwr)
                        net_flow[h] -= charge
                        needed_gap -= charge
                        total_gap_charge_vol += charge

                    # 3. APPLY TO DATAFRAME
                    for h in range(24):
                        df_schedule.at[idx, hr_cols[h]] = round(net_flow[h], 4)
                        df_sim.at[idx, hr_cols[h]] = max(0, row[hr_cols[h]] - net_flow[h])

                # 4. FINAL CALCS FOR MODULE
                m_net, m_peak_mw = calculate_network_charge_average(df_sim, biz_mask, hr_cols)
                m_gen_p = get_gen_peak_mean(df_sim, target_mask_list, biz_mask)
                m_gen_c = m_gen_p * KW_TO_MWH * TOTAL_RATE_RUB_M_WH
                m_en_c = calculate_total_energy_cost(df_sim, price_map, hr_cols)

                summary_results.append({
                    "Setup": module_names[m], 
                    "Total Monthly kWh": round(df_sim[hr_cols].sum().sum(), 2),
                    "Generating Peak (kW)": round(m_gen_p, 4), 
                    "Avg Assessment Peak (MW)": m_peak_mw*1000,
                    "Night Charge (Daily Avg kWh)": round(total_night_charge_vol/max(1, days_count), 1),
                    "Gap Charge (Daily Avg kWh)": round(total_gap_charge_vol/max(1, days_count), 1),
                    "Generating cost": round(m_gen_c, 2), 
                    "Max network charge": m_net,
                    "Total Consumption Cost": m_en_c, 
                    "GRAND TOTAL COST": round(m_gen_c + m_net + m_en_c, 2),
                    "Success Rate (%)": calculate_success_rate(df_sim, target_mask_list)
                })
                excel_data[f"{m}_Modules_Load"] = df_sim
                excel_data[f"{m}_Schedule"] = df_schedule

# --- 3. EXECUTIVE REPORT ---
            def get_weighted_avg_price(res_entry):
                if res_entry['Total Monthly kWh'] == 0: return 0
                # res_entry['Total Consumption Cost'] is RUB
                # res_entry['Total Monthly kWh'] * KW_TO_MWH is MWh
                # Result is RUB / MWh
                return res_entry['Total Consumption Cost'] / (res_entry['Total Monthly kWh'] * KW_TO_MWH)

            v_report = [
                {"": "Потребление", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                {"": "Объем потребления, кВт×ч", **{res['Setup']: res['Total Monthly kWh'] for res in summary_results}},
                {"": "Генераторная (покупная) мощность, кВт", **{res['Setup']: res['Generating Peak (kW)'] for res in summary_results}},
                {"": "Сетевая мощность, кВт", **{res['Setup']: round(res['Avg Assessment Peak (MW)'] * 1000, 2) for res in summary_results}},
                {"": "", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                {"": "Тарифы", "ФАКТ": "", "5_Modules 73kW": "", "6_Modules 87,6kW": "", "7_Modules 102,2kW": "", "8_Modules 116,8kW": ""},
                # FIXED: Removed extra /1000 since function already returns RUB/MWh
                {"": "Средняя стоимость электроэнергии, руб/MВтч", **{res['Setup']: round(get_weighted_avg_price(res)/1000, 2) for res in summary_results}},
                # FIXED: Removed /1000 to keep it as RUB/MWh (standard reporting)
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
            base_filename = u_input.name.split('.')[0]
            dynamic_name = f"{base_filename}_{region_choice}_{month_choice}.xlsx"

            
# --- 2.5 VALIDATION DISPLAY ---
            st.subheader("📊 Анализ циклов заряда")
            cols = st.columns(len(MODULE_COUNTS))
            for i, m in enumerate(MODULE_COUNTS):
                res = summary_results[i+1] # +1 to skip Baseline
                with cols[i]:
                    st.metric(f"Конфиг {m}", f"{res['Night Charge (Daily Avg kWh)']} кВтч")
                    st.caption("Заряд Ночь + Заряд Перерыв")
                    if res['Night Charge (Daily Avg kWh)'] > (cap * 0.5 * len(DYN_NIGHT_WINDOW)):
                        st.warning("⚠️ Ограничение мощности заряда достигнуто")
            st.success("✅ Готово!")
            st.download_button(
                label="📥 Скачать результаты Excel",
                data=output.getvalue(),
                file_name=dynamic_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("Для начала расчета, выберите файлы.")

