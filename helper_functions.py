import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklean.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error, r2_score

# Extract data form excel spreadsheets based on date and fix how it was originally stored
def extract_data(cs2_dates:list, foulder_name:str, sheet_name:str) -> pd.DataFrame:
    df_subcycle = pd.DataFrame()
    c = 1
    for n in cs2_dates:
        filename = foulder_name + n + '.xlsx'
        df_temp = pd.read_excel(filename, sheet_name=sheet_name) 
        cycles = df_temp.Cycle_Index.unique()
        df_add = pd.DataFrame()
        for x in cycles:
            df_temp_short = df_temp[df_temp['Cycle_Index']==x] 
            df_temp_short['Cycle_Index'] = c
            charge_info = df_temp_short[df_temp_short['Step_Index'].isin([2, 4])]
            discharge_info = df_temp_short[df_temp_short['Step_Index'] == 7]
            if (df_temp_short.shape[0] <= 400) and (df_temp_short.shape[0] >= 275) and (charge_info.shape[0] >= 50) and (discharge_info.shape[0] >= 50):
                if x != 1:
                    df_temp_short_previous_cycle = df_temp[df_temp['Cycle_Index']==x-1] 
                    charge_cap_max = max(df_temp_short_previous_cycle['Charge_Capacity(Ah)']) 
                    discharge_cap_max = max(df_temp_short_previous_cycle['Discharge_Capacity(Ah)']) 
                    charge_energy_max = max(df_temp_short_previous_cycle['Charge_Energy(Wh)']) 
                    discharge_energy_max = max(df_temp_short_previous_cycle['Discharge_Energy(Wh)']) 
                    test_time_max = max(df_temp_short_previous_cycle['Test_Time(s)']) 
                    df_temp_short['Charge_Capacity(Ah)'] = df_temp_short['Charge_Capacity(Ah)'] - charge_cap_max 
                    df_temp_short['Discharge_Capacity(Ah)'] = df_temp_short['Discharge_Capacity(Ah)'] - discharge_cap_max
                    df_temp_short['Charge_Energy(Wh)'] = df_temp_short['Charge_Energy(Wh)'] - charge_energy_max
                    df_temp_short [ 'Discharge_Energy(Wh)'] = df_temp_short['Discharge_Energy(Wh)'] - discharge_energy_max
                    df_temp_short['Test_Time(s)'] = df_temp_short['Test_Time(s)'] - test_time_max
                df_add = pd.concat([df_add, df_temp_short])
                c += 1
        df_subcycle = pd.concat([df_subcycle, df_add]) 
    return df_subcycle

# Extract the Cycling data from the Subcycle data
def extract_cycle_data(df_subcycle:pd.DataFrame, cell_num:int) -> pd.DataFrame:
    cycles = df_subcycle.Cycle_Index.unique()
    df = pd.DataFrame(columns = ['cell_number', 'cycle', 'test_time', 'current', 'voltage', 'charge_capacity', 'discharge_capacity', 'charge_energy', 'discharge_energy', 'internal_resistance', 'charge_time', 'discharge_time'])
    for c in cycles:
        df_temp = df_subcycle[df_subcycle['Cycle_Index']==c]
        try:
            charge_time = max(df_temp[df_temp['Step_Index'].isin([2, 4])]['Step_Time(s)'])
            discharge_time = max(df_temp[df_temp['Step_Index'] == 7]['Step_Time(s)']) - charge_time
        except:
            print(c)
            continue
        last_row = df_temp.iloc[-1]
        df.loc[len(df)] = [cell_num, c, last_row['Test_Time(s)'], max(df_temp['Current(A)']), last_row['Voltage(V)'], last_row['Charge_Capacity(Ah)'], last_row['Discharge_Capacity(Ah)'], last_row['Charge_Energy(Wh)'], last_row['Discharge_Energy(Wh)'], last_row['Internal_Resistance(Ohm)'], charge_time, discharge_time]
    return df

def my_diff(x):
    middle = int(float(len(x))/2-0.5)
    diffs = abs(x.iloc[middle] - x)

    return np.median(diffs)

# Remove unusual cycling data measurements
def filter_outliers_local(df: pd.DataFrame, cols: list, std_window: int, diff_window: int, iqr_cut: int) -> pd.DataFrame:
    df_cleaned = []
    rolling_data = df.copy()
    for count, col in enumerate(cols):
        rolling_sd = rolling_data[col].rolling(std_window, min_periods=1, center=True).std()
        rolling_diff = rolling_sd.rolling(diff_window, min_periods=1, center=True).apply(my_diff)
        rolling_data["ave_diff"] = rolling_diff
        q3rd, q1st = np.nanpercentile(rolling_data["ave_diff"], [75, 25])
        iqr = q3rd-q1st
        clean = rolling_data[~(rolling_data["ave_diff"] > q3rd + iqr_cut * iqr)] 
        clean = clean.loc[:, ~clean.columns.isin(["ave_diff"])]
        if count != 0:
            cleaned_merge = pd.merge(cleaned_merge, clean, how="inner")
        else:
            cleaned_merge = clean 
    df_cleaned.append(cleaned_merge)

    return pd.concat(df_cleaned)

# Plot the subcycle data colored based on cycle
def plot_subcycle_data(df_subcycle:pd.DataFrame, column_name:str, colormap, cmap, t:str): 
    cm = plt.cm.get_cmap(colormap)
    cycles = df_subcycle.Cycle_Index.unique()
    colors = [cm(val/len(cycles)) for val in range(len(cycles))]
    counter=0
    for cycle, color in zip(cycles, colors):
        df_temp = df_subcycle[df_subcycle[ 'Cycle_Index']==cycle]
        df_temp = df_temp[df_temp[ 'Step_Index'].isin([2, 4])]
        plt.plot(df_temp ['Test_Time(s)'], df_temp[column_name], color=color, alpha=0.5)
    plt.xlabel('Time(s)')
    plt.ylabel(column_name)
    plt.title(t)
    plt.show()
    fig, ax = plt.subplots (figsize=(7, 0.25))
    fig.subplots_adjust (bottom=0.5)
    norm = mpl.colors.Normalize(vmin=1, vmax=max(cycles))
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb1.set_label('Cycle')
    plt.show()

# Calculate which cycle each battery reaches 80% state of health (eol)
def create_target(df:pd.DataFrame, eol:float) -> pd.DataFrame:
    df_remove = df.loc[df['discharge_capacity'] <= eol]
    cycle = min(df_remove['cycle'])
    df['eol_cycle'] = cycle
    df = df[df['cycle'] <= df['eol_cycle']]
    df['eol_cycle'] = df['eol_cycle'].astype(int)
    return df

# Add target features of state of health and state of charge
# Exclude unusually short/long running experiments (partial charges)
def calculate_soc_soh(df:pd.DataFrame, df_subcycle:pd.DataFrame, nominal_capacity:float):
    df_final = pd.DataFrame()
    df['state_of_health'] = df['charge_capacity']/nominal_capacity
    for c in df.cycle.unique():
        df_subcycle_temp = df_subcycle[df_subcycle['Cycle_Index']==c]
        df_temp = df[df['cycle']==c]
        df_subcycle_temp['State_of_Health'] = df_temp['state_of_health'].iloc[0]
        df_subcycle_temp['Full_Charge_Capacity'] = df_temp['charge_capacity'].iloc[0]
        df_sub_charge = df_subcycle_temp[df_subcycle_temp['Step_Index'].isin([2, 4])] 
        df_sub_discharge = df_subcycle_temp[df_subcycle_temp['Step_Index']==7] 
        df_sub_charge['State_of_Charge'] = df_sub_charge['Charge_Capacity(Ah)']/df_temp['charge_capacity'].iloc[0]
        df_sub_discharge['State_of_Charge'] = (df_temp['discharge_capacity'].iloc[0]-df_sub_discharge['Discharge_Capacity(Ah)'])/df_temp['discharge_capacity'].iloc[0]
        charge_time = max(df_sub_charge['Step_Time(s)']) - min(df_sub_charge['Step_Time(s)'])
        discharge_time = max(df_sub_discharge['Step_Time(s)']) - min(df_sub_discharge['Step_Time(s)'])
        if charge_time >= 4000 and charge_time <= 7000 and discharge_time >= 2000 and discharge_time <= 4000:
            df_sub = pd.concat([df_sub_charge, df_sub_discharge])
            df_final = pd.concat([df_final, df_sub])
    return df_final, df


def get_X_y_soh(df, scaler, features, add_soh, add_soc, seq_len, soh_lower_bound, soh_upper_bound, soc_cycle_list, soh_break): 
    df_soh = pd.DataFrame(columns=['cycle_num', 'for_soc', 'charge_time', 'soh', 'soc'])
    X = np.array([])
    cycle_list = list(df.Cycle_Index.unique())
    for c in cycle_list:
        if add_soc:
            df_cycle = df[df['Cycle_Index']==c] 
            if c in soc_cycle_list:
                for x in range(df_cycle.shape[0]): 
                    if x+1<seq_len:
                        len_diff = seq_len-x-1
                        add_list = [-999] * len_diff 
                        df_scale = df_cycle.iloc[:x+1]
                        soh_label = float(df_scale['State_of_Health'].iloc[-1])
                        #soh_label = float(0)
                        df_scale[features] = scaler.transform(df_scale[features])
                    else:
                        start = x+1-seq_len
                        add_list = []
                        df_scale = df_cycle.iloc[start:x+1]
                        if len(df_scale[(df_scale['Step_Index'].isin([2, 4])) & (df_scale['Voltage(V)'] >= soh_lower_bound) & (df_scale['Voltage(V)'] <= soh_upper_bound)]) == seq_len:
                            soh_label = float(df_scale['State_of_Health'].iloc[-1])
                        else:
                            soh_label = float(df_scale['State_of_Health'].iloc[-1])
                            #soh_label = float(0)
                        df_scale[features] -scaler.transform(df_scale[features]) 
                    values_to_add_current = add_list + list(df_scale[features[0]])
                    values_to_add_voltage = add_list + list(df_scale[features[1]])
                    values_to_add_temp = add_list + list(df_scale[features[2]])
                    values_to_add = np.array([np.array(values_to_add_current), np.array(values_to_add_voltage), np.array(values_to_add_temp)]) 
                    values_to_add = values_to_add.transpose()
                    X = np.append(X, values_to_add)
                    df_soh.loc[len(df_soh)] = [c, True, float(df_scale['Test_Time(s)'].iloc[-1]), soh_label, float(df_scale['State_of_Charge'].iloc[-1])]
        if add_soh:
            df_cycle = df[df['Cycle_Index']==c] 
            df_voltage_range = df_cycle[(df_cycle['Step_Index'].isin([2, 4])) & (df_cycle['Voltage(V)'] >= soh_lower_bound) & (df_cycle['Voltage(V)'] <= soh_upper_bound)]
            length_vr = df_voltage_range.shape[0]
            if length_vr <= seq_len:
                df_scale = df_cycle[(df_cycle['Step_Index'].isin([2, 4])) & (df_cycle['Voltage(V)'] >= soh_lower_bound)].iloc[:seq_len] 
                df_scale[features] = scaler.transform(df_scale[features])
                X = np.append(X, df_scale[features].to_numpy())
                df_soh.loc[len(df_soh)] [c, False, float(df_scale['Test_Time(s)'].iloc[-1]), float(df_scale['State_of_Health'].iloc[-1]), float(df_scale['State_of_Charge'].iloc[-1])]
            else:
                for x in range(0, length_vr-seq_len+1, soh_break):
                    df_scale = df_voltage_range.iloc[x:seq_len+x]
                    df_scale[features] = scaler.transform(df_scale[features])
                    X = np.append(X, df_scale[features].to_numpy())
                    df_soh.loc[len(df_soh)] [c, False, float(df_scale['Test_Time(s)'].iloc[-1]), float(df_scale['State_of_Health'].iloc[-1]), float(df_scale['State_of_Charge'].iloc[-1])]

    X = np.reshape(X, (-1, seq_len, len(features)))
    y_soh = df_soh['soh'].to_numpy()
    y_soc = df_soh['soc'].to_numpy()

    return df_soh, X, y_soh, y_soc


def print_results(df_results, title_name, show_soc, show_soh):
    if show_soh:
        y_soh = df_results[df_results['for_soc'] == False]['soh'] 
        pred_soh = df_results[df_results['for_soc'] == False]['soh prediction']
        print('SOH Predictions:')
        print('Mean Absolute Error:', mean_absolute_error(y_soh, pred_soh))
        print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_soh, pred_soh))) 
        print('Median Absolute Error:', median_absolute_error(y_soh, pred_soh))
        print('Mean Absolute Percentage Error:', mean_absolute_percentage_error(y_soh, pred_soh)) 
        print('R Squared:', r2_score(y_soh, pred_soh))
        print()
    if show_soc:
        y_soc = df_results[df_results['for_soc'] == True]['soc']
        pred_soc = df_results[df_results['for_soc'] == True]['soc_prediction']
        print('SOC Predictions:')
        print('Mean Absolute Error:', mean_absolute_error(y_soc, pred_soc))
        print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_soc, pred_soc))) 
        print('Median Absolute Error:', median_absolute_error(y_soc, pred_soc))
        print('Mean Absolute Percentage Error:', mean_absolute_percentage_error(y_soc, pred_soc)) 
        print('R Squared:', r2_score(y_soc, pred_soc))


def plot_soc_stuff(df, data_name):
    df['soc_abs_error'] = abs(df['soc']-df['soc_prediction'])
    soc_results_by_cycle = df.groupby('cycle_num')['soc_abs_error'].mean().reset_index() 
    plt.scatter(soc_results_by_cycle['cycle_num'], soc_results_by_cycle['soc_abs_error'], color='blue')
    plt.title('SOC Mean Absolute Error Over Cycle')
    plt.xlabel('Cycle')
    plt.ylabel('SOC Mean Absolute Error')
    plt.show()

    df_temp_testing = df[df['is_charge'] == True]
    df_temp_testing['bin'] = pd.qcut(np.array(df_temp_testing['soc']), 100)
    soc_results_by_soc = df_temp_testing.groupby(['bin'])['soc_abs_error'].median().reset_index() 
    temp_mean = df_temp_testing.groupby(['bin'])['soc_abs_error'].mean().reset_index()
    temp_75 = df_temp_testing.groupby(['bin'])['soc_abs_error'].quantile(0.95).reset_index()
    temp_25 = df_temp_testing.groupby(['bin'])['soc_abs_error'].quantile(0.05).reset_index()
    soc_results_by_soc['soc_median_abs_err'] = soc_results_by_soc['soc_abs_error'] 
    soc_results_by_soc['soc_mean_abs_err'] = temp_mean['soc_abs_error'] 
    soc_results_by_soc['soc_q75_abs_err'] = temp_75['soc_abs_error'] 
    soc_results_by_soc['soc_q25_abs_err'] = temp_25['soc_abs_error']
    midpoints = []
    for index, row in soc_results_by_soc.iterrows():
        midpoints.append(row['bin'].mid)
    soc_results_by_soc['midpoint'] = midpoints
    plt.plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_median_abs_err'], color='blue', label='median error') 
    plt.plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_mean_abs_err'], color='cyan', label='mean error') 
    plt.plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_925_abs_err'], color='magenta', linestyle='--', label='5th quantile') 
    plt.plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_q75_abs_err'], color='magenta', linestyle='--', label='95th quantile') 
    plt.xlabel('State Of Charge') 
    plt.ylabel('Absolute Error')
    t = 'Absolute Error over State of Charge (Charging)'
    plt.title(t)
    plt.legend()
    plt.show()

    df_temp_testing = df[df['is_charge'] == False]
    df_temp_testing['bin'] = pd.qcut(np.array(df_temp_testing['soc']), 100)
    soc_results_by_soc = df_temp_testing.groupby(['bin'])['soc_abs_error'].median().reset_index() 
    temp_mean = df_temp_testing.groupby(['bin'])['soc_abs_error'].mean().reset_index()
    temp_75 = df_temp_testing.groupby(['bin'])['soc_abs_error'].quantile(0.95).reset_index()
    temp_25 = df_temp_testing.groupby(['bin'])['soc_abs_error'].quantile(0.05).reset_index()
    soc_results_by_soc['soc_median_abs_err'] = soc_results_by_soc['soc_abs_error'] 
    soc_results_by_soc['soc_mean_abs_err'] = temp_mean['soc_abs_error'] 
    soc_results_by_soc['soc_q75_abs_err'] = temp_75['soc_abs_error'] 
    soc_results_by_soc['soc_q25_abs_err'] = temp_25['soc_abs_error']
    midpoints = []
    for index, row in soc_results_by_soc.iterrows():
        midpoints.append(row['bin'].mid)
    soc_results_by_soc['midpoint'] = midpoints
    plt.plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_median_abs_err'], color='blue', label='median error') 
    plt.plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_mean_abs_err'], color='cyan', label='mean error') 
    plt.plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_925_abs_err'], color='magenta', linestyle='--', label='5th quantile') 
    plt.plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_q75_abs_err'], color='magenta', linestyle='--', label='95th quantile') 
    plt.xlabel('State Of Charge') 
    plt.ylabel('Absolute Error')
    t = 'Absolute Error over State of Charge (Discharging)'
    plt.title(t)
    plt.legend()
    plt.show()