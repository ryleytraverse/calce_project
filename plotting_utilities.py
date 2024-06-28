import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

# Plot the subcycle data colored based on cycle
def plot_subcycle_data(df_subcycle:pd.DataFrame, current_name:str, voltage_name:str, bat:int): 
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    cm = plt.cm.get_cmap('winter')
    cycles = df_subcycle.Cycle_Index.unique()
    colors = [cm(val/len(cycles)) for val in range(len(cycles))]
    for cycle, color in zip(cycles, colors):
        df_temp = df_subcycle[df_subcycle['Cycle_Index']==cycle]
        df_temp = df_temp[df_temp['Step_Index'].isin([2, 4])]
        axs[0].plot(df_temp['Test_Time(s)'].to_numpy(), df_temp[current_name].to_numpy(), color=color, alpha=0.5)
    axs[0].set_xlabel('Time(s)')
    axs[0].set_ylabel(current_name)
    t = 'Current over Time Colored by Cycle Number for Cell #' + str(bat)
    axs[0].set_title(t)

    cm = plt.cm.get_cmap('autumn')
    colors = [cm(val/len(cycles)) for val in range(len(cycles))]
    for cycle, color in zip(cycles, colors):
        df_temp = df_subcycle[df_subcycle['Cycle_Index']==cycle]
        df_temp = df_temp[df_temp['Step_Index'].isin([2, 4])]
        axs[1].plot(df_temp['Test_Time(s)'].to_numpy(), df_temp[voltage_name].to_numpy(), color=color, alpha=0.5)
    axs[1].set_xlabel('Time(s)')
    axs[1].set_ylabel(voltage_name)
    t = 'Voltage over Time Colored by Cycle Number for Cell #' + str(bat)
    axs[1].set_title(t)
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15.5, 0.25))
    #fig.subplots_adjust(bottom=0.5)
    norm = mpl.colors.Normalize(vmin=1, vmax=max(cycles))
    cb1 = mpl.colorbar.ColorbarBase(ax[0], cmap=mpl.cm.winter, norm=norm, orientation='horizontal')
    ax[0].set_xlabel('Cycle')

    norm = mpl.colors.Normalize(vmin=1, vmax=max(cycles))
    cb1 = mpl.colorbar.ColorbarBase(ax[1], cmap=mpl.cm.autumn, norm=norm, orientation='horizontal')
    ax[1].set_xlabel('Cycle')
    plt.show()

# Print Accuracy Metrics for SOC or SOH
def print_results(df_results, show_soc, show_soh):
    if show_soh:
        y_soh = df_results[df_results['for_soc'] == False]['soh'] 
        pred_soh = df_results[df_results['for_soc'] == False]['soh_prediction']
        print('SOH Prediction Accuracy:')
        print('Mean Absolute Error:', mean_absolute_error(y_soh, pred_soh))
        print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_soh, pred_soh))) 
        print('Median Absolute Error:', median_absolute_error(y_soh, pred_soh))
        #print('Mean Absolute Percentage Error:', mean_absolute_percentage_error(y_soh, pred_soh)) 
        print('R Squared:', r2_score(y_soh, pred_soh))
        print()
    if show_soc:
        y_soc = df_results[df_results['for_soc'] == True]['soc']
        pred_soc = df_results[df_results['for_soc'] == True]['soc_prediction']
        print('SOC Prediction Accuracy:')
        print('Mean Absolute Error:', mean_absolute_error(y_soc, pred_soc))
        print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_soc, pred_soc))) 
        print('Median Absolute Error:', median_absolute_error(y_soc, pred_soc))
        #print('Mean Absolute Percentage Error:', mean_absolute_percentage_error(y_soc, pred_soc)) 
        print('R Squared:', r2_score(y_soc, pred_soc))

# Show plots of state of charge accuracy over cycle and state of charge
def plot_soc_stuff(df):
    # State of charge accuracy over cycle
    df['soc_abs_error'] = abs(df['soc']-df['soc_prediction'])
    soc_results_by_cycle = df.groupby('cycle_num')['soc_abs_error'].mean().reset_index() 
    plt.scatter(soc_results_by_cycle['cycle_num'], soc_results_by_cycle['soc_abs_error'], color='blue')
    plt.title('SOC Prediction Mean Absolute Error Over Cycle for Cell #37')
    plt.xlabel('Cycle')
    plt.ylabel('SOC Mean Absolute Error')
    plt.show()

    # State of charge accuracy over state of charge (charge)
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
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_mean_abs_err'], color='cyan', label='mean error')
    ax[0].plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_median_abs_err'], color='blue', label='median error') 
    ax[0].plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_q25_abs_err'], color='red', linestyle='--', label='5th quantile', alpha = 0.7) 
    ax[0].plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_q75_abs_err'], color='red', linestyle='--', label='95th quantile', alpha = 0.7) 
    ax[0].set_xlabel('State Of Charge') 
    ax[0].set_ylabel('Absolute Error')
    t = 'Prediction Absolute Error over State of Charge (Charging) for Cell #37'
    ax[0].set_title(t)
    ax[0].legend()

    # State of charge accuracy over state of charge (discharge)
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
    ax[1].plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_mean_abs_err'], color='cyan', label='mean error')
    ax[1].plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_median_abs_err'], color='blue', label='median error')
    ax[1].plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_q25_abs_err'], color='red', linestyle='--', label='5th quantile', alpha=0.7) 
    ax[1].plot(soc_results_by_soc['midpoint'], soc_results_by_soc['soc_q75_abs_err'], color='red', linestyle='--', label='95th quantile', alpha=0.7) 
    ax[1].set_xlabel('State Of Charge') 
    ax[1].set_ylabel('Absolute Error')
    t = 'Prediction Absolute Error over State of Charge (Discharging) for Cell #37'
    ax[1].set_title(t)
    ax[1].legend()
    plt.show()