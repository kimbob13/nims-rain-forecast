import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from nims_dataset import NIMSDataset
from nims_variable import *

import argparse
from multiprocessing import Process, Queue, cpu_count

def get_avg_and_max(partial_path, variables, queue=None):
    max_value_per_day = []
    avg_value_per_day = []

    for path in partial_path:
        one_day = xr.open_dataset(path)
        one_day_value = read_variable_value(one_day, variables[0])

        max_one_day_value = np.amax(one_day_value)
        max_value_per_day.append(max_one_day_value)
        if max_one_day_value == 0:
            avg_value_per_day.append(0)
        else:
            # Average over nonzero data
            avg_rain = np.mean(one_day_value[np.nonzero(one_day_value)])
            avg_value_per_day.append(avg_rain)

    if queue:
        queue.put((max_value_per_day, avg_value_per_day))
    else:
        return max_value_per_day, avg_value_per_day

def get_avg_and_max_mp(data_path, variables):
    num_processes = cpu_count() // 2
    num_path_per_process = len(data_path) // num_processes

    # Create queue
    queues = []
    for i in range(num_processes):
        queues.append(Queue())

    # Create processes
    processes = []
    for i in range(num_processes):
        start_idx = i * num_path_per_process
        end_idx = start_idx + num_path_per_process

        if i == num_processes - 1:
            processes.append(Process(target=get_avg_and_max,
                                     args=(data_path[start_idx:], variables,
                                           queues[i])))
        else:
            processes.append(Process(target=get_avg_and_max,
                                     args=(data_path[start_idx:end_idx],
                                           variables, queues[i])))
    # Start processes
    for i in range(num_processes):
        processes[i].start()

    # Get return value of each process
    max_value_per_day = []
    avg_value_per_day = []
    for i in range(num_processes):
        proc_result = queues[i].get()
        max_value_per_day.extend(proc_result[0])
        avg_value_per_day.extend(proc_result[1])

    # Join processes
    for i in range(num_processes):
        processes[i].join()

    return max_value_per_day, avg_value_per_day

def get_variable_unit(var_name):
    if var_name == 'rain':
        return 'mm/hr'

    # TODO: Add units of other variables

def plot_histogram(data, bins, variables, mode):
    # Get counts of range in bins
    counts = np.histogram(data, bins=bins)

    # Get variable name
    var_name = get_variable_name(variables[0])

    # Set title of plot
    if mode == 'avg':
        title = 'Average'
    elif mode == 'max':
        title = 'Maximum'

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot()
    avg_bars = ax.bar(range(len(bins) - 1), counts[0], width=1, edgecolor='k')
    ax.set_xticks(range(len(bins) - 1))
    ax.set_xticklabels(bins[:-1])
    ax.set_title('{} {} Distribution'.format(title, var_name.title()))
    ax.set_xlabel('{} ({})'.format(var_name, get_variable_unit(var_name)))
    #plt.hist(data, bins=bins, ec='black')
    #plt.xscale('log')
    for bar in avg_bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 15, yval)

    # Save plot
    fig.savefig('./plot/{}_{}_hist.png'.format(mode, var_name))

def get_variable_bins(variables):
    var_name = get_variable_name(variables[0])

    if var_name == 'rain':
        avg_bins = [0, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 20.0, 30.0, 40.0]
        max_bins = [0, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 500.0, 1000.0]

    # TODO: Make bins for other variables

    return avg_bins, max_bins

if __name__ == '__main__':
    # Read variable argument
    parser = argparse.ArgumentParser(description='NIMS rainfall data prediction')
    parser.add_argument('--variables', type=str, default='rain',
                        help='which variables to use (rain, cape, etc.). \
                              must specify one variable name')
    args = parser.parse_args()
    variables_args = [args.variables]
    variables = parse_variables(variables_args)
    assert len(variables) == 1

    nims_train_data_path = NIMSDataset(window_size=1,
                                       target_num=1,
                                       variables=variables,
                                       train_year=(2017, 2017),
                                       train=True,
                                       debug=False).data_path_list

    # Single core version
    #max_value_per_day, avg_value_per_day = get_avg_and_max(nims_train_data_path, variables)

    # Mutli core version
    max_value_per_day, avg_value_per_day = get_avg_and_max_mp(nims_train_data_path, variables)

    # Get appropriate bins for variables
    avg_bins, max_bins = get_variable_bins(variables)

    # Average value
    plot_histogram(avg_value_per_day, avg_bins, variables, 'avg')

    # Maximum value
    plot_histogram(max_value_per_day, max_bins, variables, 'max')

    print('Finish')