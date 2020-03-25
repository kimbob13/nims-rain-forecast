import torch
from torch.utils.data import DataLoader

import xarray as xr
import numpy as np

from multiprocessing import Process, Queue, cpu_count
import pickle
import math

from nims_dataset import NIMSDataset

def read_variable(dataset, var_idx):
    """
    Read proper variable based on var_idx.
    For example, if var_idx == 0, it should read 'rain' data,
    and if var_idx == 4, it should read 'hel' data.

    Variable List:
    [0] : rain [1] : cape [2] : cin  [3] : swe [4]: hel
    [5] : ct   [6] : vt   [7] : tt   [8] : si  [9]: ki
    [10]: li   [11]: ti   [12]: ssi  [13]: pw

    <Parameters>
    dataset [xarray dataset]: dataset for one hour to extract data
    var_idx [int]: index for variables list

    <Return>
    one_var_data [np.ndarray]: numpy array of value (CHW format)
    """
    assert var_idx >= 0 and var_idx <= 13

    if var_idx == 0:
        one_var_data = dataset.rain.values
    elif var_idx == 1:
        one_var_data = dataset.cape.values
    elif var_idx == 2:
        one_var_data = dataset.cin.values
    elif var_idx == 3:
        one_var_data = dataset.swe.values
    elif var_idx == 4:
        one_var_data = dataset.hel.values
    elif var_idx == 5:
        one_var_data = dataset.ct.values
    elif var_idx == 6:
        one_var_data = dataset.vt.values
    elif var_idx == 7:
        one_var_data = dataset.tt.values
    elif var_idx == 8:
        one_var_data = dataset.si.values
    elif var_idx == 9:
        one_var_data = dataset.ki.values
    elif var_idx == 10:
        one_var_data = dataset.li.values
    elif var_idx == 11:
        one_var_data = dataset.ti.values
    elif var_idx == 12:
        one_var_data = dataset.ssi.values
    elif var_idx == 13:
        one_var_data = dataset.pw.values

    return one_var_data

def partial_mean(pid, partial_path, unused, queue):
    mean_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0,
                 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
                 10: 0, 11: 0, 12: 0, 13: 0}
    data_num = len(partial_path)

    for i, path in enumerate(partial_path):
        dataset = xr.open_dataset(path)
        for var_idx in mean_dict:
            data = read_variable(dataset, var_idx)

            one_hour_mean = np.mean(data, dtype=np.float64)
            #print('[PID {} - i: {}] var_idx: {} - mean: {}'.format(pid, i, var_idx, one_hour_mean))

            # Calcuate cumulative moving average
            mean_dict[var_idx] = mean_dict[var_idx] + \
                                 ((one_hour_mean - mean_dict[var_idx]) / (i + 1))

    print('[partial_mean] Process {} finished'.format(pid))
    queue.put((mean_dict, data_num))

def partial_variance(pid, partial_path, total_mean, queue):
    """
    <Parameters>
    partial_path [list[str]]
    total_mean [dict]
    queue [Queue]
    """
    variance_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0,
                     5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
                     10: 0, 11: 0, 12: 0, 13: 0}
    data_num = len(partial_path)

    for i, path in enumerate(partial_path):
        dataset = xr.open_dataset(path)
        for var_idx in variance_dict:
            data = read_variable(dataset, var_idx)

            one_hour_bias_square= np.square((data - total_mean[var_idx]),
                                            dtype=np.float64)
            one_hour_bias_square_mean = np.mean(one_hour_bias_square,
                                                dtype=np.float64)

            # Calculating cumulative mean for square mean of bais
            variance_dict[var_idx] = variance_dict[var_idx] + \
                                     ((one_hour_bias_square_mean -
                                       variance_dict[var_idx]) / (i + 1))

    print('[partial_variance] Process {} finished'.format(pid))
    queue.put((variance_dict, data_num))

def calculation(partial_func, data_path, total_mean=None):
    num_processes = cpu_count()
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
            processes.append(Process(target=partial_func,
                                     args=(i, data_path[start_idx:],
                                           total_mean, queues[i])))
        else:
            processes.append(Process(target=partial_func,
                                     args=(i, data_path[start_idx:end_idx],
                                           total_mean, queues[i])))
    # Start processes
    for i in range(num_processes):
        processes[i].start()

    # Get return value of each process
    results = []
    for i in range(num_processes):
        results.append(queues[i].get())

    # Join processes
    for i in range(num_processes):
        processes[i].join()

    # Collect per-feature result
    total_num = 0
    total_mean = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0,
                  5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
                  10: 0, 11: 0, 12: 0, 13: 0}
    for proc_mean_dict, proc_data_num in results:
        total_num += proc_data_num

        for var_idx in proc_mean_dict:
            total_mean[var_idx] += (proc_mean_dict[var_idx] * proc_data_num)
    
    # Per-feature mean
    for var_idx in total_mean:
        total_mean[var_idx] /= total_num

    return total_mean

def get_variable_name(var_idx):
    """
    Variable List:
    [0] : rain [1] : cape [2] : cin  [3] : swe [4]: hel
    [5] : ct   [6] : vt   [7] : tt   [8] : si  [9]: ki
    [10]: li   [11]: ti   [12]: ssi  [13]: pw
    """
    assert var_idx >= 0 and var_idx <= 13

    if var_idx == 0:
        return 'rain'
    elif var_idx == 1:
        return 'cape'
    elif var_idx == 2:
        return 'cin'
    elif var_idx == 3:
        return 'swe'
    elif var_idx == 4:
        return 'hel'
    elif var_idx == 5:
        return 'ct'
    elif var_idx == 6:
        return 'vt'
    elif var_idx == 7:
        return 'tt'
    elif var_idx == 8:
        return 'si'
    elif var_idx == 9:
        return 'ki'
    elif var_idx == 10:
        return 'li'
    elif var_idx == 11:
        return 'ti'
    elif var_idx == 12:
        return 'ssi'
    elif var_idx == 13:
        return 'pw'


def print_stat(result_dict, data_type='train', stat_type='mean'):
    print('=' * 25, '{} {} RESULT'.format(data_type, stat_type), '=' * 25)

    for var_idx in result_dict:
        print('[{}] {}: {}'
              .format(get_variable_name(var_idx), stat_type,
                      result_dict[var_idx]))

    print()

if __name__ == '__main__':
    root_dir = '/home/kimbob/jupyter/weather_prediction/NIMS'
    nims_train_data_path = NIMSDataset(model='unet',
                                       window_size=1,
                                       target_num=1,
                                       variables=list(range(14)),
                                       train_year=(2009, 2017),
                                       train=True,
                                       root_dir=root_dir,
                                       debug=True).data_path_list
    
    nims_test_data_path = NIMSDataset(model='unet',
                                      window_size=1,
                                      target_num=1,
                                      variables=list(range(14)),
                                      train_year=(2009, 2017),
                                      train=False,
                                      root_dir=root_dir,
                                      debug=True).data_path_list

    print('train len: {}, test len: {}'.format(len(nims_train_data_path), len(nims_test_data_path)))

    #train_mean = calculation(partial_mean, nims_train_data_path)
    #train_variance = calculation(partial_variance, nims_train_data_path,
    #                             total_mean=train_mean)

    #train_stdv = {}
    #for var_idx in train_variance:
    #    train_stdv[var_idx] = math.sqrt(train_variance[var_idx])

    test_mean = calculation(partial_mean, nims_test_data_path)
    test_variance = calculation(partial_variance, nims_test_data_path,
                                total_mean=test_mean)

    test_stdv = {}
    for var_idx in test_variance:
        test_stdv[var_idx] = math.sqrt(test_variance[var_idx])

    #print_stat(train_mean, data_type='train', stat_type='mean')
    #print_stat(train_variance, data_type='train', stat_type='variance')
    print_stat(test_mean, data_type='test', stat_type='mean')
    print_stat(test_variance, data_type='test', stat_type='variance')

    #with open('mean_var/train_mean.pickle', 'wb') as f:
    #    pickle.dump(train_mean, f)

    #with open('mean_var/train_variance.pickle', 'wb') as f:
    #    pickle.dump(train_variance, f)

    #with open('mean_var/train_stdv.pickle', 'wb') as f:
    #    pickle.dump(train_stdv, f)

    with open('mean_var/test_mean.pickle', 'wb') as f:
        pickle.dump(test_mean, f)

    with open('mean_var/test_variance.pickle', 'wb') as f:
        pickle.dump(test_variance, f)

    with open('mean_var/test_stdv.pickle', 'wb') as f:
        pickle.dump(test_stdv, f)
