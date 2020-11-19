import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count
import os

from nims_dataset import NIMSDataset
from nims_variable import read_variable_value

YEAR_IDX = -13
MONTH_IDX = -9
DAY_IDX = -7
HOUR_IDX = -5

class DayError(Exception):
    """
    Exception raised for errors in the day.
    """
    def __init__(self, valid_day):
        self.valid_day = valid_day

def get_year():
    print()
    print('=' * 20, '2.1 Year Selection', '=' * 20)
    year = int(input('Which year do you want to see? (2009 - 2018): '))

    if (year < 2009) or (year > 2018):
        return -1

    return year

def get_month():
    print()
    print('=' * 20, '2.2 Month Selection', '=' * 20)
    month = int(input('Which month do you want to see? (1 - 12): '))

    if (month < 1) or (month > 12):
        return -1

    return month

def get_day(year, month):
    month_31_day = (1, 3, 5, 7, 8, 10, 12)
    leap_year = (2012, 2016)

    if month in month_31_day:
        valid_day = 31
    elif month == 2:
        if year in leap_year:
            valid_day = 29
        else:
            valid_day = 28
    else:
        valid_day = 30

    print()
    print('=' * 20, '2.3 Day Selection', '=' * 20)
    day = int(input('Which day do you want to see? (1 - {}): '.format(valid_day)))

    if (day < 1) or (day > valid_day):
        raise DayError(valid_day)

    return day

def _compute_stat(pid, partial_path_list, queue):
    stat = {}
    for path in partial_path_list:
        cur_month = int(path[MONTH_IDX:MONTH_IDX + 2])
        if cur_month not in stat:
            stat[cur_month] = {0: 0, 1: 0, 2: 0, 3: 0}

        one_hour = xr.open_dataset(path)
        one_hour_value = np.squeeze(read_variable_value(one_hour, 0), axis=0)
        
        for lat in range(one_hour_value.shape[0]):
            for lon in range(one_hour_value.shape[1]):
                value = one_hour_value[lat][lon]

                if value >= 0 and value < 0.1:
                    stat[cur_month][0] += 1
                elif value >= 0.1 and value < 1.0:
                    stat[cur_month][1] += 1
                elif value >= 1.0 and value < 2.5:
                    stat[cur_month][2] += 1
                elif value >= 2.5:
                    stat[cur_month][3] += 1

    print('Process {:2d} finished'.format(pid))

    queue.put(stat)

def compute_stat(path_list, stat_mode):
    start, end = path_list[0], path_list[-1]

    year = int(start[YEAR_IDX:YEAR_IDX + 4])
    date_str = '{}'.format(year)
    if (stat_mode == 2) or (stat_mode == 3):
        month = int(start[MONTH_IDX:MONTH_IDX + 2])
        date_str += '-{:02d}'.format(month)

        if stat_mode == 3:
            day = int(start[DAY_IDX:DAY_IDX + 2])
            date_str += '-{:02d}'.format(day)

    num_process = cpu_count() // 2
    num_path_per_process = len(path_list) // num_process

    # Create queue
    queues = []
    for i in range(num_process):
        queues.append(Queue())

    # Create processes
    processes = []
    for i in range(num_process):
        start_idx = i * num_path_per_process
        end_idx = start_idx + num_path_per_process

        if i == num_process - 1:
            processes.append(Process(target=_compute_stat,
                                     args=(i, path_list[start_idx:],
                                           queues[i])))
        else:
            processes.append(Process(target=_compute_stat,
                                     args=(i, path_list[start_idx:end_idx],
                                           queues[i])))

    # Start processes
    print()
    print('=' * 20, '3. Stat calculation start', '=' * 20)
    for i in range(num_process):
        processes[i].start()

    # Get result from each process
    stat = {}
    for i in range(num_process):
        proc_result = queues[i].get()
        for month, monthly_stat in proc_result.items():
            if month not in stat:
                stat[month] = {0: 0, 1: 0, 2: 0, 3: 0}

            for label, count in monthly_stat.items():
                stat[month][label] += count

    # Join processes
    for i in range(num_process):
        processes[i].join()

    # Print and save the stat as csv
    print()
    print('=' * 20, '4. Stat for {}'.format(date_str), '=' * 20)
    month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    stat_df = pd.DataFrame(columns=['Total', '0', '1', '2', '3', '0 (%)', '1 (%)', '2 (%)', '3 (%)'], index=month_name)
    for month, monthly_stat in stat.items():
        month_total_count = sum(list(monthly_stat.values()))
        stat_df.loc[month_name[month - 1]]['Total'] = month_total_count
        print('[{}] <{:^7s}> {:12,d}'.format(month_name[month - 1], 'Total', month_total_count))
        for label, count in monthly_stat.items():
            print('      <Label {}> {:12,d} ({:6.3f}%)'.format(label, count, (count / month_total_count) * 100))
            stat_df.loc[month_name[month - 1]][str(label)] = count
            stat_df.loc[month_name[month - 1]][str(label) + ' (%)'] = (count / month_total_count) * 100

    # Save only yearly data to csv
    if stat_mode == 1:
        if not os.path.isdir('./label_stat'):
            os.mkdir('./label_stat')

        stat_df.to_csv('./label_stat/{}.csv'.format(year))

if __name__ == '__main__':
    # Mode selection
    while True:
        try:
            print()
            print('=' * 20, '1. Stat Type', '=' * 20)
            print('Which mode do you want to obtain stat?')
            stat_mode = int(input('[1] Yearly    [2] Montly    [3] Daily: '))

            if stat_mode not in [1, 2, 3]:
                print('You must enter value between 1 to 3')
                continue

        except ValueError:
            print('You must enter integer only')
            continue

        break
    
    # Year selection
    while True:
        try:
            year = get_year()
            if year == -1:
                print("You must specify year between 2009 to 2018")
                continue

        except ValueError:
            print("You must enter integer for year")
            continue

        break

    if (stat_mode == 2) or (stat_mode == 3):
        # Month selection
        while True:
            try:
                month = get_month()
                if month == -1:
                    print("You must specify month between 1 to 12")
                    continue

            except ValueError:
                print("You must enter integer for month")
                continue

            break

        if stat_mode == 3:
            # Day selection
            while True:
                try:
                    day = get_day(year, month)

                except ValueError:
                    print("You must enter integer for day")
                    continue

                except DayError as d_err:
                    print("You must specifiy day between 1 to {}".format(d_err.valid_day))
                    continue
                
                break
    
    train = True
    if year == 2018:
        year -= 1
        train = False

    nims_data_path = NIMSDataset(model=None,
                                 window_size=1,
                                 target_num=1,
                                 variables=[0],
                                 train_year=(year, year),
                                 train=train,
                                 debug=False).data_path_list

    if stat_mode == 2:
        nims_data_path = [path for path in nims_data_path \
                          if int(path[MONTH_IDX:MONTH_IDX + 2]) == month]

    elif stat_mode == 3:
        nims_data_path = [path for path in nims_data_path \
                          if (int(path[MONTH_IDX:MONTH_IDX + 2]) == month) and\
                             (int(path[DAY_IDX:DAY_IDX + 2]) == day)]

    compute_stat(nims_data_path, stat_mode)