import torch
from torch.utils.data import DataLoader

from core.nims_util import *
from core.nims_dataset import NIMSDataset, ToTensor
from core.nims_trainer import NIMSTrainer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import shutil
import time
from datetime import datetime, timedelta

MONTH_DAY = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

#########################################################
# 1. Function for test setting part                     #
#########################################################

def select_experiment():
    results_dir = os.path.join('./results')
    experiment_list = sorted([f for f in os.listdir(results_dir) if f.startswith('nims')])
    print()
    print('=' * 33, 'Which experiment do you want to test?', '=' * 33)
    print()
    print('-' * 100)
    print('{:^4s}| {:^19s} | {:^65s}{:>7s}'.format('Idx', 'Last Modified', 'Experiment', '|'))
    print('-' * 100)
    for i, experiment in enumerate(experiment_list):
        path_date = time.strftime('%Y/%m/%d %H:%M:%S', time.gmtime(os.path.getmtime(os.path.join(results_dir, experiment))))
        print('[{:2d}] <{:s}> {}'.format(i + 1, path_date, experiment))
    choice = int(input('\n> '))
    experiment_name = experiment_list[choice - 1]
    chosen_weight = torch.load(os.path.join(results_dir, experiment_list[choice - 1], 'trained_weight.pt'), map_location=device)
    print('Load experiment... [{}] {}:'.format(choice, experiment_name))
    print()

    # Print trained weight info
    print('=' * 25, 'Trained Weight Information', '=' * 25)
    try:
        print('[best epoch]:', chosen_weight['best_epoch'])
    except:
        pass
    try:
        print('[best loss ]: {:5f}'.format(chosen_weight['best_loss']))
    except:
        pass
    try:
        print('[best acc  ]: {:5f}'.format(chosen_weight['best_acc']))
    except:
        pass
    try:
        print('[best csi  ]: {:5f}'.format(chosen_weight['best_csi']))
    except:
        pass
    try:
        print('[best pod  ]: {:5f}'.format(chosen_weight['best_pod']))
    except:
        pass
    try:
        print('[best far  ]: {:5f}'.format(chosen_weight['best_far']))
    except:
        pass
    try:
        print('[best f1   ]: {:5f}'.format(chosen_weight['best_f1']))
    except:
        pass
    try:
        print('[best bias ]: {:5f}'.format(chosen_weight['best_bias']))
    except:
        pass

    return chosen_weight, experiment_name

def create_output_dir(args, date, experiment_name):
    # (Re)create output directory for this year
    output_dir = os.path.join(args.dataset_dir, 'NIMS_OUTPUT', experiment_name, str(date['year']))
    if os.path.isdir(output_dir):
        yes_no = input('It is about to remove whole output directory and recreate. CONTINUE? [YES or NO] : ')
        shutil.rmtree(output_dir)
        if yes_no.upper().startswith('N'):
            print('Exit the test program')
            print()
            sys.exit()

        print()

    os.makedirs(output_dir, exist_ok=True)

    curr_date = datetime(year=date['year'], month=date['start_month'], day=date['start_day'])
    while True:
        os.mkdir(os.path.join(output_dir, curr_date.strftime('%Y%m%d')))
        if curr_date.month == date['end_month'] and curr_date.day == date['end_day']:
            break

        curr_date += timedelta(days=1)

def create_test_date_list(date, experiment_name):
    test_date_list = []
    test_months = list(range(date['start_month'], date['end_month'] + 1))
    for i, month in enumerate(test_months):
        if i == 0:
            start_day = date['start_day']
            end_day = MONTH_DAY[month]
        elif i == len(test_months) - 1:
            start_day = 1
            end_day = date['end_day']
        else:
            start_day = 1
            end_day = MONTH_DAY[month]

        current_test_date = '{:4d}{:02d}{:02d}-{:04d}{:02d}{:02d}'.format(date['year'], month, start_day,
                                                                          date['year'], month, end_day)
        current_test_path = os.path.join('./results', experiment_name, 'eval', current_test_date)
        if args.eval_only:
            assert os.path.isdir(current_test_path), \
                   'You have to run test for this date range before running eval_only mode'
        
        if not os.path.isdir(current_test_path):
            os.mkdir(current_test_path)

        test_date_list.append(current_test_path)

    return test_date_list

#########################################################
# 2. Function for plotting part                         #
#########################################################

def recreate_total_stat(total_test_path):
    one_day_stat_list = sorted([os.path.join(total_test_path, f) for f in os.listdir(total_test_path) if f.endswith('.csv')])
    total_df = pd.DataFrame(columns=['date', 'acc', 'hit', 'miss', 'false alarm', 'correct negative'])

    for one_day_stat in one_day_stat_list:
        date = one_day_stat.split('/')[-1]
        year, month, day = date[:4], date[4:6], date[6:8]

        one_day_df = pd.read_csv(one_day_stat)
        total_df = total_df.append({'date': '{}-{}-{}'.format(year, month, day),
                                    'acc': one_day_df.iloc[-1, 0],
                                    'hit': one_day_df.iloc[-1, 1],
                                    'miss': one_day_df.iloc[-1, 2],
                                    'false alarm': one_day_df.iloc[-1, 3],
                                    'correct negative': one_day_df.iloc[-1, 4]},
                                   ignore_index=True)

    total_df = total_df.append(total_df.iloc[:, 2:].sum(axis=0), ignore_index=True)
    total_hit = total_df.iloc[-1, 2]
    total_miss = total_df.iloc[-1, 3]
    total_fa = total_df.iloc[-1, 4]
    total_cn = total_df.iloc[-1, 5]

    total_df.iloc[-1, 1] = (total_hit + total_cn) / (total_hit + total_miss + total_fa + total_cn)
    total_df.iloc[-1, 0] = 'Total'

    total_df.to_csv(os.path.join(total_test_path, '..', 'total-{}.csv'.format(year)), index=False)

def get_ldaps_eval_date_files(model_utc, date):
    ldaps_eval_dir = os.path.join('./results', 'LDAPS_Logger')
    ldaps_eval_list = sorted([f for f in os.listdir(ldaps_eval_dir)])

    ldaps_eval_date_files = []
    for l in ldaps_eval_list:
        ldaps_eval_date = os.path.join(ldaps_eval_dir, l)
        ldaps_eval_date_files += sorted([os.path.join(ldaps_eval_date, f) \
                                         for f in os.listdir(ldaps_eval_date) if f.endswith('.csv')])

    if date['end_month'] == None:
        date_str = '{:4d}{:02d}'.format(date['year'], date['start_month'])
        ldaps_eval_date_files = [f for f in ldaps_eval_date_files \
                                 if int(f.split('/')[-1][9:11]) == int(model_utc) and \
                                 f.split('/')[-1][:6] == date_str]
    else:
        start_date_str = '{:4d}{:02d}'.format(date['year'], date['start_month'])
        end_date_str = '{:4d}{:02d}'.format(date['year'], date['end_month'])
        ldaps_eval_date_files = [f for f in ldaps_eval_date_files \
                                 if int(f.split('/')[-1][9:11]) == int(model_utc) and \
                                 f.split('/')[-1][:6] >= start_date_str and \
                                 f.split('/')[-1][:6] <= end_date_str]

    return ldaps_eval_date_files

def get_nims_stat(eval_date_files):
    for i, eval_date in enumerate(eval_date_files):
        df = pd.read_csv(eval_date)
        
        if i == 0:
            df_according_h = df
        else:
            df_according_h += df

    hit = np.array(df_according_h['hit'])[6: -1]
    miss = np.array(df_according_h['miss'])[6: -1]
    fa = np.array(df_according_h['false alarm'])[6: -1]
    cn = np.array(df_according_h['correct negative'])[6: -1]

    csi = np.where((hit + miss + fa) > 0, hit / (hit + miss + fa), -0.1)
    pod = np.where((hit + miss) > 0, hit / (hit + miss), -0.1)
    far = np.where((hit + fa) > 0, fa / (hit + fa), -0.1)
    f1 = np.where(((2 * hit) + fa + miss) > 0, 2 * hit / ((2 * hit) + fa + miss), 1)
    bias = np.where((hit + miss) > 0, (hit + fa) / (hit + miss), 1)
    acc = ((hit + cn) / (hit + miss + fa + cn)) * 100

    nims_stat = NIMSStat(acc, csi, pod, far, f1, bias)

    return nims_stat

def plot_stat_graph(ldaps_stat, model_stat, date, experiment_name):
    stat_list = ldaps_stat._fields
    assert stat_list == model_stat._fields

    for stat_name in stat_list:
        _ldaps = getattr(ldaps_stat, stat_name)
        _model = getattr(model_stat, stat_name)

        plt.grid()
        plt.plot(range(6, 49), _ldaps, label='LDAPS', marker='o', markersize=4)
        plt.plot(range(6, 49), _model, label='ours', marker='o', markersize=4)
        plt.xticks(range(6, 49, 6))
        plt.legend()
        if date['end_month'] != None:
            plt.title('{:4d}-{:02d} ~ {:4d}-{:02d} {}'
                      .format(date['year'], date['start_month'],
                              date['year'], date['end_month'], stat_name.upper()))
            plt.savefig('./results/{}/comparison_graph/{:4d}{:02d}-{:4d}{:02d}-{}.pdf'
                        .format(experiment_name, date['year'], date['start_month'],
                                date['year'], date['end_month'], stat_name), dpi=300)
        else:
            plt.title('{:4d}-{:02d} {}'.format(date['year'], date['start_month'], stat_name.upper()))
            plt.savefig('./results/{}/comparison_graph/{:4d}{:02d}-{}.pdf'
                        .format(experiment_name, date['year'], date['start_month'], stat_name),
                        dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    #########################################################
    # 1. Test setting part                                  #
    #########################################################

    # Parsing command line arguments
    args = parse_args()

    # Set device
    device = set_device(args)

    # Specify trained model weight
    chosen_weight, experiment_name = select_experiment()

    # Select start and end date for train
    date = select_date(test=True)

    # Create output directory
    # create_output_dir(args, date, experiment_name)

    # Replace model related arguments to the train info
    args.model = chosen_weight['model_name']
    args.n_blocks = chosen_weight['n_blocks']
    args.start_channels = chosen_weight['start_channels']
    args.window_size = chosen_weight['window_size']
    args.model_utc = chosen_weight['model_utc']
    args.pos_loc = chosen_weight['pos_loc']
    args.pos_dim = chosen_weight['pos_dim']
    args.heavy_rain = chosen_weight['heavy_rain']
    args.cross_entropy_weight = chosen_weight['cross_entropy_weight']
    args.bilinear = chosen_weight['bilinear']
    args.custom_name = chosen_weight['custom_name']
    args.batch_size = 1 # Fix batch size as 1 for test

    # Fix the seed
    fix_seed(2020)
    
    # Test dataset
    nims_test_dataset = NIMSDataset(model=args.model,
                                    reference=args.reference,
                                    model_utc=args.model_utc,
                                    window_size=args.window_size,
                                    root_dir=args.dataset_dir,
                                    date=date,
                                    lite=args.lite,
                                    heavy_rain=args.heavy_rain,
                                    train=False,
                                    transform=ToTensor())
    
    # Get normalization transform base on min/max value from training procedure
    normalization = None
    if ('norm_max' in chosen_weight) and ('norm_min' in chosen_weight) and \
       (chosen_weight['norm_max'] != None) and (chosen_weight['norm_min'] != None):
        max_values = chosen_weight['norm_max'].to(torch.device('cpu'))
        min_values = chosen_weight['norm_min'].to(torch.device('cpu'))
        transform = get_min_max_normalization(max_values, min_values)

        print('=' * 25, 'Normalization is enabled!', '=' * 25)
        print('max_values shape: {}, min_values shape: {}\n'.format(max_values.shape, min_values.shape))

        normalization = {'max_values': max_values,
                         'min_values': min_values}
        args.normalization = True

    # Get a sample for getting shape of each tensor
    sample, _, _ = nims_test_dataset[0]

    # Create a model and criterion
    model, criterion = set_model(sample, device, args, train=False, experiment_name=experiment_name)

    # Create dataloaders
    test_loader = DataLoader(nims_test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    
    # Create directory for each month currently tested date for chosen weight
    test_date_list = create_test_date_list(date, experiment_name)

    # Load trained model weight
    model.load_state_dict(chosen_weight['model'])

    # Start testing
    nims_trainer = NIMSTrainer(model, criterion, None, None, device,
                               None, test_loader, experiment_name, args,
                               normalization=normalization,
                               test_date_list=test_date_list)
    if not args.eval_only:
        nims_trainer.test()

    #########################################################
    # 2. Plotting part                                      #
    #########################################################

    # 2-1. Plot stat graph for each month
    for current_test_path in test_date_list:
        _curr_date = current_test_path.split('/')[-1].split('-')[0]
        year, month = int(_curr_date[:4]), int(_curr_date[4:6])
        curr_date = {'year': year, 'start_month': month, 'end_month': None}

        # Start evaluation with LDAPS
        ldaps_eval_date_files = get_ldaps_eval_date_files(args.model_utc, curr_date)
        ldaps_stat = get_nims_stat(ldaps_eval_date_files)

        model_eval_date_files = sorted([os.path.join(current_test_path, f) \
                                        for f in os.listdir(current_test_path) \
                                        if 'ipynb' not in f and 'total' not in f])
        model_stat = get_nims_stat(model_eval_date_files)

        # Plot graph for each stat
        plot_stat_graph(ldaps_stat, model_stat, curr_date, experiment_name)

    # 2-2. Plot stat graph for whole test range if start and end month is different
    if date['start_month'] != date['end_month']:
        total_test_date = '{:4d}{:02d}{:02d}-{:04d}{:02d}{:02d}'.format(date['year'], date['start_month'], date['start_day'],
                                                                        date['year'], date['end_month'], date['end_day'])
        total_test_path = os.path.join('./results', experiment_name, 'eval', total_test_date)
        if not os.path.isdir(total_test_path):
            os.mkdir(total_test_path)

        # Copy monthly eval file to total path
        for current_test_path in test_date_list:
            for f in os.listdir(current_test_path):
                if 'ipynb' in f or 'total' in f:
                    continue
                shutil.copy(os.path.join(current_test_path, f), total_test_path)

        # Recreate total test stat file in eval_only mode
        if args.eval_only:
            recreate_total_stat(total_test_path)
            
        total_eval_date_files = sorted([os.path.join(total_test_path, f) \
                                        for f in os.listdir(total_test_path) \
                                        if 'ipynb' not in f and 'total' not in f])
        total_stat = get_nims_stat(total_eval_date_files)

        total_ldaps_eval_date_files = get_ldaps_eval_date_files(args.model_utc, date)
        total_ldaps_stat = get_nims_stat(total_ldaps_eval_date_files)

        # Plot graph for each stat
        plot_stat_graph(total_ldaps_stat, total_stat, date, experiment_name)
        
    print('FINISHED')