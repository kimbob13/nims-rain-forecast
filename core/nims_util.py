import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import Subset
import numpy as np
import torchvision.transforms as transforms

from model.unet_model import UNet, SuccessiveUNet, AttentionUNet
from model.conv_lstm import EncoderForecaster
from .nims_loss import *

import os
import random
import argparse
import time
from pathlib import Path

from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count, Pool
from collections import namedtuple
import itertools

try:
    import setproctitle
except:
    pass

__all__ = ['NIMSStat', 'create_results_dir', 'select_date', 'parse_args', 'set_device', 'undersample', 'fix_seed',
           'set_model', 'select_pretrained_model', 'set_optimizer', 'set_experiment_name', 
           'get_min_max_values', 'get_min_max_normalization']

NIMSStat = namedtuple('NIMSStat', 'acc, csi, pod, far, f1, bias')
MONTHLY_MODE = 1
DAILY_MODE = 2

LIST_NUM_MODE = 1
PRETRAIN_NAME_MODE = 2


def create_results_dir(experiment_name):
    # Base results directory
    results_dir = os.path.join('./results', experiment_name)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # Create evaluation directory if not
    eval_dir = os.path.join(results_dir, 'eval')
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    # Create comparison_graph directory if not
    graph_dir = os.path.join(results_dir, 'comparison_graph')
    if not os.path.isdir(graph_dir):
        os.mkdir(graph_dir)


def select_date(test=False):
    format_str = '[{:<14s}] {:^25s} : '

    # Mode selection
    while True:
        try:
            print(format_str.format('1. Stat Type',  '(1) Montly (2) Daily'), end='')
            date_mode = int(input())

            if date_mode not in [MONTHLY_MODE, DAILY_MODE]:
                print('You must enter value between 1 or 2')
                continue
            # elif test == True and date_mode == DAILY_MODE:
            #     print('You must select [Monthly] mode for test')
            #     continue

        except ValueError:
            print('You must enter integer only')
            continue

        if date_mode == MONTHLY_MODE:
            date_format = 'YYYY-MM'
        elif date_mode == DAILY_MODE:
            date_format = 'YYYY-MM-DD'

        break

    VALID_MONTH = [6, 7, 8, 9]
    MONTH_DAY   = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    LEAP_YEAR   = (2020)

    def _check_valid_day(month, day):
        valid_day = MONTH_DAY[month]
        if (month == 2) and (year in LEAP_YEAR):
            valid_day = 29

        if (day < 1) or (day > valid_day):
            return False

        return True
    
    # Start date input
    while True:
        print(format_str.format('2. Start Date', '(' + date_format + ')'), end='')
        start_date = input()
        if len(start_date) != len(date_format):
            print('Please follow the format ({})'.format(date_format))
            continue

        # Parse input
        start_date = start_date.split('-')
        start_year, start_month = map(int, start_date[:2])

        # Year check
        if start_year not in [2018, 2019, 2020]:
            print("You must specify year 2018 or 2019 or 2020")
            continue

        # Month check
        if start_month not in VALID_MONTH:
            print("You must specify start month between {} to {}".format(VALID_MONTH[0], VALID_MONTH[-1]))
            continue
        
        # Day check
        start_day = 1
        if date_mode == DAILY_MODE:
            start_day = int(start_date[2])
            if not _check_valid_day(start_month, start_day):
                print("You must specifiy valid day for month '{}'".format(start_month))
                continue
        
        break

    # End date input
    while True:
        print(format_str.format('3. End Date', '(' + date_format + ')'), end='')
        end_date = input()
        if len(end_date) != len(date_format):
            print('Please follow the format ({})'.format(date_format))
            continue

        # Parse input
        end_date = end_date.split('-')
        end_year, end_month = map(int, end_date[:2])

        # Year check
        if end_year != start_year:
            print("You must specify same end year as start year")
            continue

        # Month check
        if (end_month < start_month) or (end_month not in VALID_MONTH):
            print("You must specify start month between {} to {}".format(start_month, VALID_MONTH[-1]))
            continue
        
        # Day check
        end_day = MONTH_DAY[end_month]
        if date_mode == DAILY_MODE:
            if not _check_valid_day(end_month, end_day):
                print("You must specifiy valid day for month '{}'".format(end_month))
                continue
            end_day = int(end_date[-1])
        break

    print()
    
    return {'year': start_year, 'start_month': start_month, 'start_day': start_day,
            'end_month': end_month, 'end_day': end_day}


def select_pretrained_model():
    format_str = '[{:<14s}] {:^25s} : '
    
    while True:
        try:
            print(format_str.format('1. Pre-Train Select',  '(1) List & Number (2) Insert Pre-Trained Dir Name'), end='')
            pretrain_mode = int(input())

            if pretrain_mode not in [LIST_NUM_MODE, PRETRAIN_NAME_MODE]:
                print('You must enter value between 1 or 2')
                continue
        except ValueError:
            print('You must enter integer only')
            continue

        if pretrain_mode == LIST_NUM_MODE:
            paths = sorted(Path("./results").iterdir(), key=os.path.getmtime, reverse=True)
            dir_name_lst = [str(path).replace("results/", "") for path in paths
                            if path.is_dir() and ('nims' in path.as_posix())]

            while True:
                for idx, dir_name in enumerate(dir_name_lst):
                    print("({}) {}".format(idx + 1, dir_name))

                print(format_str.format('1-1. List & Number',  'Select Model (1)~({})'.format(len(paths))), end='')
                model_idx = int(input())

                if model_idx not in list(range(len(paths))):
                    print('You must enter value from 1 to {}'.format(len(paths)))
                    continue
                else:
                    pretrain_model = dir_name_lst[model_idx - 1]
                    break
        elif pretrain_mode == PRETRAIN_NAME_MODE:
            print(format_str.format('1-2. Insert Pre-Trained Dir Name',  "Dir Name(Having 'trained_weight.pt'): "), end='')
            pretrain_model = input()
            break
        break

    return pretrain_model
    

def parse_args():
    parser = argparse.ArgumentParser(description='NIMS rainfall data prediction')

    common = parser.add_argument_group('common')
    common.add_argument('--model', default='unet', type=str, help='which model to use [unet, attn_unet, convlstm]')
    common.add_argument('--dataset_dir', default='/home/osilab12/ssd2', type=str, help='root directory of dataset')
    common.add_argument('--device', default='0', type=str, help='which device to use')
    common.add_argument('--num_workers', default=5, type=int, help='# of workers for dataloader')
    common.add_argument('--eval_only', default=False, help='when enabled, do not run test epoch, only creating graph', action='store_true')
    common.add_argument('--custom_name', default=None, type=str, help='add customize experiment name')
    # common.add_argument('--debug', help='turn on debugging print', action='store_true')

    unet = parser.add_argument_group('unet related')
    unet.add_argument('--n_blocks', default=5, type=int, help='# of blocks in Down and Up phase')
    unet.add_argument('--start_channels', default=64, type=int, help='# of channels after first block of unet')
    unet.add_argument('--pos_loc', default=0, type=int, help='index of position where learnable position inserted')
    unet.add_argument('--pos_dim', default=0, type=int, help="# of learnable position channels")
    unet.add_argument('--bilinear', default=False, help='use bilinear for upsample instead of transpose conv', action='store_true')
    unet.add_argument('--cross_entropy_weight', default=False, help='use weight for cross entropy loss', action='store_true')

    convlstm = parser.add_argument_group('convlstm related')
    convlstm.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension in ConvLSTM')

    nims_dataset = parser.add_argument_group('nims dataset related')
    nims_dataset.add_argument('--window_size', default=6, type=int, help='# of input sequences in time')
    nims_dataset.add_argument('--model_utc', default=0, type=int, help='base UTC time of data (0, 6, 12, 18)')
    nims_dataset.add_argument('--lite', default=False, help='lite version of nims dataset', action='store_true')
    # nims_dataset.add_argument('--variables', nargs='+',
    #                           help='which variables to use (rain, cape, etc.). \
    #                                 Can be single number which specify how many variables to use \
    #                                 or list of variables name')
    nims_dataset.add_argument('--sampling_ratio', default=1.0, type=float, help='the ratio of undersampling')
    nims_dataset.add_argument('--heavy_rain', default=False, help='if it is set, rain threshold becomes 10mm/hr instead of 0.1mm/hr', action='store_true')
    nims_dataset.add_argument('--normalization', default=False, help='normalize input data', action='store_true')
    nims_dataset.add_argument('--reference', default=None, type=str, help='which data to be used as a ground truth')

    hyperparam = parser.add_argument_group('hyper-parameters')
    hyperparam.add_argument('--num_epochs', default=100, type=int, help='# of training epochs')
    hyperparam.add_argument('--batch_size', default=1, type=int, help='batch size')
    hyperparam.add_argument('--optimizer', default='adam', type=str, help='which optimizer to use (rmsprop, adam, sgd)')
    hyperparam.add_argument('--lr', default=0.001, type=float, help='learning rate of optimizer')
    hyperparam.add_argument('--wd', default=0, type=float, help='weight decay')
    
    finetune = parser.add_argument_group('finetune related')
    finetune.add_argument('--final_test_time', default=None, type=str, help='the final date of fine-tuning')
    finetune.add_argument('--finetune_lr_ratio', default=0.1, type=float, help='the ratio of fine-tuning learning rate to the original learning rate')
    finetune.add_argument('--finetune_num_epochs', default=3, type=int, help='# of fine-tuning epochs')
    
    args = parser.parse_args()

    assert args.model_utc in [0, 6, 12, 18], \
           'model_utc must be one of [0, 6, 12, 18]'

    assert args.reference in ['aws', 'reanalysis'], \
           'reference must be one of [aws, reanalysis]'
    
    return args


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def set_device(args):
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        device = torch.device('cuda')

    return device


def set_model(sample, device, args, train=True,
              experiment_name=None, finetune=False, model_path=None):
    # Create a model and criterion
    num_classes = 2

    if 'unet' in args.model:
        if args.model == 'unet':
            model = UNet(n_channels=sample.shape[0],
                         n_classes=num_classes,
                         n_blocks=args.n_blocks,
                         start_channels=args.start_channels,
                         pos_loc=args.pos_loc,
                         pos_dim=args.pos_dim,
                         bilinear=args.bilinear,
                         batch_size=args.batch_size)

        elif args.model == 'attn_unet':
            model = AttentionUNet(n_channels=sample.shape[0],
                                  n_classes=num_classes,
                                  n_blocks=args.n_blocks,
                                  start_channels=args.start_channels,
                                  pos_loc=args.pos_loc,
                                  pos_dim=args.pos_dim,
                                  bilinear=args.bilinear,
                                  batch_size=args.batch_size)

        elif args.model == 'suc_unet':
            model = SuccessiveUNet(n_channels=sample.shape[0],
                                   n_classes=num_classes,
                                   n_blocks=args.n_blocks,
                                   start_channels=args.start_channels,
                                   pos_loc=args.pos_loc,
                                   pos_dim=args.pos_dim,
                                   bilinear=args.bilinear,
                                   batch_size=args.batch_size)
        
        criterion = NIMSCrossEntropyLoss(args=args,
                                         device=device,
                                         num_classes=num_classes,
                                         use_weights=args.cross_entropy_weight,
                                         experiment_name=experiment_name)
        
    elif args.model == 'convlstm':
        # assert args.window_size == args.target_num, \
        #        'window_size and target_num must be same for ConvLSTM'

        model = EncoderForecaster(input_channels=sample.shape[1],
                                  hidden_dim=args.hidden_dim,
                                  num_classes=num_classes)
        # criterion = MSELoss()
        criterion = NIMSCrossEntropyLoss(args=args,
                                         device=device,
                                         num_classes=num_classes,
                                         use_weights=args.cross_entropy_weight,
                                         experiment_name=experiment_name)

    if finetune:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint, strict=False)

    model = DataParallel(model)

    return model, criterion

def set_optimizer(model, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99,
                              weight_decay=args.wd, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
                                  alpha=0.9, eps=1e-6)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 60)

    return optimizer, scheduler


def set_experiment_name(args, date):
    """
    Set experiment name and assign to process name

    <Parameters>
    args [argparse]: parsed argument
    """
    if date != '':
        date_str = ' ({:4d}{:02d}{:02d}-{:04d}{:02d}{:02d})'.format(date['year'], date['start_month'], date['start_day'],
                                                                    date['year'], date['end_month'], date['end_day'])
    else:
        date_str = ''

    if args.wd != 0:
        wd_str = '{:1.0e}'.format(args.wd)
    else:
        wd_str = '0'

    cross_entropy_weight = ''
    if args.cross_entropy_weight:
        cross_entropy_weight = '_weight'

    normalization = ''
    if args.normalization:
        normalization = '_norm'

    bilinear = ''
    if args.bilinear:
        bilinear = '_bilinear'

    heavy_rain = ''
    if args.heavy_rain:
        heavy_rain = '_heavy'

    custom_name = ''
    if args.custom_name:
        custom_name = '_' + args.custom_name

    if 'unet' in args.model:
        experiment_name = 'nims-{}-utc{}-{}_nb{}_ch{}_ws{}_ep{}_bs{}_pos{}-{}_sr{}_{}{}_wd{}{}{}{}{}{}' \
                          .format(args.reference,
                                  args.model_utc,
                                  args.model,
                                  args.n_blocks,
                                  args.start_channels,
                                  args.window_size,
                                  args.num_epochs,
                                  args.batch_size,
                                  args.pos_loc,
                                  args.pos_dim,
                                  args.sampling_ratio,
                                  args.optimizer,
                                  args.lr,
                                  wd_str,
                                  cross_entropy_weight,
                                  normalization,
                                  bilinear,
                                  heavy_rain,
                                  custom_name)

    elif args.model == 'convlstm':
        experiment_name = 'nims-{}-convlstm_ws{}_hd{}_ep{}_bs{}_sr{}_{}{}{}{}' \
                          .format(args.reference,
                                  args.window_size,
                                  args.hidden_dim,
                                  args.num_epochs,
                                  args.batch_size,
                                  args.sampling_ratio,
                                  args.optimizer,
                                  args.lr,
                                  normalization,
                                  custom_name)

    try:
        setproctitle.setproctitle(experiment_name)
    except:
        pass

    return experiment_name


def _get_min_max_values(dataset, indices, queue=None):
    '''
        Return min and max values of ldaps_inputs in train and test dataset
            Args : NIMS_train_dataset, NIMS_test_dataset
                (ldaps_input, gt, target_time_tensor = train_dataset[i])
            Returns :
                max_values : max_values [features, ]
                min_values : min_values [features,]
    '''
    
    max_values = None
    min_values = None
    model_name = dataset.model

    # Check out training set
    for i, idx in enumerate(indices):
        # Pop out data
        ldaps_input, _, _ = dataset[idx]
        ldaps_input = ldaps_input.numpy()
        
        # Get a shape
        if 'unet' in model_name:
            features, height, width = ldaps_input.shape
            features_reshape = np.reshape(ldaps_input, (features, -1))
        elif model_name == 'convlstm':
            seq_len, features, height, width = ldaps_input.shape
            features_reshape = np.reshape(ldaps_input, (seq_len * features, -1))
        
        # Evaluate min / max on current data
        temp_max = np.amax(features_reshape, axis=-1)
        temp_min = np.amin(features_reshape, axis=-1)

        # Edge case
        if i == 0:
            max_values = temp_max
            min_values = temp_min

        # Comparing max / min values
        max_values = np.maximum(max_values, temp_max)
        min_values = np.minimum(max_values, temp_min)

    if queue:
        queue.put((max_values, min_values))
    else:
        return max_values, min_values

    
def get_min_max_values(dataset):
    # Make indices list
    indices = list(range(len(dataset)))

    num_processes = cpu_count() // 4
    num_indices_per_process = len(indices) // num_processes

    # Create queue
    queues = []
    for i in range(num_processes):
        queues.append(Queue())

    # Create processes
    processes = []
    for i in range(num_processes):
        start_idx = i * num_indices_per_process
        end_idx = start_idx + num_indices_per_process

        if i == num_processes - 1:
            processes.append(Process(target=_get_min_max_values,
                                     args=(dataset, indices[start_idx:],
                                           queues[i])))
        else:
            processes.append(Process(target=_get_min_max_values,
                                     args=(dataset, indices[start_idx:end_idx],
                                           queues[i])))

    # Start processes
    for i in range(num_processes):
        processes[i].start()

    # Join processes
    animation = "|/-\\"
    idx = 0
    alive_flag = [True] * num_processes
    while True:
        for i in range(num_processes):
            processes[i].join(timeout=0)
            if not processes[i].is_alive():
                alive_flag[i] = False
        
        if True not in alive_flag:
            print()
            break

        print('Normalization Start. Please Wait...{}'.format(animation[idx % len(animation)]), end='\r')
        idx += 1
        time.sleep(0.1)
    
    print('Normalization End!')
    print()
    
    # Get return value of each process
    max_values, min_values = None, None
    for i in range(num_processes):
        proc_result = queues[i].get()
        
        if i == 0:
            max_values = proc_result[0]
            min_values = proc_result[1]
        else:
            max_values = np.maximum(max_values, proc_result[0])
            min_values = np.minimum(min_values, proc_result[1])

    # Convert to PyTorch tensor
    max_values = torch.tensor(max_values)
    min_values = torch.tensor(min_values)

    return max_values, min_values


def get_min_max_values_no_mp(dataset):
    '''
        get_min_max_values function with no multiprocessing
    '''

    indices = list(range(len(dataset)))

    max_values, min_values = _get_min_max_values(dataset, indices)
    
    max_values = torch.tensor(max_values)
    min_values = torch.tensor(min_values)
    
    return max_values, min_values


def get_min_max_normalization(max_values, min_values):
    '''
        Transform for min_max normalization
            Args : max_values [features, ] and min_values [features, ]
            Returns :
                transform object (for broadcasting, permute is used.)
    '''
    transform = transforms.Compose([
        lambda x : x.permute(1, 2, 0),\
        lambda x : (x - min_values) / (max_values - min_values),\
        lambda x : x.permute(2, 0, 1)
    ])

    return transform

def _undersample(train_dataset, indices, pid=None, queue=None):
    target_nonzero_means = []

    for idx in indices:
        target = train_dataset.get_real_gt(idx)
        total_pixel = len(target.flatten())

        # Missing values are -99. Change these values to 0
        target[target < 0] = 0.0

        # Get average rain for target
        max_one_hour_value = np.amax(target)
        nonzero_count = np.count_nonzero(target)
        if (max_one_hour_value == 0) or \
           (nonzero_count < (total_pixel * 0.000019)):
            # If the # of nonzero pixels are less than 0.0019% of total pixels,
            # we treat it as non-rainy instance, so make avg_value as 0
            target_nonzero_means.append((idx, 0))
        else:
            # Average over nonzero data
            avg_rain = np.mean(target[np.nonzero(target)])
            target_nonzero_means.append((idx, avg_rain))

    if queue:
        queue.put(target_nonzero_means)
    else:
        return target_nonzero_means

    
def undersample(train_dataset, sampling_ratio):
    # Make indices list
    indices = list(range(len(train_dataset)))

    num_processes = cpu_count() // 2
    num_indices_per_process = len(indices) // num_processes

    # Create queue
    queues = []
    for i in range(num_processes):
        queues.append(Queue())

    # Create processes
    processes = []
    for i in range(num_processes):
        start_idx = i * num_indices_per_process
        end_idx = start_idx + num_indices_per_process

        if i == num_processes - 1:
            processes.append(Process(target=_undersample,
                                     args=(train_dataset, indices[start_idx:],
                                           i, queues[i])))
        else:
            processes.append(Process(target=_undersample,
                                     args=(train_dataset, indices[start_idx:end_idx],
                                           i, queues[i])))

    # Start processes
    for i in range(num_processes):
        processes[i].start()

    # Get return value of each process
    target_nonzero_means = []
    for i in tqdm(range(num_processes)):
        proc_result = queues[i].get()
        target_nonzero_means.extend(proc_result)

    # Join processes
    for i in range(num_processes):
        processes[i].join()

    # Sampling index for non-rainy instance
    selected_idx = []
    for idx, target_nonzero_mean in target_nonzero_means:
        if target_nonzero_mean < 2.0: # This case is not rainy hour
            if np.random.uniform() < sampling_ratio:
                selected_idx.append(idx)
        else: # This case is rainy hour
            selected_idx.append(idx)

    # Undersample train dataset
    train_dataset = Subset(train_dataset, selected_idx)

    return train_dataset