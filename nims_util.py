import torch
import torch.optim as optim
from torch.utils.data import Subset
import numpy as np

from model.unet_model import UNet, AttentionUNet
from model.conv_lstm import EncoderForecaster
from model.persistence import Persistence

from nims_loss import MSELoss, NIMSCrossEntropyLoss

import os
import sys
import random
import argparse

from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count

try:
    import setproctitle
except:
    pass

__all__ = ['create_results_dir', 'parse_args', 'set_device', 'undersample',
           'fix_seed', 'set_model', 'set_optimizer', 'set_experiment_name']

def create_results_dir(experiment_name=None):
    # Base results directory
    results_dir = './results'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # Create log directory if not
    log_dir = os.path.join(results_dir, 'log')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # Create experiment-wise directory in log directory
    if experiment_name:
        experiment_dir = os.path.join(log_dir, experiment_name)
        if not os.path.isdir(experiment_dir):
            os.mkdir(experiment_dir)

    # Create evaluation directory if not
    eval_dir = os.path.join(results_dir, 'eval')
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    # Create trained_model directory if not
    model_dir = os.path.join(results_dir, 'trained_model')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='NIMS rainfall data prediction')

    common = parser.add_argument_group('common')
    common.add_argument('--model', default='unet', type=str, help='which model to use [unet, attn_unet, convlstm, persistence]')
    common.add_argument('--dataset_dir', default='/home/osilab12/ssd/NIMS_LDPS', type=str, help='root directory of dataset')
    common.add_argument('--device', default='0', type=str, help='which device to use')
    common.add_argument('--num_workers', default=6, type=int, help='# of workers for dataloader')
    common.add_argument('--baseline_name', default=None, type=str, help='name of baseline experiment you want to compare')
    common.add_argument('--custom_name', default=None, type=str, help='add customize experiment name')
    common.add_argument('--debug', help='turn on debugging print', action='store_true')

    unet = parser.add_argument_group('unet related')
    unet.add_argument('--n_blocks', default=6, type=int, help='# of blocks in Down and Up phase')
    unet.add_argument('--start_channels', default=64, type=int, help='# of channels after first block of unet')
    unet.add_argument('--pos_dim', default=0, type=int, help="# of learnable position channels")
    unet.add_argument('--cross_entropy_weight', default=False, help='use weight for cross entropy loss', action='store_true')

    nims_dataset = parser.add_argument_group('nims dataset related')
    nims_dataset.add_argument('--test_time', default=None, type=str, help='date of test')
    nims_dataset.add_argument('--window_size', default=10, type=int, help='# of input sequences in time')
    nims_dataset.add_argument('--model_utc', default=0, type=int, help='base UTC time of data (0, 6, 12, 18)')
    # nims_dataset.add_argument('--variables', nargs='+',
    #                           help='which variables to use (rain, cape, etc.). \
    #                                 Can be single number which specify how many variables to use \
    #                                 or list of variables name')
    nims_dataset.add_argument('--sampling_ratio', default=1.0, type=float, help='the ratio of undersampling')

    hyperparam = parser.add_argument_group('hyper-parameters')
    hyperparam.add_argument('--num_epochs', default=50, type=int, help='# of training epochs')
    hyperparam.add_argument('--batch_size', default=1, type=int, help='batch size')
    hyperparam.add_argument('--optimizer', default='adam', type=str, help='which optimizer to use (rmsprop, adam, sgd)')
    hyperparam.add_argument('--lr', default=0.001, type=float, help='learning rate of optimizer')

    args = parser.parse_args()
    if args.model == 'persistence':
        args.test_only = True
        args.window_size = 1
        args.batch_size = 1

    if args.model_utc not in [0, 6, 12, 18]:
        print('model_utc must be one of [0, 6, 12, 18]')
        sys.exit()

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
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        device = torch.device('cuda:0')

    return device

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

def set_model(sample, device, args, train=True):
    # Create a model and criterion
    num_classes = 2

    if (args.model == 'unet') or (args.model == 'attn_unet'):
        if args.model == 'unet':
            model = UNet(n_channels=sample.shape[0],
                         n_classes=num_classes,
                         n_blocks=args.n_blocks,
                         start_channels=args.start_channels,
                         pos_dim=args.pos_dim)

        elif args.model == 'attn_unet':
            model = AttentionUNet(n_channels=sample.shape[0],
                                  n_classes=num_classes,
                                  n_blocks=args.n_blocks,
                                  start_channels=args.start_channels,
                                  pos_dim=args.pos_dim)

        criterion = NIMSCrossEntropyLoss(device, num_classes=num_classes,
                                         use_weights=args.cross_entropy_weight,
                                         train=train)

        num_lat = sample.shape[1] # the number of latitudes (originally 253)
        num_lon = sample.shape[2] # the number of longitudes (originally 149)

    elif args.model == 'convlstm':
        assert args.window_size == args.target_num, \
               'window_size and target_num must be same for ConvLSTM'

        model = EncoderForecaster(input_channels=sample.shape[1],
                                  hidden_channels=[64, 128],
                                  kernel_size=3,
                                  seq_len=args.window_size,
                                  device=device)
        criterion = MSELoss()

        num_lat = sample.shape[2] # the number of latitudes (originally 253)
        num_lon = sample.shape[3] # the number of longitudes (originally 149)

    elif args.model == 'persistence':
        model = Persistence(num_classes=num_classes, device=device)
        criterion = NIMSCrossEntropyLoss(device, num_classes=num_classes,
                                         use_weights=args.cross_entropy_weight,
                                         train=train)

        num_lat = sample.shape[1] # the number of latitudes (originally 253)
        num_lon = sample.shape[2] # the number of longitudes (originally 149)

    return model, criterion, num_lat, num_lon

def set_optimizer(model, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99,
                              weight_decay=5e-4, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
                                  alpha=0.9, eps=1e-6)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    return optimizer

def set_experiment_name(args):
    """
    Set experiment name and assign to process name

    <Parameters>
    args [argparse]: parsed argument
    """
    test_time = ''
    if args.test_time:
        if '-' in args.test_time:
            _start, _end = args.test_time.strip().split('-')
            start_test_time = ''
            end_test_time = ''
            try:
                start_test_time += '_{}'.format(int(_start[2:4]))
                start_test_time += '{:02d}'.format(int(_start[4:6]))
                start_test_time += '{:02d}'.format(int(_start[6:8]))
                start_test_time += '{:02d}'.format(int(_start[8:10]))
            except:
                pass
                
            try:
                end_test_time += '{}'.format(int(_end[2:4]))
                end_test_time += '{:02d}'.format(int(_end[4:6]))
                end_test_time += '{:02d}'.format(int(_end[6:8]))
                end_test_time += '{:02d}'.format(int(_end[8:10]))
            except:
                pass

            test_time = start_test_time + '-' + end_test_time

        else:
            try:
                test_time += '_{}'.format(int(args.test_time[2:4]))
                test_time += '{:02d}'.format(int(args.test_time[4:6]))
                test_time += '{:02d}'.format(int(args.test_time[6:8]))
                test_time += '{:02d}'.format(int(args.test_time[8:10]))
            except:
                pass
        

    cross_entropy_weight = ''
    if args.cross_entropy_weight:
        cross_entropy_weight = '_weight'

    custom_name = ''
    if args.custom_name:
        custom_name = '_' + args.custom_name

    if args.model == 'unet':            
        experiment_name = 'nims-utc{}-unet_nb{}_ch{}_ws{}_ep{}_bs{}_pos{}_sr{}_{}{}{}{}{}' \
                          .format(args.model_utc,
                                  args.n_blocks,
                                  args.start_channels,
                                  args.window_size,
                                  args.num_epochs,
                                  args.batch_size,
                                  args.pos_dim,
                                  args.sampling_ratio,
                                  args.optimizer,
                                  args.lr,
                                  cross_entropy_weight,
                                  custom_name,
                                  test_time)

    elif args.model == 'attn_unet':
        experiment_name = 'nims-utc{}-attn_unet_nb{}_ch{}_ws{}_ep{}_bs{}_pos{}_sr{}_{}{}{}{}{}' \
                          .format(args.model_utc,
                                  args.n_blocks,
                                  args.start_channels,
                                  args.window_size,
                                  args.num_epochs,
                                  args.batch_size,
                                  args.pos_dim,
                                  args.sampling_ratio,
                                  args.optimizer,
                                  args.lr,
                                  cross_entropy_weight,
                                  custom_name,
                                  test_time)

    # elif args.model == 'convlstm':
    #     experiment_name = 'nims-convlstm_ws{}_tn{}_ep{}_bs{}_sr{}_{}{}{}' \
    #                       .format(args.window_size,
    #                               args.target_num,
    #                               args.num_epochs,
    #                               args.batch_size,
    #                               args.sampling_ratio,
    #                               args.optimizer,
    #                               args.lr,
    #                               test_time)

    elif args.model == 'persistence':
        experiment_name = 'nims-persistence{}{}' \
                          .format(cross_entropy_weight,
                                  test_time)

    if args.debug:
        experiment_name += '_debug'

    try:
        setproctitle.setproctitle(experiment_name)
    except:
        pass

    return experiment_name