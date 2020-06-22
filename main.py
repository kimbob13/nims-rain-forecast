import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

from model.unet_model import UNet
from model.conv_lstm import EncoderForecaster

from nims_dataset import NIMSDataset, ToTensor
from nims_loss import MSELoss, NIMSCrossEntropyLoss
from nims_trainer import NIMSTrainer
from nims_variable import parse_variables

try:
    from torchsummary import summary
    import setproctitle
except:
    pass

from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count
import random
import os
import sys
import argparse

def create_results_dir():
    # Base results directory
    results_dir = './results'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # Create log directory if not
    log_dir = os.path.join(results_dir, 'log')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

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
    common.add_argument('--model', default='unet', type=str, help='which model to use')
    common.add_argument('--dataset_dir', type=str, help='root directory of dataset')
    common.add_argument('--device', default='0', type=str, help='which device to use')
    common.add_argument('--num_workers', default=6, type=int, help='# of workers for dataloader')
    common.add_argument('--test_only', default=False, action='store_true', help='Test only mode')
    common.add_argument('--baaseline_name', default=None, type=str, help='name of baseline experiment you want to compare')
    common.add_argument('--custom_name', default=None, type=str, help='add customize experiment name')
    common.add_argument('--debug', help='turn on debugging print', action='store_true')

    unet = parser.add_argument_group('unet related')
    unet.add_argument('--n_blocks', default=6, type=int, help='# of blocks in Down and Up phase')
    unet.add_argument('--start_channels', default=64, type=int, help='# of channels after first block of unet')
    unet.add_argument('--no_cross_entropy_weight', default=False, help='use weight for cross entropy loss', action='store_true')

    nims_dataset = parser.add_argument_group('nims dataset related')
    nims_dataset.add_argument('--window_size', default=10, type=int, help='# of input sequences in time')
    nims_dataset.add_argument('--target_num', default=1, type=int, help='# of output sequences to evaluate')
    nims_dataset.add_argument('--variables', nargs='+',
                              help='which variables to use (rain, cape, etc.). \
                                    Can be single number which specify how many variables to use \
                                    or list of variables name')
    nims_dataset.add_argument('--block_size', default=1, type=int, help='the size of aggregated block')
    nims_dataset.add_argument('--aggr_method', default=None, type=str, help='The method of block aggregation (max, avg)')
    nims_dataset.add_argument('--start_train_year', default=2009, type=int, help='start year for training')
    nims_dataset.add_argument('--end_train_year', default=2017, type=int, help='end year for training')
    nims_dataset.add_argument('--start_month', default=1, type=int, help='month range for train and test')
    nims_dataset.add_argument('--end_month', default=12, type=int, help='month range for train and test')
    
    nims_dataset.add_argument('--sampling_ratio', default=1.0, type=float, help='the ratio of undersampling')

    hyperparam = parser.add_argument_group('hyper-parameters')
    hyperparam.add_argument('--num_epochs', default=50, type=int, help='# of training epochs')
    hyperparam.add_argument('--batch_size', default=1, type=int, help='batch size')
    hyperparam.add_argument('--optimizer', default='adam', type=str, help='which optimizer to use (rmsprop, adam, sgd)')
    hyperparam.add_argument('--lr', default=0.001, type=float, help='learning rate of optimizer')

    args = parser.parse_args()

    return args

def _undersample(train_dataset, indices, pid=None, queue=None):
    target_nonzero_means = []

    for idx in indices:
        target = train_dataset[idx][1]

        nonzero_idx = torch.nonzero(target, as_tuple=False)
        variable_idx = nonzero_idx[:, 0]
        height_idx = nonzero_idx[:, 1]
        width_idx = nonzero_idx[:, 2]

        target_nonzero_mean = torch.mean(target[variable_idx, height_idx, width_idx])
        target_nonzero_means.append((idx, target_nonzero_mean.item()))

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

    selected_idx = []
    for idx, target_nonzero_mean in target_nonzero_means:
        if target_nonzero_mean < 2.0: # This case is not rainy hour
            if np.random.uniform() < sampling_ratio:
                selected_idx.append(idx)
        else: # This case is rainy hour
            selected_idx.append(idx)

    return selected_idx

def set_experiment_name(args):
    """
    Set experiment name and assign to process name

    <Parameters>
    args [argparse]: parsed argument
    """
    train_date = '{}{:02d}-{}{:02d}'.format(str(args.start_train_year)[-2:],
                                          args.start_month,
                                          str(args.end_train_year)[-2:],
                                          args.end_month)

    if args.model == 'unet':
        no_cross_entropy_weight = ''
        if args.no_cross_entropy_weight:
            no_cross_entropy_weight = 'noweight_'
            
        experiment_name = 'nims_unet_nb{}_ch{}_ws{}_tn{}_ep{}_bs{}_sr{}_{}{}_{}{}' \
                          .format(args.n_blocks,
                                  args.start_channels,
                                  args.window_size,
                                  args.target_num,
                                  args.num_epochs,
                                  args.batch_size,
                                  args.sampling_ratio,
                                  args.optimizer,
                                  args.lr,
                                  no_cross_entropy_weight,
                                  train_date)

    elif args.model == 'convlstm':
        experiment_name = 'nims_convlstm_ws{}_tn{}_ep{}_bs{}_sr{}_{}{}_{}' \
                          .format(args.window_size,
                                  args.target_num,
                                  args.num_epochs,
                                  args.batch_size,
                                  args.sampling_ratio,
                                  args.optimizer,
                                  args.lr,
                                  train_date)

    if args.custom_name:
        experiment_name += ('_' + args.custom_name)

    if args.debug:
        experiment_name += '_debug'

    try:
        setproctitle.setproctitle(experiment_name)
    except:
        pass

    return experiment_name

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda:0')

    # Fix the seed
    torch.manual_seed(2020)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2020)
    np.random.seed(2020)
    random.seed(2020)

    # Create necessary directory
    create_results_dir()

    # Parse NIMS dataset variables
    variables = parse_variables(args.variables)

    # Train and test dataset
    nims_train_dataset = NIMSDataset(model=args.model,
                                     window_size=args.window_size,
                                     target_num=args.target_num,
                                     variables=variables,
                                     block_size=args.block_size,
                                     aggr_method=args.aggr_method,
                                     train_year=(args.start_train_year,
                                                 args.end_train_year),
                                     month=(args.start_month,
                                            args.end_month),
                                     train=True,
                                     transform=ToTensor(),
                                     root_dir=args.dataset_dir,
                                     debug=args.debug)
    
    nims_test_dataset  = NIMSDataset(model=args.model,
                                     window_size=args.window_size,
                                     target_num=args.target_num,
                                     variables=variables,
                                     block_size=args.block_size,
                                     aggr_method=args.aggr_method,
                                     train_year=(args.start_train_year,
                                                 args.end_train_year),
                                     month=(args.start_month,
                                            args.end_month),
                                     train=False,
                                     transform=ToTensor(),
                                     root_dir=args.dataset_dir,
                                     debug=args.debug)

    # Get a sample for getting shape of each tensor
    sample, _ = nims_train_dataset[0]
    if args.debug:
        print('[main] one images sample shape:', sample.shape)

    # Create a model and criterion
    if args.model == 'unet':
        num_classes = 4
        model = UNet(n_channels=sample.shape[0],
                     n_classes=num_classes,
                     n_blocks=args.n_blocks,
                     start_channels=args.start_channels,
                     target_num=args.target_num)
        criterion = NIMSCrossEntropyLoss(device, num_classes=num_classes,
                                         no_weights=args.no_cross_entropy_weight)

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

    if args.debug:
        # XXX: Currently, torchsummary doesn't run on ConvLSTM
        print('[main] num_lat: {}, num_lon: {}'.format(num_lat, num_lon))
        if args.model == 'unet':
            model.to(device)
            try:
                summary(model, input_size=sample.shape)
            except:
                print('If you want to see summary of model, install torchsummary')

    # Undersampling
    if args.sampling_ratio != 1.0:
        print('=' * 20, 'Under Sampling', '=' * 20)
        print('Before Under sampling, train len:', len(nims_train_dataset))

        print('Please wait...')
        selected_idx = undersample(nims_train_dataset, args.sampling_ratio)
        nims_train_dataset = Subset(nims_train_dataset, selected_idx)

        print('After Under sampling, train len:', len(nims_train_dataset))
        print('=' * 20, 'Finish Under Sampling', '=' * 20)
        print()
        
    # Create dataloaders
    train_loader = DataLoader(nims_train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(nims_test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # Set the optimizer
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

    # Set experiment name and use it as process name if possible
    experiment_name = set_experiment_name(args)

    # Load model weight when test only mode
    if args.test_only:
        weight_path = os.path.join('./results', 'trained_model', experiment_name + '.pt')
        model.load_state_dict(torch.load(weight_path))

    # Start training and testing
    nims_trainer = NIMSTrainer(model, criterion, optimizer, device,
                               train_loader, test_loader,
                               len(nims_train_dataset), len(nims_test_dataset),
                               num_lat, num_lon, experiment_name, args)
    if not args.test_only:
        nims_trainer.train()
    nims_trainer.test()
