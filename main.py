import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.unet_model import UNet
from model.conv_lstm import EncoderForecaster
from nims_dataset import NIMSDataset, ToTensor
from nims_loss import MSELoss, NIMSCrossEntropyLoss
from nims_trainer import NIMSTrainer
from nims_variable import parse_variables

from torchsummary import summary
try:
    import setproctitle
except:
    pass


import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='NIMS rainfall data prediction')

    common = parser.add_argument_group('common')
    common.add_argument('--model', default='unet', type=str, help='which model to use')
    common.add_argument('--device', default='0', type=str, help='which device to use')
    common.add_argument('--dataset_dir', type=str, help='root directory of dataset')
    common.add_argument('--num_workers', default=5, type=int, help='# of workers for dataloader')
    common.add_argument('--debug', help='turn on debugging print', action='store_true')

    unet = parser.add_argument_group('unet related')
    unet.add_argument('--n_blocks', default=7, type=int, help='# of blocks in Down and Up phase')
    unet.add_argument('--start_channels', default=16, type=int, help='# of channels after first block of unet')

    nims = parser.add_argument_group('nims dataset related')
    nims.add_argument('--window_size', default=10, type=int, help='# of input sequences in time')
    nims.add_argument('--target_num', default=1, type=int, help='# of output sequences to evaluate')
    nims.add_argument('--variables', nargs='+',
                      help='which variables to use (rain, cape, etc.). \
                            Can be single number which specify how many variables to use \
                            or list of variables name')
    nims.add_argument('--start_train_year', default=2009, type=int, help='start year for training')
    nims.add_argument('--end_train_year', default=2017, type=int, help='end year for training')

    hyperparam = parser.add_argument_group('hyper-parameters')
    hyperparam.add_argument('--num_epochs', default=10, type=int, help='# of training epochs')
    hyperparam.add_argument('--batch_size', default=1, type=int, help='batch size')
    hyperparam.add_argument('--optimizer', default='adam', type=str, help='which optimizer to use (rmsprop, adam, sgd)')
    hyperparam.add_argument('--lr', default=0.001, type=float, help='learning rate of optimizer')

    args = parser.parse_args()

    return args

def set_experiment_name(args):
    """
    Set experiment name and assign to process name

    <Parameters>
    args [argparse]: parsed argument
    """
    train_year_range = str(args.start_train_year)[-2:] + \
                       str(args.end_train_year)[-2:]

    if args.model == 'unet':
        experiment_name = 'nims_unet_nb{}_ch{}_ws{}_tn{}_bs{}_{}_{}' \
                          .format(args.n_blocks,
                                  args.start_channels,
                                  args.window_size,
                                  args.target_num,
                                  args.batch_size,
                                  args.optimizer,
                                  train_year_range)

    elif args.model == 'convlstm':
        experiment_name = 'nims_convlstm_ws{}_tn{}_bs{}_{}_{}' \
                          .format(args.window_size,
                                  args.target_num,
                                  args.batch_size,
                                  args.optimizer,
                                  train_year_range)

    if args.debug:
        experiment_name += '_debug'

    try:
        setproctitle.setproctitle(experiment_name)
    except:
        pass

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda:0')

    # Parse NIMS dataset variables
    variables = parse_variables(args.variables)

    # Train and test dataset
    nims_train_dataset = NIMSDataset(model=args.model,
                                     window_size=args.window_size,
                                     target_num=args.target_num,
                                     variables=variables,
                                     train_year=(args.start_train_year,
                                                 args.end_train_year),
                                     train=True,
                                     transform=ToTensor(),
                                     root_dir=args.dataset_dir,
                                     debug=args.debug)
    
    nims_test_dataset  = NIMSDataset(model=args.model,
                                     window_size=args.window_size,
                                     target_num=args.target_num,
                                     variables=variables,
                                     train_year=(args.start_train_year,
                                                 args.end_train_year),
                                     train=False,
                                     transform=ToTensor(),
                                     root_dir=args.dataset_dir,
                                     debug=args.debug)

    # Get a sample for getting shape of each tensor
    sample, _ = nims_train_dataset[0]
    if args.debug:
        print('[unet] one images sample shape:', sample.shape)

    # Create a model and criterion
    if args.model == 'unet':
        model = UNet(n_channels=sample.shape[0],
                     n_classes=4,
                     n_blocks=args.n_blocks,
                     start_channels=args.start_channels,
                     target_num=args.target_num)
        criterion = NIMSCrossEntropyLoss()

    elif args.model == 'convlstm':
        assert args.window_size == args.target_num

        model = EncoderForecaster(input_channels=1,
                                  hidden_channels=[64, 128],
                                  kernel_size=3,
                                  seq_len=args.window_size,
                                  device=device)
        criterion = MSELoss()

    num_lat = sample.shape[1] # the number of latitudes (253)
    num_lon = sample.shape[2] # the number of longitudes (149)

    # Create dataloaders
    train_loader = DataLoader(nims_train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    test_loader  = DataLoader(nims_test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

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
    set_experiment_name(args)

    # Start training
    nims_trainer = NIMSTrainer(model, criterion, optimizer, device,
                               train_loader, test_loader,
                               len(nims_train_dataset), len(nims_test_dataset),
                               num_lat, num_lon, args)
    nims_trainer.train()
    nims_trainer.test()
