import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.unet_model import UNet
from nims_dataset import NIMSDataset, ToTensor
from nims_loss import RMSELoss, NIMSCrossEntropyLoss
from nims_trainer import NIMSTrainer

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
    common.add_argument('--device', default='0', type=str, help='which device to use')
    common.add_argument('--dataset_dir', type=str, help='root directory of dataset')
    common.add_argument('--num_workers', default=5, type=int, help='# of workers for dataloader')
    common.add_argument('--debug', help='turn on debugging print', action='store_true')

    unet = parser.add_argument_group('unet related')
    unet.add_argument('--start_channels', default=16, type=int, help='# of channels after first block of unet')

    nims = parser.add_argument_group('nims dataset related')
    nims.add_argument('--window_size', default=1, type=int, help='# of input sequences in time')
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

def parse_variables(variables_args):
    """
    Return list of variables index

    <Parameters>
    variables_args
        - [int]: How many variables to use.
        - [list[str]]: List of variable names to use

    <Return>
    variables [list[int]]: List of variable index
    """
    variables_dict = {'rain':  0, 'cape':  1, 'cin':  2, 'swe':  3, 'hel': 4,
                      'ct'  :  5, 'vt'  :  6, 'tt' :  7, 'si' :  8, 'ki' : 9,
                      'li'  : 10, 'ti'  : 11, 'ssi': 12, 'pw' : 13}

    if len(variables_args) == 1:
        assert variables_args[0].isdigit()
        variables = list(range(int(variables_args[0])))
    else:
        # List of variable names to use
        variables = set()
        if 'rain' not in variables_args:
            print("You don't add rain variable. It is added by default")
            variables.add(variables_dict['rain'])

        for var_name in variables_args:
            variables.add(variables_dict[var_name])

        variables = sorted(list(variables))

    return variables

def set_experiment_name(args):
    """
    Set experiment name and assign to process name

    <Parameters>
    args [argparse]: parsed argument
    """
    experiment_name = 'nims_ws{}_ch{}_ep{}_bs{}_{}_{}_{}' \
                      .format(args.window_size,
                              args.start_channels,
                              args.num_epochs,
                              args.batch_size,
                              args.optimizer,
                              args.start_train_year,
                              args.end_train_year)

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

    variables = parse_variables(args.variables)

    nims_train_dataset = NIMSDataset(window_size=args.window_size,
                                     target_num=args.target_num,
                                     variables=variables,
                                     train_year=(args.start_train_year,
                                                 args.end_train_year),
                                     train=True,
                                     transform=ToTensor(),
                                     root_dir=args.dataset_dir,
                                     debug=args.debug)
    
    nims_test_dataset  = NIMSDataset(window_size=args.window_size,
                                     target_num=args.target_num,
                                     variables=variables,
                                     train_year=(args.start_train_year,
                                                 args.end_train_year),
                                     train=False,
                                     transform=ToTensor(),
                                     root_dir=args.dataset_dir,
                                     debug=args.debug)

    sample, _ = nims_train_dataset[0]
    if args.debug:
        print('[{}] one images sample shape: {}'
                .format(args.model, sample.shape))

    model = UNet(n_channels=sample.shape[0],
                 n_classes=4,
                 start_channels=args.start_channels)
    criterion = NIMSCrossEntropyLoss()

    num_lat = sample.shape[1] # the number of latitudes (253)
    num_lon = sample.shape[2] # the number of longitudes (149)

    if args.debug:
        model.to(device)
        summary(model, input_size=sample.shape)

    train_loader = DataLoader(nims_train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    test_loader  = DataLoader(nims_test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

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

    set_experiment_name(args)

    # Start training
    nims_trainer = NIMSTrainer(model, criterion, optimizer, device,
                               train_loader, test_loader,
                               len(nims_train_dataset), len(nims_test_dataset),
                               num_lat, num_lon, args)
    nims_trainer.train()
    nims_trainer.test()
