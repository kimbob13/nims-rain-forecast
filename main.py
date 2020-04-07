import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.stconvs2s import STConvS2S
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

    parser.add_argument('--device', default='0' , type=str, help='which device to use')
    parser.add_argument('--dataset_dir', type=str, help='root directory of dataset')
    parser.add_argument('--num_workers', default=12, type=int, help='# of workers for dataloader')
    parser.add_argument('--debug', help='turn on debugging print', action='store_true')

    parser.add_argument('--model', default='unet', type=str, help='which model to use (stconvs2s, unet)')
    parser.add_argument('--window_size', default=1, type=int, help='# of input sequences in time')
    parser.add_argument('--target_num', default=1, type=int, help='# of output sequences to evaluate')
    parser.add_argument('--variables', nargs='+',
                        help='which variables to use (rain, cape, etc.). \
                              Can be single number which specify how many variables to use \
                              or list of variables name')

    parser.add_argument('--start_train_year', default=2009, type=int, help='start year for training')
    parser.add_argument('--end_train_year', default=2017, type=int, help='end year for training')

    parser.add_argument('--num_epochs', default=10, type=int, help='# of training epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--optimizer', default='adam', type=str, help='which optimizer to use (rmsprop, adam, sgd)')

    parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout rate')
    parser.add_argument('--upsample', help='whether to use upsample at the final layer of decoder', action='store_true')

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
    if args.model == 'stconvs2s':
        experiment_name = 'nims_stconv_ws{}_tn{}_ep{}_bs{}_{}' \
                          .format(args.window_size, args.target_num,
                                  args.num_epochs, args.batch_size,
                                  args.optimizer)
    elif args.model == 'unet':
        experiment_name = 'nims_unet_ws{}_ep{}_bs{}_{}_{}_{}' \
                          .format(args.window_size, args.num_epochs,
                                  args.batch_size, args.optimizer,
                                  args.start_train_year, args.end_train_year)

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

    if args.model == 'stconvs2s':
        nims_train_dataset = NIMSDataset(model=args.model,
                                         window_size=args.window_size,
                                         target_num=args.target_num,
                                         variables=variables,
                                         train_year=(2009, 2017),
                                         train=True,
                                         transform=ToTensor(),
                                         root_dir=args.dataset_dir,
                                         debug=args.debug)

        nims_test_dataset  = NIMSDataset(model=args.model,
                                         window_size=args.window_size,
                                         target_num=args.target_num,
                                         variables=variables,
                                         train_year=(2009, 2017),
                                         train=False,
                                         transform=ToTensor(),
                                         root_dir=args.dataset_dir,
                                         debug=args.debug)

        sample, _ = nims_train_dataset[0]
        if args.debug:
            print('[{}] one images sample shape: {}'
                  .format(args.model, sample.shape))

        model = STConvS2S(channels=sample.shape[0],
                          dropout_rate=args.dropout_rate,
                          upsample=args.upsample)
        criterion = RMSELoss()

        num_lat = sample.shape[2] # the number of latitudes (253)
        num_lon = sample.shape[3] # the number of longitudes (149)

    elif args.model == 'unet':
        nims_train_dataset = NIMSDataset(model=args.model,
                                         window_size=args.window_size,
                                         target_num=1,
                                         variables=variables,
                                         train_year=(args.start_train_year,
                                                     args.end_train_year),
                                         train=True,
                                         transform=ToTensor(),
                                         root_dir=args.dataset_dir,
                                         debug=args.debug)
        
        nims_test_dataset  = NIMSDataset(model=args.model,
                                         window_size=args.window_size,
                                         target_num=1,
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

        model = UNet(n_channels=sample.shape[0], n_classes=4)
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
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.99,
                              weight_decay=5e-4, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001,
                                  alpha=0.9, eps=1e-6)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters())

    set_experiment_name(args)

    # Start training
    nims_trainer = NIMSTrainer(model, criterion, optimizer, device,
                               train_loader, test_loader,
                               len(nims_train_dataset), len(nims_test_dataset),
                               num_lat, num_lon, args)
    nims_trainer.train()
    nims_trainer.test()
