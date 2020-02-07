import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.stconvs2s import STConvS2S
from model.unet_model import UNet
from nims_dataset import NIMSDataset, ToTensor
from nims_loss import RMSELoss, NIMSCrossEntropyLoss
from nims_trainer import NIMSTrainer

import setproctitle

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='NIMS rainfall data prediction')

    parser.add_argument('--device', default='0' , type=str, help='which device to use')
    parser.add_argument('--debug', help='turn on debugging print', action='store_true')

    parser.add_argument('--model', default='unet', type=str, help='which model to use (stconvs2s, unet)')
    parser.add_argument('--window_size', default=1, type=int, help='# of input sequences in time')
    parser.add_argument('--target_num', default=1, type=int, help='# of output sequences to evaluate')

    parser.add_argument('--start_train_year', default=2009, type=int, help='start year for training')
    parser.add_argument('--end_train_year', default=2017, type=int, help='end year for training')

    parser.add_argument('--num_epochs', default=10, type=int, help='# of training epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--optimizer', default='adam', type=str, help='which optimizer to use (rmsprop, adam, sgd)')

    parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout rate')
    parser.add_argument('--upsample', help='whether to use upsample at the final layer of decoder', action='store_true')

    args = parser.parse_args()

    return args

def set_experiment_name(args):
    if args.model == 'stconvs2s':
        experiment_name = 'nims_stconv_ws{}_tn{}_ep{}_bs{}_{}' \
                          .format(args.window_size, args.target_num,
                                  args.num_epochs, args.batch_size,
                                  args.optimizer)
    elif args.model == 'unet':
        experiment_name = 'nims_unet_ep{}_bs{}_{}_{}_to_{}' \
                          .format(args.num_epochs, args.batch_size,
                                  args.optimizer, args.start_train_year,
                                  args.end_train_year)

    if args.debug:
        experiment_name += '_debug'

    setproctitle.setproctitle(experiment_name)

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda:0')

    if args.model == 'stconvs2s':
        nims_train_dataset = NIMSDataset(model=args.model,
                                         window_size=args.window_size,
                                         target_num=args.target_num,
                                         train_year=(2009, 2017),
                                         train=True,
                                         transform=ToTensor())
        nims_test_dataset  = NIMSDataset(model=args.model,
                                         window_size=args.window_size,
                                         target_num=args.target_num,
                                         train_year=(2009, 2017),
                                         train=False,
                                         transform=ToTensor())

        sample, _ = nims_train_dataset[0]
        if args.debug:
            print('[{}] one images sample type: {}, shape: {}'
                  .format(args.model, type(sample), sample.shape))

        model = STConvS2S(channels=sample.shape[0],
                          dropout_rate=args.dropout_rate,
                          upsample=args.upsample)
        criterion = RMSELoss()

        num_lat = sample.shape[2] # the number of latitudes (253)
        num_lon = sample.shape[3] # the number of longitudes (149)

    elif args.model == 'unet':
        nims_train_dataset = NIMSDataset(model=args.model,
                                         window_size=1,
                                         target_num=1,
                                         train_year=(args.start_train_year,
                                                     args.end_train_year),
                                         train=True,
                                         transform=ToTensor(),
                                         debug=args.debug)
        nims_test_dataset  = NIMSDataset(model=args.model,
                                         window_size=1,
                                         target_num=1,
                                         train_year=(args.start_train_year,
                                                     args.end_train_year),
                                         train=False,
                                         transform=ToTensor(),
                                         debug=args.debug)

        sample, _ = nims_train_dataset[0]
        if args.debug:
            print('[{}] one images sample type: {}, shape: {}'
                  .format(args.model, type(sample), sample.shape))

        model = UNet(n_channels=sample.shape[0], n_classes=4)
        criterion = NIMSCrossEntropyLoss()

        num_lat = sample.shape[1] # the number of latitudes (253)
        num_lon = sample.shape[2] # the number of longitudes (149)

    if args.debug:
        model.to(device)
        summary(model, input_size=sample.shape)

    train_loader = DataLoader(nims_train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=5)
    test_loader  = DataLoader(nims_test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=5)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.99,
                              weight_decay=5e-4, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001,
                                  alpha=0.9, eps=1e-6)

    set_experiment_name(args)

    # Start training
    nims_trainer = NIMSTrainer(model, criterion, optimizer, device,
                               train_loader, test_loader,
                               len(nims_train_dataset), len(nims_test_dataset),
                               num_lat, num_lon, args)
    nims_trainer.train()
    nims_trainer.test()
