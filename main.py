import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

import numpy as np

from model.stconvs2s import STConvS2S
from model.unet_model import UNet
from nims_dataset import NIMSDataset, ToTensor
from nims_loss import RMSELoss, NIMSCrossEntropyLoss

from tqdm import tqdm
import setproctitle

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='NIMS rainfall data prediction')

    parser.add_argument('--device', default='0' , type=str, help='which device to use')
    parser.add_argument('--debug', help='turn on debugging print', action='store_true')

    parser.add_argument('--model', default='stconvs2s', type=str, help='which model to use (stsconv2s, unet)')
    parser.add_argument('--window_size', default=1, type=int, help='# of input sequences in time')
    parser.add_argument('--target_num', default=1, type=int, help='# of output sequences to evaluate')

    parser.add_argument('--num_epochs', default=10, type=int, help='# of training epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--optimizer', default='rmsprop', type=str, help='which optimizer to use (rmsprop, adam, sgd)')

    parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout rate')
    parser.add_argument('--upsample', help='whether to use upsample at the final layer of decoder', action='store_true')

    args = parser.parse_args()

    return args

def set_experiment_name(args):
    if args.model == 'stsconv2d':
        experiment_name = 'nims_stconv_{}_tn{}_ep{}_bs{}_{}' \
                          .format(args.window_size, args.target_num,
                                  args.num_epochs, args.batch_size,
                                  args.optimizer)
    elif args.model == 'unet':
        experiment_name = 'nims_unet_ep{}_bs{}_{}' \
                          .format(args.num_epochs, args.batch_size,
                                  args.optimizer)

    if args.debug:
        experiment_name += '_debug'

    setproctitle.setproctitle(experiment_name)

    return experiment_name

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda:0')

    if args.model == 'stconvs2s':
        nims_dataset = NIMSDataset(window_size=args.window_size,
                                   target_num=args.target_num, transform=ToTensor())
        sample, _ = nims_dataset[0]
        if args.debug:
            print('[{}] one images sample type: {}, shape: {}'
                  .format(args.model, type(sample), sample.shape))

        model = STConvS2S(model=args.model,
                          channels=sample.shape[0],
                          dropout_rate=args.dropout_rate,
                          upsample=args.upsample)
        criterion = RMSELoss()

    elif args.model == 'unet':
        nims_dataset = NIMSDataset(model=args.model,
                                   window_size=1,
                                   target_num=args.target_num,
                                   transform=ToTensor())
        sample, _ = nims_dataset[0]
        if args.debug:
            print('[{}] one images sample type: {}, shape: {}'
                  .format(args.model, type(sample), sample.shape))

        model = UNet(n_channels=sample.shape[0], n_classes=4)
        criterion = NIMSCrossEntropyLoss()

        num_lat = sample.shape[1] # the number of latitudes (253)
        num_lon = sample.shape[2] # the number of longitudes (149)

    model.to(device)
    if args.debug:
        summary(model, input_size=sample.shape)

    nims_dataloader = DataLoader(nims_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.99,
                              weight_decay=5e-4, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001,
                                  alpha=0.9, eps=1e-6)

    experiment_name = set_experiment_name(args)

    for epoch in range(1, args.num_epochs + 1):
        epoch_loss = 0.0
        if args.model == 'unet':
            running_correct = 0

        for images, target in tqdm(nims_dataloader):
            if args.debug:
                if args.model == 'stconvs2s':
                    target_nonzero = np.nonzero(target.numpy().squeeze(0)
                                                .squeeze(0).squeeze(0))
                    lat = target_nonzero[0][0]
                    lon = target_nonzero[1][0]
                    print('lat: {}, lon: {}'.format(lat, lon))


            images= images.to(device)
            if args.model == 'stconvs2s':
                output = model(images)[:, :, :args.target_num, :, :]
            elif args.model == 'unet':
                output = model(images)
                target = target.type(torch.LongTensor)

            if args.debug:
                if args.model == 'stconvs2s':
                    print('- output: {}\n- target: {}'
                          .format(output[0][0][0][lat][lon],
                                  target[0][0][0][lat][lon]))

            target = target.to(device)
            if args.model == 'stconvs2s':
                loss = criterion(output, target)
            elif args.model == 'unet':
                loss, correct = criterion(output, target)
                running_correct += correct
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if args.model == 'stconv2s':
            print('Epoch [{}]: loss = {}'
                  .format(epoch, epoch_loss / len(nims_dataset)))
        if args.model == 'unet':
            running_correct = running_correct.double()
            total_num = len(nims_dataset) * num_lat * num_lon
            print('Epoch [{}]: loss = {}, accuracy = {:.3f}%'
                  .format(epoch, epoch_loss,
                          (running_correct / total_num).item() * 100))
