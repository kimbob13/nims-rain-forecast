import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from model.stconvs2s import STConvS2S
from nims_dataset import NIMSDataset, ToTensor

import os
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='NIMS rainfall data prediction')

    parser.add_argument('--device', default='0' , type=str, help='which device to use')

    parser.add_argument('--window_size', default=10, type=int, help='# of input sequences in time')
    parser.add_argument('--target_num', default=1, type=int, help='# of output sequences to evaluate')

    parser.add_argument('--num_epochs', default=10, type=int, help='# of training epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')

    parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout rate')
    parser.add_argument('--upsample', help='whether to use upsample at the final layer of decoder', action='store_true')

    args = parser.parse_args()

    return args

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda:0')

    nims_dataset = NIMSDataset(window_size=args.window_size, target_num=args.target_num, transform=ToTensor())
    sample, _ = nims_dataset[0]
    #print('one images sample type: {}, shape: {}'.format(type(sample), sample.shape))

    nims_dataloader = DataLoader(nims_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = STConvS2S(channels=sample.shape[0], dropout_rate=args.dropout_rate, upsample=args.upsample)
    model.to(device)
    summary(model, input_size=sample.shape)

    criterion = RMSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-6)

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for images, target in tqdm(nims_dataloader):
            images, target= images.to(device), target.to(device)
            output = model(images)[:, :, :args.target_num, :, :]
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print('epoch [{}]: loss = {}'.format(epoch, epoch_loss / len(nims_dataloader)))
