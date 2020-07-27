import torch
from nims_logger import NIMSLogger

from tqdm import tqdm
import os
import numpy as np
import pandas as pd

__all__ = ['NIMSTrainer']

class NIMSTrainer:
    def __init__(self, model, criterion, optimizer, device,
                 train_loader, test_loader, train_len, test_len,
                 num_lat, num_lon, experiment_name, args):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_len = train_len
        self.test_len = test_len

        self.num_epochs = args.num_epochs
        self.debug = args.debug

        self.stn_codi = self._get_station_coordinate()
        self.num_stn = len(self.stn_codi)
        self.one_hour_pixel = num_lat * num_lon
        self.experiment_name = experiment_name

        self.model.to(self.device)

        if model.name == 'unet' or \
           model.name == 'attn_unet' or \
           model.name == 'persistence':
            self.nims_logger = NIMSLogger(loss=True, correct=True, binary_f1=True,
                                          macro_f1=False, micro_f1=False, csi=True,
                                          batch_size=args.batch_size,
                                          one_hour_pixel=self.one_hour_pixel,
                                          num_stn=self.num_stn,
                                          experiment_name=experiment_name,
                                          args=args)
        elif model.name == 'convlstm':
            self.nims_logger = NIMSLogger(loss=True, correct=False, binary_f1=True,
                                          macro_f1=False, micro_f1=False, csi=True,
                                          batch_size=args.batch_size,
                                          one_hour_pixel=self.one_hour_pixel,
                                          num_stn=self.num_stn,
                                          experiment_name=experiment_name,
                                          args=None)

    def _get_station_coordinate(self):
        codi_aws_df = pd.read_csv('/home/kimbob/jupyter/weather_prediction/pr_sample/codi_ldps_aws/codi_ldps_aws_512.csv')
        dii_info = np.array(codi_aws_df['dii']) - 1
        stn_codi = np.array([(dii // 512, dii % 512) for dii in dii_info])

        return stn_codi

    def train(self):
        self.model.train()

        for epoch in range(1, self.num_epochs + 1):
            # Run one epoch
            print('=' * 25, 'Epoch {} / {}'.format(epoch, self.num_epochs),
                  '=' * 25)
            self._epoch(self.train_loader, train=True)
            self.nims_logger.print_stat(self.train_len)

        # Save model weight
        weight_path = os.path.join('./results', 'trained_model',
                                   self.experiment_name + '.pt')
        torch.save(self.model.state_dict(), weight_path)

    def test(self):
        # self.model.eval()
        # for m in self.model.modules():
        #     if isinstance(m, torch.nn.BatchNorm2d):
        #         m.track_running_stats = False

        print('=' * 25, 'Test', '=' * 25)
        with torch.no_grad():
            self._epoch(self.test_loader, train=False)

        self.nims_logger.print_stat(self.test_len, test=True)

    def _epoch(self, data_loader, train):
        pbar = tqdm(data_loader)
        for images, target in pbar:
            if self.model.name == 'unet' or \
               self.model.name == 'attn_unet' or \
               self.model.name == 'persistence':
                images = images.type(torch.FloatTensor).to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
            
            elif self.model.name == 'convlstm':
                # Dataloader for ConvLSTM outputs images and target shape as NSCHW format.
                # We should change this to SNCHW format.
                images = images.permute(1, 0, 2, 3, 4).to(self.device)
                target = target.permute(1, 0, 2, 3, 4).to(self.device)

            output = self.model(images)
            loss = self.criterion(output, target, stn_codi=self.stn_codi,
                                  logger=self.nims_logger, test=(not train))
            
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            pbar.set_description(self.nims_logger.latest_stat)