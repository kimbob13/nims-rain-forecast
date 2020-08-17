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
                 experiment_name, args, normalization=None, test_result_path=None):
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
        self.experiment_name = experiment_name
        self.normalization = normalization

        self.train_info = {'model': None,
                           'n_blocks': args.n_blocks,
                           'start_channels': args.start_channels,
                           'pos_dim': args.pos_dim,
                           'cross_entropy_weight': args.cross_entropy_weight,
                           'window_size': args.window_size,
                           'model_utc': args.model_utc,
                           'sampling_ratio': args.sampling_ratio,
                           'num_epochs': args.num_epochs,
                           'batch_size': args.batch_size,
                           'optimizer': args.optimizer,
                           'lr': args.lr,
                           'custom_name': args.custom_name,
                           'best_loss': float("inf"),
                           'best_epoch': 0,     # epoch num at best_loss
                           'best_pod': 0.0,     # pod value at best_loss
                           'best_csi': 0.0,
                           'best_bias': 0.0}

        self.model.to(self.device)

        if model.name == 'unet' or \
           model.name == 'attn_unet' or \
           model.name == 'persistence':
            self.nims_logger = NIMSLogger(loss=True, correct=True, binary_f1=True,
                                          macro_f1=False, micro_f1=False,
                                          hit=True, miss=True, fa=True, cn=True,
                                          stn_codi=self.stn_codi,
                                          test_result_path=test_result_path)
        elif model.name == 'convlstm':
            self.nims_logger = NIMSLogger(loss=True, correct=False, binary_f1=True,
                                          macro_f1=False, micro_f1=False,
                                          hit=True, miss=True, fa=True, cn=True,
                                          stn_codi=self.stn_codi,
                                          test_result_path=test_result_path)

    def _get_station_coordinate(self):
        codi_aws_df = pd.read_csv('./codi_ldps_aws/codi_ldps_aws_512.csv')
        dii_info = np.array(codi_aws_df['dii']) - 1
        stn_codi = np.array([(dii // 512, dii % 512) for dii in dii_info])

        return stn_codi

    def train(self):
        self.model.train()

        train_info_path = os.path.join('./results', 'trained_model',
                                       self.experiment_name + '.pt')
        for epoch in range(1, self.num_epochs + 1):
            # Run one epoch
            print('=' * 25, 'Epoch {} / {}'.format(epoch, self.num_epochs), '=' * 25)
            epoch_loss = self._epoch(self.train_loader, train=True)
            pod, csi, bias = self.nims_logger.print_stat(self.train_len)

            if epoch_loss < self.train_info['best_loss']:
                self.train_info['model'] = self.model.state_dict()
                self.train_info['best_loss'] = epoch_loss
                self.train_info['best_epoch'] = epoch
                self.train_info['best_pod'] = pod
                self.train_info['best_csi'] = csi
                self.train_info['best_bias'] = bias

                # Save model weight which has minimum loss
                if os.path.isfile(train_info_path):
                    os.remove(train_info_path)
                torch.save(self.train_info, train_info_path)

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
        for images, target, target_time in pbar:
            if self.model.name == 'unet' or \
               self.model.name == 'attn_unet':
                if self.normalization:
                    b, c, h, w = images.shape
                    images = images.reshape((-1, h, w))
                    images = self.normalization(images)
                    images = images.reshape((b, c, h, w))
                
                images = images.type(torch.FloatTensor).to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
                target_time = target_time.squeeze(0).tolist()
            
            elif self.model.name == 'convlstm':
                # Dataloader for ConvLSTM outputs images and target shape as NSCHW format.
                # We should change this to SNCHW format.
                images = images.permute(1, 0, 2, 3, 4).to(self.device)
                target = target.permute(1, 0, 2, 3, 4).to(self.device)

            output = self.model(images)
            loss = self.criterion(output, target, target_time,
                                  stn_codi=self.stn_codi, logger=self.nims_logger,
                                  test=(not train))
            
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            pbar.set_description(self.nims_logger.latest_stat)

        return loss.item()