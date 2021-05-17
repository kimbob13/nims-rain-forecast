import torch
from .nims_util import get_min_max_normalization

from tqdm import tqdm
import os
import numpy as np
import pandas as pd

__all__ = ['NIMSTrainer']

class NIMSTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device,
                 train_loader, test_loader, experiment_name, args,
                 normalization=None, test_date_list=None):
        self.model = model
        self.model_name = args.model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = args.num_epochs

        self.train_loader = train_loader
        self.test_loader = test_loader
                
        self.reference = args.reference
        self.stn_codi = self._get_station_coordinate()
        self.experiment_name = experiment_name
        self.normalization = normalization

        self.trained_weight = {'model': None,
                               'model_name': args.model,
                               'n_blocks': args.n_blocks,
                               'start_channels': args.start_channels,
                               'pos_loc': args.pos_loc,
                               'pos_dim': args.pos_dim,
                               'cross_entropy_weight': args.cross_entropy_weight,
                               'bilinear': args.bilinear,
                               'window_size': args.window_size,
                               'model_utc': args.model_utc,
                               'sampling_ratio': args.sampling_ratio,
                               'heavy_rain': args.heavy_rain,
                               'num_epochs': args.num_epochs,
                               'batch_size': args.batch_size,
                               'optimizer': args.optimizer,
                               'lr': args.lr,
                               'wd': args.wd,
                               'custom_name': args.custom_name,
                               'norm_max': normalization['max_values'] if normalization else None,
                               'norm_min': normalization['min_values'] if normalization else None}
        
        self.info_str = '=' * 16 + ' {:^25s} ' + '=' * 16
        self.device_idx = int(args.device)
        self.model.to(self.device)
                
    def _get_station_coordinate(self):
        num_lat = 512
        num_lon = 512
        codi_aws_df = pd.read_csv('./codi_ldps_aws/codi_ldps_aws_512.csv')
        dii_info = np.array(codi_aws_df['dii']) - 1
        stn_codi = np.array([(dii // num_lat, dii % num_lat) for dii in dii_info]) # Need to check

        return stn_codi

    def train(self):
        self.model.train()

        train_log_path = os.path.join('./results', self.experiment_name, 'train_log.csv')
        train_log = pd.DataFrame(index=list(range(1, self.num_epochs + 1)),
                                 columns=['loss', 'hit', 'miss', 'fa', 'cn'])

        for epoch in range(1, self.num_epochs + 1):
            print(self.info_str.format('Epoch {:3d} / {:3d} (GPU {})'.format(epoch, self.num_epochs, self.device_idx)))

            # Run training epoch
            loss, hit, miss, fa, cn = self._epoch(self.train_loader, mode='train')

            # Save log
            train_log.loc[epoch] = [loss, hit, miss, fa, cn]
            train_log.to_csv(train_log_path, index=False)
            
            # Save model           
            self.trained_weight['model'] = self.model.state_dict()
            trained_weight_path = os.path.join('./results', self.experiment_name, 'trained_weight_ep{}.pt'.format(epoch))
            if os.path.isfile(trained_weight_path):
                os.remove(trained_weight_path)
            torch.save(self.trained_weight, trained_weight_path)

            if self.scheduler:
                self.scheduler.step()

    def test(self):
        print(self.info_str.format('Test (GPU {})'.format(self.device_idx)))
        with torch.no_grad():
            self._epoch(self.test_loader, mode='test')

    def _epoch(self, data_loader, mode):
        pbar = tqdm(data_loader)
        epoch_loss = []
        epoch_hit = []
        epoch_miss = []
        epoch_fa = []
        epoch_cn = []
        
        for images, target, target_time in pbar:
            if self.model_name == 'unet':
                if self.normalization:
                    b, c, h, w = images.shape
                    images = images.reshape((-1, h, w))

                    max_values_batch = self.normalization['max_values'].unsqueeze(0).repeat(b, 1).reshape(-1)
                    min_values_batch = self.normalization['min_values'].unsqueeze(0).repeat(b, 1).reshape(-1)
                    transform = get_min_max_normalization(max_values_batch, min_values_batch)

                    images = transform(images)
                    images = images.reshape((b, c, h, w))
                                
                images = images.type(torch.FloatTensor).to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
            
            elif self.model_name == 'convlstm':
                if self.normalization:
                    b, s, c, h, w = images.shape
                    images = images.reshape((-1, h, w))

                    max_values_batch = self.normalization['max_values'].unsqueeze(0).repeat(b, 1).reshape(-1)
                    min_values_batch = self.normalization['min_values'].unsqueeze(0).repeat(b, 1).reshape(-1)
                    transform = get_min_max_normalization(max_values_batch, min_values_batch)

                    images = transform(images)
                    images = images.reshape((b, s, c, h, w))
                
                images = images.type(torch.FloatTensor).to(self.device)
                target = target.type(torch.LongTensor).to(self.device)

            target_time = target_time.numpy()

            # Apply input to the model and get loss
            if self.model_name == 'unet' or self.model_name == 'attn_unet':
                output = self.model(images)
                loss, hit, miss, fa, cn = self.criterion(output, target, target_time,
                                                         stn_codi=self.stn_codi, mode=mode)
                epoch_loss.append(loss.item())
                epoch_hit.append(sum(hit)) # batch summation
                epoch_miss.append(sum(miss))
                epoch_fa.append(sum(fa))
                epoch_cn.append(sum(cn))
            
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # pbar.set_description('({}) '.format(mode.capitalize()))
            
        return np.mean(epoch_loss), sum(epoch_hit), sum(epoch_miss), sum(epoch_fa), sum(epoch_cn) # epoch summation