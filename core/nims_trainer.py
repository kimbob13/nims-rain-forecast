import torch
from .nims_util import get_min_max_normalization
from .nims_logger import NIMSLogger

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
        
        self.nims_logger = None
        self.nims_valid_logger = None
                

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
                               'norm_min': normalization['min_values'] if normalization else None,
                               'best_loss' : float("inf"),  # best valid loss
                               'best_epoch': 0,             # epoch num at best_loss
                               'best_csi'  : 0.0,           # csi value at best_loss
                               'best_pod'  : 0.0,           # pod value at best_loss
                               'best_far'  : 0.0,           # far value at best_loss
                               'best_f1'   : 0.0,           # f1 value at best_loss
                               'best_bias' : 0.0}           # bias value at best_loss

        self.info_str = '=' * 16 + ' {:^25s} ' + '=' * 16
        self.device_idx = int(args.device)
        self.model.to(self.device)

        self.nims_logger = NIMSLogger(loss=True, correct=True,
                                      macro_f1=False, micro_f1=False,
                                      hit=True, miss=True, fa=True, cn=True,
                                      reference=self.reference,
                                      stn_codi=self.stn_codi,
                                      test_date_list=test_date_list)
        
        if self.train_loader and self.test_loader:
            self.nims_valid_logger = NIMSLogger(loss=True, correct=True,
                                                macro_f1=False, micro_f1=False,
                                                hit=True, miss=True, fa=True, cn=True,
                                                reference=self.reference,
                                                stn_codi=self.stn_codi,
                                                test_date_list=test_date_list)

    def _get_station_coordinate(self):
        # codi_aws_df = pd.read_csv('./codi_ldps_aws/codi_ldps_aws_512.csv')
        codi_aws_df = pd.read_csv('./codi_ldps_aws/codi_ldps_aws_602_781.csv')
        dii_info = np.array(codi_aws_df['dii']) - 1
        # stn_codi = np.array([(dii // 512, dii % 512) for dii in dii_info])
        stn_codi = np.array([(dii // 602, dii % 602) for dii in dii_info])

        return stn_codi

    def _print_stat(self, stat_dict):
        nims_stat_list = []
        print()
        print('+' + '-' * 2, 'Stat Name', '-' * 2, end='')
        for mode_name in stat_dict:
            print('+' + '-' * 3, '{} Stat'.format(mode_name), '-' * 3, end='')
            nims_stat_list.append(stat_dict[mode_name])
        print('+')


        stat_name_fmt = '[{:^14s}] '
        stat_value_fmt = '{:^18.5f} '

        # Loss print
        print(stat_name_fmt.format('loss'), end='')
        for i in range(len(nims_stat_list)):
            print(stat_value_fmt.format(nims_stat_list[i][0]), end='')
        print()

        # Remaining stat print
        nims_stat_name = nims_stat_list[0][1]._fields
        for stat_name in nims_stat_name:
            print(stat_name_fmt.format(stat_name), end='')
            for i in range(len(nims_stat_list)):
                if stat_name == 'acc':
                    print('{}{:7.3f} %{}'.format(' ' * 3, getattr(nims_stat_list[i][1], stat_name), ' ' * 7), end='')
                else:
                    print(stat_value_fmt.format(getattr(nims_stat_list[i][1], stat_name)), end='')
            print()
        print()

    def train(self):
        self.model.train()

        trained_weight_path = os.path.join('./results', self.experiment_name, 'trained_weight.pt')
        train_log_path = os.path.join('./results', self.experiment_name, 'train_log.csv')
        train_log = pd.DataFrame(index=list(range(1, self.num_epochs + 1)),
                                 columns=['loss', 'acc', 'csi', 'pod', 'far', 'f1', 'bias'])
        valid_log_path = os.path.join('./results', self.experiment_name, 'valid_log.csv')
        valid_log = pd.DataFrame(index=list(range(1, self.num_epochs + 1)),
                                 columns=['loss', 'acc', 'csi', 'pod', 'far', 'f1', 'bias'])

        for epoch in range(1, self.num_epochs + 1):
            print(self.info_str.format('Epoch {:3d} / {:3d} (GPU {})'.format(epoch, self.num_epochs, self.device_idx)))

            # Run training epoch
            if self.train_loader:
                self._epoch(self.train_loader, mode='train', logger=self.nims_logger)

            # Run validation epoch
            if self.test_loader:
                self._epoch(self.test_loader, mode='valid', logger=self.nims_valid_logger)

            # Get stat for train and valid of this epoch
            if self.nims_logger:
                epoch_train_loss, epoch_train_stat = self.nims_logger.epoch_stat(mode='train')
            
            if self.nims_valid_logger:
                epoch_valid_loss, epoch_valid_stat = self.nims_valid_logger.epoch_stat(mode='valid')

            # Print stat
            if self.nims_logger and self.nims_valid_logger:
                self._print_stat({'Train': (epoch_train_loss, epoch_train_stat),
                                  'Valid': (epoch_valid_loss, epoch_valid_stat)})

            # Save model based on the best validation loss
            if epoch_valid_stat.csi > self.trained_weight['best_csi']:
                self.trained_weight['model']      = self.model.state_dict()
                self.trained_weight['best_loss']  = epoch_valid_loss
                self.trained_weight['best_epoch'] = epoch
                self.trained_weight['best_acc']   = epoch_valid_stat.acc
                self.trained_weight['best_csi']   = epoch_valid_stat.csi
                self.trained_weight['best_pod']   = epoch_valid_stat.pod
                self.trained_weight['best_far']   = epoch_valid_stat.far
                self.trained_weight['best_f1']    = epoch_valid_stat.f1
                self.trained_weight['best_bias']  = epoch_valid_stat.bias

                # Save model weight which has minimum loss
                if os.path.isfile(trained_weight_path):
                    os.remove(trained_weight_path)
                torch.save(self.trained_weight, trained_weight_path)

            # Save train and valid log
            train_log.loc[epoch] = [epoch_train_loss,
                                    epoch_train_stat.acc,
                                    epoch_train_stat.csi,
                                    epoch_train_stat.pod,
                                    epoch_train_stat.far,
                                    epoch_train_stat.f1,
                                    epoch_train_stat.bias]
            train_log.to_csv(train_log_path, index=False)

            valid_log.loc[epoch] = [epoch_valid_loss,
                                    epoch_valid_stat.acc,
                                    epoch_valid_stat.csi,
                                    epoch_valid_stat.pod,
                                    epoch_valid_stat.far,
                                    epoch_valid_stat.f1,
                                    epoch_valid_stat.bias]
            valid_log.to_csv(valid_log_path, index=False)

            if self.scheduler:
                self.scheduler.step()

    def test(self):
        # self.model.eval()
        # for m in self.model.modules():
        #     if isinstance(m, torch.nn.BatchNorm2d):
        #         m.track_running_stats = False

        print(self.info_str.format('Test (GPU {})'.format(self.device_idx)))
        with torch.no_grad():
            self._epoch(self.test_loader, mode='test', logger=self.nims_logger)

        # Get stat for test
        epoch_test_loss, epoch_test_stat = self.nims_logger.epoch_stat(mode='test')

        # Print stat
        self._print_stat({'Test': (epoch_test_loss, epoch_test_stat)})

    def _epoch(self, data_loader, mode, logger=None):
        pbar = tqdm(data_loader)
        for images, target, target_time in pbar:
            if 'unet' in self.model_name:
                if self.normalization:
                    b, c, h, w = images.shape
                    images = images.reshape((-1, h, w))

                    max_values_batch = self.normalization['max_values'].unsqueeze(0).repeat(b, 1).reshape(-1)
                    min_values_batch = self.normalization['min_values'].unsqueeze(0).repeat(b, 1).reshape(-1)
                    transform = get_min_max_normalization(max_values_batch, min_values_batch)

                    images = transform(images)
                    images = images.reshape((b, c, h, w))
                                
                images = images.type(torch.FloatTensor).to(self.device)
                if self.model_name == 'suc_unet':
                    target_lst = []
                    for t in target:
                        target_lst.append(t.type(torch.LongTensor).to(self.device))
                else:
                    target = target.type(torch.LongTensor).to(self.device)
                target_time = target_time.numpy()
            
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

            # Apply input to the model and get loss
            if self.model_name == 'unet':
                output = self.model(images)
                loss = self.criterion(output, target, target_time,
                                      stn_codi=self.stn_codi, mode=mode, logger=logger)
            elif self.model_name == 'suc_unet':
                output_lst = self.model(images)
                for i, (output, target) in enumerate(zip(output_lst, target_lst)):
                    if i == 0:
                        loss = self.criterion(output, target, target_time,
                                              stn_codi=self.stn_codi, mode=mode,
                                              prev_preds=None, logger=None)
                        prev_preds = output
                    else:
                        loss += self.criterion(output, target, target_time,
                                               stn_codi=self.stn_codi, mode=mode,
                                               prev_preds=prev_preds, logger=logger)
                        prev_preds = output
            elif self.model_name == 'convlstm':
                output = self.model(images, future_seq=1)
                loss = self.criterion(output, target, target_time,
                                      stn_codi=self.stn_codi, mode=mode, logger=logger)
            
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            pbar.set_description('({}) '.format(mode.capitalize()) + logger.latest_stat(target_time))