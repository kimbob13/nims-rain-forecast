import torch
from nims_util import get_min_max_normalization
from nims_logger import NIMSLogger

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

        self.stn_codi = self._get_station_coordinate()
        self.experiment_name = experiment_name
        self.normalization = normalization

        self.trained_weight = {'model': None,
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

        self.model.to(self.device)

        if self.model_name == 'unet' or self.model_name == 'attn_unet':
            self.nims_logger = NIMSLogger(loss=True, correct=True,
                                          macro_f1=False, micro_f1=False,
                                          hit=True, miss=True, fa=True, cn=True,
                                          stn_codi=self.stn_codi,
                                          test_date_list=test_date_list)
            if self.train_loader and self.test_loader:
                self.nims_valid_logger = NIMSLogger(loss=True, correct=True,
                                                    macro_f1=False, micro_f1=False,
                                                    hit=True, miss=True, fa=True, cn=True,
                                                    stn_codi=self.stn_codi,
                                                    test_date_list=test_date_list)
        elif self.model_name == 'convlstm':
            self.nims_logger = NIMSLogger(loss=True, correct=False,
                                          macro_f1=False, micro_f1=False,
                                          hit=True, miss=True, fa=True, cn=True,
                                          stn_codi=self.stn_codi,
                                          test_date_list=test_date_list)

    def _get_station_coordinate(self):
        codi_aws_df = pd.read_csv('./codi_ldps_aws/codi_ldps_aws_512.csv')
        dii_info = np.array(codi_aws_df['dii']) - 1
        stn_codi = np.array([(dii // 512, dii % 512) for dii in dii_info])

        return stn_codi

    def train(self):
        self.model.train()

        trained_weight_path = os.path.join('./results', self.experiment_name, 'trained_weight.pt')
        train_log_path = os.path.join('./results', self.experiment_name, 'train_log.csv')
        train_log = pd.DataFrame(index=list(range(1, self.num_epochs + 1)),
                                 columns=['loss', 'acc', 'csi', 'pod', 'far', 'f1', 'bias'])

        for epoch in range(1, self.num_epochs + 1):
            # Run training epoch
            print('=' * 25, 'Epoch {} / {}'.format(epoch, self.num_epochs), '=' * 25)
            self._epoch(self.train_loader, train=True, logger=self.nims_logger)
            epoch_train_loss, epoch_train_stat = self.nims_logger.print_stat()

            # Run validation epoch
            print('-' * 10, 'Validation', '-' * 10)
            self._epoch(self.test_loader, train=False, valid=True, logger=self.nims_valid_logger)
            epoch_valid_loss, epoch_valid_stat = self.nims_valid_logger.print_stat()

            if epoch_valid_loss < self.trained_weight['best_loss']:
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

            train_log.loc[epoch] = [epoch_train_loss,
                                    epoch_train_stat.acc,
                                    epoch_train_stat.csi,
                                    epoch_train_stat.pod,
                                    epoch_train_stat.far,
                                    epoch_train_stat.f1,
                                    epoch_train_stat.bias]
            train_log.to_csv(train_log_path, index=False)

            if self.scheduler:
                self.scheduler.step()

    def test(self):
        # self.model.eval()
        # for m in self.model.modules():
        #     if isinstance(m, torch.nn.BatchNorm2d):
        #         m.track_running_stats = False

        print('=' * 25, 'Test', '=' * 25)
        with torch.no_grad():
            self._epoch(self.test_loader, train=False, logger=self.nims_logger)

        self.nims_logger.print_stat(test=True)

    def _epoch(self, data_loader, train, valid=False, logger=None):
        pbar = tqdm(data_loader)
        for images, target, target_time in pbar:
            if self.model_name == 'unet' or \
               self.model_name == 'attn_unet':
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
                target_time = target_time.numpy()
            
            elif self.model_name == 'convlstm':
                # Dataloader for ConvLSTM outputs images and target shape as NSCHW format.
                # We should change this to SNCHW format.
                images = images.permute(1, 0, 2, 3, 4).to(self.device)
                target = target.permute(1, 0, 2, 3, 4).to(self.device)

            output = self.model(images)
            loss = self.criterion(output, target, target_time,
                                  stn_codi=self.stn_codi, logger=logger,
                                  test=((not train) and (not valid)))
            
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            pbar.set_description(logger.latest_stat(target_time))