import torch
from torch.utils.data import Dataset
import numpy as np

from .nims_util import *

from datetime import datetime, timedelta
import os

__all__ = ['NIMSDataset', 'ToTensor']

# TODO: missing data pre-process

class NIMSDataset(Dataset):
    def __init__(self, model, reference, model_utc, window_size, root_dir, date,
                 heavy_rain=False, train=True, transform=None):
        # assert window_size >= 6 and window_size <= 48, \
        #     'window_size must be in between 6 and 48'

        self.model = model
        self.reference = reference
        self.model_utc = model_utc
        self.window_size = window_size
        self.root_dir = root_dir
        self.date = date
        self.rain_threshold = [0.1]
        self.train = train
        self.transform = transform
        
        self._data_path_list, self._gt_path = self.__set_path()

    def __set_path(self):
        LDAPS_dir = os.path.join(self.root_dir, 'NIMS_LDPS')
        root, dirs, _ = next(os.walk(os.path.join(LDAPS_dir, str(self.date['year'])), topdown=True))

        start_date = datetime(year=self.date['year'],
                              month=self.date['start_month'],
                              day=self.date['start_day'],
                              hour=0)
                
        end_date = datetime(year=self.date['year'],
                            month=self.date['end_month'],
                            day=self.date['end_day'],
                            hour=23)
        
        # Ground truth end date
        if self.train:
            gt_end_date = end_date
        else:
            # If test mode, add (25 + model_utc) hours to end date
            # to match the prediction range with LDAPS model
            add_hour = 25 + self.model_utc
            gt_end_date = end_date + timedelta(hours=add_hour)
        
        # Set input data list
        data_dirs = sorted([os.path.join(root, d) for d in dirs \
                            if d.isnumeric() and \
                            d >= start_date.strftime("%Y%m%d") and \
                            d <= end_date.strftime("%Y%m%d")])
        
        data_path_list = []
        pres_data_path_list = []
        unis_data_path_list = []
        for data_dir in data_dirs:
            curr_data_path_list = [f for f in os.listdir(data_dir)]
            curr_data_path_list = [f for f in curr_data_path_list \
                                   if int(f.split('_')[4][8:10]) == self.model_utc]
            unis_data_path_list = sorted([f for f in curr_data_path_list \
                                          if 'unis' in f])
            pres_data_path_list = sorted([f for f in curr_data_path_list \
                                          if 'pres' in f])
            
            for p, u in list(zip(pres_data_path_list, unis_data_path_list)):
                # p name: ldps_pres_sp_h0xx_yyyymmddhh
                # u name: ldps_unis_sp_h0xx_yyyymmddhh
                # hh: LDAPS prediction start hour (00, 06, 12, 18)
                # xx: xx hour prediction from hh
                
                # Considering the calculation time of LDAPS model (5 hours)
                from_h = int(u.split('_')[3][1:])
                if from_h < 6:
                    continue
                else:
                    curr_time = datetime(year=int(u.split('_')[4][0:4]),
                                         month=int(u.split('_')[4][4:6]),
                                         day=int(u.split('_')[4][6:8]),
                                         hour=int(u.split('_')[4][8:10])) + timedelta(hours=from_h)
                    if curr_time > gt_end_date:
                        continue
                    else:
                        data_path_list.append((os.path.join(data_dir, p), os.path.join(data_dir, u)))
        
        # Set target data list
        if self.reference == 'aws':
            gt_dir = os.path.join(self.root_dir, 'OBS', str(self.date['year']))
            gt_path = os.listdir(gt_dir)
            gt_path = sorted([os.path.join(gt_dir, f) for f in gt_path if
                              f.split('_')[3][:-2] >= start_date.strftime("%Y%m%d%H") and
                              f.split('_')[3][:-2] <= gt_end_date.strftime("%Y%m%d%H")])
            return data_path_list, gt_path
        elif self.reference == 'reanalysis':
            gt_dir = os.path.join(self.root_dir, 'REANALYSIS', str(self.date['year']))
            gt_path = os.listdir(gt_dir)
            gt_path = sorted([os.path.join(gt_dir, f) for f in gt_path if
                              f.split('.')[0] >= start_date.strftime("%Y%m%d%H") and
                              f.split('.')[0] <= gt_end_date.strftime("%Y%m%d%H")])
            return data_path_list, gt_path
        else:
            return data_path_list, None
        

    def __len__(self):        
        return len(self._data_path_list)

    def __getitem__(self, idx):
        p, u = self._data_path_list[idx]
            
        # train_end_time is also target time for this x, y instance
        current_utc_date = u.split('/')[-1].split('_')
        from_h = int(current_utc_date[3][1:])
        current_utc_date = datetime(year=int(current_utc_date[4][0:4]),
                                    month=int(current_utc_date[4][4:6]),
                                    day=int(current_utc_date[4][6:8]),
                                    hour=int(current_utc_date[4][8:10]))
        train_end_time = current_utc_date + timedelta(hours=from_h)
        train_start_time = train_end_time - timedelta(hours=self.window_size)
        
        _ldaps_input = []
        for t in range(self.window_size + 1):
            curr_u = u.replace('h{}'.format(str(from_h).zfill(3)),
                               'h{}'.format(str(from_h - self.window_size + t).zfill(3)))
            data_list = [curr_u]
            
            curr_p = p.replace('h{}'.format(str(from_h).zfill(3)),
                               'h{}'.format(str(from_h - self.window_size + t).zfill(3)))
            data_list.insert(0, curr_p)
            
            # TODO: add pres_idx_list and unis_idx_list arguments using var_name
            _ldaps_input.append(self._merge_pres_unis(data_list=data_list,
                                                      pres_idx_list=[2],
                                                      unis_idx_list=[2, 14, 17],
                                                      use_kindex=True))
        
        if 'unet' in self.model:
            ldaps_input = np.concatenate(_ldaps_input, axis=0)
        elif self.model == 'convlstm':
            ldaps_input = np.stack(_ldaps_input, axis=0)
        
        if self.transform:
            ldaps_input = self.transform(ldaps_input)


        # Make tensor of current target time for test logging
        target_time_tensor = torch.tensor([current_utc_date.year,
                                           current_utc_date.month,
                                           current_utc_date.day,
                                           current_utc_date.hour,
                                           from_h])

        if self.reference in ['aws', 'reanalysis']:
            # Get ground-truth based on train mode
            gt = self._get_gt_data(train_end_time)
            return ldaps_input, gt, target_time_tensor
        else:
            return ldaps_input, target_time_tensor

    def _merge_pres_unis(self, data_list, pres_idx_list=None, unis_idx_list=None, use_kindex=True):
        p, u = data_list
        pres = np.load(p).reshape(512, 512, 20).transpose()
        unis = np.load(u).reshape(512, 512, 20).transpose()

        if pres_idx_list is not None:
            pres = pres[pres_idx_list,:,:]
        if unis_idx_list is not None:
            unis = unis[unis_idx_list,:,:]

        # LDAPS missing value pre-process
        missing_x, missing_y = np.where(unis[0] < 0)[0], np.where(unis[0] < 0)[1]
        unis[0][missing_x, missing_y] = 0.
        unis[1][missing_x, missing_y] = 0.
        unis[2][missing_x, missing_y] = 101300.
        pres[0][missing_x, missing_y] = 273.15

        concat_list = [np.expand_dims(pres[0], axis=0), unis[:3, :, :]]

        return np.concatenate(concat_list, axis=0)

    def _get_gt_data(self, train_end_time):
        gt_path = [p for p in self._gt_path \
                   if train_end_time.strftime("%Y%m%d%H") in p][0]
        gt_base = torch.tensor(np.load(gt_path), dtype=torch.float32)
        
        # missing value (<0 in aws) or wrong value (<0 in reanlysis) pre-process
        if self.reference == 'aws':
            gt_base = torch.where(gt_base < 0, -9999 * torch.ones(gt_base.shape), gt_base)
        elif self.reference == 'reanalysis':
            gt_base = torch.where(gt_base < 0, torch.zeros(gt_base.shape), gt_base)
        
        if self.model == 'unet' or self.model == 'convlstm':
            # make len(rain_threshold)+1 classes (multi-class classification)
            for i in range(len(self.rain_threshold)):
                if i == 0:
                    gt = torch.where((0 <= gt_base) & (gt_base < self.rain_threshold[i]),
                                     i * torch.ones(gt_base.shape), gt_base)
                else:
                    gt = torch.where((self.rain_threshold[i-1] <= gt_base) & (gt_base < self.rain_threshold[i]),
                                     i * torch.ones(gt_base.shape), gt)
            gt = torch.where(self.rain_threshold[i] <= gt_base,
                             (i+1) * torch.ones(gt_base.shape), gt)
            
            return gt
            
        elif self.model == 'suc_unet':
            # make len(rain_threshold) * binary classes (successive binary classification)
            gt_lst = []
            for i in range(len(self.rain_threshold)):
                gt = torch.where(gt_base < 0, gt_base,
                                 torch.where(gt_base < self.rain_threshold[i],
                                             torch.zeros(gt_base.shape),
                                             torch.ones(gt_base.shape)))
                gt_lst.append(gt)
                
            return gt_lst

    # XXX: Need to fix for new training target data
    def get_real_gt(self, idx):
        gt_path = self._gt_path[idx]
        real_gt = np.load(gt_path)

        return real_gt
    
class ToTensor(object):
    def __call__(self, images):
        return torch.from_numpy(images)