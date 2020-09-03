import torch
from torch.utils.data import Dataset

from nims_variable import read_variable_value
from nims_util import *
import torchvision.transforms as transforms

from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import xarray as xr

import os
import pwd
import copy

__all__ = ['NIMSDataset', 'ToTensor']

START_YEAR = 2009
END_YEAR = 2018
NORMAL_YEAR_DAY = 365
LEAP_YEAR_DAY = 366
MONTH_DAY = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


class NIMSDataset(Dataset):
    def __init__(self, model, model_utc, window_size, root_dir,
                 date, train=True, transform=None):
        assert window_size >= 6 and window_size <= 48, \
            'window_size must be in between 6 and 48'

        self.model = model
        self.model_utc = model_utc
        self.window_size = window_size
        self.root_dir = root_dir
        self.date = date
        self.train = train
        self.transform = transform
        
        self._data_path_list, self._gt_path_list = self.__set_path()

    @property
    def gt_path_list(self):
        return self._gt_path_list
        
    def __set_path(self):
        root, dirs, _ = next(os.walk(os.path.join(self.root_dir, str(self.date['year'])), topdown=True))
        
        # Make datetime object for start and end date
        start_date = datetime(year=self.date['year'],
                              month=self.date['start_month'],
                              day=self.date['start_day'],
                              hour=0)
        if self.date['year'] == 2019:
            if self.date['start_month'] == 6 and self.date['start_day'] == 1:
                start_date += timedelta(hours=self.window_size)

        elif self.date['year'] == 2020:
            if self.date['start_month'] == 5 and self.date['start_day'] == 1:
                start_date += timedelta(hours=self.window_size)
            
        end_date = datetime(year=self.date['year'],
                            month=self.date['end_month'],
                            day=self.date['end_day'],
                            hour=23)

        # Ground truth end date
        if self.date['end_month'] == 8 and self.date['end_day'] == 31:
            gt_end_date = end_date - timedelta(hours=9)
        else:
            gt_end_date = end_date
        
        # Set input data list
        data_dirs = sorted([os.path.join(root, d) for d in dirs \
                            if d.isnumeric() and d <= end_date.strftime("%Y%m%d")])
        data_path_list = []
        pres_data_path_list = []
        unis_data_path_list = []
        for data_dir in data_dirs:
            curr_data_path_list = [f for f in os.listdir(data_dir)]
            pres_data_path_list = sorted([f for f in curr_data_path_list \
                                          if 'pres' in f and int(f.split('_')[4][8:10]) == self.model_utc])
            unis_data_path_list = sorted([f for f in curr_data_path_list \
                                          if 'unis' in f and int(f.split('_')[4][8:10]) == self.model_utc])
            assert len(pres_data_path_list) == len(unis_data_path_list)
            
            for p, u in list(zip(pres_data_path_list, unis_data_path_list)):
                # p name: ldps_pres_sp_h0xx_yyyymmddhh
                # hh: LDAPS prediction start hour (00, 06, 12, 18)
                # xx: xx hour prediction from hh
                
                # Considering the calculation time of LDAPS model (5 hours)
                from_h = int(p.split('_')[3][1:])
                if from_h < 6:
                    continue
                else:
                    curr_time = datetime(year=int(p.split('_')[4][0:4]),
                                         month=int(p.split('_')[4][4:6]),
                                         day=int(p.split('_')[4][6:8]),
                                         hour=int(p.split('_')[4][8:10])) + timedelta(hours=from_h)
                    if curr_time > gt_end_date:
                        continue
                    else:                    
                        data_path_list.append((os.path.join(data_dir, p), os.path.join(data_dir, u)))
            
        # Set target data list
        gt_dir = os.path.join(self.root_dir, '..', 'OBS', str(self.date['year']))
        gt_path_list = os.listdir(gt_dir)
        gt_path_list = sorted([os.path.join(gt_dir, f) for f in gt_path_list if
                               f.split('_')[3][:-2] >= start_date.strftime("%Y%m%d%H") and
                               f.split('_')[3][:-2] <= end_date.strftime("%Y%m%d%H")])

        return data_path_list, gt_path_list

    def __len__(self):            
        return len(self._data_path_list)

    def __getitem__(self, idx):
        p, u = self._data_path_list[idx]
        
        # train_end_time is also target time for this x, y instance
        train_end_time = p.split('/')[-1].split('_')
        from_h = int(train_end_time[3][1:])
        train_end_time = datetime(year=int(train_end_time[4][0:4]),
                                  month=int(train_end_time[4][4:6]),
                                  day=int(train_end_time[4][6:8]),
                                  hour=int(train_end_time[4][8:10])) + timedelta(hours=from_h)
        train_start_time = train_end_time - timedelta(hours=self.window_size)
        
        _ldaps_input = []
        
        for t in range(self.window_size + 1):
            curr_p = p.replace('h{}'.format(str(from_h).zfill(3)),
                               'h{}'.format(str(from_h-self.window_size+t).zfill(3)))
            curr_u = u.replace('h{}'.format(str(from_h).zfill(3)),
                               'h{}'.format(str(from_h-self.window_size+t).zfill(3)))
            
            # TODO: add pres_idx_list and unis_idx_list arguments using var_name
            _ldaps_input.append(self._merge_pres_unis(data_list=(curr_p, curr_u),
                                                      pres_idx_list=[4, 5, 6, 7, 8, 9],
                                                      unis_idx_list=[0, 2, 3, 5, 6, 7]))
        
        if self.model == 'unet' or \
           self.model == 'attn_unet':
            for idx, l in enumerate(_ldaps_input):
                if idx == 0:
                    ldaps_input = l
                else:
                    ldaps_input = np.concatenate((ldaps_input, l), axis=0)
        else:
            ldaps_input = _ldaps_input
        
        if self.transform:
            ldaps_input = self.transform(ldaps_input)
        
        gt_path = [p for p in self._gt_path_list \
                   if train_end_time.strftime("%Y%m%d%H") in p][0]
        gt = torch.tensor(np.load(gt_path))
        gt = torch.where(gt >= 0.1, torch.ones(gt.shape), torch.zeros(gt.shape))

        # Make tensor of current target time for test logging
        target_time_tensor = torch.tensor([train_end_time.year,
                                           train_end_time.month,
                                           train_end_time.day,
                                           train_end_time.hour])

        return ldaps_input, gt, target_time_tensor

    def _merge_pres_unis(self, data_list, pres_idx_list=None, unis_idx_list=None):
        p, u = data_list
        pres = np.load(p).reshape(512, 512, 20).transpose()
        unis = np.load(u).reshape(512, 512, 20).transpose()

        if pres_idx_list is not None:
            pres = pres[pres_idx_list,:,:]
        if unis_idx_list is not None:
            unis = unis[unis_idx_list,:,:]

        return np.concatenate((pres, unis), axis=0)

    def get_real_gt(self, idx):
        gt_path = self._gt_path_list[idx]
        real_gt = np.load(gt_path)

        return real_gt
    
class ToTensor(object):
    def __call__(self, images):
        return torch.from_numpy(images)


    

    
    
    



    

    

