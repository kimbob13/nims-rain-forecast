import torch
from torch.utils.data import Dataset

from nims_variable import read_variable_value

from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import xarray as xr

import os
import pwd

__all__ = ['NIMSDataset', 'ToTensor']

START_YEAR = 2009
END_YEAR = 2018
NORMAL_YEAR_DAY = 365
LEAP_YEAR_DAY = 366
MONTH_DAY = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


class NIMSDataset(Dataset):
    def __init__(self, root_dir, test_time, window_size, numeric_thres, model, transform=None):
        self.root_dir = root_dir
        self.test_time = test_time
        self.window_size = window_size
        self.numeric_thres = numeric_thres
        self.model = model
        self.transform = transform
        
        self._data_path_dict, self._gt_path_list = self.__set_path()
        
    def __set_path(self):
        root, dirs, _ = next(os.walk(self.root_dir, topdown=True))
        
        test_time = datetime(year=int(self.test_time[0:4]),
                             month=int(self.test_time[4:6]),
                             day=int(self.test_time[6:8]),
                             hour=int(self.test_time[8:10]))
        
        data_dirs = sorted([os.path.join(root, d) for d in dirs \
                            if d.isnumeric() and d<=test_time.strftime("%Y%m%d")])
        
        data_path_dict = defaultdict(list)
        pres_data_path_list = []
        unis_data_path_list = []
        for data_dir in data_dirs:
            # remove data that predict beyond threshold hours
            curr_data_path_list = [f for f in os.listdir(data_dir) \
                                   if int(f.split('_')[3][1:]) <= self.numeric_thres]
            pres_data_path_list = sorted([f for f in curr_data_path_list if 'pres' in f])
            unis_data_path_list = sorted([f for f in curr_data_path_list if 'unis' in f])
            
            for p, u in list(zip(pres_data_path_list, unis_data_path_list)):    
                time = datetime(year=int(p.split('_')[4][0:4]),
                                month=int(p.split('_')[4][4:6]),
                                day=int(p.split('_')[4][6:8]),
                                hour=int(p.split('_')[4][8:10])) + timedelta(hours=int(p.split('_')[3][1:]))
                time = time.strftime("%Y%m%d%H")
                data_path_dict[time].append((os.path.join(data_dir, p), os.path.join(data_dir, u)))
                
        gt_dir = os.path.join(self.root_dir, 'OBS')
        gt_path_list = os.listdir(gt_dir)
        gt_path_list = sorted([os.path.join(gt_dir, f) for f in gt_path_list \
                               if '.npy' in f and
                               f.split('_')[3] < test_time.strftime("%Y%m%d%H")])
        gt_path_list = gt_path_list[self.window_size-1:]
        
        return data_path_dict, gt_path_list

    def __len__(self):
        return len(self._gt_path_list)

    def __getitem__(self, idx):
        gt_path = self._gt_path_list[idx]
        
        train_end_time = gt_path.split('/')[-1].split('_')[3]
        train_end_time = datetime(year=int(train_end_time[0:4]),
                                  month=int(train_end_time[4:6]),
                                  day=int(train_end_time[6:8]),
                                  hour=int(train_end_time[8:10]))
        train_start_time = train_end_time - timedelta(hours=self.window_size-1)
        
        _ldaps_input = []
        for t in range(self.window_size):
            time = train_start_time + timedelta(hours=t)
            time = time.strftime("%Y%m%d%H")
            # TODO: add pres_idx_list and unis_idx_list arguments using var_name
            _ldaps_input.append(self._merge_pres_unis(data_list=self._data_path_dict[time],
                                                      pres_idx_list=[0],
                                                      unis_idx_list=[0]))
            
        if self.model == 'unet':
            for idx, l in enumerate(_ldaps_input):
                if idx == 0:
                    ldaps_input = l
                else:
                    ldaps_input = np.concatenate((ldaps_input, l), axis=0)
        else:
            ldaps_input = _ldaps_input
        
        if self.transform:
            ldaps_input = self.transform(ldaps_input)
        
        gt = torch.tensor(np.load(gt_path))
        gt = torch.where(gt>=0.1, torch.ones(gt.shape), torch.zeros(gt.shape)).type(torch.LongTensor)
        return ldaps_input, gt

    def _merge_pres_unis(self, data_list, pres_idx_list, unis_idx_list):
        for idx, (p, u) in enumerate(data_list):
            if idx == 0:
                pres = np.load(p).reshape(512, 512, 20).transpose()
                unis = np.load(u).reshape(512, 512, 20).transpose()
            else:
                pres = pres + np.load(p).reshape(512, 512, 20).transpose()
                unis = unis + np.load(u).reshape(512, 512, 20).transpose()
        pres = pres / len(data_list)
        unis = unis / len(data_list)
        
        if pres_idx_list is not None:
            pres = pres[pres_idx_list,:,:]
        if unis_idx_list is not None:
            unis = unis[unis_idx_list,:,:]
        return np.concatenate((pres,unis), axis=0)
    
class ToTensor(object):
    def __call__(self, images):
        return torch.from_numpy(images)
