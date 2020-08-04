import torch
from torch.utils.data import Dataset

from nims_variable import read_variable_value

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
                 test_time=None, train=True, finetune=False, transform=None):
        assert window_size >= 0 and window_size <= 48, \
            'window_size must be in between 0 and 48'
        
        if not train:
            assert test_time != None, 'You must specify test time in test mode'

        self.model = model
        self.model_utc = model_utc
        self.window_size = window_size
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        # Initial train mode.
        # Set test time as 2020-06-01-00 to use whole May data for training
        if test_time == None:
            self.test_time = '2020060100'
        elif '-' in test_time:
            self.test_time = test_time.strip().split('-')
        else:
            self.test_time = test_time
            
        self._data_path_dict, self._gt_path_list = self.__set_path()

    @property
    def gt_path_list(self):
        return self._gt_path_list
        
    def __set_path(self):
        root, dirs, _ = next(os.walk(self.root_dir, topdown=True))
        
        start_test_time, end_test_time = self.__start_end_test_time()

        data_dirs = sorted([os.path.join(root, d) for d in dirs \
                            if d.isnumeric() and d <= end_test_time.strftime("%Y%m%d")])
        
        data_path_dict = defaultdict(list)
        pres_data_path_list = []
        unis_data_path_list = []
        for data_dir in data_dirs:
            # remove data that predict beyond threshold hours
            curr_data_path_list = [f for f in os.listdir(data_dir)]
            pres_data_path_list = sorted([f for f in curr_data_path_list if 'pres' in f])
            unis_data_path_list = sorted([f for f in curr_data_path_list if 'unis' in f])
            assert len(pres_data_path_list) == len(unis_data_path_list)
            
            for p, u in list(zip(pres_data_path_list, unis_data_path_list)):
                # p name: ldps_pres_sp_h0xx_yyyymmddhh
                # hh: LDAPS prediction start hour (00, 06, 12, 18)
                # xx: xx hour prediction from hh
                time = datetime(year=int(p.split('_')[4][0:4]),
                                month=int(p.split('_')[4][4:6]),
                                day=int(p.split('_')[4][6:8]),
                                hour=int(p.split('_')[4][8:10])) + timedelta(hours=int(p.split('_')[3][1:]))
                time = time.strftime("%Y%m%d%H")

                if int(p.split('_')[4][8:10]) == self.model_utc:
                    data_path_dict[time].append((os.path.join(data_dir, p), os.path.join(data_dir, u)))

        gt_dir = os.path.join(self.root_dir, 'OBS')
        gt_path_list = os.listdir(gt_dir)

        if self.train:
            gt_path_list = sorted([os.path.join(gt_dir, f) for f in gt_path_list \
                                if '.npy' in f and
                                f.split('_')[3][:-2] < end_test_time.strftime("%Y%m%d%H")])
            gt_path_list = gt_path_list[self.window_size:]

        else:
            gt_path_list = sorted([os.path.join(gt_dir, f) \
                                   for f in gt_path_list \
                                   if f.endswith('.npy') and \
                                   f.split('_')[3][:-2] >= start_test_time.strftime("%Y%m%d%H") and \
                                   f.split('_')[3][:-2] <= end_test_time.strftime("%Y%m%d%H")])

        return data_path_dict, gt_path_list

    def __start_end_test_time(self):
        if isinstance(self.test_time, str):
            if len(self.test_time) == 10:
                # Consider one hour
                start_test_time = datetime(year=int(self.test_time[0:4]),
                                           month=int(self.test_time[4:6]),
                                           day=int(self.test_time[6:8]),
                                           hour=int(self.test_time[8:10]))
                end_test_time = start_test_time
            elif len(self.test_time) == 8:
                # Consider one day
                start_test_time = datetime(year=int(self.test_time[0:4]),
                                           month=int(self.test_time[4:6]),
                                           day=int(self.test_time[6:8]),
                                           hour=0)
                end_test_time = start_test_time + timedelta(hours=23)
            elif len(self.test_time) == 6:
                # Consider one month
                start_test_time = datetime(year=int(self.test_time[0:4]),
                                           month=int(self.test_time[4:6]),
                                           day=1,
                                           hour=0)
                # Because original OBS files are stored in KST time,
                # the converted OBS npy file ends at 2020-06-30-14 UTC (2020-06-30-23 KST)
                end_test_time = start_test_time + timedelta(days=MONTH_DAY[int(self.test_time[4:6])] - 1,
                                                            hours=14)
            else:
                raise ValueError

        elif isinstance(self.test_time, list):
            _start = self.test_time[0]
            _end = self.test_time[1]
            assert len(_start) == len(_end)

            if len(_start) == 10:
                start_test_time = datetime(year=int(_start[0:4]),
                                           month=int(_start[4:6]),
                                           day=int(_start[6:8]),
                                           hour=int(_start[8:10]))

                end_test_time = datetime(year=int(_end[0:4]),
                                         month=int(_end[4:6]),
                                         day=int(_end[6:8]),
                                         hour=int(_end[8:10]))

            elif len(_start) == 8:
                start_test_time = datetime(year=int(_start[0:4]),
                                           month=int(_start[4:6]),
                                           day=int(_start[6:8]),
                                           hour=0)

                end_test_time = datetime(year=int(_end[0:4]),
                                         month=int(_end[4:6]),
                                         day=int(_end[6:8]),
                                         hour=23)
            
            elif len(_start) == 6:
                raise NotImplementedError

        return start_test_time, end_test_time

    def __len__(self):
        return len(self._gt_path_list)

    def __getitem__(self, idx):
        gt_path = self._gt_path_list[idx]
        
        # train_end_time is also target time for this x, y instance
        train_end_time = gt_path.split('/')[-1].split('_')[3]
        train_end_time = datetime(year=int(train_end_time[0:4]),
                                  month=int(train_end_time[4:6]),
                                  day=int(train_end_time[6:8]),
                                  hour=int(train_end_time[8:10]))
        train_start_time = train_end_time - timedelta(hours=self.window_size)

        # Get the latest UTC time before current train end time
        latest_utc = datetime(year=train_end_time.year,
                              month=train_end_time.month,
                              day=train_end_time.day,
                              hour=self.model_utc)
        if train_end_time < latest_utc:
            latest_utc = latest_utc - timedelta(days=1)

        _ldaps_input = []
        for t in range(self.window_size + 1):
            curr_time = train_start_time + timedelta(hours=t)
            curr_time_str = curr_time.strftime("%Y%m%d%H")

            data_list = copy.deepcopy(self._data_path_dict[curr_time_str])

            # Considering the calculation time of LDAPS model (5 hours)
            # If current time is within 6 hours from latest_utc,
            # this data can't be used, so we need to remove them.
            if curr_time >= latest_utc:
                train_end_time_from_latest_utc = train_end_time - latest_utc
                if (train_end_time_from_latest_utc >= timedelta(hours=0)) and \
                   (train_end_time_from_latest_utc <= timedelta(hours=5)):
                    assert len(data_list) > 1
                    data_list.sort(key=lambda x: x[0].split('/')[-1].split('_')[-1])
                    data_list.pop(-1)

            # TODO: add pres_idx_list and unis_idx_list arguments using var_name
            _ldaps_input.append(self._merge_pres_unis(data_list=data_list,
                                                      pres_idx_list=[4, 5, 6, 7, 8, 9],
                                                      unis_idx_list=[0, 2, 3, 5, 6, 7]))
            
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
        gt = torch.where(gt >= 0.1, torch.ones(gt.shape), torch.zeros(gt.shape))

        return ldaps_input, gt

    def get_real_gt(self, idx):
        gt_path = self._gt_path_list[idx]
        real_gt = np.load(gt_path)

        return real_gt

    def _merge_pres_unis(self, data_list, pres_idx_list=None, unis_idx_list=None):
        p, u = sorted(data_list)[-1]
        pres = np.load(p).reshape(512, 512, 20).transpose()
        unis = np.load(u).reshape(512, 512, 20).transpose()

        if pres_idx_list is not None:
            pres = pres[pres_idx_list,:,:]
        if unis_idx_list is not None:
            unis = unis[unis_idx_list,:,:]

        return np.concatenate((pres, unis), axis=0)
    
class ToTensor(object):
    def __call__(self, images):
        return torch.from_numpy(images)
