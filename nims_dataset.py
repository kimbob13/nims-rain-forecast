import torch
from torch.utils.data import Dataset
import numpy as np

from nims_util import *

from datetime import datetime, timedelta
import os

__all__ = ['NIMSDataset', 'ToTensor']

VALID_YEAR = [2019, 2020]
VALID_MONTH = [5, 6, 7, 8]
MONTH_DAY = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


class NIMSDataset(Dataset):
    def __init__(self, model, model_utc, window_size, root_dir, date,
                 lite=False, heavy_rain=False, train=True, transform=None):
        # assert window_size >= 6 and window_size <= 48, \
        #     'window_size must be in between 6 and 48'

        self.model = model
        self.model_utc = model_utc
        self.window_size = window_size
        self.root_dir = root_dir
        self.date = date
        self.lite = lite
        self.rain_threshold = 10 if heavy_rain else 0.1
        self.train = train
        self.transform = transform
        
        self._data_path_list, self._gt_path = self.__set_path()

    def __set_path(self):
        root, dirs, _ = next(os.walk(os.path.join(self.root_dir, str(self.date['year'])), topdown=True))
        
        # Make datetime object for start and end date
        assert self.date['year']        in VALID_YEAR
        assert self.date['start_month'] in VALID_MONTH
        assert self.date['end_month']   in VALID_MONTH

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
            unis_data_path_list = sorted([f for f in curr_data_path_list \
                                          if 'unis' in f and int(f.split('_')[4][8:10]) == self.model_utc])
            if self.lite:
                pres_data_path_list = unis_data_path_list
            else:
                pres_data_path_list = sorted([f for f in curr_data_path_list \
                                              if 'pres' in f and int(f.split('_')[4][8:10]) == self.model_utc])
                assert len(pres_data_path_list) == len(unis_data_path_list)
            
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
                        if self.lite:
                            data_path_list.append(os.path.join(data_dir, u))
                        else:
                            data_path_list.append((os.path.join(data_dir, p), os.path.join(data_dir, u)))
        
        # Set target data list
        # For training, use gt data which is same size with LDAPS grid
        # For testing, use gt data which contains observation from only 705 stations
        if self.train:
            gt_dir = os.path.join(self.root_dir, '..', 'OBS', 'train', str(self.date['year']))
            gt_path = os.path.join(gt_dir, 'rainr.npy')
            gt_path = torch.tensor(np.load(gt_path))
            gt_path = torch.where(gt_path >= self.rain_threshold, torch.ones(gt_path.shape), torch.zeros(gt_path.shape))
        else:
            gt_dir = os.path.join(self.root_dir, '..', 'OBS', 'test', str(self.date['year']))
            gt_path = os.listdir(gt_dir)
            gt_path = sorted([os.path.join(gt_dir, f) for f in gt_path if
                              f.split('_')[3][:-2] >= start_date.strftime("%Y%m%d%H") and
                              f.split('_')[3][:-2] <= gt_end_date.strftime("%Y%m%d%H")])
        
        return data_path_list, gt_path

    def __len__(self):        
        return len(self._data_path_list)

    def __getitem__(self, idx):
        if self.lite:
            u = self._data_path_list[idx]
        else:
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
            if not self.lite:
                curr_p = p.replace('h{}'.format(str(from_h).zfill(3)),
                                   'h{}'.format(str(from_h - self.window_size + t).zfill(3)))
                data_list.insert(0, curr_p)
            
            # TODO: add pres_idx_list and unis_idx_list arguments using var_name
            _ldaps_input.append(self._merge_pres_unis(data_list=data_list,
                                                      #pres_idx_list=[4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15],
                                                      #unis_idx_list=[0, 2, 3, 5, 6, 7, 8, 9, 12, 14, 15, 16]))
                                                      unis_idx_list=[2, 14, 17],
                                                      use_kindex=True))
        
        if self.model == 'unet' or self.model == 'attn_unet':
            for idx, l in enumerate(_ldaps_input):
                if idx == 0:
                    ldaps_input = l
                else:
                    ldaps_input = np.concatenate((ldaps_input, l), axis=0)
        else:
            ldaps_input = _ldaps_input
        
        if self.transform:
            ldaps_input = self.transform(ldaps_input)

        # Get ground-truth based on train mode
        gt = self._get_gt_data(train_end_time)

        # Make tensor of current target time for test logging
        target_time_tensor = torch.tensor([current_utc_date.year,
                                           current_utc_date.month,
                                           current_utc_date.day,
                                           current_utc_date.hour,
                                           from_h])

        return ldaps_input, gt, target_time_tensor

    def _merge_pres_unis(self, data_list, pres_idx_list=None, unis_idx_list=None, use_kindex=True):
        assert (pres_idx_list != None) or (unis_idx_list != None)

        if self.lite:
            u = data_list[0]
            unis = np.load(u)

            return unis
        else:
            p, u = data_list
            pres = np.load(p).reshape(512, 512, 20).transpose()
            unis = np.load(u).reshape(512, 512, 20).transpose()

            if use_kindex:
                T_850 = np.expand_dims(pres[9], axis=0)
                T_700 = np.expand_dims(pres[10], axis=0)
                T_500 = np.expand_dims(pres[11], axis=0)

                rh_850 = np.expand_dims(np.where(pres[13] > 0, pres[13], 0.), axis=0)
                rh_700 = np.expand_dims(np.where(pres[14] > 0, pres[14], 0.), axis=0)

                t_850 = T_850 - 273.15
                t_700 = T_700 - 273.15

                D_850 = (rh_850 / 100) ** (1 / 8) * (112 + (0.9 * t_850)) - 112 + (0.1 * t_850) + 273.15
                D_700 = (rh_700 / 100) ** (1 / 8) * (112 + (0.9 * t_700)) - 112 + (0.1 * t_700) + 273.15
                kindex = (T_850 - T_500) + D_850 - (T_700 - D_700)

            concat_list = []
            if pres_idx_list != None:
                pres = pres[pres_idx_list, :, :]
                concat_list.append(pres)
            if unis_idx_list != None:
                unis = unis[unis_idx_list, :, :]
                concat_list.append(unis)
            if use_kindex:
                concat_list.append(kindex)

            return np.concatenate(concat_list, axis=0)

    def _get_index(self, train_end_time):
        # Declare new_year date
        new_year_date = datetime(year=self.date['year'], month=1, day=1, hour=0)
    
        # getting passing days (train_end_date) from new_years
        index = int((train_end_time - new_year_date).total_seconds() // 3600)

        return index

    def _get_gt_data(self, train_end_time):
        if self.train:
            # Getting indices of ground truth data
            train_end_time_index = self._get_index(train_end_time)
            gt_index = train_end_time_index

            # Slicing gt data 
            gt = self._gt_path[gt_index]
        else:
            gt_path = [p for p in self._gt_path \
                       if train_end_time.strftime("%Y%m%d%H") in p][0]
            gt = torch.tensor(np.load(gt_path))
            gt = torch.where(gt >= self.rain_threshold, torch.ones(gt.shape), torch.zeros(gt.shape))

        return gt

    # XXX: Need to fix for new training target data
    def get_real_gt(self, idx):
        gt_path = self._gt_path[idx]
        real_gt = np.load(gt_path)

        return real_gt
    
class ToTensor(object):
    def __call__(self, images):
        return torch.from_numpy(images)