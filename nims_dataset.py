import torch
from torch.utils.data import Dataset

from nims_variable import read_variable_value

import numpy as np
import xarray as xr

import os
import pwd

__all__ = ['NIMSDataset', 'ToTensor']

START_YEAR = 2009
END_YEAR = 2018
NORMAL_YEAR_DAY = 365


class NIMSDataset(Dataset):
    def __init__(self, window_size, target_num, variables,
                 train_year=(2009, 2017), train=True, transform=None,
                 root_dir=None, debug=False):
        self.window_size = window_size
        self.target_num = target_num
        self.variables = variables

        self.start_train_year = train_year[0]       # start year for training
        self.end_train_year = train_year[1]         # end year for training
        self.test_year = self.end_train_year + 1    # year for testing

        self.train = train
        self.transform = transform

        self.debug = debug

        if root_dir == None:
            self.root_dir = self.__set_default_root_dir()
        else:
            self.root_dir = root_dir

        self._data_path_list = self.__set_data_path_list()

        assert len(self.variables) <= 14
        assert self.start_train_year <= self.end_train_year
        assert self.start_train_year >= START_YEAR
        assert self.end_train_year <= END_YEAR - 1

    @property
    def data_path_list(self):
        return self._data_path_list
        
    def __set_default_root_dir(self):
        for p in pwd.getpwall():
            if p[0].startswith('osilab'):
                data_user = p[0]
                break

        return os.path.join('/home', data_user, 'hdd/NIMS')

    def __set_data_path_list(self):
        root, dirs, _ = next(os.walk(self.root_dir, topdown=True))

        data_dirs = [os.path.join(root, d) for d in sorted(dirs)]

        # Get proper data directory for specified train and test year
        if self.train:
            start_day = (self.start_train_year - START_YEAR) * NORMAL_YEAR_DAY
            end_day = -((END_YEAR - self.end_train_year) * NORMAL_YEAR_DAY)

            if self.start_train_year > 2012 and self.start_train_year <= 2016:
                start_day += 1
            elif self.start_train_year > 2016:
                start_day += 2

            if self.end_train_year >= 2012 and self.end_train_year < 2016:
                end_day -= 1
            elif self.end_train_year < 2012:
                end_day -= 2

            data_dirs = data_dirs[start_day:end_day]

            if self.debug:
                print('train start: {}, end: {}'.format(data_dirs[0], data_dirs[-1]))
        else:
            start_day = (self.test_year - START_YEAR) * NORMAL_YEAR_DAY

            if self.test_year > 2012 and self.test_year <= 2016:
                start_day += 1
            elif self.test_year > 2016:
                start_day += 2

            if self.test_year in (2012, 2016):
                end_day = start_day + 366
            else:
                end_day = start_day + 365

            data_dirs = data_dirs[start_day:end_day]

            if self.debug:
                print('test start: {}, end: {}'.format(data_dirs[0], data_dirs[-1]))

        data_path_list = []
        for data_dir in data_dirs:
            curr_data_path_list = [os.path.join(data_dir, f) \
                                   for f in sorted(os.listdir(data_dir))]
            data_path_list += curr_data_path_list

        if self.debug:
            print('[{}] first day: {}, last day: {}'
                  .format('train' if self.train else 'test',
                          data_path_list[0], data_path_list[-1]))

        return data_path_list

    def __len__(self):
        return len(self._data_path_list) - self.window_size

    def __getitem__(self, idx):
        images_window_path = self._data_path_list[idx:idx + self.window_size]

        target_start_idx = idx + self.window_size
        target_end_idx = target_start_idx + self.target_num
        target_window_path = self._data_path_list[target_start_idx:target_end_idx]

        images = self._merge_window_data(images_window_path)
        target = self._merge_window_data(target_window_path, target=True)
        target = self._to_pixel_wise_label(target)

        if self.transform:
            images = self.transform(images)
            target = self.transform(target)

        return images, target

    def _merge_window_data(self, window_path, target=False):
        """
        Merge data in current window into single numpy array

        <Parameters>
        window_path [list[str]]: list of file path in current window
        target [bool]: whether merged data is for target.
                       target data only use rain data

        <Return>
        results [np.ndarray]
            - model == stconvs2s: result array in CDHW format
            - model == unet: result array in CHW format
        """
        for i, data_path in enumerate(window_path):
            one_hour_dataset = xr.open_dataset(data_path)

            if target:
                # Use rain value as target
                one_hour_data = read_variable_value(one_hour_dataset, 0)
            else:
                for var_idx in self.variables:
                    if var_idx == 0:
                        one_hour_data = read_variable_value(one_hour_dataset,
                                                            var_idx)
                    else:
                        one_var_data = read_variable_value(one_hour_dataset,
                                                           var_idx)
                        one_hour_data = np.concatenate([one_hour_data,
                                                        one_var_data],
                                                       axis=0)

            one_hour_data = np.expand_dims(one_hour_data, axis=0)

            if i == 0:
                results = one_hour_data
            else:
                results = np.concatenate([results, one_hour_data], axis=0)

        # We change each tensor to CWH format when the model is UNet
        # window_size is serves as channel in UNet
        #assert self.window_size == 1
        results = results.squeeze(1)

        return results

    def _to_pixel_wise_label(self, target):
        """
        Based on value in each target pixel,
        change target tensor to pixel-wise label value

        <Parameters>
        target [np.ndarray]: CHW format, C must be 1

        <Return>
        results [np.ndarray]: HW format
        """
        assert target.shape[0] == 1
        assert self.target_num == 1

        target = target.squeeze(0)
        target_label = np.zeros([target.shape[0], target.shape[1]])

        for lat in range(target.shape[0]):
            for lon in range(target.shape[1]):
                value = target[lat][lon]

                if value >= 0 and value < 0.1:
                    target_label[lat][lon] = 0
                elif value >= 0.1 and value < 1.0:
                    target_label[lat][lon] = 1
                elif value >= 1.0 and value < 2.5:
                    target_label[lat][lon] = 2
                elif value >= 2.5:
                    target_label[lat][lon] = 3
                # if value >= 0 and value < 0.1:
                #     target_label[lat][lon] = 0
                # elif value >= 0.1:
                #     target_label[lat][lon] = 1
                else:
                    #print('Invalid target value:', value)
                    target_label[lat][lon] = 0

        return target_label

class ToTensor(object):
    def __call__(self, images):
        return torch.from_numpy(images)
