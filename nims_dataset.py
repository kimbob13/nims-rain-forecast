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
LEAP_YEAR_DAY = 366
MONTH_DAY = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


class NIMSDataset(Dataset):
    def __init__(self, model, window_size, target_num, variables, block_size, aggr_method,
                 train_year=(2009, 2017),  month=(1, 12), train=True, transform=None,
                 root_dir=None, debug=False):
        
        self.model = model
        self.window_size = window_size
        self.target_num = target_num
        self.variables = variables
        self.block_size = block_size
        self.aggr_method = aggr_method

        self.start_train_year = train_year[0]       # start year for training
        self.end_train_year = train_year[1]         # end year for training
        self.test_year = self.end_train_year + 1    # year for testing

        assert len(self.variables) <= 14
        assert self.start_train_year <= self.end_train_year
        assert self.start_train_year >= START_YEAR
        assert self.end_train_year <= END_YEAR - 1
        assert month[0] >= 1 and month[0] <= 12
        assert month[1] >= 1 and month[1] <= 12
        assert month[0] <= month[1]

        self.train = train
        self.transform = transform

        self.debug = debug

        if root_dir == None:
            self.root_dir = self.__set_default_root_dir()
        else:
            self.root_dir = root_dir

        self._data_path_list = self.__set_data_path_list(month[0], month[1])

    @property
    def data_path_list(self):
        return self._data_path_list
        
    def __set_default_root_dir(self):
        for p in pwd.getpwall():
            if p[0].startswith('osilab'):
                data_user = p[0]
                break

        return os.path.join('/home', data_user, 'hdd/NIMS')

    def __set_data_path_list(self, start_month, end_month):
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

        else:
            start_day = (self.test_year - START_YEAR) * NORMAL_YEAR_DAY

            if self.test_year > 2012 and self.test_year <= 2016:
                start_day += 1
            elif self.test_year > 2016:
                start_day += 2

            if self.test_year in (2012, 2016):
                end_day = start_day + LEAP_YEAR_DAY
            else:
                end_day = start_day + NORMAL_YEAR_DAY

        if start_month > 1:
            start_day += sum(MONTH_DAY[:start_month - 1])

            if (start_month != 2) and (self.start_train_year in (2012, 2016)):
                start_day += 1

        if end_month < 12:
            end_day -= sum(MONTH_DAY[end_month:])

            if (end_month == 1) and (self.end_train_year in (2012, 2016)):
                end_day -= 1

        data_dirs = data_dirs[start_day:end_day]

        if self.debug:
            if self.train:
                print('[NIMSDataset] train start: {}, end: {}'.format(data_dirs[0], data_dirs[-1]))
            else:
                print('[NIMSDataset] test start: {}, end: {}'.format(data_dirs[0], data_dirs[-1]))

        # Build data path list
        data_path_list = []
        for data_dir in data_dirs:
            curr_data_path_list = [os.path.join(data_dir, f) \
                                   for f in sorted(os.listdir(data_dir))]
            data_path_list += curr_data_path_list

        if self.debug:
            print('[NIMSDataset] {} - first day: {}, last day: {}'
                  .format('Train' if self.train else 'Test',
                          data_path_list[0], data_path_list[-1]))

        return data_path_list

    def __len__(self):
        return len(self._data_path_list) - self.window_size - \
               (self.target_num - 1)

    def __getitem__(self, idx):
        # Get images (train data) window path
        images_window_path = self._data_path_list[idx:idx + self.window_size]

        # Get target window path
        target_start_idx = idx + self.window_size
        target_end_idx = target_start_idx + self.target_num
        target_window_path = self._data_path_list[target_start_idx:target_end_idx]
        
        images = self._merge_window_data(images_window_path)
        target = self._merge_window_data(target_window_path, target=True)
        
        images, target = self._to_model_specific_tensor(images, target, self.train,
                                                        self.block_size, self.aggr_method)

        if self.transform:
            images = self.transform(images)
            target = self.transform(target)
            
        return images, target

    def get_real_target(self, idx):
        # TODO: Fix undersampling when target_num > 1
        assert self.target_num == 1, \
               'Currently, undersampling only works on target_num == 1'

        # Get target window path
        target_start_idx = idx + self.window_size
        target_end_idx = target_start_idx + self.target_num
        target_window_path = self._data_path_list[target_start_idx:target_end_idx]
        
        target = self._merge_window_data(target_window_path, target=True)

        return target

    def _merge_window_data(self, window_path, target=False):
        """
        Merge data in current window into single numpy array

        <Parameters>
        window_path [list[str]]: list of file path in current window
        target [bool]: whether merged data is for target.
                       target data only use rain data

        <Return>
        results [np.ndarray]: SCHW format
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

        return results
    
    def _to_model_specific_tensor(self, images, target, train, block_size=1, aggr_method=None):
        """
        Change images and target tensor to model specific tensor
        i.e. change the shape of images and target

        <Parameters>
        images [np.ndarray]: SCHW format (S: window size)
        target [np.ndarray]: SCHW format (S: window size)
        block_size [int]: the size of aggregated block.
                          if block_size == 1, then original images and target.
        aggr_method [str]: block aggregation method - 'max' or 'avg'

        <Return>
        model == unet:
            images [np.ndarray]: SHW format
            target [np.ndarray]: SHW format
        model == convlstm
            images [np.ndarray]: SCHW format
            target [np.ndarray]: SCHW format
        """
        
        if train and block_size > 1:
            reduced_height = images.shape[2] // block_size
            reduced_width = images.shape[3] // block_size

            reduced_images = np.zeros([images.shape[0], images.shape[1], reduced_height, reduced_width])
            reduced_target = np.zeros([target.shape[0], target.shape[1], reduced_height, reduced_width])

            for i in range(reduced_height):
                for j in range(reduced_width):
                    tmp_images = images[:, :, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    tmp_target = target[:, :, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    
                    if aggr_method == 'max':
                        aggregated_images = np.max(tmp_images, axis=(2,3))
                        aggregated_target = np.max(tmp_target, axis=(2,3))
                    elif aggr_method == 'avg':
                        aggregated_images = np.average(tmp_images, axis=(2,3))
                        aggregated_target = np.average(tmp_target, axis=(2,3))
                    
                    reduced_images[:,:,i,j] = aggregated_images
                    reduced_target[:,:,i,j] = aggregated_target
                    
            images = reduced_images
            target = reduced_target
        
        if self.model == 'unet':
            # We change each tensor to CWH format when the model is UNet
            # window_size is serves as channel in UNet
            #assert self.window_size == 1
            images = images.squeeze(1)
            target = target.squeeze(1)
            target = self._to_pixel_wise_label(target)
            
        elif self.model == 'convlstm':
            # Just return original images and target
            pass

        return images, target

    def _to_pixel_wise_label(self, target):
        """
        Based on value in each target pixel,
        change target tensor to pixel-wise label value.
        Used for UNet model.

        <Parameters>
        target [np.ndarray]: SHW format (S: window size)

        <Return>
        results [np.ndarray]: SHW format
        """
        target_label = np.zeros([target.shape[0], target.shape[1], target.shape[2]])

        for seq in range(target.shape[0]):
            for lat in range(target.shape[1]):
                for lon in range(target.shape[2]):
                    value = target[seq][lat][lon]

                    if value >= 0 and value < 0.1:
                        target_label[seq][lat][lon] = 0
                    elif value >= 0.1 and value < 1.0:
                        target_label[seq][lat][lon] = 1
                    elif value >= 1.0 and value < 2.5:
                        target_label[seq][lat][lon] = 2
                    elif value >= 2.5:
                        target_label[seq][lat][lon] = 3
                    # if value >= 0 and value < 0.1:
                    #     target_label[seq][lat][lon] = 0
                    # elif value >= 0.1:
                    #     target_label[seq][lat][lon] = 1
                    else:
                        #print('Invalid target value:', value)
                        target_label[seq][lat][lon] = 0

        return target_label

class ToTensor(object):
    def __call__(self, images):
        return torch.from_numpy(images)
