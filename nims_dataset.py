import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import numpy as np
import xarray as xr

import os
import pwd

__all__ = ['NIMSDataset', 'ToTensor']

class NIMSDataset(Dataset):
    def __init__(self, model, window_size, target_num, train=True,
                 transform=None, root_dir=None):
        self.model = model
        self.window_size = window_size
        self.target_num = target_num
        self.train = train
        self.transform = transform

        if root_dir == None:
            self.root_dir = self.__set_default_root_dir()

        self.data_path_list = self.__set_data_path_list()

    def __set_default_root_dir(self):
        for p in pwd.getpwall():
            if p[0].startswith('osilab'):
                data_user = p[0]
                break

        return os.path.join('/home', data_user, 'hdd/NIMS')

    def __set_data_path_list(self):
        root, dirs, _ = next(os.walk(self.root_dir, topdown=True))

        data_dirs = [os.path.join(root, d) for d in sorted(dirs)]

        data_path_list = []
        for data_dir in data_dirs:
            curr_data_path_list = [os.path.join(data_dir, f) \
                                   for f in sorted(os.listdir(data_dir))]
            data_path_list += curr_data_path_list

        return data_path_list

    def __len__(self):
        return len(self.data_path_list) - self.window_size

    def __getitem__(self, idx):
        images_window_path = self.data_path_list[idx:idx + self.window_size]

        target_start_idx = idx + self.window_size
        target_end_idx = target_start_idx + self.target_num
        target_window_path = self.data_path_list[target_start_idx:target_end_idx]

        images = self._merge_window_data(images_window_path)
        target = self._merge_window_data(target_window_path)
        if self.model == 'unet':
            target = self._to_pixel_wise_label(target)

        if self.transform:
            images = self.transform(images)
            target = self.transform(target)

        return images, target

    def _merge_window_data(self, window_path):
        """
        Merge data in current window into single numpy array

        <Parameters>
        window_path [list[str]]: list of file path in current window

        <Return>
        results [np.ndarray]: result array in CDHW format
        """
        for i, data_path in enumerate(window_path):
            one_hour_dataset = xr.open_dataset(data_path)
            one_hour_rain_data = np.expand_dims(one_hour_dataset.rain.values, axis=0)
            if i == 0:
                results = one_hour_rain_data
            else:
                results = np.concatenate([results, one_hour_rain_data], axis=0)

        # window_size serves as first dimension of each image,
        # and each image has one channel.
        # Therefore, we change dimension order to match CDHW format.
        results = np.transpose(results, (1, 0, 2, 3))

        # We change each tensor to CWH format when the model is UNet
        if self.model == 'unet':
            assert self.window_size == 1
            results = results.squeeze(0)

        return results

    def _to_pixel_wise_label(self, target):
        """
        Based on value in each target pixel,
        change target tensor to pixel-wise label value

        <Parameters>
        target [np.ndarray]: CHW format, C must be 1

        <Return>
        results [np.ndarray]: CHW format, C must be 1
        """
        assert target.shape[0] == 1
        assert self.target_num == 1

        target = target.squeeze(0)
        target_label = np.zeros([target.shape[0], target.shape[1]])

        for lat in range(target.shape[0]):
            for lon in range(target.shape[1]):
                value = target[lat][lon]

                if value >= 0 and value < 0.1:
                    target_label[lat, lon] = 0
                elif value >= 0.1 and value < 1.0:
                    target_label[lat, lon] = 1
                elif value >= 1.0 and value < 2.5:
                    target_label[lat, lon] = 2
                elif value >= 2.5:
                    target_label[lat, lon] = 3
                else:
                    raise InvalidTargetValue("Invalid target value:", value)

        return target_label

class ToTensor(object):
    def __call__(self, images):
        return torch.from_numpy(images)

class InvalidTargetValue(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(*args, **kwargs)
