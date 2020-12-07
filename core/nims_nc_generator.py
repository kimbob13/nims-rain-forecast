import torch
from .nims_util import get_min_max_normalization
from .nims_logger import NIMSLogger

from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import xarray as xr

from scipy.special import softmax


__all__ = ['NIMSNCGenerator']

class NIMSNCGenerator:
    def __init__(self, model, device, data_loader, nc_dir, 
                 experiment_name, args, normalization=None, test_date_list=None):
        self.model = model
        self.model_name = args.model
        self.device = device
        self.data_loader = data_loader
        self.nc_dir = nc_dir
        self.experiment_name = experiment_name
        self.normalization = normalization
        self.device_idx = int(args.device)
        self.model.to(self.device)
        self.codi = self._get_coordinate()
        self.stn_codi = self._get_station_coordinate()

    def _get_coordinate(self):
        codi_df = pd.read_csv('./codi_ldps_aws/codi_ldps_602_781.csv')
        
        return codi_df
        
    def _get_station_coordinate(self):
        # codi_aws_df = pd.read_csv('./codi_ldps_aws/codi_ldps_aws_512.csv')
        codi_aws_df = pd.read_csv('./codi_ldps_aws/codi_ldps_aws_602_781.csv')
        dii_info = np.array(codi_aws_df['dii']) - 1
        # stn_codi = np.array([(dii // 512, dii % 512) for dii in dii_info])
        stn_codi = np.array([(dii // 602, dii % 602) for dii in dii_info])

        return stn_codi

    def gen_nc(self):
        pbar = tqdm(self.data_loader)
        for images, target_time in pbar:
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
            if self.model_name == 'unet' or self.model_name == 'attn_unet':
                output = self.model(images)
            elif self.model_name == 'suc_unet':
                output_lst = self.model(images)
            elif self.model_name == 'convlstm':
                output = self.model(images, future_seq=1)
                
            year   = target_time[0][0].item()
            month  = target_time[0][1].item()
            day    = target_time[0][2].item()
            hour   = target_time[0][3].item()
            from_h = target_time[0][4].item()
            
            curr_time = "{:04d}{:02d}{:02d}{:02d}".format(year, month, day, from_h)
            
            logit = output[0].cpu().detach().numpy()
            prob = softmax(logit, axis=0)[1].astype(np.float32)
            prcp = np.where(prob>=0.5, 1., 0.).astype(np.float32)
            
            X = np.arange(1., 603.)
            Y = np.arange(1., 782.)
            
            prob_pd = pd.DataFrame(prob, index=Y, columns=X)
            prcp_pd = pd.DataFrame(prcp, index=Y, columns=X)
            
            if from_h == 6:
                nc_data = xr.Dataset(coords={'X': X, 'Y': Y})
                
                longitude = self.codi['lon_l'].to_numpy().reshape(781, 602)
                latitude = self.codi['lat_l'].to_numpy().reshape(781, 602)
                
                nc_data = nc_data.assign({'longitude': (('Y', 'X'), longitude),
                                          'latitude': (('Y', 'X'), latitude)})
                for i in range(from_h):
                    nc_data = nc_data.assign({'prob_h{:02d}'.format(i): (('Y', 'X'), prob_pd),
                                              'prcp_h{:02d}'.format(i): (('Y', 'X'), prcp_pd)})
                nc_data = nc_data.assign({'prob_h{:02d}'.format(from_h): (('Y', 'X'), prob_pd),
                                          'prcp_h{:02d}'.format(from_h): (('Y', 'X'), prcp_pd)})
            else:
                nc_data = nc_data.assign({'prob_h{:02d}'.format(from_h): (('Y', 'X'), prob_pd),
                                          'prcp_h{:02d}'.format(from_h): (('Y', 'X'), prcp_pd)})
            
            if from_h == 48:
                nc_data.to_netcdf('./NC/{}.nc'.format(curr_time[:-2]))
                nc_data.close()