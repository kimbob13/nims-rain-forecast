import torch
from .nims_util import get_min_max_normalization
from .nims_logger import NIMSLogger

from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import netCDF4 as nc

from scipy.special import softmax


__all__ = ['NIMSNCGenerator']

class NIMSNCGenerator:
    def __init__(self, model, device, data_loader, nc_result_dir, 
                 experiment_name, args, normalization=None, test_date_list=None):
        self.model = model
        self.model_name = args.model
        self.device = device
        self.data_loader = data_loader
        self.nc_result_dir = nc_result_dir
        self.experiment_name = experiment_name
        self.args = args
        self.normalization = normalization
        self.model.to(self.device)
        self.codi = self._get_coordinate()
        self.stn_codi = self._get_station_coordinate()

        os.makedirs(self.nc_result_dir, exist_ok=True)


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

            logit = output[0].cpu().detach().numpy()
            prob = softmax(logit, axis=0)[1].astype(np.float32)
            prcp = np.where(prob>=0.5, 1., 0.).astype(np.float32)

            if from_h == 6:
                fname = self.nc_result_dir + '/AIW_UNET_PRCP01_KR_V01_{:04d}{:02d}{:02d}_{:02d}0000.nc'.format(year,
                                                                                                               month,
                                                                                                               day,
                                                                                                               self.args.model_utc)
                nc_data = nc.Dataset(fname, mode='w', format='NETCDF4')
                complevel=5

                X_dim = nc_data.createDimension('X', 602)
                Y_dim = nc_data.createDimension('Y', 781)

                X_var = nc_data.createVariable('X', np.float64, ('X',), zlib=True, complevel=complevel)
                X_var[:] = np.arange(1., 603.)
                X_var.units = 'units'
                X_var.long_name = 'X'
                Y_var = nc_data.createVariable('Y', np.float64, ('Y',), zlib=True, complevel=complevel)
                Y_var[:] = np.arange(1., 782.)
                Y_var.units = 'units'
                Y_var.long_name = 'Y'

                lon_var = nc_data.createVariable('longitude', np.float32, ('Y','X'), zlib=True, complevel=complevel)
                lon_var[:, :] = self.codi['lon_l'].to_numpy().reshape(781, 602)
                lon_var.units = 'deg'
                lat_var = nc_data.createVariable('latitude', np.float32, ('Y','X'), zlib=True, complevel=complevel)
                lat_var[:, :] = self.codi['lat_l'].to_numpy().reshape(781, 602)
                lat_var.units = 'deg'

                prob_group = nc_data.createGroup('PROB_01')
                prcp_group = nc_data.createGroup('PRCP_01')

                for i in range(from_h):
                    tmp_prob_var = prob_group.createVariable('h{:03d}'.format(i), np.float32, ('Y','X'), zlib=True, complevel=complevel)
                    tmp_prob_var[:, :] = prob
                    tmp_prob_var.units = '0-1'
                    tmp_prob_var.long_name = 'PROB_01 T+{:03d}'.format(i)
                    tmp_prob_var.coordinates = 'longitude latitude'

                    tmp_prcp_var = prcp_group.createVariable('h{:03d}'.format(i), np.float32, ('Y','X'), zlib=True, complevel=complevel)
                    tmp_prcp_var[:, :] = prcp
                    tmp_prcp_var.units = '0-1'
                    tmp_prcp_var.long_name = 'PRCP_01 T+{:03d}'.format(i)
                    tmp_prcp_var.coordinates = 'longitude latitude'

                tmp_prob_var = prob_group.createVariable('h{:03d}'.format(from_h), np.float32, ('Y','X'), zlib=True, complevel=complevel)
                tmp_prob_var[:, :] = prob
                tmp_prob_var.units = '0-1'
                tmp_prob_var.long_name = 'PROB_01 T+{:03d}'.format(from_h)
                tmp_prob_var.coordinates = 'longitude latitude'

                tmp_prcp_var = prcp_group.createVariable('h{:03d}'.format(from_h), np.float32, ('Y','X'), zlib=True, complevel=complevel)
                tmp_prcp_var[:, :] = prcp
                tmp_prcp_var.units = '0-1'
                tmp_prcp_var.long_name = 'PRCP_01 T+{:03d}'.format(from_h)
                tmp_prcp_var.coordinates = 'longitude latitude'
            else:
                tmp_prob_var = prob_group.createVariable('h{:03d}'.format(from_h), np.float32, ('Y','X'), zlib=True, complevel=complevel)
                tmp_prob_var[:, :] = prob
                tmp_prob_var.units = '0-1'
                tmp_prob_var.long_name = 'PROB_01 T+{:03d}'.format(from_h)
                tmp_prob_var.coordinates = 'longitude latitude'

                tmp_prcp_var = prcp_group.createVariable('h{:03d}'.format(from_h), np.float32, ('Y','X'), zlib=True, complevel=complevel)
                tmp_prcp_var[:, :] = prcp
                tmp_prcp_var.units = '0-1'
                tmp_prcp_var.long_name = 'PRCP_01 T+{:03d}'.format(from_h)
                tmp_prcp_var.coordinates = 'longitude latitude'

            if from_h == 48:
                nc_data.close()













                        
