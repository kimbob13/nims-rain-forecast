import os
import argparse
from datetime import datetime, timezone
from dateutil import tz

import numpy as np
import pandas as pd
from tqdm import tqdm

from nims_util import select_date

# codi_aws_df = pd.read_csv('./codi_ldps_aws/codi_ldps_aws_512.csv')
codi_aws_df = pd.read_csv('./codi_ldps_aws/codi_ldps_aws_602_781.csv')

def get_station_coordinate(stn_id):
    stn_info = codi_aws_df[codi_aws_df['stn'] == stn_id]
    stn_dii = stn_info['dii'] - 1

    # x, y = stn_dii // 512, stn_dii % 512
    x, y = stn_dii // 602, stn_dii % 602

    return x, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LDAPS Observations Converter')
    parser.add_argument('--root_dir', default='/home/osilab12/ssd', type=str, help='root directory of dataset')
    
    args = parser.parse_args()

    date = select_date()
    root_dir = args.root_dir

    KST = tz.gettz('Asia/Seoul')
    obs_txt_dir = os.path.join(root_dir, 'AWS', str(date['year']))  
    obs_txt_list = sorted([f for f in os.listdir(obs_txt_dir) if
                           f.split('_')[3][:-4] >= '{:4d}{:02d}{:02d}'.format(date['year'], date['start_month'], date['start_day']) and
                           f.split('_')[3][:-4] <= '{:4d}{:02d}{:02d}'.format(date['year'], date['end_month'], date['end_day'])])
    pbar = tqdm(obs_txt_list)
    for obs_txt in pbar:
        pbar.set_description('[current_file] {}'.format(obs_txt))

        # result_array = np.full([512, 512], -9999)
        result_array = np.full([781, 602], -9999)
        with open(os.path.join(obs_txt_dir, obs_txt), 'r', encoding='euc-kr') as f:
            for line in f:
                if line.startswith('#'):
                    continue

                # Get proper rain data and station coordinate
                line_list = line.strip().split()
                stn_id = int(line_list[1])
                one_hour_rain = float(line_list[6])
                stn_x, stn_y = get_station_coordinate(stn_id)

                result_array[stn_x, stn_y] = one_hour_rain

        # Convert KST time to UTC time
        date_str = line_list[0]
        kst_date = datetime(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]),
                            hour=int(date_str[8:10]), minute=int(date_str[10:12]),
                            tzinfo=KST)
        utc_date = kst_date.astimezone(tz=timezone.utc)
        #print('[date] KST = {}, UTC = {}'.format(date, utc_date))
        utc_year, utc_month, utc_day, utc_hour, utc_minute = \
                utc_date.year, utc_date.month, utc_date.day, utc_date.hour, utc_date.minute
        utc_str = '{:4d}{:02d}{:02d}{:02d}{:02d}'.format(utc_year, utc_month, utc_day,
                                                        utc_hour, utc_minute)
        file_name = 'AWS_HOUR_ALL_{}_{}.npy'.format(utc_str, utc_str)

        # Save npy file
        obs_npy_dir = os.path.join(root_dir, 'OBS', str(date['year']))
        os.makedirs(obs_npy_dir, exist_ok=True)
        with open(os.path.join(obs_npy_dir, file_name), 'wb') as npyf:
            np.save(npyf, result_array)
