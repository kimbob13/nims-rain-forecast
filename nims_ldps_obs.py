import os
import pytz
from datetime import datetime, timezone
from dateutil import tz
import numpy as np
import pandas as pd

codi_aws_df = pd.read_csv('/home/kimbob/jupyter/weather_prediction/pr_sample/codi_ldps_aws/codi_ldps_aws_512.csv')

def get_station_coordinate(stn_id):
    stn_info = codi_aws_df[codi_aws_df['stn'] == stn_id]
    stn_dii = stn_info['dii'] - 1

    x, y = stn_dii // 512, stn_dii % 512

    return x, y

if __name__ == '__main__':
    root_dir = '/home/osilab11/hdd/NIMS_LDPS/OBS'

    KST = tz.gettz('Asia/Seoul')
    obs_list = sorted(os.listdir(root_dir))
    for obs_file in obs_list:
        if obs_file.endswith('.npy'):
            continue

        print('[current_file]:', obs_file)
        result_array = np.zeros([512, 512])
        with open(os.path.join(root_dir, obs_file), 'r', encoding='euc-kr') as f:
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
        with open(os.path.join(root_dir, file_name), 'wb') as npyf:
            np.save(npyf, result_array)
