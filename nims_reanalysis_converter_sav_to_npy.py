import os
import scipy.io
import numpy as np

from datetime import datetime, timedelta

def convert_sav_to_npy(sav_filename, save_path, year):
    data = scipy.io.readsav(sav_filename, python_dict=True, verbose=True)
    # (time, latitude, longitude) [8760, 781, 602]
    rain_data = data['rainr'].astype(np.float32)

    # Get shape
    shape = rain_data.shape
    print(shape)

    # Slicing
    # start_x = shape[1] - 512
    # rain_data = rain_data[:, start_x:, :512] # rain_data[:, 90:, :512]

    # Save
    os.makedirs(save_path, exist_ok=True)
    for i in range(len(rain_data)):
        if i == 0:
            current_date = datetime(year=year, month=1, day=1, hour=0)
        else:
            current_date = current_date + timedelta(hours=1)
        
        save_filename = os.path.join(save_path, current_date.strftime("%Y%m%d%H")+'.npy')
        np.save(save_filename, rain_data[i])
        
if __name__ == "__main__":
    
    '''
        User Defined : file_path
    '''
    
    # 2018 year converter
    sav_filename = "/home/osilab12/ssd2/REANALYSIS/rain_18r.sav"
    save_path = "/home/osilab12/ssd2/REANALYSIS/2018"
    convert_sav_to_npy(sav_filename, save_path, year=2018)
    
    # 2019 year converter
    sav_filename = "/home/osilab12/ssd2/REANALYSIS/rain_19r.sav"
    save_path = "/home/osilab12/ssd2/REANALYSIS/2019"
    convert_sav_to_npy(sav_filename, save_path, year=2019)