import os
import scipy.io
import numpy as np

if __name__ == "__main__":
    
    '''
        User Defined : file_path
    '''
    # File load
    sav_filename_19 = "/home/osilab12/ssd/NIMS_LDPS_SAV/rain_19r.sav"
    sav_filename_18 = "/home/osilab12/ssd/NIMS_LDPS_SAV/rain_18r.sav"

    # File save
    save_path_19 = "/home/osilab12/ssd/NIMS_LDPS_SAV/2019"
    save_path_18 = "/home/osilab12/ssd/NIMS_LDPS_SAV/2018"

    column_name = ['xx', 'yy', 'rainr']
    data = scipy.io.readsav(sav_filename_19, python_dict=True, verbose=True)

    for name in column_name:
        if name == "rainr":
            #Debug (time, latitude, longitude) [8760, 781, 602]
            rain_data = data[name].astype(np.float32)
            datalist = []
            datalist.append(rain_data)

            # Tranpose 
                #(time, latitude, longitude) [8760, 781, 602] --> (time, longitude, latitude) [8760, 602, 781]
            rain_data = np.transpose(rain_data, (0, 2, 1))

            # Get shape
            shape = rain_data.shape
            print(shape)

            #Slicing
            start_x = shape[1] - 512
            rain_data = rain_data[:, start_x:, :512] # rain_data[:, 90:, :512]

            #Debug
            print(rain_data.shape)
            
            # Save
            save_filename = os.path.join(save_path_19, name + ".npy")
            np.save(save_filename, rain_data)
            
    
    # File test using .npy
    npy_filename_19 = "/home/osilab12/ssd/NIMS_LDPS_SAV/rainr.npy"
    rainr_npy = np.load(npy_filename_19)

    print(rainr_npy.shape)
    

    

        

    

    

    
    






    