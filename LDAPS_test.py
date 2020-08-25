import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from LDPS_logger import LDPSLogger
from nims_util import create_results_dir
from nims_dataset import NIMSDataset

from tqdm import tqdm
import os
import argparse
import pdb

import datetime
from collections import defaultdict

try:
    import setproctitle
except:
    pass

def get_stat(pred, target):
    binary_f1 = f1_score(target, pred, zero_division=0)

    """
    confusion matrix
    tp: Hit
    fn: Miss
    fp: False Alarm
    tn: Correct Negative
    """
    conf_mat = confusion_matrix(target, pred, labels=np.arange(2))
    hit, miss, fa, cn = conf_mat[1, 1], conf_mat[1, 0], conf_mat[0, 1], conf_mat[0, 0]
    correct = hit + cn

    return correct, binary_f1, hit, miss, fa, cn

def find_gt_path(gt_path_list, target_time):
    for gt_path in gt_path_list:
        gt_time = gt_path.split('/')[-1].split('_')[3][:-2]
        if gt_time == target_time:
            return gt_path
    raise NotImplemented

if __name__ == '__main__':
    
    ### set argument

    parser = argparse.ArgumentParser(description='NIMS rainfall data prediction')
#   parser.add_argument('--dataset_dir', default='/home/osilab12/ssd/OBS/2020', type=str, help='root directory of dataset')
    parser.add_argument('--test_time', default='20200601', type=str, help='date of test')
    #parser.add_argument('--test_time_start', default='20200601', type=str, help='start date of test')
    #parser.add_argument('--test_time_end', default=None, type=str, help='end date of test')
    args = parser.parse_args()
    
    test_dates = []
    test_date = datetime.datetime.strptime(args.test_time, "%Y%m%d")
    test_dates.append(test_date.strftime("%Y%m%d"))
    test_dates.append((test_date + datetime.timedelta(days=1)).strftime("%Y%m%d"))
    test_dates.append((test_date + datetime.timedelta(days=2)).strftime("%Y%m%d")) # Because LDPS data uses 48h after 0, 6, 12, 18h

    ### set LDAPS, OBS dir path
    LDAPS_root_dir = '/home/osilab12/hdd2/NIMS_LDPS'
    OBS_root_dir = '/home/osilab12/hdd2/OBS'
    
    test_time = args.test_time
    LDAPS_year_dir = os.path.join(LDAPS_root_dir, test_time[:4])
    OBS_year_dir = os.path.join(OBS_root_dir, test_time[:4])
    
    LDAPS_dir = os.path.join(LDAPS_year_dir, test_time)
    
    
    result_path = './results'
    logger_folder = 'LDAPS_Logger'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    test_result_path = os.path.join(result_path, logger_folder)
    if not os.path.isdir(test_result_path):
        os.mkdir(test_result_path)
    
    codi_aws_df = pd.read_csv('./codi_ldps_aws/codi_ldps_aws_512.csv')
    dii_info = np.array(codi_aws_df['dii']) - 1
    stn_codi = np.array([(dii // 512, dii % 512) for dii in dii_info])

    ldps_logger = LDPSLogger(loss=False, correct=False, binary_f1=False,
                             macro_f1=False, micro_f1=False,
                             hit=True, miss=True, fa=True, cn=True,
                             stn_codi=stn_codi,
                             test_result_path=test_result_path)

    ### get ground truth data for test time
    
    gt_dir = OBS_year_dir
    
    #ground truth data list(file name) /
    gt_path_list = os.listdir(gt_dir)
    gt_path_list = sorted([os.path.join(gt_dir, f) \
                           for f in gt_path_list \
                           if f.endswith('.npy') and \
                           f.split('_')[3][:8] in test_dates]) # when recorded data is located between test time start and end
    dataset_len = len(gt_path_list)
    
    #load data in dictionary key = time(0~23) / value = np.array // Total data load
    gt_data_dict = defaultdict(list)
    for i, data_dir in enumerate(gt_path_list):
        tmp_data = np.load(data_dir).reshape(512, 512, 1).transpose()
        gt_data_dict[i] = tmp_data
        
    
    ### get LDAPS data (LCPCP, index=2)
    
    LDAPS_path_dir = os.listdir(LDAPS_dir)
    for LDAPS_data in LDAPS_path_dir:
        curr_data_path = [f for f in LDAPS_path_dir]
        unis_data_path_list = sorted([f for f in curr_data_path if 'unis' in f])

    #load dadta in dictionary key = time(ex h000_00, h000_06) value=  np.array
    unis_data_dict = defaultdict(list)

    for a in unis_data_path_list:
        tmp_h, tmp_t =a.split('_')[3], a.split('_')[4][8:10]
        tmp_list = [tmp_h, tmp_t]
        tmp_key = '_'.join(tmp_list)

        tmp_data = np.load(os.path.join(LDAPS_dir, a)).reshape(512,512,20).transpose()
        tmp_data = tmp_data[2, :, :]
        unis_data_dict[tmp_key] = tmp_data

    for start_hour in range(4):
        start_hour = start_hour * 6 # 0, 6, 12, 18
        pbar = tqdm(range(49)) # 48h prediction for each start hour / h000~h0048
        for i in pbar:
            target_hour = i + start_hour
            target_time = datetime.datetime.strptime(args.test_time, "%Y%m%d") + datetime.timedelta(hours=target_hour)
            target_time = target_time.strftime("%Y%m%d%H")
            test_time = [int(args.test_time[0:4]), int(args.test_time[4:6]), int(args.test_time[6:8])]
            gt_path = find_gt_path(gt_path_list, target_time)
            
        ### preprocessing reference data

            reference_data = np.load(gt_path)
            reference_data = np.where(reference_data >= 0.1, np.ones(reference_data.shape), np.zeros(reference_data.shape))
            reference_data = reference_data[stn_codi[:, 0], stn_codi[:, 1]]

        ### preprocessing LDPS data
            LDPS_data = unis_data_dict["h00{0}_{1}".format(i, '00' if start_hour==0 else start_hour)] if (i//10)==0 \
            else unis_data_dict["h0{0}_{1}".format(i, '00' if start_hourt==0 else start_hour)]
            LDPS_data = np.asarray(LDPS_data) #LDPS_Data.size = (512,512)
#             pdb.set_trace()
#             print(LDPS_data.shape)
#             LDPS_data = np.squeeze(LDPS_data, axis=0)
            LDPS_data = np.where(LDPS_data >= 0.1, np.ones(LDPS_data.shape), np.zeros(LDPS_data.shape))
            LDPS_data = LDPS_data[stn_codi[:, 0], stn_codi[:, 1]]

        ### get confusion matrix

            _, _, hit, miss, fa, cn = get_stat(LDPS_data, reference_data)

        ### update logger

            ldps_logger.update(hit=hit, miss=miss, fa=fa, cn=cn, target_time=test_time, start_hour=start_hour, target_hour_48=i, test=True)
            pbar.set_description(nims_logger.latest_stat)

    #nims_logger.print_stat(dataset_len, test=True)
