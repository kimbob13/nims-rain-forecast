import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from nims_logger import NIMSLogger
from nims_util import create_results_dir
from nims_dataset import NIMSDataset

from tqdm import tqdm
import os
import argparse

import datetime

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

if __name__ == '__main__':
    
    ### set argument

    parser = argparse.ArgumentParser(description='NIMS rainfall data prediction')
    parser.add_argument('--dataset_dir', default='/home/osilab12/ssd/OBS/2020', type=str, help='root directory of dataset')
    parser.add_argument('--test_time', default='20200601', type=str, help='date of test')
    #parser.add_argument('--test_time_start', default='20200601', type=str, help='start date of test')
    #parser.add_argument('--test_time_end', default=None, type=str, help='end date of test')
    args = parser.parse_args()
    
    #if int(args.test_time_start) > int(args.test_time_end):
    #    print("End test time is earlier than start test time, [set] end test time = start test time")
    #    args.test_time_end = args.test_time_start    

    ### set LDAPS dir path

    LDAPS_root_dir = '/home/osilab12/hdd2/NIMS_LDPS'
    test_result_path = os.path.join('./results', 'eval', args.test_time[0:4])
    if not os.path.isdir(test_result_path):
        os.mkdir(test_result_path)
    
    #test_time_start = datetime.datetime.strptime(args.test_time_start, "%Y%m%d")
    #test_time_end = datetime.datetime.strptime(args.test_time_end, "%Y%m%d")
    #time_delta = test_time_end - test_time_start

    nims_logger = NIMSLogger(loss=False, correct=False, binary_f1=False,
                             macro_f1=False, micro_f1=False,
                             hit=True, miss=True, fa=True, cn=True,
                             stn_codi=stn_codi,
                             test_result_path=test_result_path)

    ### get ground truth data for test time
    
    gt_dir = args.dataset_dir
    gt_path_list = os.listdir(gt_dir)
    gt_path_list = sorted([os.path.join(gt_dir, f) \
                           for f in gt_path_list \
                           if f.endswith('.npy') and \
                           f.split('_')[3][:8] == args.test_time]) # when recorded data is located between test time start and end
    dataset_len = len(gt_path_list)

    ### get LDPS data (LCPCP)

    pbar = tqdm(range(dataset_len))
    for i in pbar:
        target_hour = gt_path_list[i].split('/')[-1].split('_')[3][8:10]
        num_start_hour = int(target_hour) // 6 + 1 # if 2, uses LDAPS data predicted from 00, 06. If 3, use from 00, 06, 12
        
        ### preprocessing reference data

        reference_data = np.load(gt_path_list[i])
        reference_data = np.where(reference_data >= 0.1, np.ones(reference_data.shape), np.zeros(reference_data.shape))
        reference_data = reference_data[stn_codi[:, 0], stn_codi[:, 1]]

        for start_hour in range(num_start_hour):
            start_hour = start_hour * 6 # 00, 06, 12, 18
            prediction_period_hour = int(target_hour) - start_hour # when LDPS 06 + h02 -> target_hour = 8, start_hour = 0, 6, prediction_period_hour = 8, 2

        ### preprocessing LDPS data

            LDPS_data = np
            LDPS_data = np.where(LDPS_data >= 0.1, np.ones(LDPS_data.shape), np.zeros(LDPS_data.shape))
            LDPS_data = LDPS_data[stn_codi[:, 0], stn_codi[:, 1]]

        ### get confusion matrix

            _, _, hit, miss, fa, cn = get_stat(LDPS_data, reference_data)

        ### update logger

            nims_logger.updata(hit=hit, miss=miss, fa=fa, cn=cn, target_time=target_time, prediction_hour=prediction_period_hour, test=True)
            pbar.set_description(nims_logger.latest_stat)

    nims_logger.print_stat(dataset_len, test=True)
