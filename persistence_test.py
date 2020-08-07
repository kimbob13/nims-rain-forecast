import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from nims_logger import NIMSLogger
from nims_util import create_results_dir

from tqdm import tqdm
import os
import argparse

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
    tp, fp, tn, fn = conf_mat[1, 1], conf_mat[0, 1], conf_mat[0, 0], conf_mat[1, 0]
    
    correct = tp + tn
    csi = tp / (tp + fn + fp) if tp + fn + fp > 0 else -1.0
    pod = tp / (tp + fn) if tp + fn > 0 else -1.0
    bias = (tp + fp) / (tp + fn) if tp + fn > 0 else -1.0

    # ratio = (tp + fp) / (tn + fn)
    # tn *= ratio
    # fn *= ratio
    # csi = tp / (tp + fn + fp) if tp + fn + fp > 0 else -1.0
    # pod = tp / (tp + fn) if tp + fn > 0 else -1.0
    # bias = (tp + fp) / (tp + fn) if tp + fn > 0 else -1.0

    return correct, binary_f1, csi, pod, bias

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NIMS rainfall data prediction')
    parser.add_argument('--dataset_dir', default='/home/osilab12/ssd/NIMS_LDPS', type=str, help='root directory of dataset')
    parser.add_argument('--test_time', default=None, type=str, help='date of test')
    args = parser.parse_args()

    codi_aws_df = pd.read_csv('./codi_ldps_aws/codi_ldps_aws_512.csv')
    dii_info = np.array(codi_aws_df['dii']) - 1
    stn_codi = np.array([(dii // 512, dii % 512) for dii in dii_info])

    experiment_name = 'nims-persistence_{}'.format(args.test_time[2:])
    try:
        setproctitle.setproctitle(experiment_name)
    except:
        pass
    create_results_dir(experiment_name)

    nims_logger = NIMSLogger(loss=False, correct=True, binary_f1=True,
                             macro_f1=False, micro_f1=False,
                             csi=True, pod=True, bias=True,
                             stn_codi=stn_codi,
                             experiment_name=experiment_name)

    gt_dir = os.path.join(args.dataset_dir, 'OBS')
    gt_path_list = os.listdir(gt_dir)
    gt_path_list = sorted([os.path.join(gt_dir, f) \
                           for f in gt_path_list \
                           if f.endswith('.npy') and \
                           f.split('_')[3][:6] == args.test_time])
    dataset_len = len(gt_path_list)

    pbar = tqdm(range(dataset_len - 1))
    for i in pbar:
        today = np.load(gt_path_list[i])
        tomorrow = np.load(gt_path_list[i + 1])
        target_time = gt_path_list[i + 1].split('/')[-1].split('_')[3][:-2]
        target_time = [int(target_time[0:4]), int(target_time[4:6]), int(target_time[6:8]), int(target_time[8:10])]

        today = np.where(today >= 0.1, np.ones(today.shape), np.zeros(today.shape))
        tomorrow = np.where(tomorrow >= 0.1, np.ones(tomorrow.shape), np.zeros(tomorrow.shape))

        today = today[stn_codi[:, 0], stn_codi[:, 1]]
        tomorrow = tomorrow[stn_codi[:, 0], stn_codi[:, 1]]

        correct, binary_f1, csi, pod, bias = get_stat(today, tomorrow)

        nims_logger.update(correct=correct, binary_f1=binary_f1,
                           csi=csi, pod=pod, bias=bias,
                           target_time=target_time, test=True)
        pbar.set_description(nims_logger.latest_stat)

    nims_logger.print_stat(dataset_len, test=True)