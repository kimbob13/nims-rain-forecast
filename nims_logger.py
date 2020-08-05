import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

from nims_eval_converter import save_nims_metric

__all__ = ['NIMSLogger']

class NIMSLogger:
    def __init__(self, loss, correct, binary_f1, macro_f1, micro_f1,
                 csi, pod, bias, batch_size, one_hour_pixel, num_stn,
                 experiment_name, num_classes=2, args=None):
        """
        <Parameter>
        loss, correct, macro_f1, micro_f1 [bool]: whether to record each variable
        batch_size [int]: self-explanatory
        one_hour_pixel [int]: # of total pixels in each one hour data
        experiment_name [str]: self-explanatory
        args [argparse]: parsed arguments from main

        <Public Method>
        update: Update specified stat for one instance(batch)
        print_stat: Print one epoch stat
        latest_stat: Return straing of one "instance(batch)" stat
        """
        self.batch_size = batch_size
        self.one_hour_pixel = one_hour_pixel
        self.one_instance_pixel = batch_size * one_hour_pixel
        self.num_stn = num_stn
        self.num_classes = num_classes

        self.num_update = 0
        self.csi_update = 0
        self.pod_update = 0
        self.bias_update = 0

        # Initialize one epoch stat dictionary
        self.one_epoch_stat = OneTargetStat(loss, correct, binary_f1, macro_f1, micro_f1, csi, pod, bias)

        # Used for one data instance stat
        self._latest_stat = OneTargetStat(loss, correct, binary_f1, macro_f1, micro_f1, csi, pod, bias)

        # # Store monthly stat for label-wise accuracy for classification model
        # if args:
        #     self.experiment_name = experiment_name
        #     self.baseline_name = args.baseline_name
        #     self.month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        #     self.month_label_stat = dict()
        #     self.cur_test_time = datetime(year=args.end_train_year + 1,
        #                                   month=args.start_month,
        #                                   day=1,
        #                                   hour=args.window_size)

        #     for target_idx in range(target_num):
        #         self.month_label_stat[target_idx + 1] = dict()
            
        #     self.micro_eval = np.zeros((len(self.month_name) + 1, 2, 2))
        #     self.macro_eval = np.zeros((len(self.month_name) + 1, 4, 4))                                              

    def update(self, loss=None, correct=None,
               binary_f1=None, macro_f1=None, micro_f1=None,
               csi=None, pod=None, bias=None,
               test=False, pred_tensor=None, target_tensor=None):

        if loss != None:
            try:
                self.one_epoch_stat.loss += loss
                self._latest_stat.loss = loss
            except:
                print("You don't specify the loss to be logged")
        if correct != None:
            try:
                self.one_epoch_stat.correct += correct
                self._latest_stat.correct = correct
            except:
                print("You don't specify the coorect to be logged")
        if binary_f1 != None:
            try:
                self.one_epoch_stat.binary_f1 += binary_f1
                self._latest_stat.binary_f1 = binary_f1
            except:
                print("You don't specify the binary_f1 to be logged")
        if macro_f1 != None:
            try:
                self.one_epoch_stat.macro_f1 += macro_f1
                self._latest_stat.macro_f1 = macro_f1
            except:
                print("You don't specify the macro_f1 to be logged")
        if micro_f1 != None:
            try:
                self.one_epoch_stat.micro_f1 += micro_f1
                self._latest_stat.micro_f1 = micro_f1
            except:
                print("You don't specify the micro_f1 to be logged")
        if csi != None:
            try:
                if csi >= 0.0:
                    self.one_epoch_stat.csi += csi
                    self.csi_update += 1
                
                self._latest_stat.csi = csi
            except:
                print("You don't specify the csi to be logged")
        if pod != None:
            try:
                if pod >= 0.0:
                    self.one_epoch_stat.pod += pod
                    self.pod_update += 1
                
                self._latest_stat.pod = pod
            except:
                print("You don't specify the pod to be logged")
        if bias != None:
            try:
                if bias >= 0.0:
                    self.one_epoch_stat.bias += bias
                    self.bias_update += 1
                
                self._latest_stat.bias = bias
            except:
                print("You don't specify the bias to be logged")

        self.num_update += 1

        # if test:
        #     cur_target_time = self.cur_test_time + timedelta(hours=target_idx)
        #     cur_month = cur_target_time.month

        #     self._update_label_stat(target_idx, cur_month, pred_tensor, target_tensor)
        #     self.cur_test_time += timedelta(hours=self.batch_size)

    def print_stat(self, dataset_len, test=False):
        #total_pixel = dataset_len * self.one_hour_pixel
        total_stn = dataset_len * self.num_stn

        stat_str = ''
        
        try:
            stat_str += "loss = {:.5f}".format(self.one_epoch_stat.loss / self.num_update)
        except:
            pass

        try:
            stat_str += ", accuracy = {:.3f}%".format((self.one_epoch_stat.correct / total_stn) * 100)
        except:
            pass

        try:
            stat_str += ", f1 (binary) = {:.5f}".format(self.one_epoch_stat.binary_f1 / self.num_update)
        except:
            pass

        try:
            stat_str += ", f1 (macro) = {:.5f}".format(self.one_epoch_stat.macro_f1 / self.num_update)
        except:
            pass

        try:
            stat_str += ", f1 (micro) = {:.5f}".format(self.one_epoch_stat.micro_f1 / self.num_update)
        except:
            pass

        try:
            stat_str += ", csi = {:.5f}".format(self.one_epoch_stat.csi / self.csi_update)
        except:
            pass

        try:
            stat_str += ", pod = {:.5f}".format(self.one_epoch_stat.pod / self.pod_update)
        except:
            pass

        try:
            stat_str += ", bias = {:.5f}".format(self.one_epoch_stat.bias / self.bias_update)
        except:
            pass

        print(stat_str)
        print()
        self._clear_one_target_stat(self.one_epoch_stat)

        # if test:
        #     self._save_test_result()

    @property
    def latest_stat(self):
        try:
            accuracy = (self._latest_stat.correct / self.num_stn) * 100
            assert accuracy <= 100.0
        except:
            pass

        stat_str = ""
        try:
            stat_str += "loss = {:.5f}".format(self._latest_stat.loss)
        except:
            pass

        try:
            stat_str += ", accuracy = {:.3f}%".format(accuracy)
        except:
            pass

        try:
            stat_str += ", f1 (binary) = {:.5f}".format(self._latest_stat.binary_f1)
        except:
            pass

        try:
            stat_str += ", f1 (macro) = {:.5f}".format(self._latest_stat.macro_f1)
        except:
            pass
        
        try:
            stat_str += ", f1 (micro) = {:.5f}".format(self._latest_stat.micro_f1)
        except:
            pass

        try:
            stat_str += ", csi = {:.5f}".format(self._latest_stat.csi)
        except:
            pass

        try:
            stat_str += ", pod = {:.5f}".format(self._latest_stat.pod)
        except:
            pass

        try:
            stat_str += ", bias = {:.5f}".format(self._latest_stat.bias)
        except:
            pass

        return stat_str

    def _update_label_stat(self, target_idx, cur_month, pred_tensor, target_tensor):
        num_class = pred_tensor.shape[1]
        _, pred_label  = pred_tensor.topk(1, dim=1, largest=True, sorted=True)
        pred_label = pred_label.squeeze(1).flatten().detach().cpu().numpy()
        target = target_tensor.flatten().detach().cpu().numpy()

        # Create new entry if cur_month is not encountered
        if cur_month not in self.month_label_stat[target_idx + 1]:
            self.month_label_stat[target_idx + 1][cur_month] = dict()
            for label in range(self.num_classes):
                self.month_label_stat[target_idx + 1][cur_month][label] = {'count': 0, 'total': 0}

            # self.month_label_stat[target_idx + 1][cur_month] = {0: {'count': 0, 'total': 0},
            #                                                     1: {'count': 0, 'total': 0},
            #                                                     2: {'count': 0, 'total': 0},
            #                                                     3: {'count': 0, 'total': 0}}

        # 전체 0은 target에 있는 0의 개수여야 하고, 맞춘 건 그중에서 target이 0인 위치를 pred도 0으로 맞춘 개수
        # 1, 2, 3도 마찬가지
        for i in range(num_class):
            cur_label_target_idx = np.where(target == i)[0]
            cur_label_pred_idx = np.where(pred_label[cur_label_target_idx] == i)[0]

            self.month_label_stat[target_idx + 1][cur_month][i]['count'] += len(cur_label_pred_idx)
            self.month_label_stat[target_idx + 1][cur_month][i]['total'] += len(cur_label_target_idx)

            # Update macro eval table
            for j in range(num_class):
                pred_idx = np.where(pred_label[cur_label_target_idx] == j)[0]
                self.macro_eval[cur_month - 1][i][j] += len(pred_idx)   # month specific
                self.macro_eval[-1][i][j] += len(pred_idx)              # year total

    def _clear_one_target_stat(self, _stat):
        try:
            if _stat.loss:
                _stat.loss = 0.0
        except:
            pass

        try:
            if _stat.correct:
                _stat.correct = 0
        except:
            pass

        try:
            if _stat.binary_f1:
                _stat.binary_f1 = 0.0
        except:
            pass
            
        try:
            if _stat.macro_f1:
                _stat.macro_f1 = 0.0
        except:
            pass

        try:
            if _stat.micro_f1:
                _stat.micro_f1 = 0.0
        except:
            pass

        try:
            if _stat.csi:
                _stat.csi = 0.0
        except:
            pass

        try:
            if _stat.pod:
                _stat.pod = 0.0
        except:
            pass

        try:
            if _stat.bias:
                _stat.bias = 0.0
        except:
            pass

        self.num_update = 0
        self.csi_update = 0
        self.pod_update = 0
        self.bias_update = 0

    def _save_test_result(self):
        columns = ['label {}'.format(label) for label in range(self.num_classes)]
        stat_df = pd.DataFrame(columns=columns, index=self.month_name)

        log_file = os.path.join('./results', 'log', 'test-{}.log'.format(self.experiment_name))
        csv_file = os.path.join('./results', 'log', 'test-{}.csv'.format(self.experiment_name))
        with open(log_file, 'w') as f:
            sys.stdout = f

            print('=' * 25, 'Monthly label accuracy', '=' * 25)
            for hour_after in self.month_label_stat:
                print('-' * 10, '{} hour after'.format(hour_after), '-' * 10)

                year_count = [0] * self.num_classes
                year_total = [0] * self.num_classes
                for month, label_stat in self.month_label_stat[hour_after].items():
                    print('[{}]'.format(self.month_name[month - 1]))

                    for label, stat in label_stat.items():
                        count = stat['count']
                        total = stat['total']

                        if total == 0:
                            accuracy = 'NO TARGET VALUE'
                            print('\t(label {}): {} ({:11,d} / {:11,d})'.format(label, accuracy, count, total))
                            stat_df.loc[self.month_name[month - 1]]['label {}'.format(label)] = np.nan

                        else:
                            accuracy = (count / total) * 100
                            print('\t(label {}): {:7.3f}% ({:11,d} / {:11,d})'.format(label, accuracy, count, total))
                            stat_df.loc[self.month_name[month - 1]]['label {}'.format(label)] = accuracy

                        year_count[label] += count
                        year_total[label] += total

                print('[Year Total]')
                for label, (y_count, y_total) in enumerate(zip(year_count, year_total)):
                    accuracy = (y_count / y_total) * 100
                    print('\t(label {}): {:7.3f}% ({:11,d} / {:11,d})'.format(label, accuracy, y_count, y_total))

        # Save stat df
        stat_df = stat_df.T
        stat_df.to_csv(csv_file, na_rep='nan')

        # Update micro confusion table
        self.micro_eval[:, 1, 1] = self.macro_eval[:, 0, 0]                        # Correct negative
        self.micro_eval[:, 1, 0] = np.sum(self.macro_eval[:, 0, 1:], axis=1)       # False Alarm
        self.micro_eval[:, 0, 1] = np.sum(self.macro_eval[:, 1:, 0], axis=1)       # Misses
        self.micro_eval[:, 0, 0] = np.sum(self.macro_eval[:, 1:, 1:], axis=(1, 2)) # Hits

        # Save macro confusion table
        macro_file = os.path.join('./results', 'eval', 'macro-{}.npy'.format(self.experiment_name))
        with open(macro_file, 'wb') as f:
            np.save(f, self.macro_eval)

        # Save micro confusion table
        micro_file = os.path.join('./results', 'eval', 'micro-{}.npy'.format(self.experiment_name))
        with open(micro_file, 'wb') as f:
            np.save(f, self.micro_eval)

        # Save metric for NIMS to file
        save_nims_metric(self.experiment_name, self.baseline_name)

class OneTargetStat:
    def __init__(self, loss, correct, binary_f1, macro_f1, micro_f1, csi, pod, bias):
        """
        <Parameter>
        loss, correct, macro_f1, micro_f1 [bool]: whether to record each variable
        """
        if loss:
            self._loss = 0.0
        if correct:
            self._correct = 0
        if binary_f1:
            self._binary_f1 = 0.0
        if macro_f1:
            self._macro_f1 = 0.0
        if micro_f1:
            self._micro_f1 = 0.0
        if csi:
            self._csi = 0.0
        if pod:
            self._pod = 0.0
        if bias:
            self._bias = 0.0
    
    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss_val):
        self._loss = loss_val
    
    @property
    def correct(self):
        return self._correct

    @correct.setter
    def correct(self, correct_val):
        self._correct = correct_val

    @property
    def binary_f1(self):
        return self._binary_f1

    @binary_f1.setter
    def binary_f1(self, binary_f1_val):
        self._binary_f1 = binary_f1_val

    @property
    def macro_f1(self):
        return self._macro_f1

    @macro_f1.setter
    def macro_f1(self, macro_f1_val):
        self._macro_f1 = macro_f1_val

    @property
    def micro_f1(self):
        return self._micro_f1

    @micro_f1.setter
    def micro_f1(self, micro_f1_val):
        self._micro_f1 = micro_f1_val

    @property
    def csi(self):
        return self._csi

    @csi.setter
    def csi(self, csi_val):
        self._csi = csi_val

    @property
    def pod(self):
        return self._pod

    @pod.setter
    def pod(self, pod_val):
        self._pod = pod_val

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias_val):
        self._bias = bias_val