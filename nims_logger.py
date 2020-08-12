import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

from nims_eval_converter import save_nims_metric

__all__ = ['NIMSLogger']

class NIMSLogger:
    def __init__(self, loss, correct, binary_f1, macro_f1, micro_f1,
                 csi, pod, bias, stn_codi, experiment_name, num_classes=2):
        """
        <Parameter>
        loss, correct, macro_f1, micro_f1 [bool]: whether to record each variable
        args [argparse]: parsed arguments from main

        <Public Method>
        update: Update specified stat for one instance(batch)
        print_stat: Print one epoch stat
        latest_stat: Return straing of one "instance(batch)" stat
        """
        self.stn_codi = stn_codi
        self.num_stn = len(stn_codi)
        self.num_classes = num_classes
        self.experiment_name = experiment_name

        self.num_update = 0
        self.csi_update = 0
        self.pod_update = 0
        self.bias_update = 0

        # Initialize one epoch stat dictionary
        self.one_epoch_stat = OneTargetStat(loss, correct, binary_f1, macro_f1, micro_f1, csi, pod, bias)

        # Used for one data instance stat
        self._latest_stat = OneTargetStat(loss, correct, binary_f1, macro_f1, micro_f1, csi, pod, bias)

        # Test stat dataframe
        self.test_df = pd.DataFrame(index=['acc', 'csi', 'pod', 'bias', 'correct_update', 'csi_update', 'pod_update', 'bias_update'])
        self.daily_df = pd.DataFrame(index=['acc', 'csi', 'pod', 'bias'], columns=list(range(24)))

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
               target_time=None, test=False):

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

        if test:
            target_year = target_time[0]
            target_month = target_time[1]
            target_day = target_time[2]
            target_hour = target_time[3]

            csi_update = 1
            pod_update = 1
            bias_update = 1
            if csi < 0:
                csi = np.nan
                csi_update = 0
            if pod < 0:
                pod = np.nan
                pod_update = 0
            if bias < 0:
                bias = np.nan
                bias_update = 0

            self.daily_df[target_hour] = [(correct / self.num_stn) * 100, csi, pod, bias]
            if target_hour == 23:
                daily_t = self.daily_df.T
                daily_t = daily_t.append(daily_t.mean(axis=0, skipna=True), ignore_index=True)

                save_dir = os.path.join('./results', 'log', self.experiment_name)
                daily_t.to_csv(os.path.join(save_dir, '{:4d}{:02d}{:02d}.csv'.format(target_year, target_month, target_day)))

            # Update total test dataframe
            if str(target_day) in self.test_df:
                self.test_df[str(target_day)] += [correct, csi, pod, bias, self.num_stn,
                                                  csi_update, pod_update, bias_update]
            else:
                self.test_df[str(target_day)] = [correct, csi, pod, bias, self.num_stn,
                                                 csi_update, pod_update, bias_update]

    def print_stat(self, dataset_len, test=False):
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

        if test:
            self._save_test_result()

        self._clear_one_target_stat(self.one_epoch_stat)

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
        self.test_df.loc["acc"] /= self.test_df.loc["correct_update"]
        self.test_df.loc["csi"] /= self.test_df.loc["csi_update"]
        self.test_df.loc["pod"] /= self.test_df.loc["pod_update"]
        self.test_df.loc["bias"] /= self.test_df.loc["bias_update"]

        self.test_df = self.test_df.T
        self.test_df.to_csv(os.path.join("./results", "log", self.experiment_name, 'total.csv'))

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
