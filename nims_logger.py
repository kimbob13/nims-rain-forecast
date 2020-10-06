import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

from nims_eval_converter import save_nims_metric

__all__ = ['NIMSLogger']

class NIMSLogger:
    def __init__(self, loss, correct, macro_f1, micro_f1,
                 hit, miss, fa, cn, stn_codi, test_date_list=None):
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
        self.test_date_dict = dict()

        if test_date_list != None:
            self.test_result_path = '/'.join(test_date_list[0].split('/')[:-1])

            for test_date_path in test_date_list:
                month = int(test_date_path.split('/')[-1][4:6])
                self.test_date_dict[month] = test_date_path

            self.test_year = int(test_date_path.split('/')[-1][0:4])
        
        self.num_update = 0

        # Initialize one epoch stat dictionary
        self.one_epoch_stat = OneTargetStat(loss, correct, macro_f1, micro_f1, hit, miss, fa, cn)

        # Used for one data instance stat
        self._latest_stat = OneTargetStat(loss, correct, macro_f1, micro_f1, hit, miss, fa, cn)

        # Test stat dataframe
        self.test_df = pd.DataFrame(columns=['date', 'acc', 'hit', 'miss', 'false alarm', 'correct negative'])
        self.daily_df = pd.DataFrame(0, index=['acc', 'hit', 'miss', 'false alarm', 'correct negative'], columns=list(range(49)))

    def update(self, loss=None, correct=None,
               macro_f1=None, micro_f1=None,
               hit=None, miss=None, fa=None, cn=None,
               target_time=None, test=False):

        if loss != None:
            try:
                self.one_epoch_stat.loss += loss
                self._latest_stat.loss = loss
            except:
                print("You don't specify the loss to be logged")
        if correct != None:
            try:
                self.one_epoch_stat.correct += sum(correct)
                self._latest_stat.correct = sum(correct)
            except:
                print("You don't specify the correct to be logged")
        if macro_f1 != None:
            try:
                self.one_epoch_stat.macro_f1 += sum(macro_f1)
                self._latest_stat.macro_f1 = sum(macro_f1)
            except:
                print("You don't specify the macro_f1 to be logged")
        if micro_f1 != None:
            try:
                self.one_epoch_stat.micro_f1 += sum(micro_f1)
                self._latest_stat.micro_f1 = sum(micro_f1)
            except:
                print("You don't specify the micro_f1 to be logged")
        if hit != None:
            try:
                self.one_epoch_stat.hit += sum(hit)
                self._latest_stat.hit = sum(hit)
            except:
                print("You don't specify the hit to be logged")
        if miss != None:
            try:
                self.one_epoch_stat.miss += sum(miss)
                self._latest_stat.miss = sum(miss)
            except:
                print("You don't specify the miss to be logged")
        if fa != None:
            try:
                self.one_epoch_stat.fa += sum(fa)
                self._latest_stat.fa = sum(fa)
            except:
                print("You don't specify the false alarm to be logged")
        if cn != None:
            try:
                self.one_epoch_stat.cn += sum(cn)
                self._latest_stat.cn = sum(cn)
            except:
                print("You don't specify the correct negative to be logged")

        num_batch = target_time.shape[0]
        self.num_update += num_batch

        if test:
            for b in range(num_batch):
                utc_year = int(target_time[b][0])
                utc_month = int(target_time[b][1])
                utc_day = int(target_time[b][2])
                utc_hour = int(target_time[b][3]) # same as model_utc
                from_h = int(target_time[b][4])

                # Save daily_df if we reach last prediction for one day(48 hours) or
                # end of date for current test data (August 31 for both 2019 and 2020)
                save = False
                if from_h == 48:
                    save = True

                # Update entry for target test time and save to file if save == True
                self.daily_df[from_h] = [(correct[b] / self.num_stn) * 100, hit[b], miss[b], fa[b], cn[b]]
                if save:
                    daily_t = self.daily_df.T

                    # Add last row that contains one day statistics
                    acc_mean = daily_t.iloc[:, 0].mean(axis=0)
                    daily_t = daily_t.append(daily_t.sum(axis=0), ignore_index=True)
                    daily_t.iloc[-1, 0] = acc_mean

                    # Save to file
                    daily_t.to_csv(os.path.join(self.test_date_dict[utc_month],
                                                '{:4d}{:02d}{:02d}+{:02d}.csv'
                                                .format(utc_year, utc_month, utc_day, utc_hour)),
                                   index=False)

                    # Update total test dataframe
                    initial_test_time = datetime(year=utc_year,
                                                 month=utc_month,
                                                 day=utc_day,
                                                 hour=utc_hour)
                    self.test_df = self.test_df.append({'date': initial_test_time.strftime('%Y-%m-%d'),
                                                        'acc': daily_t.iloc[-1, 0],
                                                        'hit': daily_t.iloc[-1, 1],
                                                        'miss': daily_t.iloc[-1, 2],
                                                        'false alarm': daily_t.iloc[-1, 3],
                                                        'correct negative': daily_t.iloc[-1, 4]},
                                                       ignore_index=True)

                    # Reset all of daily_df values to 0
                    for col in self.daily_df.columns:
                        self.daily_df[col].values[:] = 0
    
    def print_stat(self, test=False):
        total_stn = self.num_update * self.num_stn

        stat_str = ''
        
        try:
            epoch_loss = self.one_epoch_stat.loss / self.num_update
            stat_str += "[{:12s}] {:.5f}\n".format('loss', epoch_loss)
        except:
            pass

        try:
            stat_str += "[{:12s}] {:.3f}%\n".format('accuracy', (self.one_epoch_stat.correct / total_stn) * 100)
        except:
            pass

        try:
            stat_str += "[{:12s}] {:.5f}\n".format('f1 (macro)', self.one_epoch_stat.macro_f1 / self.num_update)
        except:
            pass

        try:
            stat_str += "[{:12s}] {:.5f}\n".format('f1 (micro)', self.one_epoch_stat.micro_f1 / self.num_update)
        except:
            pass

        try:
            pod = self.one_epoch_stat.hit / (self.one_epoch_stat.hit + self.one_epoch_stat.miss)
            csi = self.one_epoch_stat.hit / (self.one_epoch_stat.hit + self.one_epoch_stat.miss + self.one_epoch_stat.fa)
            bias = (self.one_epoch_stat.hit + self.one_epoch_stat.fa) / (self.one_epoch_stat.hit + self.one_epoch_stat.miss)
            far = self.one_epoch_stat.fa / (self.one_epoch_stat.hit + self.one_epoch_stat.fa)
            f1 = (2 * self.one_epoch_stat.hit) / ((2 * self.one_epoch_stat.hit) + self.one_epoch_stat.fa + self.one_epoch_stat.miss)

            stat_str += "[{:12s}] {:.5f}\n".format('pod', pod)
            stat_str += "[{:12s}] {:.5f}\n".format('csi', csi)
            stat_str += "[{:12s}] {:.5f}\n".format('far', far)
            stat_str += "[{:12s}] {:.5f}\n".format('f1 score', f1)
            stat_str += "[{:12s}] {:.5f}\n".format('bias', bias)
        except:
            pass

        print(stat_str)
        print()

        if test:
            self._save_test_result()

        self._clear_one_target_stat(self.one_epoch_stat)

        return epoch_loss, pod, csi, bias

    def latest_stat(self, target_time):
        num_batch = target_time.shape[0]
        batch_stn = self.num_stn * num_batch
        try:
            accuracy = (self._latest_stat.correct / batch_stn) * 100
            assert accuracy <= 100.0
        except:
            pass

        stat_str = '[{:4d}-{:02d}-{:02d}:{:02d}+{:02d}h'.format(target_time[0][0],
                                                        target_time[0][1],
                                                        target_time[0][2],
                                                        target_time[0][3],
                                                        target_time[0][4])
        if num_batch > 1:
            stat_str += ' ~ {:4d}-{:02d}-{:02d}:{:02d}+{:02d}h] '.format(target_time[-1][0],
                                                                 target_time[-1][1],
                                                                 target_time[-1][2],
                                                                 target_time[-1][3],
                                                                 target_time[-1][4])
        else:
            stat_str += '] '

        try:
            stat_str += "loss = {:.5f}".format(self._latest_stat.loss)
        except:
            pass

        try:
            stat_str += ", accuracy = {:.3f}%".format(accuracy)
        except:
            pass

        try:
            stat_str += ", f1 (macro) = {:.5f}".format(self._latest_stat.macro_f1 / num_batch)
        except:
            pass
        
        try:
            stat_str += ", f1 (micro) = {:.5f}".format(self._latest_stat.micro_f1 / num_batch)
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
            if _stat.hit:
                _stat.hit = 0
        except:
            pass

        try:
            if _stat.miss:
                _stat.miss = 0
        except:
            pass

        try:
            if _stat.fa:
                _stat.fa = 0
        except:
            pass

        try:
            if _stat.cn:
                _stat.cn = 0
        except:
            pass

        self.num_update = 0

    def _save_test_result(self):
        self.test_df = self.test_df.append(self.test_df.iloc[:, 2:].sum(axis=0), ignore_index=True)
        total_hit = self.test_df.iloc[-1, 2]
        total_miss = self.test_df.iloc[-1, 3]
        total_fa = self.test_df.iloc[-1, 4]
        total_cn = self.test_df.iloc[-1, 5]

        self.test_df.iloc[-1, 1] = (total_hit + total_cn) / (total_hit + total_miss + total_fa + total_cn)
        self.test_df.iloc[-1, 0] = 'Total'

        self.test_df.to_csv(os.path.join(self.test_result_path,
                                         'total-{}.csv'.format(self.test_year)), index=False)

class OneTargetStat:
    def __init__(self, loss, correct, macro_f1, micro_f1, hit, miss, fa, cn):
        """
        <Parameter>
        loss, correct, macro_f1, micro_f1 [bool]: whether to record each variable
        """
        if loss:
            self._loss = 0.0
        if correct:
            self._correct = 0
        if macro_f1:
            self._macro_f1 = 0.0
        if micro_f1:
            self._micro_f1 = 0.0
        if hit:
            self._hit = 0
        if miss:
            self._miss = 0
        if fa:
            self._fa = 0
        if cn:
            self._cn = 0
    
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
    def hit(self):
        return self._hit

    @hit.setter
    def hit(self, hit_val):
        self._hit = hit_val

    @property
    def miss(self):
        return self._miss

    @miss.setter
    def miss(self, miss_val):
        self._miss = miss_val

    @property
    def fa(self):
        return self._fa

    @fa.setter
    def fa(self, fa_val):
        self._fa = fa_val

    @property
    def cn(self):
        return self._cn

    @cn.setter
    def cn(self, cn_val):
        self._cn = cn_val
