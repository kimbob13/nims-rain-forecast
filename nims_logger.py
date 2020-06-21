import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

__all__ = ['NIMSLogger']

class NIMSLogger:
    def __init__(self, loss, correct, macro_f1, micro_f1,
                 target_num, batch_size, one_hour_pixel,
                 experiment_name, args=None):
        """
        <Parameter>
        loss, correct, macro_f1, micro_f1 [bool]: whether to record each variable
        target_num [int]: # of hours to predict
        batch_size [int]: self-explanatory
        one_hour_pixel [int]: # of total pixels in each one hour data
        experiment_name [str]: self-explanatory
        args [argparse]: parsed arguments from main
        """
        self.target_num = target_num
        self.batch_size = batch_size
        self.one_hour_pixel = one_hour_pixel
        self.one_instance_pixel = batch_size * one_hour_pixel
        self.num_update = 0

        # Initialize one epoch stat dictionary
        self.one_epoch_stat = dict()
        for target_idx in range(target_num):
            self.one_epoch_stat[target_idx + 1] = OneTargetStat(loss, correct, macro_f1, micro_f1)

        # Used for one data instance stat
        self._latest_stat = dict()
        for target_idx in range(target_num):
            self._latest_stat[target_idx + 1] = OneTargetStat(loss, correct, macro_f1, micro_f1)

        # Store monthly stat for label-wise accuracy for classification model
        if args:
            self.experiment_name = experiment_name
            self.month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            self.month_label_stat = dict()
            self.cur_test_time = datetime(year=args.end_train_year + 1,
                                          month=args.start_month,
                                          day=1,
                                          hour=args.window_size)

            for target_idx in range(target_num):
                self.month_label_stat[target_idx + 1] = dict()
            
            self.micro_eval = np.zeros((len(self.month_name) + 1, 2, 2))
            self.macro_eval = np.zeros((len(self.month_name) + 1, 4, 4))                                              

    def update(self, target_idx, loss=None, correct=None,
               macro_f1=None, micro_f1=None, test=False,
               pred_tensor=None, target_tensor=None):
        assert (target_idx >= 0) and (target_idx < self.target_num)

        if loss:
            try:
                self.one_epoch_stat[target_idx + 1].loss += loss
                self._latest_stat[target_idx + 1].loss = loss
            except:
                print("You don't specify the loss to be logged")
        if correct:
            try:
                self.one_epoch_stat[target_idx + 1].correct += correct
                self._latest_stat[target_idx + 1].correct = correct
            except:
                print("You don't specify the coorect to be logged")
        if macro_f1:
            try:
                self.one_epoch_stat[target_idx + 1].macro_f1 += macro_f1
                self._latest_stat[target_idx + 1].macro_f1 = macro_f1
            except:
                print("You don't specify the macro_f1 to be logged")
        if micro_f1:
            try:
                self.one_epoch_stat[target_idx + 1].micro_f1 += micro_f1
                self._latest_stat[target_idx + 1].micro_f1 = micro_f1
            except:
                print("You don't specify the micro_f1 to be logged")

        self.num_update += 1

        if test:
            cur_target_time = self.cur_test_time + timedelta(hours=target_idx)
            cur_month = cur_target_time.month

            self._update_label_stat(target_idx, cur_month, pred_tensor, target_tensor)
            self.cur_test_time += timedelta(hours=self.batch_size)

    def print_stat(self, dataset_len, test=False):
        total_pixel = dataset_len * self.one_hour_pixel

        for target_idx in range(self.target_num):
            cur_target_stat = self.one_epoch_stat[target_idx + 1]

            cur_stat_str = "[{:2d} hour] ".format(target_idx + 1)
            try:
                cur_stat_str += "loss = {:.5f}".format(cur_target_stat.loss / self.num_update)
            except:
                pass

            try:
                cur_stat_str += ", accuracy = {:.3f}%".format((cur_target_stat.correct / total_pixel) * 100)
            except:
                pass

            try:
                cur_stat_str += ", f1 (macro) = {:.5f}".format(cur_target_stat.macro_f1 / self.num_update)
            except:
                pass

            try:
                cur_stat_str += ", f1 (micro) = {:.5f}".format(cur_target_stat.micro_f1 / self.num_update)
            except:
                pass

            print(cur_stat_str)
            print()
            self._clear_one_target_stat(cur_target_stat)

        if test:
            stat_df = pd.DataFrame(columns=['class 0', 'class 1', 'class 2', 'class 3'], index=self.month_name)

            log_file = os.path.join('./results', 'log', 'test-{}.log'.format(self.experiment_name))
            csv_file = os.path.join('./results', 'log', 'test-{}.csv'.format(self.experiment_name))
            with open(log_file, 'w') as f:
                sys.stdout = f

                print('=' * 25, 'Monthly label accuracy', '=' * 25)
                for hour_after in self.month_label_stat:
                    print('-' * 10, '{} hour after'.format(hour_after), '-' * 10)

                    for month, label_stat in self.month_label_stat[hour_after].items():
                        print('[{}]'.format(self.month_name[month - 1]))

                        for label, stat in label_stat.items():
                            count = stat['count']
                            total = stat['total']

                            # Update micro eval table
                            if label == 0:
                                # Month specific
                                self.micro_eval[month - 1][1][1] += count
                                self.micro_eval[month - 1][1][0] += (total - count)

                                # Year total
                                self.micro_eval[-1][1][1] += count
                                self.micro_eval[-1][1][0] += (total - count)

                            else:
                                # Month specific
                                self.micro_eval[month - 1][0][0] = count
                                self.micro_eval[month - 1][0][1] = (total - count)

                                # Year total
                                self.micro_eval[-1][0][0] += count
                                self.micro_eval[-1][0][1] += (total - count)

                            if total == 0:
                                accuracy = 'NO TARGET VALUE'
                                print('\t(label {}): {} ({:10,d} / {:10,d})'.format(label, accuracy, count, total))
                                stat_df.loc[self.month_name[month - 1]]['class {}'.format(label)] = np.nan

                            else:
                                accuracy = (count / total) * 100
                                print('\t(label {}): {:7.3f}% ({:10,d} / {:10,d})'.format(label, accuracy, count, total))
                                stat_df.loc[self.month_name[month - 1]]['class {}'.format(label)] = accuracy

            # Save stat df
            stat_df = stat_df.T
            stat_df.to_csv(csv_file, na_rep='nan')

            # Save macro confusion table
            macro_file = os.path.join('./results', 'eval', 'macro-{}.npy'.format(self.experiment_name))
            with open(macro_file, 'wb') as f:
                np.save(f, self.macro_eval)

            # Save micro confusion table
            micro_file = os.path.join('./results', 'eval', 'micro-{}.npy'.format(self.experiment_name))
            with open(micro_file, 'wb') as f:
                np.save(f, self.micro_eval)

    @property
    def latest_stat(self):
        # TODO: Currently, only show one hour after.
        # Need to extend to multiple hours
        try:
            accuracy = self._latest_stat[1].correct / self.one_instance_pixel
            assert accuracy <= 1.0
        except:
            pass

        stat_str = ""
        try:
            stat_str += "loss = {:.5f}".format(self._latest_stat[1].loss)
        except:
            pass

        try:
            stat_str += ", accuracy = {:.3f}%".format(accuracy * 100)
        except:
            pass

        try:
            stat_str += ", f1 (macro) = {:.5f}".format(self._latest_stat[1].macro_f1)
        except:
            pass
        
        try:
            stat_str += ", f1 (micro) = {:.5f}".format(self._latest_stat[1].micro_f1)
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
            self.month_label_stat[target_idx + 1][cur_month] = {0: {'count': 0, 'total': 0},
                                                                1: {'count': 0, 'total': 0},
                                                                2: {'count': 0, 'total': 0},
                                                                3: {'count': 0, 'total': 0}}

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
            if _stat.macro_f1:
                _stat.macro_f1 = 0.0
        except:
            pass

        try:
            if _stat.micro_f1:
                _stat.micro_f1 = 0.0
        except:
            pass

        self.num_update = 0

class OneTargetStat:
    def __init__(self, loss, correct, macro_f1, micro_f1):
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
