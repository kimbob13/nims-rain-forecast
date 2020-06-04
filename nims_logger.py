__all__ = ['NIMSLogger']

class NIMSLogger:
    def __init__(self, loss, correct, macro_f1, micro_f1,
                 target_num, batch_size, one_hour_pixel):
        """
        <Parameter>
        loss, correct, macro_f1, micro_f1 [bool]: whether to record each variable
        target_num [int]
        one_hour_pixel [int]: # of total pixels in each one hour data
        """
        self._target_num = target_num
        self._one_hour_pixel = one_hour_pixel
        self._one_instance_pixel = batch_size * one_hour_pixel

        # Initialize one epoch stat dictionary
        self._one_epoch_stat = dict()
        for target_idx in range(target_num):
            self._one_epoch_stat[target_idx + 1] = OneTargetStat(loss, correct, macro_f1, micro_f1)

        # Used for one data instance stat
        self._latest_stat = dict()
        for target_idx in range(target_num):
            self._latest_stat[target_idx + 1] = OneTargetStat(loss, correct, macro_f1, micro_f1)

    def update(self, target_idx, loss=None, correct=None,
               macro_f1=None, micro_f1=None):
        assert (target_idx >= 0) and (target_idx < self._target_num)

        if loss:
            try:
                self._one_epoch_stat[target_idx + 1].loss += loss
                self._latest_stat[target_idx + 1].loss = loss
            except:
                print("You don't specify the loss to be logged")
        if correct:
            try:
                self._one_epoch_stat[target_idx + 1].correct += correct
                self._latest_stat[target_idx + 1].correct = correct
            except:
                print("You don't specify the coorect to be logged")
        if macro_f1:
            try:
                self._one_epoch_stat[target_idx + 1].macro_f1 += macro_f1
                self._latest_stat[target_idx + 1].macro_f1 = macro_f1
            except:
                print("You don't specify the macro_f1 to be logged")
        if micro_f1:
            try:
                self._one_epoch_stat[target_idx + 1].micro_f1 += micro_f1
                self._latest_stat[target_idx + 1].micro_f1 = micro_f1
            except:
                print("You don't specify the micro_f1 to be logged")

    def print_stat(self, dataset_len):
        total_pixel = dataset_len * self._one_hour_pixel

        for target_idx in range(self._target_num):
            cur_target_stat = self._one_epoch_stat[target_idx + 1]

            cur_stat_str = "[{:2d} hour] ".format(target_idx + 1)
            try:
                cur_stat_str += "loss = {:.3f}".format(cur_target_stat.loss / dataset_len)
            except:
                pass

            try:
                cur_stat_str += ", accuracy = {:.3f}%".format((cur_target_stat.correct / total_pixel) * 100)
            except:
                pass

            try:
                cur_stat_str += ", f1 (macro) = {:.5f}".format(cur_target_stat.macro_f1 / dataset_len)
            except:
                pass

            try:
                cur_stat_str += ", f1 (micro) = {:.5f}".format(cur_target_stat.micro_f1 / dataset_len)
            except:
                pass

            print(cur_stat_str)
            self._clear_one_target_stat(cur_target_stat)

    @property
    def latest_stat(self):
        # TODO: Currently, only show one hour after.
        # Need to extend to multiple hours
        try:
            accuracy = self._latest_stat[1].correct / self._one_instance_pixel
            assert accuracy <= 1.0
        except:
            pass

        stat_str = ""
        try:
            stat_str += "loss = {:.3f}".format(self._latest_stat[1].loss)
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

    def _clear_one_target_stat(self, _stat):
        try:
            _stat.loss = 0.0
        except:
            pass

        try:
            _stat.correct = 0
        except:
            pass
            
        try:
            _stat.macro_f1 = 0.0
        except:
            pass

        try:
            _stat.micro_f1 = 0.0
        except:
            pass

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