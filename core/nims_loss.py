import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process, Queue
from pytorch_lightning.metrics.functional import confusion_matrix

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from scipy.special import softmax

import warnings
warnings.filterwarnings(action='ignore')

__all__ = ['MSELoss', 'NIMSCrossEntropyLoss', 'BinaryDiceLoss']

def _save_output_plot(preds, target_time, dataset_dir, experiment_name, queue):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    year   = target_time[0][0]
    month  = target_time[0][1]
    day    = target_time[0][2]
    hour   = target_time[0][3]
    from_h = target_time[0][4]

    preds_plot = softmax(preds[0], axis=0)[0]
    curr_pred_date = '{:4d}-{:02d}-{:02d}:{:02d}+{:02d}'.format(year,
                                                                month,
                                                                day,
                                                                hour,
                                                                from_h)
    heatmap_plot = sns.heatmap(preds_plot, cmap='Blues')
    plt.title(curr_pred_date)
    output_path = os.path.join(dataset_dir, 'NIMS_OUTPUT', experiment_name, str(year),
                               '{:4d}{:02d}{:02d}'.format(year, month, day), curr_pred_date + '.jpg')
    plt.savefig(output_path, optimize=True)

    queue.put(0)

def save_output_plot(preds_list, dataset_dir, experiment_name):
    num_proc = len(preds_list)

    queues = []
    for i in range(num_proc):
        queues.append(Queue())

    processes = []
    for i in range(num_proc):
        processes.append(Process(target=_save_output_plot,
                                 args=(preds_list[i][0], preds_list[i][1],
                                       dataset_dir, experiment_name, queues[i])))

    for i in range(num_proc):
        processes[i].start()

    for i in range(num_proc):
        processes[i].join()

class ClassificationStat(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.reference = args.reference

    def get_stat(self, preds, targets, mode):
        if mode == 'train':
            _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)
            b = pred_labels.shape[0]
            if b == 0:
                return
            
            pred_labels = pred_labels.squeeze(1).detach().reshape(b, -1)
            target_labels = targets.data.detach().reshape(b, -1)
        
        elif (mode == 'valid') or (mode == 'test'):
            # Old
            _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)

            # Current
            # preds = F.softmax(preds, dim=1)
            # true_probs = preds[:, 1, :].unsqueeze(1)
            # pred_labels = torch.where(true_probs > 0.05,
            #                           torch.ones(true_probs.shape).to(0),
            #                           torch.zeros(true_probs.shape).to(0))
            b, _, num_stn = pred_labels.shape
            assert (b, num_stn) == targets.shape

        pred_labels = pred_labels.squeeze(1).detach()
        target_labels = targets.data.detach()

        correct = [0] * b
        hit = [0] * b
        miss = [0] * b
        fa = [0] * b
        cn = [0] * b

        for i in range(b):
            pred, target = pred_labels[i], target_labels[i]

            """
            [Confusion matrix]
            - tp: Hit
            - fn: Miss
            - fp: False Alarm
            - tn: Correct Negative
            """
            if -1 in target:
                print('invalid target:', target.shape)
                print('target:\n', target)
            conf_mat = confusion_matrix(pred, target, num_classes=self.num_classes)
            _hit, _miss, _fa, _cn = conf_mat[1, 1], conf_mat[1, 0], conf_mat[0, 1], conf_mat[0, 0]
            _hit, _miss, _fa, _cn = int(_hit), int(_miss), int(_fa), int(_cn)
            _correct = _hit + _cn

            correct[i] = _correct
            hit[i] = _hit
            miss[i] = _miss
            fa[i] = _fa
            cn[i] = _cn

        return correct, hit, miss, fa, cn

    def remove_missing_station(self, targets):
        _targets = targets.squeeze(0)
        targets_norain_idx = (_targets == 0).nonzero().cpu().tolist() # [(x, y)'s]
        targets_rain_idx = (_targets == 1).nonzero().cpu().tolist()
        targets_idx = targets_norain_idx + targets_rain_idx
        
        return np.array(targets_idx)

class NIMSCrossEntropyLoss(ClassificationStat):
    def __init__(self, args, device, num_classes=2, use_weights=False, experiment_name=None):
        super().__init__(args=args, num_classes=num_classes)
        self.device = device
        self.use_weights = use_weights
        self.dataset_dir = args.dataset_dir
        self.experiment_name = experiment_name

        self.num_preds_batch = 100
        self.preds_list = []

    def _get_class_weights(self, targets):
        _targets = targets.flatten().detach().cpu().numpy()
        targets_classes = np.unique(_targets)
        weights = compute_class_weight('balanced', classes=targets_classes, y=_targets) # shape: (C)

        if len(self.classes) != len(targets_classes):
            for label in self.classes:
                if label not in targets_classes:
                    weights = np.insert(weights, label, 0.0)

        return torch.from_numpy(weights).type(torch.FloatTensor).to(self.device)

    def forward(self, preds, targets, target_time, stn_codi, mode,
                prev_preds=None, logger=None):
        """
        <Parameter>
        preds [torch.tensor]: NCHW format (N: batch size, C: class num)
        targets [torch.tensor]:  NHW format (same as preds)
        target_time [np.ndarray]: datetime of current target ([year, month, day, hour])
        stn_codi [np.ndarray]: coordinates of station
        logger [NIMSLogger]: Collect stat for this data instance
        """
        assert preds.shape[0] == targets.shape[0]

        class_weights = None
        if self.use_weights:
            self._get_class_weights(targets)

        # print('[cross_entropy] preds shape:', preds.shape)
        # print('[cross_entropy] targets shape:', targets.shape)

        # Save preds plot in test mode
        # if mode == 'test':
        #     _preds_cpu = preds.detach().cpu().numpy()
        #     self.preds_list.append((_preds_cpu, target_time))
        #     if (len(self.preds_list) % self.num_preds_batch) == 0:
        #         save_output_plot(self.preds_list, self.dataset_dir, self.experiment_name)
        #         self.save_count = 0
        #         self.preds_list = []

        if self.reference == 'aws':
            stn_codi = self.remove_missing_station(targets)
            stn_targets = targets[:, stn_codi[:, 0], stn_codi[:, 1]]
            
        if prev_preds == None:
            if self.reference == 'aws':
                curr_preds = preds[:, :, stn_codi[:, 0], stn_codi[:, 1]]
                curr_targets = stn_targets
                final_preds = curr_preds
            elif self.reference == 'reanalysis':
                curr_preds = preds
                curr_targets = targets
                final_preds = curr_preds
        else:
            if self.reference == 'aws':
                prev_preds = prev_preds[:, :, stn_codi[:, 0], stn_codi[:, 1]]
                curr_preds = preds[:, :, stn_codi[:, 0], stn_codi[:, 1]]
                
                prev_preds_max_idx = torch.argmax(prev_preds, dim=1, keepdims=True)
                prev_preds_min_idx = torch.argmin(prev_preds, dim=1, keepdims=True)
                prev_preds[:, 0, :] = prev_preds[:, 0, :] + 20. # for calibration when testing
                final_preds = prev_preds_min_idx * prev_preds + prev_preds_max_idx * curr_preds
                
                curr_codi = (prev_preds_max_idx.squeeze(1) == 1).nonzero().cpu().numpy()
                curr_preds = curr_preds[:, :, curr_codi[:, 1]]
                curr_targets = stn_targets[:, curr_codi[:, 1]]
            elif self.reference == 'reanalysis':
                raise NotImplementedError
        
        if prev_preds != None and len(curr_codi) == 0:
            loss = torch.tensor(0., device=self.device)
            correct, hit, miss, fa, cn = self.get_stat(prev_preds, stn_targets, mode=mode)
            if logger:
                logger.update(loss=loss.item(), correct=correct,
                              hit=hit, miss=miss, fa=fa, cn=cn,
                              target_time=target_time, mode=mode)
        else:
            if self.reference == 'aws':
                correct, hit, miss, fa, cn = self.get_stat(final_preds, stn_targets, mode=mode)
            elif self.reference == 'reanalysis':
                correct, hit, miss, fa, cn = self.get_stat(final_preds, targets, mode=mode)

            loss = F.cross_entropy(curr_preds, curr_targets, weight=class_weights, reduction='none')
            loss = torch.mean(torch.mean(loss, dim=1))

            if logger:
                logger.update(loss=loss.item(), correct=correct,
                              hit=hit, miss=miss, fa=fa, cn=cn,
                              target_time=target_time, mode=mode)

        # print('[cross_entropy] loss: {}'.format(loss.item()))

        return loss

class BinaryDiceLoss(ClassificationStat):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, args, num_classes, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__(args=args, num_classes=num_classes)
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, preds, targets, target_time, stn_codi, mode, logger=None):
        assert preds.shape[0] == targets.shape[0], "predict & target batch size don't match"
        
        stn_codi = self.remove_missing_station(targets) # Disable for Old
        stn_targets = targets[:, stn_codi[:, 0], stn_codi[:, 1]]
        stn_preds = preds[:, :, stn_codi[:, 0], stn_codi[:, 1]]
        correct, hit, miss, fa, cn = self.get_stat(stn_preds, stn_targets, mode=mode)
        
        stn_preds = F.softmax(stn_preds, dim=1)
        stn_preds = torch.max(stn_preds, dim=1).values
        predict = stn_preds.contiguous().view(stn_preds.shape[0], -1)
        target = stn_targets.contiguous().view(targets.shape[0], -1)


        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

        if logger:
            logger.update(loss=loss.item(), correct=correct,
                          hit=hit, miss=miss, fa=fa, cn=cn,
                          target_time=target_time, mode=mode)

        return loss

class MSELoss(ClassificationStat):
    def __init__(self, device):
        super().__init__(num_classes=2)
        self.device = device

    def forward(self, preds, targets, target_time,
                stn_codi, logger=None, test=False):
        stn_preds = preds[:, :, stn_codi[:, 0], stn_codi[:, 1]]
        stn_preds = stn_preds.squeeze(0)
        stn_targets = targets[:, stn_codi[:, 0], stn_codi[:, 1]]

        # stn_targets_label = torch.where(stn_targets >= 0.1,
        #                                 torch.ones(stn_targets.shape).to(self.device),
        #                                 torch.zeros(stn_targets.shape).to(self.device)).to(self.device)
        # correct, hit, miss, fa, cn = self.get_stat(stn_preds, stn_targets_label)

        loss = F.mse_loss(stn_preds, stn_targets)
        
        if logger:
            logger.update(loss=loss.item(), target_time=target_time, test=test)

        return loss