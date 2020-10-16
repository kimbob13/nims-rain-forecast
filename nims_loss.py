import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import confusion_matrix

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings(action='ignore')

__all__ = ['MSELoss', 'NIMSCrossEntropyLoss', 'NIMSBinaryFocalLoss']

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class ClassificationStat(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def get_stat(self, preds, targets, mode):
        if mode == 'train':
            _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)
            b = pred_labels.shape[0]

            pred_labels = pred_labels.squeeze(1).detach().reshape(b, -1)
            target_labels = targets.data.detach().reshape(b, -1)
        elif (mode == 'valid') or (mode == 'test'):
            _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)
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

    def remove_missing_station(self, targets, stn_codi):
        _targets = targets.squeeze(0)
        # targets_missing_idx = (_targets == -1).nonzero().cpu().tolist()
        targets_norain_idx = (_targets == 0).nonzero().cpu().tolist()
        targets_rain_idx = (_targets == 1).nonzero().cpu().tolist()

        filtered_stn_codi = targets_norain_idx + targets_rain_idx

        # filtered_stn_codi = stn_codi.tolist()
        # for missing_idx in targets_missing_idx:
        #     filtered_stn_codi.pop(filtered_stn_codi.index(missing_idx))
        
        return np.array(filtered_stn_codi)

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

class NIMSCrossEntropyLoss(ClassificationStat):
    def __init__(self, device, num_classes=2, use_weights=False):
        super().__init__(num_classes=num_classes)
        self.device = device
        self.use_weights = use_weights

    def _get_class_weights(self, targets):
        _targets = targets.flatten().detach().cpu().numpy()
        targets_classes = np.unique(_targets)
        weights = compute_class_weight('balanced', classes=targets_classes, y=_targets) # shape: (C)

        if len(self.classes) != len(targets_classes):
            for label in self.classes:
                if label not in targets_classes:
                    weights = np.insert(weights, label, 0.0)

        return torch.from_numpy(weights).type(torch.FloatTensor).to(self.device)

    def forward(self, preds, targets, target_time,
                stn_codi, mode, logger=None):
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
            class_weights = self._get_class_weights(targets)

        # print('[cross_entropy] preds shape:', preds.shape)
        # print('[cross_entropy] targets shape:', targets.shape)

        if (mode == 'valid') or (mode == 'test'):
            filtered_stn_codi = self.remove_missing_station(targets, stn_codi)
            preds = preds[:, :, filtered_stn_codi[:, 0], filtered_stn_codi[:, 1]]
            targets = targets[:, filtered_stn_codi[:, 0], filtered_stn_codi[:, 1]]

        correct, hit, miss, fa, cn = self.get_stat(preds, targets, mode=mode)
        
        loss = F.cross_entropy(preds, targets, weight=class_weights, reduction='none')
        loss = torch.mean(torch.mean(loss, dim=0))

        if logger:
            logger.update(loss=loss.item(), correct=correct,
                          hit=hit, miss=miss, fa=fa, cn=cn,
                          target_time=target_time, mode=mode)

        # print('[cross_entropy] loss: {}'.format(loss.item()))

        return loss

class NIMSBinaryFocalLoss(ClassificationStat):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=[1.0, 1.0], gamma=2):
        super(NIMSBinaryFocalLoss, self).__init__(num_classes=2)
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        # self.ignore_index = ignore_index
        # self.reduction = reduction

        # assert self.reduction in ['none', 'mean', 'sum']

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert self.alpha.shape[0] == 2, \
                'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, preds, targets, target_time,
                stn_codi, logger=None, test=False):
        stn_preds = preds[:, :, stn_codi[:, 0], stn_codi[:, 1]]
        stn_targets = targets[:, stn_codi[:, 0], stn_codi[:, 1]]
        
        correct, hit, miss, fa, cn = self.get_stat(stn_preds, stn_targets)

        prob = torch.sigmoid(stn_preds)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (stn_targets == 1).float()
        neg_mask = (stn_targets == 0).float()

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * \
                   torch.log(torch.sub(1.0, prob)) * neg_mask

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg

        if logger:
            logger.update(loss=loss.item(), correct=correct,
                          hit=hit, miss=miss, fa=fa, cn=cn,
                          target_time=target_time, test=test)
        
        return loss