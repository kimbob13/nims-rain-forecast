import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

__all__ = ['MSELoss', 'RMSELoss', 'NIMSCrossEntropyLoss']

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y, logger=None, test=False):
        #print('[mse_loss] yhat shape: {}, y shape: {}'.format(yhat.shape, y.shape))
        loss = self.mse(yhat, y)
        
        if logger:
            target_num = y.shape[0]
            for target_idx in range(target_num):
                one_target_loss = self.mse(yhat[target_idx, ...],
                                           y[target_idx, ...])
                logger.update(target_idx, loss=one_target_loss.item())

        return loss

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class NIMSCrossEntropyLoss(nn.Module):
    def __init__(self, device, num_classes=2, use_weights=False, train=True):
        super().__init__()
        self.device = device
        self.classes = np.arange(num_classes)
        self.use_weights = use_weights
        self.train = train

    def _get_stat(self, preds, targets):
        _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)
        pred_labels = pred_labels.squeeze(1).detach().cpu().numpy().squeeze(axis=0)
        target_labels = targets.data.detach().cpu().numpy().squeeze(axis=0)

        binary_f1 = f1_score(target_labels, pred_labels, zero_division=0)

        """
        [Confusion matrix]
        - tp: Hit
        - fn: Miss
        - fp: False Alarm
        - tn: Correct Negative
        """
        conf_mat = confusion_matrix(target_labels, pred_labels, labels=self.classes)
        hit, miss, fa, cn = conf_mat[1, 1], conf_mat[1, 0], conf_mat[0, 1], conf_mat[0, 0]
        correct = hit + cn

        return correct, binary_f1, hit, miss, fa, cn

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
                stn_codi, logger=None, test=False):
        """
        <Parameter>
        preds [torch.tensor]: NCHW format (N: batch size, C: class num)
        targets [torch.tensor]:  NHW format (same as preds)
        target_time [torch.tensor]: datetime of current target ([year, month, day, hour])
        stn_codi [np.ndarray]: coordinates of station
        logger [NIMSLogger]: Collect stat for this data instance
        """
        assert preds.shape[0] == targets.shape[0]

        class_weights = None
        if self.use_weights:
            class_weights = self._get_class_weights(targets)

        # print('[cross_entropy] preds shape:', preds.shape)
        # print('[cross_entropy] targets shape:', targets.shape)
        # print('[cross_entropy] correct: {}, totalnum: {}'
        #      .format(correct, preds.shape[0] * preds.shape[2] * preds.shape[3]))

        stn_preds = preds[:, :, stn_codi[:, 0], stn_codi[:, 1]]
        stn_targets = targets[:, stn_codi[:, 0], stn_codi[:, 1]]
        
        correct, binary_f1, hit, miss, fa, cn = self._get_stat(stn_preds, stn_targets)
        
        loss = F.cross_entropy(stn_preds, stn_targets, weight=class_weights, reduction='none')
        loss = torch.mean(torch.mean(loss, dim=0))

        if logger:
            logger.update(loss=loss.item(), correct=correct,
                          binary_f1=binary_f1, hit=hit, miss=miss, fa=fa, cn=cn,
                          target_time=target_time, test=test)

        #print('[cross_entropy] loss: {}'.format(loss.item()))

        return loss
