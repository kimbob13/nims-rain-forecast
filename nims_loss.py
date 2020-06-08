import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

__all__ = ['MSELoss', 'RMSELoss', 'NIMSCrossEntropyLoss']

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y, logger=None):
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
    def __init__(self, device, num_classes=4, use_weights=False):
        super().__init__()
        self.device = device
        self.classes = np.arange(num_classes)
        self.use_weights = use_weights

    def _get_num_correct(self, preds, targets):
        _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)
        pred_labels = pred_labels.squeeze(1)

        correct = torch.sum(pred_labels == targets.data)

        return correct.item()

    def _get_f1_score(self, preds, targets):
        _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)
        pred_labels = pred_labels.squeeze(1).flatten().detach().cpu().numpy()
        _targets = targets.flatten().detach().cpu().numpy()

        # Remove 0 class for micro f1 score evaluation.
        # However, by doing this, macro f1 score becomes nan,
        # so we just keep 0 class for macro f1 score evaluation.
        nonzero_target_idx = _targets.nonzero()
        nonzero_pred_labels = pred_labels[nonzero_target_idx]
        nonzero_targets = _targets[nonzero_target_idx]

        _macro_f1_score = f1_score(_targets, pred_labels,
                                   average='macro', zero_division=0)
        _micro_f1_score = f1_score(nonzero_targets, nonzero_pred_labels,
                                   average='micro', zero_division=0)
        #print('[_get_f1_score] macro f1: {}, micro f1: {}'.format(_macro_f1_score, _micro_f1_score))

        return _macro_f1_score, _micro_f1_score

    def _get_class_weights(self, targets):
        _targets = targets.flatten().detach().cpu().numpy()
        targets_classes = np.unique(_targets)
        weights = compute_class_weight('balanced', targets_classes, _targets) # shape: (C)

        if len(self.classes) != len(targets_classes):
            for label in self.classes:
                if label not in targets_classes:
                    weights = np.insert(weights, label, 0.0)

        return torch.from_numpy(weights).type(torch.FloatTensor).to(self.device)

    def forward(self, preds, targets, logger=None):
        """
        <Parameter>
        preds [torch.tensor]: NS'CHW format (N: batch size, S': target num, C: class num)
        targets [torch.tensor]:  NS'HW format (same as preds)
        logger [NIMSLogger]: Collect stat for this data instance
        """
        # convert preds to S'NCHW, and targets to S'NHW format
        preds = preds.permute(1, 0, 2, 3, 4)
        targets = targets.permute(1, 0, 2, 3)
        assert preds.shape[0] == targets.shape[0]

        target_num = targets.shape[0]
        loss = 0.0
        for target_idx in range(target_num):
            cur_pred = preds[target_idx, ...] # NCHW
            cur_target = targets[target_idx, ...] # NHW

            correct = self._get_num_correct(cur_pred, cur_target)
            macro_f1, micro_f1 = self._get_f1_score(cur_pred, cur_target)

            class_weights = None
            if self.use_weights:
                class_weights = self._get_class_weights(cur_target)

            # print('[cross_entropy] cur_pred shape:', cur_pred.shape)
            # print('[cross_entropy] cur_target shape:', cur_target.shape)
            # print('[cross_entropy] correct: {}, totalnum: {}'
            #      .format(correct, cur_pred.shape[0] * cur_pred.shape[2] * cur_pred.shape[3]))

            cur_loss = 0.0
            for lat in range(cur_pred.shape[2]):
                for lon in range(cur_pred.shape[3]):
                    pred = cur_pred[:, :, lat, lon]     # (N, C)
                    target = cur_target[:, lat, lon]    # (N)

                    #print('[cross_entropy] pred: {}, target: {}'.format(pred.shape, target.shape))
                    #print('[cross_entropy] pred [:, 0]: {}'.format(pred[:, 0]))
                    #print('[cross_entropy] target: {}'.format(target))
                    #import sys; sys.exit()

                    pixel_loss = F.cross_entropy(pred, target, weight=class_weights)
                    # if torch.isnan(pixel_loss):
                    #     print('[cross_entropy] nan loss: lat = {}, lon = {}'.format(lat, lon))

                    cur_loss += pixel_loss

            if logger:
                logger.update(target_idx, loss=cur_loss.item(), correct=correct,
                              macro_f1=macro_f1, micro_f1=micro_f1,
                              pred_tensor=cur_pred, target_tensor=cur_target)

            loss += cur_loss

        #print('[cross_entropy] loss: {}'.format(loss.item()))

        return loss
