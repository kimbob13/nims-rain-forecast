import torch
import torch.nn as nn
from sklearn.metrics import f1_score

__all__ = ['RMSELoss', 'NIMSCrossEntropyLoss']

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class NIMSCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)

    def _get_num_correct(self, preds, targets):
        _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)
        pred_labels = pred_labels.squeeze(1)

        correct = torch.sum(pred_labels == targets.data)

        return correct

    def _get_f1_score(self, preds, targets):
        _, pred_labels = preds.topk(1, dim=1, largest=True, sorted=True)
        pred_labels = pred_labels.squeeze(1).flatten().detach().cpu().numpy()
        targets = targets.flatten().detach().cpu().numpy()

        _f1_score = f1_score(targets, pred_labels, average='macro')

        return _f1_score

    def forward(self, preds, targets):
        correct = self._get_num_correct(preds, targets)
        f1_score = self._get_f1_score(preds, targets)
        #print('[cross_entropy] pred_labels:', pred_labels.shape)
        #print('[cross_entropy] targets:', targets.shape)
        #print('[cross_entropy] correct: {}, totalnum: {}'
        #      .format(correct, pred_labels.shape[0] * pred_labels.shape[1] * pred_labels.shape[2]))

        loss = 0.0
        for lat in range(preds.shape[2]):
            for lon in range(preds.shape[3]):
                pred = preds[:, :, lat, lon]     # (N, 4)
                target = targets[:, lat, lon]    # (N)

                loss += self.cross_entropy(pred, target)

        return loss, correct, f1_score
