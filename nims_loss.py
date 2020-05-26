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

        _f1_score = f1_score(targets, pred_labels, average='micro')
        #print('[_get_f1_score] f1 score: {}'.format(_f1_score))

        return _f1_score

    def forward(self, preds, targets):
        """
        <Parameters>
        preds [torch.tensor]: NCHW format. N(batch size)
        target [torch.tensor]: NHW format. N(batch size)

        <Return>
        loss [float]
        correct [float]: % of correct prediction
        f1_score [float]: micro f1 score
        """
        correct = self._get_num_correct(preds, targets)
        f1_score = self._get_f1_score(preds, targets)
        #print('[cross_entropy] preds shape:', preds.shape)
        #print('[cross_entropy] targets shape:', targets.shape)
        #print('[cross_entropy] correct: {}, totalnum: {}'
        #      .format(correct, preds.shape[0] * preds.shape[2] * preds.shape[3]))

        loss = 0.0
        for lat in range(preds.shape[2]):
            for lon in range(preds.shape[3]):
                pred = preds[:, :, lat, lon]     # (N, 4)
                target = targets[:, lat, lon]    # (N)

                #print('[cross_entropy] pred: {}, target: {}'.format(pred.shape, target.shape))
                #print('[cross_entropy] pred [:, 0]: {}'.format(pred[:, 0]))
                #print('[cross_entropy] target: {}'.format(target))
                #import sys; sys.exit()

                pixel_loss = self.cross_entropy(pred, target)
                # if torch.isnan(pixel_loss):
                #     print('[cross_entropy] nan loss: lat = {}, lon = {}'.format(lat, lon))

                loss += pixel_loss

        #print('[cross_entropy] loss: {}'.format(loss.item()))

        return loss, correct, f1_score
