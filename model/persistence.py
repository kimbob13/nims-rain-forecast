import torch
import torch.nn as nn

__all__ = ['Persistence']

class Persistence(nn.Module):
    def __init__(self, num_classes, device):
        super(Persistence, self).__init__()

        self.num_classes = num_classes
        self.device = device

        self._dummy = nn.Linear(1, 1)

    @property
    def name(self):
        return 'persistence'

    def forward(self, x):
        b_size, seq, height, width = x.shape
        #print('x shape:', x.shape)
        assert seq == 1

        zero_out = torch.zeros(x.shape).type(torch.LongTensor).to(self.device)
        one_out = torch.ones(x.shape).type(torch.LongTensor).to(self.device)
        two_out = (torch.ones(x.shape).type(torch.LongTensor) * 2).to(self.device)
        three_out = (torch.ones(x.shape).type(torch.LongTensor) * 3).to(self.device)

        # zero_bound = torch.tensor(0.1).to(self.device)
        # one_bound = torch.tensor(1.0).to(self.device)
        # two_bound = torch.tensor(2.5).to(self.device)

        _out = torch.where(x < 0.1, zero_out,
                           torch.where(x < 1.0, one_out,
                                       torch.where(x < 2.5, two_out, three_out))).to(self.device)

        # concat_tensor = torch.zeros(_out.shape).type(torch.LongTensor)
        # out = torch.cat([concat_tensor, concat_tensor, concat_tensor, concat_tensor], dim=1)

        # for label in range(self.num_classes):
        #     b, _, lat, lon = torch.where(_out == label)
        #     out[b, label, lat, lon] = 1

        out = torch.zeros((b_size, self.num_classes, height, width)).to(self.device)
        out = out.scatter_(1, _out, 1)

        # Add S dimension to make out be NSCHW
        out = out.unsqueeze(1).type(torch.FloatTensor).to(self.device)

        return out