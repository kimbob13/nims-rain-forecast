import torch

from torchsummary import summary
from tqdm import tqdm

__all__ = ['NIMSTrainer']

class NIMSTrainer:
    def __init__(self, model, criterion, optimizer, device,
                 train_loader, test_loader, train_len, test_len,
                 num_lat, num_lon, args):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_len = train_len
        self.test_len = test_len

        self.num_epochs = args.num_epochs
        self.target_num = args.target_num
        self.debug = args.debug

        self.num_lat = num_lat
        self.num_lon = num_lon

        self.model.to(self.device)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            # Run one epoch
            print('=' * 25, 'Epoch [{}]'.format(epoch), '=' * 25)
            epoch_loss, epoch_correct, epoch_f1_score = \
                    self._unet_epoch(self.train_loader, train=True)

            epoch_correct = epoch_correct.double()
            total_num = self.train_len * self.num_lat * self.num_lon
            print('loss = {:.10f}\naccuracy = {:.3f}%'
                    .format(epoch_loss / self.train_len,
                            (epoch_correct / total_num).item() * 100))
            print('f1 score = {:.10f}'
                  .format(epoch_f1_score / self.train_len))

    def test(self):
        test_loss, test_correct, test_f1_score = \
                self._unet_epoch(self.test_loader, train=False)

        test_correct = test_correct.double()
        total_num = self.test_len * self.num_lat * self.num_lon
        print('Test loss = {:.10f}, accuracy = {:3f}%, f1_score = {:.10f}'
                .format(test_loss / self.test_len, 
                        (test_correct / total_num).item() * 100,
                        test_f1_score / self.test_len))

    def _unet_epoch(self, data_loader, train):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_f1_score = 0.0

        for images, target in tqdm(data_loader):
            images = images.to(self.device)
            target = target.type(torch.LongTensor).to(self.device)

            output = self.model(images)
            loss, correct, f1_score = self.criterion(output, target)
            epoch_correct += correct
            epoch_f1_score += f1_score

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss, epoch_correct, epoch_f1_score
