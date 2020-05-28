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
            epoch_loss = self._conv_lstm_epoch(self.train_loader, train=True)

            print('loss = {:.10f}'.format(epoch_loss / self.train_len))

    def test(self):
        test_loss = self._conv_lstm_epoch(self.test_loader, train=False)

        test_correct = test_correct.double()
        total_num = self.test_len * self.num_lat * self.num_lon
        print('Test loss = {:.10f}'.format(test_loss / self.test_len))

    def _conv_lstm_epoch(self, data_loader, train):
        epoch_loss = 0.0

        pbar = tqdm(data_loader)
        for images, target in pbar:
            images = images.unsqueeze(2).numpy().transpose([1, 0, 2, 3, 4])
            images = (torch.from_numpy(images)).to(self.device)

            target = target.unsqueeze(2).numpy().transpose([1, 0, 2, 3, 4])
            target = (torch.from_numpy(target)).to(self.device)
            #print('images: {}, target: {}'.format(images.shape, target.shape))

            output = self.model(images)
            #print('output:', output.shape)
            loss = self.criterion(output, target)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_description("loss = %.5f" %(loss))

        return epoch_loss
