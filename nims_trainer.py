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

        self.model_type = args.model
        self.num_epochs = args.num_epochs
        self.target_num = args.target_num
        self.debug = args.debug

        self.num_lat = num_lat
        self.num_lon = num_lon

        self.model.to(self.device)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            if self.model_type == 'stconvs2s':
                epoch_loss = \
                        self._stconvs2s_epoch(self.train_loader, train=True)

                print('Epoch [{}]: loss = {}'
                      .format(epoch, epoch_loss / self.train_len))

            elif self.model_type == 'unet':
                print('=' * 25, 'Epoch [{}]'.format(epoch), '=' * 25)
                epoch_loss, running_correct, running_f1_score = \
                        self._unet_epoch(self.train_loader, train=True)

                running_correct = running_correct.double()
                total_num = self.train_len * self.num_lat * self.num_lon
                print('loss = {:.10f}, accuracy = {:.3f}%, f1 score = {:.10f}'
                      .format(epoch_loss / self.train_len,
                              (running_correct / total_num).item() * 100,
                              running_f1_score / self.train_len))

    def test(self):
        if self.model_type == 'stconvs2s':
            test_loss = self._stconvs2s_epoch(self.test_loader, train=False)

            print('Test loss = {}'.format(test_loss / self.test_len))

        elif self.model_type == 'unet':
            test_loss, test_correct, test_f1_score = \
                    self._unet_epoch(self.test_loader, train=False)

            test_correct = test_correct.double()
            total_num = self.test_len * self.num_lat * self.num_lon
            print('Test loss = {:.10f}, accuracy = {:3f}%, f1_score = {:.10f}'
                  .format(test_loss / self.test_len, 
                          (test_correct / total_num).item() * 100,
                          test_f1_score / self.test_len))

    def _stconvs2s_epoch(self, data_loader, train):
        epoch_loss = 0.0

        for images, target in tqdm(data_loader):
            if self.debug:
                target_nonzero = np.nonzero(target.numpy().squeeze(0)
                                            .squeeze(0).squeeze(0))
                lat = target_nonzero[0][0]
                lon = target_nonzero[1][0]
                print('lat: {}, lon: {}'.format(lat, lon))

            images, target = images.to(self.device), target.to(self.device)
            output = self.model(images)[:, :, :self.target_num, :, :]

            if self.debug:
                print('- output: {}\n- target: {}'
                      .format(output[0][0][0][lat][lon],
                              target[0][0][0][lat][lon]))

            loss = self.criterion(output, target)
            
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss


    def _unet_epoch(self, data_loader, train):
        epoch_loss = 0.0
        running_correct = 0
        running_f1_score = 0.0

        for images, target in tqdm(data_loader):
            images = images.to(self.device)
            target = target.type(torch.LongTensor).to(self.device)

            output = self.model(images)
            loss, correct, f1_score = self.criterion(output, target)
            running_correct += correct
            running_f1_score += f1_score

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss, running_correct, running_f1_score
