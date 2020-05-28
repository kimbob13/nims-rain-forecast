import torch

from nims_logger import NIMSLogger

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
        self.one_hour_pixel = num_lat * num_lon

        self.model.to(self.device)

        if model.name == 'unet':
            self.nims_logger = NIMSLogger(loss=True, correct=True,
                                          macro_f1=True, micro_f1=True,
                                          target_num=self.target_num,
                                          batch_size=args.batch_size,
                                          one_hour_pixel=self.one_hour_pixel)
        elif model.name == 'convlstm':
            self.nims_logger = NIMSLogger(loss=True, correct=False,
                                          macro_f1=False, micro_f1=False,
                                          target_num=self.target_num,
                                          batch_size=args.batch_size,
                                          one_hour_pixel=self.one_hour_pixel)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            # Run one epoch

            print('=' * 25,
                  'Epoch {} / {}'.format(epoch, self.num_epochs),
                  '=' * 25)
            self._epoch(self.train_loader, train=True)
            self.nims_logger.print_stat(self.train_len)

    def test(self):
        print('=' * 25, 'Test', '=' * 25)
        self._epoch(self.test_loader, train=False)
        self.nims_logger.print_stat(self.test_len)

    def _epoch(self, data_loader, train):
        pbar = tqdm(data_loader)
        for images, target in pbar:
            if self.model.name == 'unet':
                images = images.to(self.device)
                target = target.type(torch.LongTensor).to(self.device)
            
            elif self.model.name == 'convlstm':
                # Dataloader for ConvLSTM outputs images and target shape as NSCHW format.
                # We should change this to SNCHW format.
                images = images.permute(1, 0, 2, 3, 4).to(self.device)
                target = target.permute(1, 0, 2, 3, 4).to(self.device)

            output = self.model(images)
            loss = self.criterion(output, target, logger=self.nims_logger)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            pbar.set_description(self.nims_logger.latest_stat)