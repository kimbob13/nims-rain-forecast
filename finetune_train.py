import os
import torch
from torch.utils.data import DataLoader, Subset
from datetime import datetime, timedelta

from nims_util import *
from nims_dataset import NIMSDataset, ToTensor
from nims_trainer import NIMSTrainer
#from nims_variable import parse_variables

try:
    from torchsummary import summary
except:
    pass

if __name__ == '__main__':
    args = parse_args()
    args.num_epochs = args.finetune_num_epochs
    args.lr = args.finetune_lr_ratio * args.lr
    
    # Set device
    device = set_device(args)

    # Fix the seed
    fix_seed(2020)

    # Create necessary directory
    create_results_dir()

    # Parse NIMS dataset variables
    # variables = parse_variables(args.variables)
    
    # Make finetune list from 20200602 to final test time
    first_date = datetime(year=2020,
                          month=6,
                          day=2)
    if int(args.final_test_time) < 20200602 or args.final_test_time is None:
        args.final_test_time = '20200602'
    final_date = datetime(year=int(args.final_test_time[0:4]),
                          month=int(args.final_test_time[4:6]),
                          day=int(args.final_test_time[6:8]))
    finetune_period = (final_date-first_date).days + 1
    
    for i in range(finetune_period):
        # Train dataset
        test_time = first_date + timedelta(days=i)
        test_time_str = test_time.strftime("%Y%m%d%H")        
        nims_train_dataset = NIMSDataset(model=args.model,
                                         model_utc=args.model_utc,
                                         window_size=args.window_size,
                                         root_dir=args.dataset_dir,
                                         test_time=test_time_str,
                                         train=True,
                                         finetune=True,
                                         transform=ToTensor())
    
        # Get a sample for getting shape of each tensor
        sample, _ = nims_train_dataset[0]
        if args.debug:
            print('[main] one images sample shape:', sample.shape)
        
        # Set experiment name and use it as process name if possible
        experiment_name = set_experiment_name(args)

        if test_time == first_date:
            pretrained_model = 'nims-utc0-unet_nb6_ch64_ws6_ep200_bs1_sr1.0_adam0.001'
        else:
            yesterday_time = test_time - timedelta(days=1)
            yesterday_time_str = yesterday_time.strftime("%Y%m%d%H")
            pretrained_model = experiment_name + '_{}'.format(yesterday_time_str[:8])
        model_path = os.path.join('./results', 'trained_model', '{}.pt'.format(pretrained_model))
        
        # Create a model and criterion
        model, criterion, num_lat, num_lon = set_model(sample, device, args, train=True,
                                                       finetune=True, model_path=model_path)

        if args.debug:
            # XXX: Currently, torchsummary doesn't run on ConvLSTM
            print('[main] num_lat: {}, num_lon: {}'.format(num_lat, num_lon))
            if args.model == 'unet':
                model.to(device)
                try:
                    summary(model, input_size=sample.shape)
                except:
                    print('If you want to see summary of model, install torchsummary')

        # Undersampling
        if args.sampling_ratio < 1.0:
            print('=' * 20, 'Under Sampling', '=' * 20)
            print('Before Under sampling, train len:', len(nims_train_dataset))

            print('Please wait...')
            nims_train_dataset = undersample(nims_train_dataset, args.sampling_ratio)

            print('After Under sampling, train len:', len(nims_train_dataset))
            print('=' * 20, 'Finish Under Sampling', '=' * 20)
            print()

        # Create dataloaders
        train_loader = DataLoader(nims_train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  pin_memory=True)

        # Set the optimizer
        optimizer = set_optimizer(model, args)

        # Start training
        print ("fine-tuning for {}".format(test_time_str[:8]))
        experiment_name += '_{}'.format(test_time_str[:8])
        nims_trainer = NIMSTrainer(model, criterion, optimizer, device,
                                   train_loader, None,
                                   len(nims_train_dataset), 0,
                                   num_lat, num_lon, experiment_name, args)
        nims_trainer.train()
