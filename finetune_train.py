import os
import torch
from torch.utils.data import DataLoader, Subset
from datetime import datetime, timedelta

from nims_util import *
from nims_util import get_min_max_values_pool, get_min_max_values_no_mp
from nims_dataset import NIMSDataset, ToTensor
from nims_trainer import NIMSTrainer
#from nims_variable import parse_variables

try:
    from torchsummary import summary
except:
    pass

if __name__ == '__main__':
    date = select_date()

    args = parse_args()
    args.num_epochs = args.finetune_num_epochs
    args.lr = args.finetune_lr_ratio * args.lr
    
    # Set device
    device = set_device(args)

    # Fix the seed
    fix_seed(2020)

    # Parse NIMS dataset variables
    # variables = parse_variables(args.variables)
    
    # Make finetune list from 20200602 to final test time
    finetune_start = datetime(year=date['year'],
                              month=date['start_month'],
                              day=date['start_day'])
    fintune_end = datetime(year=date['year'],
                           month=date['end_month'],
                           day=date['end_day'])
    finetune_period = (fintune_end - finetune_start).days + 1
    
    for i in range(finetune_period):
        # Train dataset
        test_time = finetune_start + timedelta(days=i)
        train_time = test_time - timedelta(days=1)
        assert train_time.year == test_time.year

        curr_date = {'year': train_time.year,
                     'start_month': train_time.month,
                     'start_day': train_time.day,
                     'end_month': train_time.month,
                     'end_day': train_time.day}
        
        nims_train_dataset = NIMSDataset(model=args.model,
                                         model_utc=args.model_utc,
                                         window_size=args.window_size,
                                         root_dir=args.dataset_dir,
                                         date=curr_date,
                                         lite=args.lite,
                                         train=True,
                                         transform=ToTensor())

        # Get normalization transform
        normalization = None
        if args.normalization:
            print('=' * 25, 'Normalization Start...', '=' * 25)
            max_values, min_values = get_min_max_values_no_mp(nims_train_dataset)
            transform = get_min_max_normalization(max_values, min_values)
            print('=' * 25, 'Normalization End!', '=' * 25)
            print()

            normalization = {'transform': transform,
                            'max_values': max_values,
                            'min_values': min_values}
    
        # Get a sample for getting shape of each tensor
        sample, _, _ = nims_train_dataset[0]
        
        # Set experiment name and use it as process name if possible
        experiment_name = set_experiment_name(args, '')

        # XXX: Need to change using curr_date
        if test_time == finetune_start:
            pretrained_model = 'nims-utc0-unet_nb5_ch32_ws6_ep2_bs1_pos0-0_sr1.0_adam0.001_wd5e-05_norm'
        else:
            train_time_str = train_time.strftime("%Y%m%d")
            pretrained_model = experiment_name + '_{}'.format(train_time_str)
        model_path = os.path.join('./results', 'trained_model', '{}.pt'.format(pretrained_model))
        
        # Create a model and criterion
        model, criterion = set_model(sample, device, args, train=True,
                                     finetune=True, model_path=model_path)

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
        optimizer, scheduler = set_optimizer(model, args)

        # Start training
        print ("fine-tuning for {}".format(test_time.strftime("%Y%m%d")))
        experiment_name += '_{}'.format(test_time.strftime("%Y%m%d"))
        nims_trainer = NIMSTrainer(model, criterion, optimizer, scheduler, device,
                                   train_loader, None, len(nims_train_dataset), 0,
                                   experiment_name, args,
                                   normalization=normalization)
        nims_trainer.train()
