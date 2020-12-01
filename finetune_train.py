import os
import torch
from torch.utils.data import DataLoader, Subset
from datetime import datetime, timedelta

from core.nims_util import *
from core.nims_util import get_min_max_values_no_mp
from core.nims_dataset import NIMSDataset, ToTensor
from core.nims_trainer import NIMSTrainer
#from nims_variable import parse_variables

try:
    from torchsummary import summary
except:
    pass

if __name__ == '__main__':
    print()
    phase_str = '=' * 16 + ' {:^25s} ' + '=' * 16
    # Select start and end date for train and valid
    print(phase_str.format('Train Range'))
    train_date = select_date()
    print(phase_str.format('Valid Range'))
    valid_date = select_date()

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
    # finetune_start = datetime(year=date['year'],
    #                           month=date['start_month'],
    #                           day=date['start_day'])
    # fintune_end = datetime(year=date['year'],
    #                        month=date['end_month'],
    #                        day=date['end_day'])
    # finetune_period = (fintune_end - finetune_start).days + 1

    # Train dataset for fine-tuning
    nims_train_dataset = NIMSDataset(model=args.model,
                                     reference=args.reference,
                                     model_utc=args.model_utc,
                                     window_size=args.window_size,
                                     root_dir=args.dataset_dir,
                                     date=train_date,
                                     lite=args.lite,
                                     train=True,
                                     transform=ToTensor())
    
    # Valid dataset
    nims_valid_dataset = NIMSDataset(model=args.model,
                                     reference=args.reference,
                                     model_utc=args.model_utc,
                                     window_size=args.window_size,
                                     root_dir=args.dataset_dir,
                                     date=valid_date,
                                     lite=args.lite,
                                     heavy_rain=args.heavy_rain,
                                     train=False,
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

    # Get directories in results
    pretrained_model = select_pretrained_model()
    model_path = os.path.join('./results', pretrained_model, 'trained_weight.pt')
    experiment_name = pretrained_model
    
    # Create a model and criterion
    model, criterion = set_model(sample, device, args, train=True,
                                 finetune=True, model_path=model_path)

    # Create dataloaders
    train_loader = DataLoader(nims_train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)    
    valid_loader = DataLoader(nims_valid_dataset, batch_size=1,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    
    # Set the optimizer
    optimizer, scheduler = set_optimizer(model, args)
    
    # Start training
    print("fine-tuning for {:4d}-{:02d}-{:02d} ~ {:4d}-{:02d}-{:02d}".format(train_date['year'], train_date['start_month'], train_date['start_day'],
                                                                             train_date['year'], train_date['end_month'], train_date['end_day']))
    experiment_name += "_FT_{:4d}-{:02d}-{:02d}-{:4d}-{:02d}-{:02d}".format(train_date['year'], train_date['start_month'], train_date['start_day'],
                                                                            train_date['year'], train_date['end_month'], train_date['end_day'])
    
    # Create necessary directory
    create_results_dir(experiment_name)

    nims_trainer = NIMSTrainer(model, criterion, optimizer, scheduler, device,
                               train_loader, valid_loader, experiment_name, args,
                               normalization=normalization)
    
    nims_trainer.train()

    # for i in range(1, finetune_period + 1):
    #     # Train dataset
    #     test_time = finetune_start + timedelta(days=i)
    #     train_time = test_time - timedelta(days=1)
    #     assert train_time.year == test_time.year

    #     curr_date = {'year': train_time.year,
    #                  'start_month': train_time.month,
    #                  'start_day': train_time.day,
    #                  'end_month': train_time.month,
    #                  'end_day': train_time.day}
    #     print('curr_date:', curr_date)
    #     # Train dataset for fine-tuning
    #     nims_train_dataset = NIMSDataset(model=args.model,
    #                                      reference=args.reference,
    #                                      model_utc=args.model_utc,
    #                                      window_size=args.window_size,
    #                                      root_dir=args.dataset_dir,
    #                                      date=curr_date,
    #                                      lite=args.lite,
    #                                      train=True,
    #                                      transform=ToTensor())
        
    #     # Valid dataset
    #     nims_valid_dataset = NIMSDataset(model=args.model,
    #                                  reference=args.reference,
    #                                  model_utc=args.model_utc,
    #                                  window_size=args.window_size,
    #                                  root_dir=args.dataset_dir,
    #                                  date=valid_date,
    #                                  lite=args.lite,
    #                                  heavy_rain=args.heavy_rain,
    #                                  train=False,
    #                                  transform=ToTensor())

    #     # Get normalization transform
    #     normalization = None
    #     if args.normalization:
    #         print('=' * 25, 'Normalization Start...', '=' * 25)
    #         max_values, min_values = get_min_max_values_no_mp(nims_train_dataset)
    #         transform = get_min_max_normalization(max_values, min_values)
    #         print('=' * 25, 'Normalization End!', '=' * 25)
    #         print()

    #         normalization = {'transform': transform,
    #                         'max_values': max_values,
    #                         'min_values': min_values}
    
    #     # Get a sample for getting shape of each tensor
    #     sample, _, _ = nims_train_dataset[0]

    #     # Get directories in results
    #     pretrained_model = select_pretrained_model()
    #     model_path = os.path.join('./results', pretrained_model, 'trained_weight.pt')
    #     experiment_name = pretrained_model
        
    #     # Create a model and criterion
    #     model, criterion = set_model(sample, device, args, train=True,
    #                                  finetune=True, model_path=model_path)

    #     # Undersampling
    #     if args.sampling_ratio < 1.0:
    #         print('=' * 20, 'Under Sampling', '=' * 20)
    #         print('Before Under sampling, train len:', len(nims_train_dataset))

    #         print('Please wait...')
    #         nims_train_dataset = undersample(nims_train_dataset, args.sampling_ratio)

    #         print('After Under sampling, train len:', len(nims_train_dataset))
    #         print('=' * 20, 'Finish Under Sampling', '=' * 20)
    #         print()

    #     # Create dataloaders
    #     train_loader = DataLoader(nims_train_dataset, batch_size=args.batch_size,
    #                               shuffle=False, num_workers=args.num_workers,
    #                               pin_memory=True)
        
    #     valid_loader = DataLoader(nims_valid_dataset, batch_size=1,
    #                               shuffle=False, num_workers=args.num_workers,
    #                               pin_memory=True)
    #     # Set the optimizer
    #     optimizer, scheduler = set_optimizer(model, args)
        
    #     # Start training
    #     print ("fine-tuning for {}".format(test_time.strftime("%Y%m%d")))
    #     experiment_name += '_FT_{}'.format(test_time.strftime("%Y%m%d"))
        
    #     # Create necessary directory
    #     create_results_dir(experiment_name)

    #     nims_trainer = NIMSTrainer(model, criterion, optimizer, scheduler, device,
    #                        train_loader, valid_loader, experiment_name, args,
    #                        normalization=normalization)
        
    #     nims_trainer.train()
