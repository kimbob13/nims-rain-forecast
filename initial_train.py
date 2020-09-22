import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from nims_util import *
from nims_dataset import NIMSDataset, ToTensor
from nims_trainer import NIMSTrainer
#from nims_variable import parse_variables

try:
    from torchsummary import summary
except:
    pass

if __name__ == '__main__':
    # Select start and end date for train
    date = select_date()

    # Parsing command line arguments
    args = parse_args()

    # Set device
    device = set_device(args)

    # Fix the seed
    fix_seed(2020)

    # Create necessary directory
    create_results_dir()

    # Parse NIMS dataset variables
    # variables = parse_variables(args.variables)

    # Train dataset
    nims_train_dataset = NIMSDataset(model=args.model,
                                     model_utc=args.model_utc,
                                     window_size=args.window_size,
                                     root_dir=args.dataset_dir,
                                     date=date,
                                     lite=args.lite,
                                     train=True,
                                     transform=ToTensor())
    
    # Undersampling (the raining points >= 40)
    if date['year'] == 2019 and \
       date['start_month'] == 6 and \
       date['start_day'] == 1 and \
       date['end_month'] == 8 and \
       date['end_day'] == 31:
        subset_indices = np.load('./subset_indcies.npy')
        nims_train_dataset = Subset(nims_train_dataset, subset_indices)

    # Get normalization transform
    normalization = None
    if args.normalization:
        print('=' * 25, 'Normalization Start...', '=' * 25)
        max_values, min_values = get_min_max_values(nims_train_dataset)
        print('=' * 25, 'Normalization End!', '=' * 25)
        print()

        normalization = {'max_values': max_values,
                         'min_values': min_values}
    
    # Get a sample for getting shape of each tensor
    sample, _, _ = nims_train_dataset[0]
    if args.debug:
        print('[main] one images sample shape:', sample.shape)

    # Create a model and criterion
    model, criterion = set_model(sample, device, args)

    if args.debug:
        # XXX: Currently, torchsummary doesn't run on ConvLSTM
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
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    # Set the optimizer
    optimizer, scheduler = set_optimizer(model, args)

    # Set experiment name and use it as process name if possible
    experiment_name = set_experiment_name(args, date)

    # Start training
    nims_trainer = NIMSTrainer(model, criterion, optimizer, scheduler, device,
                               train_loader, None, len(nims_train_dataset), 0,
                               experiment_name, args,
                               normalization=normalization)
    nims_trainer.train()
