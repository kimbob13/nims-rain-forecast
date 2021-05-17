import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from core.nims_util import *
from core.nims_dataset import NIMSDataset, ToTensor
from core.nims_trainer import NIMSTrainer
#from nims_variable import parse_variables

import os


if __name__ == '__main__':
    print()

    phase_str = '=' * 16 + ' {:^25s} ' + '=' * 16
    # Select start and end date for train
    print(phase_str.format('Train Range'))
    train_date = select_date()

    # Parsing command line arguments
    args = parse_args()

    # Set device
    device = set_device(args)

    # Fix the seed
    fix_seed(2020)
    
    # Train dataset
    nims_train_dataset = NIMSDataset(model=args.model,
                                     reference=args.reference,
                                     model_utc=args.model_utc,
                                     window_size=args.window_size,
                                     root_dir=args.dataset_dir,
                                     date=train_date,
                                     heavy_rain=args.heavy_rain,
                                     train=True,
                                     transform=ToTensor())
    
    """
    # Undersampling
    if args.sampling_ratio < 1.0:
        print('=' * 25, 'Undersampling Start...', '=' * 25)
        rain_points = []
        for i in tqdm(range(len(nims_train_dataset))):
            rain_points.append(torch.sum(nims_train_dataset[i][1]).item())
            
        subset_indices = np.where(np.array(rain_points) >= np.quantile(rain_points, 1 - args.sampling_ratio))[0]
        nims_train_dataset = Subset(nims_train_dataset, subset_indices)
        print('=' * 25, 'Undersampling End!', '=' * 25)
        print()
    """
        
    """ TODO: Need to consider range of variables
    # Get normalization transform
    normalization = None
    if args.normalization:
        print(phase_str.format('Normalization'))
        max_values, min_values = get_min_max_values(nims_train_dataset)

        normalization = {'max_values': max_values,
                         'min_values': min_values}
    """
        
    # Get a sample for getting shape of each tensor
    LDAPS_sample, gt_sample, _ = nims_train_dataset[0]

    # Create a model and criterion
    model, criterion = set_model(LDAPS_sample, device, args)
    
    # Create dataloaders
    train_loader = DataLoader(nims_train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    
    # Set the optimizer
    optimizer, scheduler = set_optimizer(model, args)

    # Set experiment name and use it as process name if possible
    experiment_name = set_experiment_name(args, train_date)

    # Create necessary directory
    create_results_dir(experiment_name)

    # Start training
    nims_trainer = NIMSTrainer(model, criterion, optimizer, scheduler, device,
                               train_loader, None, experiment_name, args,
                               normalization=None)
    nims_trainer.train()