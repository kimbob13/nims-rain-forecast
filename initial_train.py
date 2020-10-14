import numpy as np
import torch
from tqdm import tqdm
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
    # Set the number of threads in pytorch
    torch.set_num_threads(3)

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
                                     heavy_rain=args.heavy_rain,
                                     train=True,
                                     transform=ToTensor())
    
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

    # Create a model and criterion
    model, criterion = set_model(sample, device, args)

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
