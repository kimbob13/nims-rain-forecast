import torch
from torch.utils.data import DataLoader

from core.nims_util import *
from core.nims_dataset import NIMSDataset, ToTensor
from core.nims_nc_generator import NIMSNCGenerator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import shutil
import time
from datetime import datetime, timedelta

MONTH_DAY = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

if __name__ == '__main__':
    # Parsing command line arguments
    args = parse_args()

    # Set device
    device = set_device(args)

    # Specify trained model weight
    results_dir = os.path.join('./results')
    experiment_name = 'unet-utc{:02d}'.format(args.model_utc)
    chosen_weight = torch.load(os.path.join(results_dir, experiment_name, 'trained_weight.pt'), map_location=device)

    # Select start and end date for train
    if "-" in  args.date:
        date_str = "".join(args.date.split("-"))
    else:
        date_str = args.date
 
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])
    date = {'year': year, 'start_month': month, 'start_day': day, 'end_month': month, 'end_day': day}   

    # Create output directory
    os.makedirs("./NC", exist_ok=True)

    # Replace model related arguments to the train info
    args.model = chosen_weight['model_name']
    args.n_blocks = chosen_weight['n_blocks']
    args.start_channels = chosen_weight['start_channels']
    args.window_size = chosen_weight['window_size']
    args.model_utc = chosen_weight['model_utc']
    args.pos_loc = chosen_weight['pos_loc']
    args.pos_dim = chosen_weight['pos_dim']
    args.heavy_rain = chosen_weight['heavy_rain']
    args.cross_entropy_weight = chosen_weight['cross_entropy_weight']
    args.bilinear = chosen_weight['bilinear']
    args.custom_name = chosen_weight['custom_name']
    args.batch_size = 1 # Fix batch size as 1 for test

    # Fix the seed
    fix_seed(2020)
    
    # Test dataset
    nims_test_dataset = NIMSDataset(model=args.model,
                                    reference=None,
                                    model_utc=args.model_utc,
                                    window_size=args.window_size,
                                    root_dir=args.dataset_dir,
                                    date=date,
                                    lite=args.lite,
                                    heavy_rain=args.heavy_rain,
                                    train=False,
                                    transform=ToTensor())
    
    # Get normalization transform base on min/max value from training procedure
    normalization = None
    if ('norm_max' in chosen_weight) and ('norm_min' in chosen_weight) and \
       (chosen_weight['norm_max'] != None) and (chosen_weight['norm_min'] != None):
        max_values = chosen_weight['norm_max'].to(torch.device('cpu'))
        min_values = chosen_weight['norm_min'].to(torch.device('cpu'))
        transform = get_min_max_normalization(max_values, min_values)

        print('=' * 25, 'Normalization is enabled!', '=' * 25)
        print('max_values shape: {}, min_values shape: {}\n'.format(max_values.shape, min_values.shape))

        normalization = {'max_values': max_values,
                         'min_values': min_values}
        args.normalization = True

    # Get a sample for getting shape of each tensor
    sample, _ = nims_test_dataset[0]

    # Create a model and criterion
    model, criterion = set_model(sample, device, args, train=False, experiment_name=experiment_name)

    # Create dataloaders
    test_loader = DataLoader(nims_test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    
    # Create directory for each month currently tested date for chosen weight
    test_date_list = [args.date]

    # Load trained model weight
    if args.device == 'cpu':
        chosen_weight['model'] = {k[7:]: v for k, v in chosen_weight['model'].items()}
    model.load_state_dict(chosen_weight['model'])

    # Start testing
    nims_nc_gen = NIMSNCGenerator(model, 
                                  device, 
                                  test_loader, 
                                  "./NC/result", 
                                  experiment_name, 
                                  args, 
                                  normalization=normalization, 
                                  test_date_list=test_date_list)
    
    nims_nc_gen.gen_nc()
    print('FINISHED')
