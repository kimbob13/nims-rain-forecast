import torch
from torch.utils.data import DataLoader

from nims_util import *
from nims_dataset import NIMSDataset, ToTensor
from nims_trainer import NIMSTrainer

import os
import time

if __name__ == '__main__':
    # Parsing command line arguments
    args = parse_args()

    # Set device
    device = set_device(args)

    # Specify trained model weight
    weight_dir = os.path.join('./results', 'trained_model')
    weight_list = sorted([f for f in os.listdir(weight_dir) if f.endswith('.pt')])
    print()
    print('=' * 33, 'Which model do you want to test?', '=' * 33)
    print()
    print('-' * 100)
    print('{:^4s}| {:^19s} | {:^65s}{:>7s}'.format('Idx', 'Last Modified', 'Trained Weight', '|'))
    print('-' * 100)
    for i, weight in enumerate(weight_list):
        path_date = time.strftime('%Y/%m/%d %H:%M:%S', time.gmtime(os.path.getmtime(os.path.join(weight_dir, weight))))
        print('[{:2d}] <{:s}> {}'.format(i + 1, path_date, weight))
    choice = int(input('\n> '))
    chosen_info = torch.load(os.path.join(weight_dir, weight_list[choice - 1]), map_location=device)
    print('Load weight...:', weight_list[choice - 1])
    print()

    # Select start and end date for train
    date = select_date()

    # Replace model related arguments to the train info
    args.n_blocks = chosen_info['n_blocks']
    args.start_channels = chosen_info['start_channels']
    args.pos_dim = chosen_info['pos_dim']
    args.cross_entropy_weight = chosen_info['cross_entropy_weight']
    args.bilinear = chosen_info['bilinear']
    args.window_size = chosen_info['window_size']
    args.model_utc = chosen_info['model_utc']
    args.sampling_ratio = chosen_info['sampling_ratio']
    args.num_epochs = chosen_info['num_epochs']
    args.batch_size = chosen_info['batch_size']
    args.optimizer = chosen_info['optimizer']
    args.lr = chosen_info['lr']
    args.custom_name = chosen_info['custom_name']

    if (args.custom_name != None) and ('transpose_conv' in args.custom_name):
        args.bilinear = False
        args.custom_name = ''

    # Fix the seed
    fix_seed(2020)
    
    # Test dataset
    nims_test_dataset = NIMSDataset(model=args.model,
                                    model_utc=args.model_utc,
                                    window_size=args.window_size,
                                    root_dir=args.dataset_dir,
                                    date=date,
                                    train=False,
                                    transform=ToTensor())

    # Get normalization transform base on min/max value from training procedure
    normalization = None
    if ('norm_max' in chosen_info) and ('norm_min' in chosen_info) and \
       (chosen_info['norm_max'] != None) and (chosen_info['norm_min'] != None):
        max_values = chosen_info['norm_max'].to(torch.device('cpu'))
        min_values = chosen_info['norm_min'].to(torch.device('cpu'))
        transform = get_min_max_normalization(max_values, min_values)

        print('=' * 25, 'Normalization is enabled!', '=' * 25)
        print('max_values shape: {}, min_values shape: {}\n'.format(max_values.shape, min_values.shape))

        normalization = {'transform': transform,
                         'max_values': max_values,
                         'min_values': min_values}
        args.normalization = True

    # Get a sample for getting shape of each tensor
    sample, _, _ = nims_test_dataset[0]
    if args.debug:
        print('[main] one images sample shape:', sample.shape)

    # Create a model and criterion
    model, criterion = set_model(sample, device, args, train=False)

    # Create dataloaders
    test_loader = DataLoader(nims_test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    
    # Set the optimizer
    optimizer, _ = set_optimizer(model, args)

    # Set experiment name and use it as process name if possible
    experiment_name = set_experiment_name(args, date)

    # Create necessary directory
    create_results_dir()

    # Create directory for eval data based on trained model name
    weight_name = weight_list[choice - 1][:-3]
    test_result_path = os.path.join('./results', 'eval', weight_name)
    if not os.path.isdir(test_result_path):
        os.mkdir(test_result_path)

    # Create directory for currently tested date for chosen weight
    current_test_date = '{:4d}{:02d}{:02d}-{:04d}{:02d}{:02d}'.format(date['year'], date['start_month'], date['start_day'],
                                                                      date['year'], date['end_month'], date['end_day'])
    current_test_path = os.path.join(test_result_path, current_test_date)
    if not os.path.isdir(current_test_path):
        os.mkdir(current_test_path)

    # Load trained model weight
    model.load_state_dict(chosen_info['model'])

    # Start testing
    nims_trainer = NIMSTrainer(model, criterion, optimizer, None, device,
                               None, test_loader, 0, len(nims_test_dataset),
                               experiment_name, args,
                               normalization=normalization,
                               test_result_path=current_test_path)
    nims_trainer.test()