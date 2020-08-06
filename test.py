import torch
from torch.utils.data import DataLoader

from nims_util import *
from nims_dataset import NIMSDataset, ToTensor
from nims_trainer import NIMSTrainer

import os

if __name__ == '__main__':
    args = parse_args()

    # Set device
    device = set_device(args)

    # Fix the seed
    fix_seed(2020)
    
    # Test dataset
    nims_test_dataset = NIMSDataset(model=args.model,
                                    model_utc=args.model_utc,
                                    window_size=args.window_size,
                                    root_dir=args.dataset_dir,
                                    test_time=args.test_time,
                                    train=False,
                                    transform=ToTensor())

    # Get a sample for getting shape of each tensor
    sample, _, _ = nims_test_dataset[0]
    if args.debug:
        print('[main] one images sample shape:', sample.shape)

    # Create a model and criterion
    model, criterion, num_lat, num_lon = set_model(sample, device, args, train=False)

    # Create dataloaders
    test_loader = DataLoader(nims_test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    
    # Set the optimizer
    optimizer = set_optimizer(model, args)

    # Set experiment name and use it as process name if possible
    experiment_name = set_experiment_name(args)

    # Create necessary directory
    create_results_dir(experiment_name)

    # Load trained model weight
    # Persistence model doesn't have trained model
    if not args.model == 'persistence':
        weight_name = '_'.join(experiment_name.split('_')[:-1])
        weight_path = os.path.join('./results', 'trained_model', weight_name + '.pt')
        model.load_state_dict(torch.load(weight_path))

    # Start testing
    nims_trainer = NIMSTrainer(model, criterion, optimizer, device,
                               None, test_loader,
                               0, len(nims_test_dataset),
                               num_lat, num_lon, experiment_name, args)
    nims_trainer.test()