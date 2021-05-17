#!/bin/bash

# Please refer to 'parse_args' function in './core/nims_util.py'

python train.py --dataset_dir=/home/osilab12/ssd2 \
                --model=unet \
                --reference=aws \
                --window_size=6 \
                --n_blocks=2 \
                --start_channels=16