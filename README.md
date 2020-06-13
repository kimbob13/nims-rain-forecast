# NIMS Rain Forecaster

## 1. Introduction

This repository contains Python code for running rain forcasting model for NIMS dataset based on deep learning technology. NIMS dataset was provided by [National Institute of Meteorogical Sciences(국립기상과학원)](http://www.nimr.go.kr/MA/main.jsp), which belongs to the [Korea Meteorogical Administrator(대한민국 기상청)](https://www.kma.go.kr/home/index.jsp). This project aims to nowcast rainfall in Korea area.

## 2. Prerequisite

The following version of Python, PyTorch, and other Python modules are required to run. Our recommendation is to **use the latest version** of following modules.

- Python 3.6 or above
- PyTorch 1.3 or above: Install it through [Anaconda](https://www.anaconda.com/)
- numpy
- xarray
- torchsummary
- tqdm
- setproctitle (not necessary)

You can install required modules other than PyTorch by running following command
```
python3 -m pip install -r requirements.txt
```

## 3. Model

Currently, it supports two models which are came from following papers

- [U-Net: Convolutional Networks for Biomedical Image Segmentation, Ronneberger et al. 2015](http://arxiv.org/abs/1505.04597)
- [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting, Shi et al. 2015](https://arxiv.org/abs/1506.04214)

All the model implementations are in `model` directory. 

### 3.1 U-Net

Firstly, **U-Net** model is used for **classification** task. Therefore, it expects following formats for input tensor: `NS'HW`. Here are the meaning of each dimension.

- `N`: batch size
- `S'`: # of targets for prediction. Specified by `target_num` argument. But in U-Net case, `S'` is used as it serves as **# of channels** in the model.
- `H`: # of height (=253)
- `W`: # of width (=149)

It uses **NIMSCrossEntropyLoss** for loss function. It is just a cross entropy loss, but it computes **pixel-wise** loss and calculate **f1 score** and **# of correct pixels** as well. You can find it from **nims_loss.py**.

Currently, for the classification task, we discretize the rain value in **4 classess**, and here is the associated range.

- **class 0**: `0 mm/hr` to `0.1 mm/hr`
- **class 1**: `0.1 mm/hr` to `1.0 mm/hr`
- **class 2**: `1.0 mm/hr` to `2.5 mm/hr`
- **class 3**: Above `2.5mm/hr`

You can see this discretization in `_to_pixel_wise_label` method in `NIMSDataset` class in `nims_dataset.py`.

### 3.2 ConvLSTM 

Second model that is supported is **ConvLSTM**. It is used for **regression** task, so it expects following formats for input tensor: `NSCHW`. Here are the meaning of each dimension.

- `N`: batch size
- `S`: # of targets for prediction. Specified by `target_num` argument. 
- `C`: # of channels
- `H`: # of height (=253)
- `W`: # of width (=149)

It currently uses `MSELoss` for loss function, and you can also find it from `nims_loss.py` as well.

## 4. Dataset Implementation

Firstly, our NIMS dataset is composed of NetCDF file type, so current implementation read these data in numpy array and convert into PyTorch tensor. This dataset contains **10 years** of rainfall data in **one hour period**. Timestamp of each hour is recorded in its file name, and its base is UTC+0. Each hour's data contains **14 variables** and each variable is recorded in **253 by 149** grid. The resoulution of each grid is 5km by 5km, so it covers whole Korean Peninsula.

Our dataset implementation has following interface, and it inherits PyTorch Dataset class. Here are the interface of our `NIMSDataset` class constructor.

```python
NIMSDataset(model, window_size, target_num, variables,
            train_year=(2009, 2017), train=True, transform=None,
            root_dir=None, debug=False):
```

- model: Which model to use. (Currently, `unet` or `convlstm`)
- window_size: How many sequnces in one instance. (eg. 10 to use 10 hour sequences)
- target_num: How many output sequences to forecast.
- variables: How many variables to use out of 14. It can be single integer or list of variable name
- train_year: Which year to use as training data. It is tuple of start year and end year.
- train: If `true`, it returns training data. Otherwise, returns test data.
- transform: Which transform to apply to the output of dataset instance (eg. ToTensor() to transform numpy array into PyTorch tensor)
- root_dir: Base directory for dataset
- debug: If `true`, it'll print several messages that is intended to help debugging.

## 5. Training

In the simplest form, it can be run by executing following command.
```
python3 main.py
```

There are several argument you can specify. It can be shown as follow.
```
python3 main.py --help
```

The recommendation is that you specify `model` argument and `variables` argument as 1. For example,
```
python3 main.py --model=unet --variables=1
```
will run U-Net model and use only one variable, which is `rain` variable that is main concerns of our project.

While training, the statistics for one epoch will be printed. It is managed by `NIMSLogger` class in `nims_logger.py`. Currently, it only supports printing, but we will extend it to be able to save these logs to the log file.

After finishing the trainig, the trained model will be saved in `trained_model` directory. Its name is specified by the `experiment_name` in the `main.py`.

## 6. Testing

You can run test only mode as well. If you want to do this, you have to specify the same arguments used in training and add `test_only` arguments. For example, if you've trained the model with following command,
```
python3 main.py --model=unet --variables=1 --n_blocks=5 --start_channels=64 --window_size=5 --target_num=1 --num_epochs=50
```

then just add `test_only` as follow
```
python3 main.py --model=unet --variables=1 --n_blocks=5 --start_channels=64 --window_size=5 --target_num=1 --num_epochs=50 --test_only
```

It'll automatically find the trained model weights, and run a testing code.