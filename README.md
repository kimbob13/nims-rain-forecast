# NIMS Rain Forecaster

## 1. Introduction

This repository contains Python code for running rain forcasting model for NIMS dataset based on deep learning technology. NIMS dataset was provided by [National Institute of Meteorogical Sciences(국립기상과학원)](http://www.nimr.go.kr/MA/main.jsp), which belongs to the [Korea Meteorogical Administrator(대한민국 기상청)](https://www.kma.go.kr/home/index.jsp). This project aims to nowcast rainfall in Korea area.

## 2. Prerequisite

The following version of Python, PyTorch, and other Python modules are required to run.

- Python 3.6 or above
- PyTorch 1.2 or above (1.3 is recommended): Install it through [Anaconda](https://www.anaconda.com/)
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
- [STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for Weather Forecasting, Nascimento et al. 2019](http://arxiv.org/abs/1912.00134)

Note that each model expects different tensors in terms of dimension. Since STConvS2S uses 3D convolution layer, it expects CDHW tensor (except batch dimension). But U-Net expects usual CHW format tensor. You can see this difference from `main.py`

## 4. Dataset Implementation

Firstly, our NIMS dataset is composed of NetCDF file type, so current implementation read these data in numpy array and convert into PyTorch tensor. This dataset contains **10 years** of rainfall data in **one hour period**. Timestamp of each hour is recorded in its file name, and its base is UTC+0. Each hour's data contains **14 variables** and each variable is recorded in **253 by 149** grid. The resoulution of each grid is 5km by 5km, so it covers whole Korean Peninsula.

Our dataset implementation has following interface, and it inherits PyTorch Dataset class.

```python
NIMSDataset(self, model, window_size, target_num, variables,
            train_year=(2009, 2017), train=True, transform=None,
            root_dir=None, debug=False):
```

- model: Which model to use. (Currently, `unet` or `stconvs2s`)
- window_size: How many sequnces in one instance. (eg. 10 to use 10 hour sequences)
- target_num: How many output sequences to forecast.
- variables: How many variables to use out of 14. It can be single integer or list of variable name
- train_year: Which year to use as training data. It is tuple of start year and end year.
- train: If `true`, it returns training data. Otherwise, returns test data.
- transform: Which transform to apply to the output of dataset instance (eg. ToTensor() to transform numpy array into PyTorch tensor)
- root_dir: Base directory for dataset
- debug: If `true`, it'll print several messages that is intended to help debugging.

## 5. Usage

In the simplest form, it can be run by executing following command.
```
python3 main.py
```

There are several argument you can specify. It can be show as follow.
```
python3 main.py --help
```

The recommendation is that you may have to specify `model` arugment and `variables` arugment as 1. For example,
```
python3 main.py --model=unet --variables=1
```
will run U-Net model and use only one variable, which is `rain` variable that is main concerns of our project.