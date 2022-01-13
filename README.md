[![](https://github.com/lh9171338/Outline/blob/master/icon.jpg)](https://github.com/lh9171338/Outline) MNIST Classify
===

# Introduction
The repository contains the PyTorch implementation of image classification models for MNIST dataset, based on MLP, CNN, RNN, LSTM, and GRU.

# Results

## Metrics
| Model | #FLOPs (M) | #Params (K) | Acc (%) |
| :--- | :---: | :---: | :---: |
| RNN | 0.3 | 6.5 | 94.4 |
| LSTM | 1.3 | 24.1 | 98.5 |
| GRU | 1.0 | 18.3 | 98.9 |
| MLP | 0.5 | 238.0 | 98.7 |
| CNN | 0.6 | 43.8 | 99.3 |

## Loss & Accuracy Curves

<p align="center">
    <img width="100%" src="figure/loss.png"/>
</p>
<p align="center">
    <img width="100%" src="figure/accuracy.png"/>
</p>

# Requirements

 - python3
 - pytorch==1.6.0
 - CUDA==10.1
 - argparse, yacs, tqdm, tensorboardX

# Training & Testing

## Training
```shell
python train.py --arch <ARCH> [--model_name <MODEL_NAME>] [--gpu <GPU_ID>]
```

## Test
```shell
python test.py --arch <ARCH> [--model_name <MODEL_NAME>] [--gpu <GPU_ID>]
```
