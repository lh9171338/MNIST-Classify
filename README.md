[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) MNIST Classify
===

# Introduction
The repository contains five MNIST classification models implemented by PyTorch, which are based on MLP, CNN, RNN, LSTM, and GRU respectively.

# Results

## Metrics
| Model | #FLOPs (K) | #Params (K) | Acc (%) |
| :---: | :---: | :---: | :---: |
| RNN | 333.8 | 6.5 | 94.4 |
| LSTM | 1345.2 | 24.1 | 98.5 |
| RGU | 1012.8 | 18.3 | 98.9 |
| MLP | 475.1 | 238.0 | 98.7 |
| CNN | 577.1 | 43.8 | 99.3 |


## Loss & Accuracy Curves

<p align="center">
    <img width="49%" src="figure/loss.png"/>
    <img width="49%" src="figure/accuracy.png"/>
</p>

# Requirements

 - python3
 - pytorch==1.6.0
 - CUDA==10.1
 - argparse, yacs, tqdm, tensorboardX

# Training & Testing

## Training
```shell
python train.py --model_name <MODEL_NAME> [--gpu <GPU_ID>]
```

## Test
```shell
python test.py --model_name <MODEL_NAME> [--gpu <GPU_ID>]
```
