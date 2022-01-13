#!/bin/bash
for arch in "RNN" "LSTM" "GRU" "MLP" "CNN"
do
  python test.py -a $arch
done