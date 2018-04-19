#!/usr/bin/env bash

wget 'https://www.csie.ntu.edu.tw/~b05902042/model_ensemble.pt'
python3 predict.py $1 $2
