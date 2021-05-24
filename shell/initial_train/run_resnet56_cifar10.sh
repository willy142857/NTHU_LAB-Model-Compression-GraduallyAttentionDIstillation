#!/usr/bin/env bash
python3 initial_train.py \
    --model resnet56 \
    --dataset cifar10 \
    --lr 0.01 \
    --schedule 1 60 120 160 \
    --lr_drops 10 0.2 0.2 0.2 \
    --batch-size 128 \
    --seed 8152 \
