#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

# Run training
python train.py \
    --data_dir /home1/chenhaoyang/data/accentDB/data \
    --save_dir ./checkpoints/conformer \
    --batch_size 16 \
    --epochs 20 \
    --lr 0.0001 \
    --n_mels 80 \
    --encoder_dim 256 \
    --num_encoder_layers 6 \
    --num_attention_heads 4 \
    --num_workers 0 \
    --augment True
