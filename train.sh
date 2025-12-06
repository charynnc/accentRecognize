#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
MODEL="custom_model"

# Run training
python train.py \
    --data_dir /home1/chenhaoyang/data/accentDB/data \
    --save_dir ./checkpoints/$MODEL \
    --model $MODEL \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.0001 \
    --n_mels 80 \
    --encoder_dim 256 \
    --num_encoder_layers 6 \
    --num_attention_heads 4 \
    --num_workers 0 \
    --augment True
