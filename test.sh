#!/bin/bash

export CUDA_VISIBLE_DEVICES=3,4

# Run testing
python test.py \
    --data_dir /home1/chenhaoyang/data/accentDB/data \
    --model_path ./checkpoints/conformer/best_model.pth \
    --batch_size 16 \
    --n_mels 80 \
    --encoder_dim 256 \
    --num_encoder_layers 6 \
    --num_attention_heads 4 \
    --num_workers 0 \
    --augment True
