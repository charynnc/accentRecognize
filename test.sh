#!/bin/bash

export CUDA_VISIBLE_DEVICES=3,4
MODEL="conformer"
# Run testing
python test.py \
    --data_dir /home1/chenhaoyang/data/ST-CMDS/ \
    --model_path ./checkpoints/$MODEL/best_model.pth \
    --model $MODEL \
    --batch_size 32 \
    --n_mels 80 \
    --encoder_dim 256 \
    --num_encoder_layers 6 \
    --num_attention_heads 4 \
    --num_workers 0 \
    --augment False
