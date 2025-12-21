#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
MODEL="custom_model"

# Run training
python train.py \
    --data_dir /data_extend/fcy/dl/datasets/ \
    --save_dir ./checkpoints/5$MODEL \
    --model $MODEL \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.0001 \
    --n_mels 80 \
    --encoder_dim 256 \
    --num_encoder_layers 6 \
    --num_attention_heads 4 \
    --num_workers 4 \
    --augment False \
    --dropout 0.4 \
    --use_pinyin
    # --win_enable
    # --resume ./checkpoints/$MODEL/best_model.pth
