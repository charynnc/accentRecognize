#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
MODEL="custom_model"
# Run testing
python test.py \
    --data_dir /data_extend/fcy/dl/datasets/ \
    --model_path ./checkpoints/2$MODEL/best_model.pth \
    --model $MODEL \
    --batch_size 32 \
    --n_mels 80 \
    --encoder_dim 256 \
    --num_encoder_layers 6 \
    --num_attention_heads 4 \
    --num_workers 4 \
    --use_pinyin
