#!/bin/zsh

python scripts/run_lm_traning.py \
    --d_model 64 \
    --d_ff 256 \
    --num_heads 4 \
    --num_layers 2 \
    --vocab_size 5000 \
    --context_length 512 \
    --device cpu \
    --dtype float32 \
    --batch_size 4 \
    --total_steps 100 \
    --warmup_steps 10 \
    --eval_period 20 \
    --checkpoint_period 50 \
    --max_lr 1e-3 \
    --min_lr 1e-4 \
    --datasets tokenized/tinystories_sample_5M.bin \
    --datasets_sampling_probs 1.0 \
    --validation_datasets tokenized/tinystories_sample.bin