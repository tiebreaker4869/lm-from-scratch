#!/bin/zsh

python scripts/model_inference_completion.py \
    --checkpoint checkpoints/best.bin \
    --tokenizer_dir tokenizer_output \
    --prompt "Once upon a time" \
    --max_tokens 50 \
    --temperature 0.8 \
    --top_p 0.9 \
    --d_model 64 \
    --d_ff 256 \
    --num_heads 4 \
    --num_layers 2 \
    --vocab_size 5000 \
    --device cpu \
    --dtype float32