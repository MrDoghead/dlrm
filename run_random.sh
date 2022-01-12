#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$HOME/project/mlcommons/inference/recommendation/dlrm

fake_model_dir="./fake_tb00_40M"
mkdir ${fake_model_dir} 

python dlrm_s_pytorch.py \
	--arch-sparse-feature-size=128 \
    --arch-sparse-feature-num=30 \
    --random-emb-size \
    --max-ind-range=50000000 \
	--arch-mlp-bot="20-512-256-128" \
	--arch-mlp-top="1024-1024-512-256-1" \
    --data-generation="random" \
    --data-size=1000 \
	--mini-batch-size=1 \
    --num-indices-per-lookup=10 \
    --numpy-rand-seed=123 \
    --rand-data-dist="exp" \
	--test-num-workers=16 \
	--memory-map \
	--round-targets=True \
	--data-sub-sample-rate=0.875 \
	--inference-only \
    --onnx-path=${fake_model_dir} \
    --save-onnx \
    --onnx-runtime

