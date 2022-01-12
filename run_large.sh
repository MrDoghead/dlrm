#!/bin/bash
set -x

export DATA_DIR=$HOME/project/mlcommons/inference/recommendation/dlrm/pytorch/fake_criteo
export MODEL_DIR=$HOME/project/mlcommons/inference/recommendation/model_large
export DLRM_DIR=$HOME/project/mlcommons/inference/recommendation/dlrm
export PYTHONPATH=$PYTHONPATH:$HOME/project/mlcommons/inference/recommendation/dlrm

python dlrm_s_pytorch.py \
	--arch-sparse-feature-size=128 \
	--arch-mlp-bot="13-512-256-128" \
	--arch-mlp-top="1024-1024-512-256-1" \
    --arch-embedding-size="39884406-39043-17289-7420-20263-3-7120-1543-63-38532951-2953546-403346-10-2208-11938-155-4-976-14-39979771-25641295-39664984-585935-12972-108-36" \
	--max-ind-range=40000000 \
    --mini-batch-size=1 \
	--data-generation=dataset \
	--data-set=terabyte \
	--raw-data-file=/home/ubuntu/project/mlcommons/inference/recommendation/criteo/day \
	--round-targets=True \
	--test-mini-batch-size=16384 \
	--test-num-workers=16 \
	--load-model="$MODEL_DIR/dlrm_terabyte.pytorch" \
	--save-onnx \
    --onnx-path="./tb00_40M" \
	--inference-only \
	--test-freq=10240 \
	--memory-map \
	--data-sub-sample-rate=0.875
