#!/bin/bash

#export DATA_DIR=$HOME/project/mlcommons/inference/recommendation/dlrm/pytorch/fake_criteo
export DATA_DIR=$HOME/project/mlcommons/inference/recommendation/dlrm/fake_terabyte0875
export MODEL_DIR=$HOME/project/mlcommons/inference/recommendation/model
export DLRM_DIR=$HOME/project/mlcommons/inference/recommendation/dlrm
export PYTHONPATH=$PYTHONPATH:$HOME/project/mlcommons/inference/recommendation/dlrm

python dlrm_s_pytorch.py \
	--load-model=/home/ubuntu/project/mlcommons/inference/recommendation/model/dlrm_terabyte.pytorch \
	--arch-sparse-feature-size=64 \
	--arch-mlp-bot="13-512-256-64" \
	--arch-mlp-top="512-512-256-1" \
	--arch-embedding-size="9980333-36084-17217-7378-20134-3-7112-1442-61-9758201-1333352-313829-10-2208-11156-122-4-970-14-9994222-7267859-9946608-415421-12420-101-36" \
    --data-generation="random" \
    --data-size=1000 \
	--mini-batch-size=1 \
    --num-indices-per-lookup=10 \
    --numpy-rand-seed=123 \
    --rand-data-dist="exp" \
    --rand-data-min=0 \
    --rand-data-max=1 \
    --rand-data-sigma=1 \
	--test-num-workers=16 \
	--test-freq=10240 \
	--memory-map \
	--loss-function=bce \
	--round-targets=True \
	--data-sub-sample-rate=0.875 \
	--inference-only \
    --onnx-path="./fake_tb0875_10M/dlrm_s_pytorch.onnx" \
    #--save-onnx \
