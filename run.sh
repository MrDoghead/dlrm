#!/bin/bash

#export DATA_DIR=$HOME/project/mlcommons/inference/recommendation/dlrm/pytorch/fake_criteo
export DATA_DIR=$HOME/project/mlcommons/inference/recommendation/dlrm/fake_terabyte0875
export MODEL_DIR=$HOME/project/mlcommons/inference/recommendation/model
export DLRM_DIR=$HOME/project/mlcommons/inference/recommendation/dlrm
export PYTHONPATH=$PYTHONPATH:$HOME/project/mlcommons/inference/recommendation/dlrm

python dlrm_s_pytorch.py \
	--arch-sparse-feature-size=64 \
	--arch-mlp-bot="13-512-256-64" \
	--arch-mlp-top="512-512-256-1" \
	--max-ind-range=10000000 \
	--data-generation=dataset \
	--data-set=terabyte \
	--raw-data-file=/home/ubuntu/project/mlcommons/inference/recommendation/dlrm/fake_terabyte0875/day  \
	--processed-data-file=/home/ubuntu/project/mlcommons/inference/recommendation/dlrm/fake_terabyte0875/day \
	--loss-function=bce \
	--round-targets=True \
	--learning-rate=0.1 \
	--mini-batch-size=2 \
	--print-freq=1024 \
	--print-time \
	--test-mini-batch-size=2 \
	--test-num-workers=16 \
	--load-model=/home/ubuntu/project/mlcommons/inference/recommendation/model/dlrm_terabyte.pytorch \
	--test-freq=10240 \
	--memory-map \
	--data-sub-sample-rate=0.875 \
	--inference-only \
	#--save-onnx \
