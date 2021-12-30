#!/bin/bash

export DATA_DIR=$HOME/project/mlcommons/inference/recommendation/dlrm/pytorch/fake_criteo
export MODEL_DIR=$HOME/project/mlcommons/inference/recommendation/model
export DLRM_DIR=$HOME/project/mlcommons/inference/recommendation/dlrm
export PYTHONPATH=$PYTHONPATH:$HOME/project/mlcommons/inference/recommendation/dlrm
export PYTHONPATH=$PYTHONPATH:$HOME/project/mlcommons/inference/loadgen

./run_local.sh pytorch dlrm terabyte cpu --scenario Offline --max-ind-range=10000000 --data-sub-sample-rate=0.875 --samples-to-aggregate-fix=1 --max-batchsize=2048 

#./run_local.sh pytorch dlrm terabyte cpu --scenario Offline --max-ind-range=40000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --samples-per-query-offline=204800 

