#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/nfs/workspace/dcao/work/dlrm

fake_model_dir="./fake_tb00_40M_m1"
mkdir ${fake_model_dir} 

python3 dlrm_s_pytorch.py \
	--arch-sparse-feature-size=128 \
    	--arch-sparse-feature-num=26 \
	--arch-mlp-bot="13-512-256-128" \
	--arch-mlp-top="1024-1024-512-256-1" \
	--arch-embedding-size="119653218,117129,51867,22260,60789,9,21360,4629,189,115598853,8860638,1210038,30,6624,35814,465,12,2928,42,119939313,76923885,118994952,1757805,38916,324,108" \
    	--data-generation="random" \
    	--data-size=1000 \
	--mini-batch-size=8 \
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

