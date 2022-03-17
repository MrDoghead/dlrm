#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/nfs/workspace/dcao/work/dlrm

fake_model_dir="./synthetic_m1"
mkdir ${fake_model_dir} 

python3 dlrm_s_pytorch.py \
	--arch-sparse-feature-size=128 \
    	--arch-sparse-feature-num=104 \
	--arch-mlp-bot="13-512-256-128" \
	--arch-mlp-top="1024-1024-512-256-1" \
	--arch-embedding-size="39884406-39043-17289-7420-20263-3-7120-1543-63-38532951-2953546-403346-10-2208-11938-155-4-976-14-39979771-25641295-39664984-585935-12972-108-36-39884406-39043-17289-7420-20263-3-7120-1543-63-38532951-2953546-403346-10-2208-11938-155-4-976-14-39979771-25641295-39664984-585935-12972-108-36-39884406-39043-17289-7420-20263-3-7120-1543-63-38532951-2953546-403346-10-2208-11938-155-4-976-14-39979771-25641295-39664984-585935-12972-108-36-39884406-39043-17289-7420-20263-3-7120-1543-63-38532951-2953546-403346-10-2208-11938-155-4-976-14-39979771-25641295-39664984-585935-12972-108-36" \
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

