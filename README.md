# DLRM Synthetic Models 

* [Overview](#Overview)
* [Quickstart](#Quickstart)
* [Performance](#Reference)

## Overview

This is a tool for generating DLRM Synthetic Models, which is developed based on facebook [DLRM](facebook_dlrm.md). Now it supports fake data generating with multi-hot features.

## Quickstart

Create a new enviornment for development. You can either use `conda` or `docker`.

```bash
conda create dlrm_v1 python=3.6
conda activate dlrm_v1
```

install packages,

```bash
pip install -r requirements.txt
```

**Notice: as we are using the pytorch version 1.7.1+cpu, install may fail. pls install it seperately via `pip isntall torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html`. If some modules are missing, you can also install them seperately**

### Datasets

Please follow the guide [here](./pytorch/README.md) to download **criteo** datasets.

Also, you can use fake data for quick start.

### Examples

run synthesis models with fixed table size or number.

```bash
bash run_syn_m1.sh

bash run_syn_m2.sh
```

run synthesis models with random table configs.

```bash
bash run.sh

# mlperf offical model
bash run_large.sh
```

tips:

1. to generate onnx model, make sure you have --save-onnx, --onnx-runtime, --onnx-path. 

2. to generate multi-hot inputs, choose --data-generation="random" and specify --num-indices-per-lookup.

## Reference

for more information, check [here](https://confluence.int.lightelligence.co/display/SOL/DLRM+Synthetic+Models). 


