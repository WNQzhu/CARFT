#!/bin/bash

export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/data/code/pp2/code/vae:$PYTHONPATH

export ROOT=/data/code/pp2/code/vae

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ${ROOT}/accelerate_configs/deepspeed_zero3_8gpus.yaml ${ROOT}/scripts/run_my_cl2.py ${ROOT}/Eight_gpus/training_configs/codellama/gsm8k/cl2_v3.yaml
