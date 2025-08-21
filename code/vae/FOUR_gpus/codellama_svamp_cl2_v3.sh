#!/bin/bash

export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/data/code/pp2/code/vae:$PYTHONPATH

export ROOT=/data/code/pp2/code/vae

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ${ROOT}/accelerate_configs/deepspeed_zero3_4gpus.yaml ${ROOT}/scripts/run_my_cl2.py training_configs/codellama/svamp/cl2_v3.yaml
