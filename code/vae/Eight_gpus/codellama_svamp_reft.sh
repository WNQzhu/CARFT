#!/bin/bash

export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/data/code/pp2/code/vae:$PYTHONPATH

export ROOT=/data/code/pp2/code/vae

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ${ROOT}/accelerate_configs/deepspeed_zero3_8gpus.yaml ${ROOT}/scripts/run_my_reft.py training_configs/codellama/svamp/reft.yaml
