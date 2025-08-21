#!/bin/bash

export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/data/code/pp2/code/vae:$PYTHONPATH

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3_8gpus.yaml scripts/run_my_cl2.py training_configs/codellama/svamp/cl2_v3.yaml
