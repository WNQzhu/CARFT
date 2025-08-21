#!/bin/bash

export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/home/wnq/bank/bundle.rollout/code/vae:$PYTHONPATH

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero2_1gpus.yaml scripts/run_my_cl2_test.py training_configs/pythia-cl-test.yaml
