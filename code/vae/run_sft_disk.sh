#!/bin/bash

#export PATH=/opt/conda/bin:$PATH
#export PYTHONPATH=/home/wnq/bank/proximal/code:$PYTHONPATH
#export PYTHONPATH=/home/wnq/pp1/code:$PYTHONPATH

source cmd.sh

launch run "0,1" "codellama-7b-sft-nl.log" PATH=/opt/conda/bin:$PATH ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_my_sft.py training_configs/codellama-sft-test.yaml



