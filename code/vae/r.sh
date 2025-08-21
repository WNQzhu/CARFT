#!/bin/bash

export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/home/wnq/bank/bundle.rollout/code/vae:$PYTHONPATH

python scripts/run_my_reft.py training_configs/pythia-sft-test.yaml

