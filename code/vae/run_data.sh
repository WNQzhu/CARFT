export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/home/wnq/bank/proximal/code:$PYTHONPATH


#ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/mistral-7b-base-simpo.yaml


python3 scripts/run_data.py training_configs/data-test.yaml
