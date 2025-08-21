export PATH=/opt/conda/bin:$PATH
#export PYTHONPATH=/home/wnq/bank/proximal/code:$PYTHONPATH
export PYTHONPATH=/home/wnq/pp1/code:$PYTHONPATH


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_sam_disk.py training_configs/llama-3-8b-instruct-ld.yaml



