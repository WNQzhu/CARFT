export PATH=/opt/conda/bin:$PATH
export VAE_ROOT=/home/wnq/bank/bundle.rollout/code/vae
export PYTHONPATH=$VAR_ROOT:$PYTHONPATH
export ACCELERATE_LOG_LEVEL=info
python run_reft.py --world_size 1 \
       --train_mode test \
       --model_path /home/wnq/models/EleutherAI/pythia-14m \
       --model_name pythia \
       --attn_implementation eager \
       --train_path /home/wnq/bank/bundle.rollout/code/mwp_ReFT/data/svamp_nl.json \
       --eval_path  /home/wnq/bank/bundle.rollout/code/mwp_ReFT/data/svamp_test_set.json \
       --dataset_name svamp \
       --max_length 700 \
       --gradient_accumulation_steps 2 \
       --lr 5e-7 \
       --logging_steps 1 \
       --num_train_epochs 8 \
       --output_dir sft-outputs/pythia-base-sft \
       --per_device_train_batch_size 2 \
       --per_device_eval_batch_size  2 \
       --save_total_limit 2 
       
