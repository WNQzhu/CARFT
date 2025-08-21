export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/home/wnq/bank/bundle.rollout/code/vae:$PYTHONPATH

python scripts/parse_token_hotmap.py \
       --input_path /home/wnq/output/sft-gen.jsonl \
       --output_path /home/wnq/output/sft-gen-hotmap.jsonl \
       --model_path /home/wnq/models/EleutherAI/pythia-14m \
       --model_name 'pythia' 
