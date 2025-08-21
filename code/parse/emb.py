import os
import torch
import math
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM

def show(args):
    model_name_or_path = os.path.join(args.root, args.model_path)    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    text = "This is a test"
    inputs = tokenizer(text, return_tensors='pt')

    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    print("len: ", len(hidden_states))
    for hidden_state in hidden_states:
        print("shape: ", hidden_state.shape)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='embedding parse')
    parser.add_argument('--root', type=str, default="/home/wnq/models", help='')
    parser.add_argument('--model_path',
                        type=str,
                        default="ppo_test/gsm8k_python_sdp_galactica/global_step_460_epoch_1",
                        help='')
    args = parser.parse_args()

    show(args)
