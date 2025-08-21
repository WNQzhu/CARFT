import copy
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_tok(model_path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True)
    if model_name == 'pythia':
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 0
    elif model_name == 'codellama':
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 2
    elif model_name == 'qwen_2_5':
        tokenizer.pad_token_id = 151643
        tokenizer.eos_token_id = 151645
    elif model_name == 'galactica':
        tokenizer.pad_token_id = 1
        tokenizer.eos_token_id = 2
    elif model_name == 'deepseek':
        tokenizer.pad_token_id = 32014
        tokenizer.eos_token_id = 32014        
    else:
        raise ValueError("not supprted")
    return tokenizer
    
def get_rl_model_and_tok(model_args):
    # print("model_args: ", model_args)
    model_args_copied = copy.copy(model_args)
    model_path = model_args.pop('model_name_or_path', 'unknown')
    model_name = model_args.pop('model_name', 'unknown')
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path, **model_args)

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path, **model_args)

    tokenizer = get_tok(model_path, model_name)

    # model.resize_token_embeddings(len(tokenizer))
        
    return model, ref_model, tokenizer


def get_model_and_tok(model_args):
    model_path = model_args.pop('model_name_or_path', 'unknown')
    model_name = model_args.pop('model_name', 'unknown')
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, **model_args)

    tokenizer = get_tok(model_path, model_name)

    # model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer

