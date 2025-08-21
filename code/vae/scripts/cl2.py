# encoding=utf-8
import copy
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


from tok import batch_decode_to_tokens
from modeling_hidden_head_v2 import AutoModelForCausalLMWithValueHeadV2
from models import get_tok


def get_cl_model_and_tok(model_args):
    # print("model_args: ", model_args)
    model_args_copied = copy.copy(model_args)
    model_path = model_args.pop('model_name_or_path', 'unknown')
    model_name = model_args.pop('model_name', 'unknown')
    
    model = AutoModelForCausalLMWithValueHeadV2.from_pretrained(
        model_path, **model_args)

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path, **model_args)

    tokenizer = get_tok(model_path, model_name)

    # model.resize_token_embeddings(len(tokenizer))
        
    return model, ref_model, tokenizer
