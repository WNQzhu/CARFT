#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import random
import sys
import re

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
from collections import defaultdict

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    #SFTConfig,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)

from my_sft_trainer import MySFTTrainer
from my_sft_config  import MySFTConfig

from trl import setup_chat_format

from utils import read_raw_dataset
from models import get_model_and_tok
from transformers.integrations.deepspeed import deepspeed_init
import argparse
from pathlib import Path
import json
from models import get_tok

logger = logging.getLogger(__name__)

def pretty(token):
    return token

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def pretty_result(token_list, value_list):
    ret_tokens, ret_values = [], []
    for _token, _val in zip(token_list, value_list):
        token = remove_non_ascii(_token)
        if token is None or token == '':
            continue
        ret_tokens.append(token)
        ret_values.append(_val)
    return ret_tokens, ret_values

def interval(token_seq, tokenizer, tag_list):
    start_found, end_found = False, False
    start_idx, end_idx = 0, len(token_seq) - 1
    for i, token in enumerate(token_seq):
        if tag_list[0] in token and not start_found:
            if tag_list[1] in token_seq[i+1] and \
               tag_list[2] in token_seq[i+2]:
                start_found = True
                start_idx = i + 3
        if token == tokenizer.eos_token:
            end_idx = i - 1
    return start_idx, end_idx

def compute_color_value(logits):
    min_val = min(logits)
    max_val = max(logits)
    return [
        (val - min_val)/(max_val - min_val) for val in logits
    ]

def parse(instance, tokenizer,
          tag_list=["Answer", "reasoning", ":"]):
    token_seq = instance['token_seq'][1:]
    index_logits = instance['index_logits']
    start_idx, end_idx = interval(token_seq, tokenizer, tag_list)

    ret_token_seq = token_seq[start_idx:end_idx + 1]
    ret_color_values = compute_color_value(index_logits[start_idx:end_idx + 1])

    return pretty_result(ret_token_seq, ret_color_values)
    # Answer reasoning:
    

def main(args):
    tokenizer = get_tok(args.model_path,
                        args.model_name)
    input_path = args.input_path
    output_path = args.output_path
    with open(output_path, 'w', encoding='utf-8') as w:
        with open(input_path, 'r', encoding='utf-8') as f:
            for l in f:
                instance = json.loads(l)
                print('instance:', instance)
                token_seq, color_values = parse(instance,
                                tokenizer)
                d = {
                    "token_seq": token_seq,
                    "color_values": color_values
                    }
                ss = json.dumps(d, ensure_ascii=False)
                w.write(ss + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hot map')
    parser.add_argument('--input_path', type=str, help='parser input')
    parser.add_argument('--output_path', type=str, help='parser output')
    parser.add_argument('--model_path',
                        type=str, help='parser output')
    parser.add_argument('--model_name',
                        type=str, help='parser output')

    
    args = parser.parse_args() 
    main(args)
