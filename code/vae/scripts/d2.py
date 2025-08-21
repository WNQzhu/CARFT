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

logger = logging.getLogger(__name__)


def _prepare_train_dataset(ds, args, tokenizer):
    instruction = args.cot["instruction"]
    cot_trigger = args.cot["cot_trigger"]
    answer_trigger = args.cot["answer_trigger"]    
    logger.info(f'Using instruction: {instruction}')
    logger.info(f'Using cot_trigger: {cot_trigger}')
    logger.info(f'Using answer_trigger: {answer_trigger}')
    
    def tokenize_fn(batch, args, tokenizer):
        assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
        new_batch = defaultdict(list)
        all_keys = list(batch.keys())
        for item_values in zip(*(batch[k] for k in all_keys)):
            item = {k: item_values[i] for i, k in enumerate(all_keys)}
            item_id, question, answer_value, answer_cot = \
                item['item_id'], \
                item['question'], \
                item['answer_value'], \
                item.get('answer_cot', None), \

            question = question.strip()
            if answer_value is not None:
                answer_value = answer_value.strip()

            if answer_cot is not None:
                answer_cot = answer_cot.strip()
                if args.engine == 'nl':
                    answer_cot += f'{answer_trigger}{answer_value}'

            input = f'{instruction}{question}{cot_trigger}'
            output = f'{answer_cot}'

            input_encode = tokenizer(input, add_special_tokens=False)
            output_encode = tokenizer(output, add_special_tokens=False)
            input_ids = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
            labels = [-100]*len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]
            attention_mask = [1]* len(input_ids)

            # Truncation
            input_ids_max_length = len(input_ids)
            # assert input_ids_max_length <= args['max_input_length'], input_ids_max_length
            input_ids = input_ids[:args.max_length]
            labels = labels[:args.max_length]
            attention_mask = attention_mask[:args.max_length]

            ##
            new_batch['input_ids'].append(input_ids)
            new_batch['labels'].append(labels)
            new_batch['attention_mask'].append(attention_mask)

        return new_batch
    
    return ds.map(
        tokenize_fn, fn_kwargs={'args': args, 'tokenizer': tokenizer},
        batched=True,
        remove_columns=ds.column_names, 
        num_proc=4, load_from_cache_file=False)

def _prepare_eval_dataset(ds, args, tokenizer):
    instruction = args.cot["instruction"]
    cot_trigger = args.cot["cot_trigger"]
    answer_trigger = args.cot["answer_trigger"]    
    logger.info(f'Using instruction: {instruction}')
    logger.info(f'Using cot_trigger: {cot_trigger}')
    logger.info(f'Using answer_trigger: {answer_trigger}')
    
    def tokenize_fn(batch, args, tokenizer):
        assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
        new_batch = defaultdict(list)
        all_keys = list(batch.keys())
        for item_values in zip(*(batch[k] for k in all_keys)):
            item = {k: item_values[i] for i, k in enumerate(all_keys)}
            item_id, question, answer_value, answer_cot = \
                item['item_id'], \
                item['question'], \
                item['answer_value'], \
                item.get('answer_cot', None), \

            question = question.strip()
            if answer_value is not None:
                answer_value = answer_value.strip()

            if answer_cot is not None:
                answer_cot = answer_cot.strip()
                if args.engine == 'nl':
                    answer_cot += f'{answer_trigger}{answer_value}'

            output = f'{answer_cot}'
            prefix_text = f'{instruction}{question}{cot_trigger}'

            output_encode = tokenizer(output, add_special_tokens=False)
            prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

            prefix = prefix_encode['input_ids']
            prefix_attention_mask = prefix_encode['attention_mask']

            # Truncation
            # input_ids_max_length = len(input_ids)
            # assert input_ids_max_length <= args['max_input_length'], input_ids_max_length
            prefix = prefix[:args.max_length]
            prefix_attention_mask = prefix_attention_mask[:args.max_length]

            ##
            new_batch['input_ids'].append(prefix)
            new_batch['attention_mask'].append(prefix_attention_mask)
            ##
            new_batch['item_id'].append(item_id)
            new_batch['question'].append(question)
            new_batch['answer_cot'].append(answer_cot)
            new_batch['answer_value'].append(answer_value)
            #  new_batch['input_ids_max_length'].append(input_ids_max_length)
            
        return new_batch
    
    return ds.map(
        tokenize_fn, fn_kwargs={'args': args, 'tokenizer': tokenizer},
        batched=True,
        remove_columns=ds.column_names, 
        num_proc=4, load_from_cache_file=False)



def main():
    from tt import T
    t = T()    
    parser = H4ArgumentParser((MySFTConfig,))
    training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    #logger.info(f"Model parameters {model_args}")
    #logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    
    #print("t.acc: ", t.acc)
    
    # Set seed for reproducibility
    # set_seed(training_args.seed)
    

if __name__ == "__main__":
    main()
