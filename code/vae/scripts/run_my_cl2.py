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

from my_cl_trainer import MyCLTrainer
#from my_sft_trainer import MySFTTrainer
# from my_sft_config  import MySFTConfig
from my_cl_config import MyCLConfig

from trl import setup_chat_format

from utils import read_raw_dataset
from cl2 import get_cl_model_and_tok


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
            prefix_text = f'{instruction}{question}{cot_trigger}'

            src_name = args.dataset_name
            
            # Modify for particular datasets and engine
            if src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric'] and args.engine == 'python':
                prefix_text += f'def solution():\n    """{question}"""\n'

            input_encode = tokenizer(input, add_special_tokens=False)
            output_encode = tokenizer(output, add_special_tokens=False)
            prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

            input_ids = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
            labels = [-100]*len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]
            attention_mask = [1]* len(input_ids)
            prefix = prefix_encode['input_ids']
            prefix_attention_mask = prefix_encode['attention_mask']
            
            # Truncation
            # input_ids_max_length = len(input_ids)
            # assert input_ids_max_length <= args['max_input_length'], input_ids_max_length
            input_ids = input_ids[:args.max_length]
            labels = labels[:args.max_length]
            attention_mask = attention_mask[:args.max_length]
            prefix = prefix[:args.max_input_length]
            prefix_attention_mask = prefix_attention_mask[:args.max_input_length]

            ##
            new_batch['input_ids'].append(input_ids)
            new_batch['labels'].append(labels)
            new_batch['attention_mask'].append(attention_mask)

            new_batch['prefix'].append(prefix)
            new_batch['prefix_attention_mask'].append(prefix_attention_mask)
            ##
            new_batch['item_id'].append(item_id)
            new_batch['question'].append(question)
            new_batch['prefix_text'].append(prefix_text)
            new_batch['answer_cot'].append(answer_cot)
            new_batch['answer_value'].append(answer_value)
        return new_batch
    
    return ds.map(
        tokenize_fn, fn_kwargs={'args': args, 'tokenizer': tokenizer},
        batched=True,
        remove_columns=ds.column_names, 
        num_proc=16, load_from_cache_file=False)

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

            input = f'{instruction}{question}{cot_trigger}'
            output = f'{answer_cot}'
            prefix_text = f'{instruction}{question}{cot_trigger}'

            src_name = args.dataset_name
            # Modify for particular datasets and engine
            if src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric'] and args.engine == 'python':
                prefix_text += f'def solution():\n    """{question}"""\n'

            input_encode = tokenizer(input, add_special_tokens=False)
            output_encode = tokenizer(output, add_special_tokens=False)
            prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

            input_ids = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
            labels = [-100]*len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]
            attention_mask = [1]* len(input_ids)
            prefix = prefix_encode['input_ids']
            prefix_attention_mask = prefix_encode['attention_mask']
            
            # Truncation
            # input_ids_max_length = len(input_ids)
            # assert input_ids_max_length <= args['max_input_length'], input_ids_max_length
            input_ids = input_ids[:args.max_length]
            labels = labels[:args.max_length]
            attention_mask = attention_mask[:args.max_length]
            prefix = prefix[:args.max_length]
            prefix_attention_mask = prefix_attention_mask[:args.max_length]

            ##
            new_batch['input_ids'].append(input_ids)
            new_batch['labels'].append(labels)
            new_batch['attention_mask'].append(attention_mask)

            new_batch['prefix'].append(prefix)
            new_batch['prefix_attention_mask'].append(prefix_attention_mask)
            ##
            new_batch['item_id'].append(item_id)
            new_batch['question'].append(question)
            new_batch['prefix_text'].append(prefix_text)
            new_batch['answer_cot'].append(answer_cot)
            new_batch['answer_value'].append(answer_value)
        return new_batch
    
    return ds.map(
        tokenize_fn, fn_kwargs={'args': args, 'tokenizer': tokenizer},
        batched=True,
        remove_columns=ds.column_names, 
        num_proc=16, load_from_cache_file=False)

def main():
    parser = H4ArgumentParser((MyCLConfig,))
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

    # Check for last checkpoint
    # last_checkpoint = get_checkpoint(training_args)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #    logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############

    raw_datasets = read_raw_dataset(
        training_args.train_path,
        training_args.eval_path)
    logger.info(f"raw_datasets: {raw_datasets}")

    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    train_column_names = list(raw_datasets["train"].features)
    eval_column_names = list(raw_datasets["test"].features)    
    logger.info(f"*** train_column_names : {train_column_names} ***")
    logger.info(f"*** eval_column_names : {eval_column_names} ***")
    logger.info("*** Load pretrained model ***")
    
    model, ref_model, tokenizer = get_cl_model_and_tok(
        training_args.model_init_kwargs)

    tok_train_dataset = _prepare_train_dataset(
        raw_datasets["train"],
        training_args,
        tokenizer)
    tok_eval_dataset = _prepare_eval_dataset(
        raw_datasets["test"],
        training_args,
        tokenizer)

    logger.info(f"tok_trian: {tok_train_dataset}")
    logger.info(f"tok_eval:  {tok_eval_dataset}")
    print('---' * 30)
    with training_args.main_process_first(desc="Log a few random samples from the src data set"):
        for index in random.sample(range(len(tok_train_dataset)), 2):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]}")
        for index in random.sample(range(len(tok_eval_dataset)), 2):
            logger.info(f"Sample {index} of the processed eval set:\n\n{raw_datasets['test'][index]}")
    

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(tok_train_dataset)), 2):
            logger.info(f"Sample {index} of the processed training set:\n\n{tok_train_dataset[index]}")
        for index in random.sample(range(len(tok_eval_dataset)), 2):
            logger.info(f"Sample {index} of the processed eval set:\n\n{tok_eval_dataset[index]}")

    ########################
    # Initialize the Trainer
    ########################
    trainer = MyCLTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=tok_train_dataset,
        eval_dataset=tok_eval_dataset,
        processing_class = tokenizer,
    )
    # exit(0)
    ###############
    # Training loop
    ###############
    #logger.info("*** Train ***")
    #checkpoint = None
    #if training_args.resume_from_checkpoint is not None:
    #    checkpoint = training_args.resume_from_checkpoint
    #elif last_checkpoint is not None:
    #    checkpoint = last_checkpoint
    #train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = trainer.train()    
    # metrics = train_result.metrics
    metrics["train_samples"] = len(tok_train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    #kwargs = {
    #    "finetuned_from": model_args.model_name_or_path,
    #    "dataset": list(data_args.dataset_mixer.keys()),
    #    "dataset_tags": list(data_args.dataset_mixer.keys()),
    #    "tags": ["alignment-handbook"],
    #}
    #if trainer.accelerator.is_main_process:
        # trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        #trainer.model.config.use_cache = True
        #trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(tok_eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
