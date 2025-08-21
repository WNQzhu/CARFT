# wnq::cl_trainer_v5(from cl_trainer_v3)
# this version takes only the positive part of the generated cot
# unlike v3, we extracted all the cot tokens from from the ref and postive generated cot with 1 as the reward
## IMP:
# we aug the number of ref cot by random mask some numbers:
## that is we have changed the `not_cot_mask` function:
### (random masked some number values)

# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Optional, Union
import random
from deepspeed.accelerator import get_accelerator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object, pad_across_processes
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.utils import logging
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available

from trl.core import masked_mean, masked_var, masked_whiten
from trl.models import create_reference_model
from trl.models.utils import (
    unwrap_model_for_generation
)
from transformers.trainer_utils import (
    EvalPrediction,
    seed_worker,
    EvalLoopOutput,
    has_length,   
    denumpify_detensorize,
    speed_metrics,
)

from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    log_table_to_comet_experiment,
    peft_module_casting_to_bf16,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
    selective_log_softmax,    
)

from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)

from my_cl_config import MyCLConfig
from functools import partial
from reft import get_rewards
from tok import batch_decode_to_tokens

from transformers.debug_utils import ( DebugOption,)


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb

from utils import discount_cumsum, do_gather, allgather, allgather_masked_whiten
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from utils import floatify, extract_pred_values, compute_acc
import torch.nn.functional as F

from cl import (get_cur_list_value,
                check_string,
                find_num_idx)


INVALID_LOGPROB = 1.0

logger = logging.get_logger(__name__)

# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(**kwargs)
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits

def train_collate_fn(batch, args, tokenizer):
    max_prefix_length = max([len(item['prefix']) for item in batch])
    
    max_base_length = max([len(item['input_ids']) for item in batch])

    prefix, prefix_left_padded = [], []
    prefix_attention_mask, prefix_attention_mask_left_padded = [], []
    
    base, base_left_padded = [], []
    base_attention_mask, base_attention_mask_left_padded = [], []

    query_list, ans_list = [], []

    base_input_prefix_length_list = []
    
    for item in batch:
        prefix_left_padded.append([tokenizer.pad_token_id] * (max_prefix_length - len(item['prefix'])) + item['prefix'])
        prefix_attention_mask_left_padded.append(
                [0] * (max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask'])
        base_left_padded.append([tokenizer.pad_token_id] * (max_base_length - len(item['input_ids'])) + item['input_ids'])
        query_list.append(item['prefix_text'])
        ans_list.append(item['answer_value'])
        base_input_prefix_length_list.append(
            max_base_length - len(item['input_ids'])  + len(item['prefix'])
        )
                
    new_batch = {
        'input_ids': torch.LongTensor(prefix_left_padded),
        'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
        'base_input_ids': torch.LongTensor(base_left_padded),
        #'base_attention_mask': torch.BoolTensor(base_attention_mask_left_padded),
        #torch.LongTensor(base_input_prefix_length_list).unsqueeze(-1)
    }
    text_batch = {
        'query': query_list,
        'answer_values': ans_list,
        'max_prefix_length': max_prefix_length,
        'base_input_prefix_length': base_input_prefix_length_list
    }
    return new_batch, text_batch

    
def eval_collate_fn(batch, args, tokenizer):
    max_prefix_length = max([len(item['prefix']) for item in batch])
    prefix_left_padded = []
    prefix_attention_mask_left_padded = []

    item_id_list, question_list, \
        answer_value_list, answer_cot_list = [], [], [], []
    
    for item in batch:
        prefix_left_padded.append([tokenizer.pad_token_id] * (max_prefix_length - len(item['prefix'])) + item['prefix'])
        prefix_attention_mask_left_padded.append(
                [0] * (max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask'])
        item_id_list.append(item['item_id'])
        question_list.append(item['question'])
        answer_value_list.append(item['answer_value'])
        answer_cot_list.append(item['answer_cot'])
        
                
    new_batch = {
        'input_ids': torch.LongTensor(prefix_left_padded),
        'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
    }
    
    text_batch = {
        'item_id': item_id_list,
        'question': question_list,
        'answer_value': answer_value_list,
        'answer_cot':   answer_cot_list
    }    
    
    return new_batch, text_batch

class MyCLTrainer(Trainer):
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        args: MyCLConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        model: nn.Module,
        ref_model: Optional[nn.Module],
        # reward_model: nn.Module,
        train_dataset: Dataset,
        # value_model: Optional[nn.Module] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ) -> None:
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must make a copy of it, or `None` if you use peft."
            )

        self.args = args
        self.processing_class = processing_class
        #self.policy_model = model
        self.model = model

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        if ref_model:
            self.ref_model = ref_model
        else:
            raise ValueError("pls provide ref model")

        # self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        # self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
            
        #accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        #self.accelerator = accelerator
        
        self.create_accelerator_and_postprocess()        
        args.world_size = self.accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        
        self.accelerator.print(f"world_size: {args.world_size}, local_batch_size: {args.local_batch_size}, micro_batch_size: {args.micro_batch_size}")
        self.accelerator.print(f"batch_size: {args.batch_size}, mini_batch_size: {args.mini_batch_size}, local_mini_batch_size: {args.local_mini_batch_size}")

        if args.whiten_rewards:
            assert args.local_mini_batch_size >= 8, (
                f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
            )
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=self.accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + self.accelerator.process_index * 100003  # Prime
        #if args.num_sample_generations > 0:
        #    self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        # for module in [self.policy_model, self.ref_model, self.value_model, self.reward_model]:
        for module in [self.model, self.ref_model]:
            if module is not None:
                disable_dropout_in_model(module)
        # self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        # self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            # collate_fn=self.data_collator,
            collate_fn=partial(train_collate_fn,
                                  args=self.args, tokenizer=self.processing_class),
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=partial(eval_collate_fn,
                                  args=self.args,
                                          tokenizer=self.processing_class),
            drop_last=self.args.dataloader_drop_last
        )  # no need to shuffle eval dataset
        self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            #self.reward_model = prepare_deepspeed(
            #    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            #)
            if self.ref_model is None:
                raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is None:
                raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = accelerator.prepare(self.ref_model)
                # self.ref_model = self.ref_model.to(self.accelerator.device)
            #self.reward_model = self.reward_model.to(self.accelerator.device)
        #logger.info(" at the end of __init__")
        #print("at the end of __Init__")
        
    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model.policy).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.model_adapter_name or "default")

    def rollout(self, args, new_batch, text_batch, generation_config):
        self.model.eval()
        with torch.no_grad():
            #_,_,_, base_hidden_emb = self.model(
            #    input_ids=new_batch['base_input_ids'],
            #    attention_mask=new_batch['base_attention_mask']
            #)
            #del _
            #torch.cuda.empty_cache()
            with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    completed_tensors, logitss = batch_generation(
                        unwrapped_model,
                        new_batch['input_ids'],
                        #args.local_rollout_forward_batch_size,
                        self.local_dataloader_batch_size,
                        self.processing_class.pad_token_id,
                        generation_config,
                    )
                    self.accelerator.print(f"rollout.completed_tensor.shape: {completed_tensors.shape}")
                    del logitss
                    torch.cuda.empty_cache()
        completed_tensors = pad_across_processes(
            completed_tensors,
            dim=1,
            pad_index=self.processing_class.pad_token_id,
            pad_first=False)

        #gen_size = completed_tensors.size(1)
        #base_size = new_batch['base_input_ids'].size(1)
        #batch_size = self.local_dataloader_batch_size
        #target_dim1 = max(gen_size, base_size)
        #if gen_size > base_size:
        #    padding = (0, target_dim1 - base_size)
        #    base_tensors = F.pad(new_batch['base_input_ids'], padding, mode='constant',
        #                         value=self.processing_class.pad_token_id)
        #    full_tensors = torch.cat((base_tensors,  completed_tensors), dim = 0)
        #    pad_completed_tensors = completed_tensors
        # else:
        #    padding = (0, target_dim1 - gen_size)
        #    pad_completed_tensors = F.pad(completed_tensors, padding, mode='constant',
        #                         value=self.processing_class.pad_token_id)
        #    full_tensors = torch.cat((new_batch['base_input_ids'],  pad_completed_tensors), dim=0)
        #full_tensors_mask = (full_tensors != self.processing_class.pad_token_id)
        
        # reward:
        # Evaluate score
        completed_texts = self.processing_class.batch_decode(
            completed_tensors.cpu().numpy().tolist(), skip_special_tokens=True)
        programs = [text.strip().split(args.cot['cot_trigger'])[-1].strip()
                    for text in completed_texts]
        correctness = get_rewards(args, programs, text_batch['answer_values'])

        model_input_ids = completed_tensors
        model_attention_mask = (completed_tensors != self.processing_class.pad_token_id)
        #model_input_ids = pad_completed_tensors
        #model_attention_mask = (pad_completed_tensors != self.processing_class.pad_token_id)
        base_input_ids = pad_across_processes(
            new_batch['base_input_ids'],
            dim=1,
            pad_index=self.processing_class.pad_token_id,
            pad_first=False)
        
        base_attention_mask = (base_input_ids!= self.processing_class.pad_token_id)
       
        with torch.no_grad():
            # Get old logprob and val
            # lm_logits, _dummy2, val = self.model(input_ids=model_input_ids, attention_mask=model_attention_mask) 
            lm_logits, _, val, _ = self.model(input_ids=model_input_ids, attention_mask=model_attention_mask)
            
            _, _, base_val, base_hidden_emb = self.model(input_ids=base_input_ids, attention_mask=base_attention_mask)
            
            #base_lm_logits, lm_logits = torch.chunk(full_lm_logits, 2, dim=0)
            #base_val, val = torch.chunk(full_val, 2, dim=0)
            #base_hidden_emb, gen_hidden_emb = torch.chunk(full_hidden_emb, 2, dim=0)
            
            old_logprob = selective_log_softmax(lm_logits[:, :-1, :],
                                                model_input_ids[:, 1:])  # (bs, seqlen-1)

            del _
            torch.cuda.empty_cache()
        
            # Get the ref model logprob
            ref_logprob = None
            ref_output = self.ref_model(
                input_ids=model_input_ids,
                attention_mask=model_attention_mask)
            ref_lm_logits = ref_output.logits
            ref_logprob = selective_log_softmax(
                ref_lm_logits[:, :-1, :],
                model_input_ids[:, 1:])  # (bs, seqlen-1)
        
        # Masking the last prompt token up untils the token before eos_token_id
        prompt_len = new_batch['input_ids'].size(1)
        mask = torch.zeros_like(model_input_ids, dtype=torch.bool)  # (bs, seqlen)
        mask[:, prompt_len - 1: -1] = 1
        score_rew = np.zeros(mask.shape)  # (bs, seqlen)
        score_rew[:, -2] = np.array(correctness)
        nonzero = (model_input_ids == self.processing_class.eos_token_id).nonzero()
        for (bidx, tidx) in nonzero:
            mask[bidx][tidx:] = 0
            score_rew[bidx][tidx:] = 0
            score_rew[bidx][tidx - 1] = correctness[bidx]
        
        # Make the kl reward and the full reward
        kl_rew = None
        rew = score_rew
        if ref_logprob is not None:
            kl = old_logprob - ref_logprob  # (bs, seqlen-1)
            kl = (kl.float() * mask[:, :-1]).cpu().numpy()
            kl_rew = np.zeros(mask.shape)  # (bs, seqlen)
            kl_rew[:, :-1] = -kl # NOTE the minus sign
        
            #kl_coef = args["kl_coef"]
            kl_coef = args.kl_coef
            rew = score_rew + kl_coef * kl_rew
        # Process val ret adv logprob
        val = (val.float() * mask).cpu().numpy()
        gamma = args.gamma
        #gamma = args["gamma"]
        #lam = args["lam"]
        lam = args.lam
        # ret = np.zeros_like(rew)
        adv = np.zeros_like(rew)
        for i in range(len(rew)):
            cur_rew, cur_val = rew[i], val[i]
            cur_delta = -cur_val[:-1] + cur_rew[:-1] + gamma * cur_val[1:]
            cur_adv = discount_cumsum(cur_delta, discount=gamma * lam)
            cur_adv[:prompt_len - 1] = 0
            adv[i][:-1] = cur_adv
        
        # lambda_return = GAE + values
        ret = adv + val  # (bs, seqlen)
        
        rew = torch.tensor(rew, device=mask.device, dtype=old_logprob.dtype) * mask
        score_rew = torch.tensor(score_rew, device=mask.device, dtype=old_logprob.dtype) * mask
        if kl_rew is not None:
            kl_rew = torch.tensor(kl_rew, device=mask.device, dtype=old_logprob.dtype) * mask
        ret = torch.tensor(ret, device=mask.device, dtype=old_logprob.dtype) * mask
        val = torch.tensor(val, device=mask.device, dtype=old_logprob.dtype) * mask
        adv = torch.tensor(adv, device=mask.device, dtype=old_logprob.dtype) * mask
        old_logprob = old_logprob * mask[:, :-1]
        
        gc.collect()
        
        self.model.train()
        # return model_input_ids, model_attention_mask, mask, rew, score_rew, kl_rew, ret, correctness, val, old_logprob, ref_logprob, adv, base_lm_logits, base_hidden_emb
        
        return model_input_ids, model_attention_mask, mask, rew, score_rew, kl_rew, ret, correctness, val, old_logprob, ref_logprob, adv, \
            base_input_ids, base_attention_mask, \
                base_val, base_hidden_emb
        
        
        
    def train(self):
        args = self.args
        vf_coef = args.vf_coef
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        # reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            #max_new_tokens=args.response_length,
            #temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            # bos_token_id=tokenizer.bos_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            #max_length=args.max_gen_length,
            max_length=args.max_length,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        cl_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        #+begin_src for eval
        one_epoch_total_batches = args.num_total_batches // args.num_train_epochs
        #+end_src

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            new_batch, text_batch = next(iter_dataloader)
            # self.accelerator.print(f"new_batch: {new_batch}, text_batch: {text_batch}")
            model_input_ids, model_attention_mask, mask, rew, score_rew, kl_rew, ret, \
                correctness, val, old_logprob, ref_logprob, adv, \
                base_input_ids, base_attention_mask, \
                base_val, base_hidden_emb = self.rollout(
                self.args, new_batch, text_batch,
                generation_config)
            self.accelerator.print(f"adv: {adv.shape}")
            # preprocess
            raw_adv = adv
            if args.adv_whitening == 'global':
                adv = allgather_masked_whiten(adv, mask) # (mini_bs, seqlen)
            elif args.adv_whitening == 'local':
                adv = masked_whiten(adv, mask)

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            # Subset to batch
                            cur_val = val[micro_batch_inds].contiguous()  # mini_bs x seqlen
                            self.accelerator.print(f"ppo.cur_val.shape: {cur_val.shape}")
                            cur_old_logprob = old_logprob[micro_batch_inds].contiguous()  # mini_bs x seqlen
                            cur_mask = mask[micro_batch_inds].contiguous()  # mini_bs x seqlen
                            cur_rew = rew[micro_batch_inds].contiguous()  # mini_bs x seqlen
                            cur_score_rew = score_rew[micro_batch_inds].contiguous() # mini_bs x seqlen
                            cur_kl_rew = None if kl_rew is None else kl_rew[micro_batch_inds].contiguous()  # mini_bs x seqlen
                            cur_ret = ret[micro_batch_inds].contiguous()  # mini_bs x seqlen
                            cur_adv = adv[micro_batch_inds].contiguous()  # mini_bs x seqlen
                            cur_raw_adv = raw_adv[micro_batch_inds].contiguous()  # mini_bs x seqlen
                            cur_model_input_ids = model_input_ids[micro_batch_inds].contiguous()  # mini_bs x seqlen
                            cur_model_attention_mask = model_attention_mask[micro_batch_inds].contiguous()  # mini_bs x seqlen
                            
                            resp_len_per_sample = torch.clamp(torch.sum(cur_mask, dim=1), min=1.0)  # (mini_bs,)
                            cur_query_mask = torch.logical_xor(cur_mask, cur_model_attention_mask)  # (mini_bs, seqlen)
                            query_len_per_sample = torch.clamp(torch.sum(cur_query_mask, dim=1), min=1.0)  # (mini_bs,)
                            
                            # Preprocess advantage and get metrics  
                            cur_mask = cur_mask.type(cur_adv.dtype).contiguous()
                            mean_adv, var_adv = masked_mean(cur_adv, cur_mask), masked_var(cur_adv, cur_mask)
                            # Forward current model
                            # model.eval()
                            lm_logits, _, vpreds, pred_hidden_emb = model(
                                input_ids=cur_model_input_ids,
                                attention_mask=cur_model_attention_mask)
                            logprob = selective_log_softmax(
                                lm_logits[:, :-1, :],
                                cur_model_input_ids[:, 1:])  # (mini_bs, seqlen-1)
                            
                            #+begin_src cl compute_cl_loss
                            cur_base_val = base_val[
                                micro_batch_inds].contiguous()
                            cur_base_hidden_emb = base_hidden_emb[
                                micro_batch_inds].contiguous()
                            cur_base_input_ids = base_input_ids[
                                micro_batch_inds].contiguous()
                            cur_base_attention_mask = base_attention_mask[
                                micro_batch_inds].contiguous()
                            cur_correctness = get_cur_list_value(
                                micro_batch_inds, correctness)
                            cur_base_prefix_len_list = get_cur_list_value(
                                micro_batch_inds,
                                text_batch['base_input_prefix_length'])

                            # loss
                            cur_bs = args.per_device_train_batch_size
                            model_len  = cur_model_input_ids.size(1)
                            base_len =   cur_base_input_ids.size(1)
                            
                            cur_base_input_ids_cpu = cur_base_input_ids.cpu().numpy().tolist()
                            cur_model_input_ids_cpu = cur_model_input_ids.cpu().numpy().tolist()
                            
                            cur_base_token_seq_list = batch_decode_to_tokens(
                                self.processing_class,
                                cur_base_input_ids_cpu)
                            cur_model_token_seq_list = batch_decode_to_tokens(
                                self.processing_class,
                                cur_model_input_ids_cpu)

                            base_emb_mask = self.not_cot_mask(
                                cur_base_token_seq_list,
                                cur_base_prefix_len_list,
                                aug=True
                            )
                            
                            base_emb_mask = torch.logical_and(
                                base_emb_mask.to(device),
                                cur_base_attention_mask)

                            ref_weight  = self.masked_softmax(cur_base_val,
                                                    base_emb_mask)        
                            z_1 = torch.sum(cur_base_hidden_emb * ref_weight.unsqueeze(-1), dim=1)
                            
                            cur_model_prefix_len_list = [
                                text_batch['max_prefix_length']
                            ] * cur_bs
                            
                            model_emb_mask = self.not_cot_mask(
                                cur_model_token_seq_list,
                                cur_model_prefix_len_list                                
                            )
                            
                            model_emb_mask = torch.logical_and(
                                model_emb_mask.to(device),
                                cur_model_attention_mask)

                            model_emb_mask.to(device)
                            # print("vpreds:", vpreds)
                            pred_weight  = self.masked_softmax(vpreds,
                                                    model_emb_mask)
                            # print("vpreds:(after)", vpreds) 
                            
                            z_2 = torch.sum(pred_hidden_emb * pred_weight.unsqueeze(-1), dim=1)

                            self.accelerator.print(f"z_2.shape: {z_2.shape}")

                            def f(x): return torch.exp(x / self.args.cl_tau)
                            
                            # batch_size
                            between_sim = f(self.sim_1(z_1, z_2))
                            # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
                            
                            z_all = torch.reshape(allgather(z_2), (-1, 64))
                            
                            self.accelerator.print(f"z_all.shape: {z_all.shape}")
                            
                            
                            all_sim = f(self.sim_2(z_1, z_all))

                            corr_mask = (torch.Tensor(cur_correctness).unsqueeze(-1) > 0.5)
                            corr_mask = corr_mask.to(device)
                            # batch_size
                            positive_pairs = between_sim 
                            # batch_size
                            negative_pairs = torch.sum(all_sim, 1)
                            #loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
                            #cl_loss = torch.mean(-torch.log(positive_pairs / negative_pairs))
                            cl_loss_a = -torch.log(
                                positive_pairs / negative_pairs
                            ) * corr_mask  
                            cl_loss_b = torch.sum(corr_mask) + 1e-8

                            cl_loss = torch.sum(cl_loss_a) / cl_loss_b
                            
                            #cl_loss = torch.sum(-torch.log(positive_pairs / negative_pairs)) / (torch.sum(corr_mask) + 1e-8)

                            #+end_src cl compute_cl_loss
                            
                            # Compute losses
                            loss = 0
                            
                            # policy gradient loss
                            ratio = torch.exp(logprob - cur_old_logprob)
                            pg_losses = -cur_adv[:, :-1] * ratio
                            pg_losses2 = -cur_adv[:, :-1] * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                            pg_loss = ((torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum(dim=-1) / resp_len_per_sample).mean()
                            # pg_loss = (torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum() / cur_mask[:, :-1].sum()
                            # pg_loss = (-logprob * cur_ret[:,:-1]).sum() / cur_mask[:, :-1].sum()
                            
                            # value loss
                            vpredclipped = torch.max(torch.min(vpreds, cur_val + 0.2), cur_val - 0.2)
                            vf_losses1 = (vpreds - cur_ret) ** 2
                            vf_losses2 = (vpredclipped - cur_ret) ** 2
                            vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum(dim=-1) / resp_len_per_sample).mean()
                            # vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum() / cur_mask.sum())
                            
                            # total loss
                            loss += pg_loss + \
                                vf_coef * vf_loss + \
                                args.cl_coef * cl_loss 
                            
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                                                        
                            with torch.no_grad():
                                #pg_clipfrac = masked_mean(
                                #    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                #)
                                #prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                #entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                #approxkl = 0.5 * (logprobs_diff**2).mean()
                                #approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                #pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                #    pg_clipfrac
                                #)
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                cl_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = cl_loss
                                
                                #vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                #    vf_clipfrac
                                #)
                                #entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                #ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        cur_val, cur_old_logprob, cur_mask,
                        cur_rew, cur_score_rew,
                        cur_kl_rew, cur_ret,
                        cur_adv, cur_raw_adv,
                        cur_model_input_ids,
                        cur_model_attention_mask,
                        cur_query_mask,
                        mean_adv, var_adv,
                        lm_logits, _,
                        vpreds, pred_hidden_emb,
                        logprob,
                        cur_base_val, cur_base_hidden_emb,
                        cur_base_input_ids,
                        cur_base_attention_mask,
                        cur_base_input_ids_cpu,
                        cur_model_input_ids_cpu,
                        cur_base_token_seq_list,
                        cur_model_token_seq_list,
                        base_emb_mask, ref_weight,
                        model_emb_mask, pred_weight,
                        z_1, z_2, z_all,
                        vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, ratio, pg_losses, pg_losses2,
                        cl_loss,
                        pg_loss, loss
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
                    gc.collect()
                    get_accelerator().empty_cache()

            with torch.no_grad():
                metrics = {}
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["loss/cl_avg"] = self.accelerator.gather_for_metrics(cl_loss_stats).mean().item()                
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            #if self.state.episode % self.train_dataset_len == 0:
            # eval_metrics = None
            if update % one_epoch_total_batches == 0:
                eval_metrics = self.evaluate()
                # disable save for fast check
                #self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                #if self.control.should_save:
                #    self._save_checkpoint(model, trial=None, metrics=eval_metrics)
                #self.control = self.callback_handler.on_save(self.args, self.state, self.control)
                
            self.lr_scheduler.step()
            del (
                model_input_ids, model_attention_mask,
                mask, rew, score_rew,
                kl_rew, ret, correctness,
                val, old_logprob, ref_logprob, adv)
            
            torch.cuda.empty_cache()
            get_accelerator().empty_cache()            
            gc.collect()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
        ret_metrics = speed_metrics(
            "train",
            start_time,
            num_samples=len(self.dataloader),
            num_steps=self.state.episode,
            num_tokens=1,
        )
        return ret_metrics


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        #+begin_src remove  and test
        """
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        """
        #+end_src remove  and test
                

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        # model.eval()
        self.model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)

        #preds_host = []
        #labels_host = []

        # losses/preds/labels on CPU (final containers)
        #all_preds = None
        #all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0
        corr_num, total_num = 0, 0
        #+begin_src +
        generation_config = GenerationConfig(
            max_length=self.args.max_length,
            # temperature=(args.temperature + 1e-7),
            output_scores = True,
            return_dict_in_generate=True,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,    
        )        
        #+end_src 
        # Main evaluation loop
        for step, (inputs, text_dict) in enumerate(dataloader):
            #print("input is : ", inputs)
            #print("text_dict: ", text_dict)
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            #+ begin_src new
            device = self.accelerator.device
            with torch.no_grad():
                queries = inputs["input_ids"].to(device)
                context_length = queries.shape[1]
                with unwrap_model_for_generation(
                    #model,
                        self.model,
                        self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model,
                        queries,
                        batch_size,
                        self.processing_class.pad_token_id,
                        generation_config,
                    )
                    del logitss
                    torch.cuda.empty_cache()
                    #print("q _ r : ", query_responses)
                    #print("query_response.shape: ", query_responses.shape)
                    #for query in query_responses:
                    #    print("query: ", query)
                    logger.info(f"##### p_idx {self.accelerator.process_index} qr shape {query_responses.shape} #####")
                    extract_value_list  =  extract_pred_values([
                        self.processing_class.decode(
                            qr,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True).strip()
                        for qr in query_responses                                 
                        ], self.args)

                    label_list = [floatify(a) for a in text_dict['answer_value']]
                    
                    logger.info(f"##### p_idx {self.accelerator.process_index} extract len {len(extract_value_list)} #####")

                    gathered_value_list = self.accelerator.gather_for_metrics(extract_value_list)
                    
                    gathered_label_list = self.accelerator.gather_for_metrics(label_list)

                    # logger.info(f"##### p_idx {self.accelerator.process_index} gathered_value_list {gathered_value_list} #####")
                    # logger.info(f"##### p_idx {self.accelerator.process_index} gathered_label_list {gathered_label_list} #####")


                    t_acc, t_total = compute_acc(gathered_value_list, gathered_label_list, self.args)
                    corr_num += t_acc
                    total_num += t_total
                    
                    logger.info(f"##### p_idx {self.accelerator.process_index} corr {corr_num} total {total_num} #####")                    

                    #preds_host.extend(extrat_value_list)
                    #logger.info(f"##### p_idx {self.accelerator.process_index} ph len {len(preds_host)} #####")
                    
                    # print("preds_host", preds_host)
                    # print("1,2", query_responses, logitss)
                    #labels_host.extend([
                    #    floatify(a) for a in text_dict['answer_value']
                    #])
                    #logger.info(f"##### p_idx {self.accelerator.process_index} ph len {len(labels_host)} #####")
                    
                    #print("labels host: ", labels_host)
            #+ end_src
        #+begin_src
        #all_preds = self.accelerator.gather_for_metrics(preds_host)
        #all_labels= self.accelerator.gather_for_metrics(labels_host)
        #logger.info(f"##### p_idx {self.accelerator.process_index} all preds len {len(all_preds)} #####")
        #logger.info(f"##### p_idx {self.accelerator.process_index} all labels len {len(all_labels)} #####")
        
        #acc = compute_acc(all_preds, all_labels, self.args)
        #+end_src

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        metrics['myacc'] = 1.0 * corr_num /total_num
        metrics['num_samples']   = num_samples
        metrics['gather_num'] = total_num
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        torch.cuda.empty_cache()
        gc.collect()

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics, num_samples=num_samples)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        # self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader()

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            #prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        #self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics


    def sim_1(self, z1: torch.Tensor, z2: torch.Tensor):
        return torch.clamp(F.cosine_similarity(z1, z2), max=0.9)
    
    def sim_2(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def masked_softmax(self, logits, mask):
        """
        计算带 mask 的 softmax
        :param logits: 输入的 logits，形状为 (batch_size, seq_len)
        :param mask: 掩码，形状为 (batch_size, seq_len)，值为 0 或 1
        :return: 带 mask 的 softmax 结果
        """
        dtype = logits.dtype
        device = logits.device
        min_dtype = torch.finfo(dtype).min
        #weights = torch.ones_like(logits, dtype=dtype)
        #weights = weights.to(device)
        weights = logits.clone()
        # 将 mask 中为 0 的位置对应的 logits 设置为负无穷
        weights.masked_fill_(~mask.bool(), min_dtype)
        #logits.masked_fill_(~mask.bool(), min_dtype)
        
        # 计算 softmax
        return F.softmax(weights, dim=-1)

    def not_cot_mask(self, cur_token_seq_list, prefix_len_list, aug=False):
        assert len(cur_token_seq_list) == len(prefix_len_list), \
            f"len(cur_token_seq_lit): {len(cur_token_seq_list)} != len(prefix_len_list): {len(prefix_len_list)}"
        emb_mask = []
        for i, prefix_len in enumerate(prefix_len_list):
            row_mask = []
            num_idx_list = []
            for j, token in enumerate(cur_token_seq_list[i]):
                if j < prefix_len:
                    row_mask.append(0)
                else:
                    rand = random.randint(1, 10)
                    if rand <= 1:
                        row_mask.append(0)
                    else:
                        row_mask.append(1)
            if(sum(row_mask) == 0):
                if len(num_idx_list) != 0:
                    for idx in num_idx_list:
                        row_mask[idx] = 1
                else:
                    row_mask[-1] = 1                    
            emb_mask.append(row_mask)
        return torch.BoolTensor(emb_mask)
