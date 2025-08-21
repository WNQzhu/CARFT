# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from transformers import TrainingArguments
# from trl.trainer.utils import OnPolicyConfig


@dataclass
class MyReftConfig(TrainingArguments):
    # (OnPolicyConfig):
    r"""
    Configuration class for the [`PPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[:-3]`):
            Name of this experiment.
        reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the reward model.
        model_adapter_name (`str` or `None`, *optional*, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str` or `None`, *optional*, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        num_ppo_epochs (`int`, *optional*, defaults to `4`):
            Number of epochs to train.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whether to whiten the rewards.
        kl_coef (`float`, *optional*, defaults to `0.05`):
            KL coefficient.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Clip range.
        vf_coef (`float`, *optional*, defaults to `0.1`):
            Value function coefficient.
        cliprange_value (`float`, *optional*, defaults to `0.2`):
            Clip range for the value function.
        gamma (`float`, *optional*, defaults to `1.0`):
            Discount factor.
        lam (`float`, *optional*, defaults to `0.95`):
            Lambda value for GAE.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation.
    """

    exp_name: str = field(
        default=os.path.basename(__file__)[:-3],
        metadata={"help": "Name of this experiment."},
    )
    reward_model_path: str = field(
        default="EleutherAI/pythia-160m",
        metadata={"help": "Path to the reward model."},
    )
    model_adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the train target PEFT adapter, when using LoRA with multiple adapters."},
    )
    ref_adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the reference PEFT adapter, when using LoRA with multiple adapters."},
    )
    num_ppo_epochs: int = field(
        default=2,
        metadata={"help": "Number of epochs to train."},
    )
    whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the rewards."},
    )
    kl_coef: float = field(
        default=0.05,
        metadata={"help": "KL coefficient."},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    vf_coef: float = field(
        default=5,
        metadata={"help": "Value function coefficient."},
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "Clip range for the value function."},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor."},
    )
    lam: float = field(
        default=0.95,
        metadata={"help": "Lambda value for GAE."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation."
        },
    )
    # + 
    train_path: str = field(
        default="text",
        metadata={"help": "train path."},
    )
    
    eval_path: str = field(
        default="text",
        metadata={"help": "eval path."},
    )

    dataset_name: str = field(
        default="text",
        metadata={"help": "dataset name."},
    )
    
    model_name_or_path: str = field(
        default="text",
        metadata={"help": "model_name_or_path."},
    )
    model_name: str = field(
        default="text",
        metadata={"help": "model_name."},
    )

    cot: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "cot prefix"
        },
    )
    engine: str = field(
        default='nl',
        metadata={
            "help": "engine"
        },
    )
    max_input_length: int = field(
        default=300,
        metadata={
            "help": "engine"
        },
    )
    max_gen_length: int = field(
        default=700,
        metadata={
            "help": "engine"
        },
    )
    dataset_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Dictionary of optional keyword arguments for the dataset preparation. The only supported key is "
            "`skip_prepare_dataset`."
        },
    )
    packing: bool = field(
        default=False,
        metadata={
            "help": "Whether to pack multiple sequences into a fixed-length format. Uses `max_length` to define "
            "sequence length."
        },
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from"
            "the right. If `None`, no truncation is applied. When packing is enabled, this value sets the "
            "sequence length."
        },
    )
    # Parameters that control the model
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `SFTTrainer` is provided as a string."
        },
    )
    adv_whitening: str = field(
        default='global',
        metadata={
            "help": "engine"
        },
    )
    stop_token: Optional[Literal["eos"]] = field(
        default=None,
        metadata={
            "help": "Specifies the stop token to use for text generation. This parameter is mutually exclusive with "
            "`stop_token_id`."
        },
    )
    stop_token_id: Optional[int] = field(
        default=None,
        metadata={
            "help": "Specifies the ID of the stop token to use for text generation. If `None`, no stop token ID is "
            "applied, unless `stop_token` is specified. This parameter is mutually exclusive with `stop_token`."
        },
    )
    total_episodes: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of episodes in the dataset."},
    )
    world_size: int = field(
        default=2,
        metadata={
            "help": "world size"
        },
    )
    num_mini_batches: int = field(
        default=1,
        metadata={"help": "Number of minibatches to split a batch into."},
    )
