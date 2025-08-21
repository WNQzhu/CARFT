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
from pathlib import Path
import gc
import dataclasses
import os
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Type, Union

import json
import torch
import torch.nn as nn
import transformers
from accelerate import PartialState
from accelerate.utils import gather_object
from accelerate.utils import broadcast, gather_object, pad_across_processes
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    DataCollatorForLanguageModeling,
    DataCollatorWithFlattening,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import (
    EvalPrediction,
    seed_worker,
    EvalLoopOutput,
    has_length,   
    denumpify_detensorize
)
from transformers.integrations.deepspeed import deepspeed_init
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

from trl.models.utils import unwrap_model_for_generation
from transformers.utils import (
    is_peft_available, logging,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled, 
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
)

from trl.data_utils import is_conversational, maybe_apply_chat_template, maybe_convert_to_chatml, pack_examples
from trl.trainer.utils import (
    ConstantLengthDataset, generate_model_card, get_comet_experiment_url, peft_module_casting_to_bf16,
    batch_generation)

from my_sft_config import MySFTConfig

from transformers import (
    GenerationConfig,
)

from torch.utils.data import DataLoader
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from utils import floatify, extract_pred_values, compute_acc
from tok import batch_decode_to_tokens

if is_peft_available():
    import peft
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

logger = logging.get_logger(__name__)
    
def train_collate_fn(batch, args, tokenizer):
    max_input_length = max([len(item['input_ids']) for item in batch])
    max_target_length = max([len(item['labels']) for item in batch])
    input_ids  = []
    attention_mask  = []
    labels = []
    for item in batch:
        input_ids.append(item['input_ids'] + [tokenizer.pad_token_id]*(max_input_length - len(item['input_ids'])))
        attention_mask.append(item['attention_mask'] + [0]*(max_input_length - len(item['attention_mask'])))
        labels.append(item['labels'] + [-100]*(max_target_length - len(item['labels'])))
        
    new_batch = {
        'input_ids': torch.LongTensor(input_ids),
        'attention_mask': torch.BoolTensor(attention_mask),
        'labels': torch.LongTensor(labels)
    }
    return new_batch

def eval_collate_fn(batch, args, tokenizer):
    max_input_length = max([len(item['input_ids']) for item in batch])
    
    input_ids  = []
    attention_mask  = []

    item_id_list, question_list, \
        answer_value_list, answer_cot_list = [], [], [], []
    for item in batch:
        # print("item is: ", item)
        input_ids.append(item['input_ids'] + [tokenizer.pad_token_id]*(max_input_length - len(item['input_ids'])))
        attention_mask.append(item['attention_mask'] + [0]*(max_input_length - len(item['attention_mask'])))

        item_id_list.append(item['item_id'])
        question_list.append(item['question'])
        answer_value_list.append(item['answer_value'])
        answer_cot_list.append(item['answer_cot'])
        
    new_batch = {
        'input_ids': torch.LongTensor(input_ids),
        'attention_mask': torch.BoolTensor(attention_mask),
    }

    text_batch = {
        'item_id': item_id_list,
        'question': question_list,
        'answer_value': answer_value_list,
        'answer_cot':   answer_cot_list
    }    
    return new_batch, text_batch


class MySFTTrainer(Trainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) method.

    This class is a wrapper around the [`transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    from datasets import load_dataset
    from trl import SFTTrainer

    dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

    trainer = SFTTrainer(model="Qwen/Qwen2-0.5B-Instruct", train_dataset=dataset)
    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        args ([`MySFTConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator (`DataCollator`, *optional*):
            Function to use to form a batch from a list of elements of the prcessed `train_dataset` or `eval_dataset`.
            Will default to [`~transformers.default_data_collator`] if no `processing_class` is provided, an instance
            of [`~transformers.DataCollatorWithPadding`] otherwise if the processing_class is a feature extractor or
            tokenizer.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. SFT supports both [language modeling](#language-modeling) type and
            [prompt-completion](#prompt-completion) type. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            The trainer also supports processed datasets (tokenized) as long as they contain an `input_ids` field.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoTokenizer.from_pretrained`].
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*, defaults to `None`):
            A tuple containing the optimizer class and keyword arguments to use.
            Overrides `optim` and `optim_args` in `args`. Incompatible with the `optimizers` argument.

            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*, defaults to `None`):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        formatting_func (`Optional[Callable]`):
            Formatting function applied to the dataset before tokenization.
    """

    _tag_names = ["trl", "sft"]

    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[Union[MySFTConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[Type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Union[Callable[[dict], str], Callable[[dict], list[str]]]] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = MySFTConfig(f"{model_name}-SFT")
        elif isinstance(args, TrainingArguments) and not isinstance(args, MySFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            dict_args.pop("push_to_hub_token")
            args = MySFTConfig(**dict_args)

        # Model
        if args.model_init_kwargs is not None and not isinstance(model, str):
            warnings.warn(
                "You passed model_init_kwargs to the `MySFTConfig`, but your model is already instantiated. "
                "The `model_init_kwargs` will be ignored."
            )
        if isinstance(model, str):
            model = self._create_model_from_path(model, args)

        # PEFT configuration and model wrapping
        if peft_config is not None:
            model = self._prepare_peft_model(model, peft_config, args)

        # Handle the tokenizer
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if processing_class.pad_token is None:
                processing_class.pad_token = processing_class.eos_token  # required for padding when collating data

        # Dataset
        preprocess_dataset = args.dataset_kwargs is None or not args.dataset_kwargs.get("skip_prepare_dataset", False)
        if preprocess_dataset:
            train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, args.packing, formatting_func, "train"
            )
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, packing, formatting_func, "eval"
                    )

        # Data collator
        if args.padding_free:
            if data_collator is not None:
                raise ValueError("Passing a custom data collator is not supported when using padding-free.")
            if args.packing:
                warnings.warn(
                    "You are passing `packing=True` and `padding_free=True` which is not recommended. Please refer "
                    "to the documentation to understand why this is not recommended."
                )
            if model.config._attn_implementation != "flash_attention_2":
                warnings.warn(
                    "Padding-free training is enabled, but the attention implementation is not set to "
                    "'flash_attention_2'. Padding-free training flattens batches into a single sequence, and "
                    "'flash_attention_2' is the only known attention mechanism that reliably supports this. Using "
                    "other implementations may lead to unexpected behavior. To ensure compatibility, set "
                    "`attn_implementation='flash_attention_2'` in the model configuration, or verify that your "
                    "attention mechanism can handle flattened sequences."
                )
            if args.per_device_train_batch_size == 1:
                warnings.warn(
                    "You are using a per_device_train_batch_size of 1 with padding-free training. Using a batch size "
                    "of 1 anihilate the benefits of padding-free training. Please consider increasing the batch size "
                    "to at least 2."
                )
            data_collator = DataCollatorWithFlattening()

        if data_collator is None:
            data_collator = DataCollatorForLanguageModeling(tokenizer=processing_class, mlm=False)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Initialize the Trainer. Parent class will handle:
        # - DeepSpeed configuration (through create_accelerator_and_postprocess)
        # - FSDP setup
        # - Distributed training setup
        # - Optimizer and scheduler creation
        # Some arguments are only available for transformers>=4.47.0. Can be removed when the min version is bumped.
        super_init_kwargs = {}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super_init_kwargs["optimizer_cls_and_kwargs"] = optimizer_cls_and_kwargs
        else:
            if optimizer_cls_and_kwargs is not None:
                warnings.warn(
                    "The `optimizer_cls_and_kwargs` argument is only available for `transformers>=4.47.0`. "
                    "The default optimizer will be used. "
                    "Remove the `optimizer_cls_and_kwargs` or upgrade to `transformers>=4.47.0`."
                )
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **super_init_kwargs,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def _create_model_from_path(self, model_path: str, args: MySFTConfig) -> PreTrainedModel:
        """Creates a model from a path or model identifier."""
        model_init_kwargs = args.model_init_kwargs or {}
        # Handle torch dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `MySFTConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        if args.gradient_checkpointing:
            model_init_kwargs["use_cache"] = False

        # Create model
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        return model

    def _prepare_peft_model(self, model: PreTrainedModel, peft_config: Any, args: MySFTConfig) -> PreTrainedModel:
        """Prepares a model for PEFT training."""
        if not is_peft_available():
            raise ImportError("To use PeftModel, you need to install the `peft` library.")

        if not isinstance(peft_config, PeftConfig):
            raise ValueError(
                f"Expected PeftConfig object but got {type(peft_config)}. If you want to use the PeftModel, you need "
                "to pass a PeftConfig object to the SFTTrainer."
            )

        if isinstance(model, PeftModel):
            return model

        # Handle quantized models (QLoRA)
        is_qlora = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

        is_sharded_qlora = False
        if getattr(model, "is_loaded_in_4bit", False):
            # Check if model is sharded (FSDP/DS-Zero3)
            for _, param in model.named_parameters():
                if param.__class__.__name__ == "Params4bit":
                    is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                    break

        # Prepare model for kbit training if needed
        if is_qlora and not is_sharded_qlora:
            model = self._prepare_model_for_kbit_training(model, args)
            # Disable gradient checkpointing as it's handled by prepare_model_for_kbit_training
            args = dataclasses.replace(args, gradient_checkpointing=False)
        elif args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Create PEFT model
        if (
            version.parse(peft.__version__) >= version.parse("0.12")  # autocast_adapter_dtype introduced in 0.12
            and getattr(model, "is_loaded_in_4bit", False)
            and is_sharded_qlora
        ):
            model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
        else:
            model = get_peft_model(model, peft_config)

        # Handle bf16 casting for 4-bit models
        if args.bf16 and getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora:
            peft_module_casting_to_bf16(model)

        return model

    def _prepare_model_for_kbit_training(self, model: PreTrainedModel, args: MySFTConfig) -> PreTrainedModel:
        """Prepares a quantized model for kbit training."""
        prepare_model_kwargs = {
            "use_gradient_checkpointing": args.gradient_checkpointing,
            "gradient_checkpointing_kwargs": args.gradient_checkpointing_kwargs or {},
        }

        return prepare_model_for_kbit_training(model, **prepare_model_kwargs)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: MySFTConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        return model

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: MySFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Convert the dataset to an IterableDataset if it is a ConstantLengthDataset
        if isinstance(dataset, ConstantLengthDataset):
            return dataset

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                batched = isinstance(formatting_func(next(iter(dataset))), list)

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=batched, **map_kwargs)

            # If the dataset is prompt-completion, convert it to language modeling type
            first_example = next(iter(dataset))
            if "prompt" in first_example.keys() and "completion" in first_example.keys():
                key = "messages" if is_conversational(first_example) else "text"

                def concat_prompt_completion(example):
                    return {key: example["prompt"] + example["completion"]}

                dataset = dataset.map(concat_prompt_completion, remove_columns=["prompt", "completion"])

            if not is_processed:
                # Convert the dataset to ChatML if needed
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                column_names = next(iter(dataset)).keys()
                dataset = dataset.map(
                    maybe_convert_to_chatml,
                    remove_columns="conversations" if "conversations" in column_names else None,
                    **map_kwargs,
                )

                # Apply the chat template if needed
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
                column_names = next(iter(dataset)).keys()
                dataset = dataset.map(
                    maybe_apply_chat_template,
                    fn_kwargs={"tokenizer": processing_class},
                    remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                    **map_kwargs,
                )

                # Tokenize the dataset if needed
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize(example, processing_class, dataset_text_field):
                    processed = processing_class(text=example[dataset_text_field])
                    if (
                        processing_class.eos_token_id is not None
                        and processed["input_ids"][-1] != processing_class.eos_token_id
                    ):
                        processed["input_ids"] = processed["input_ids"] + [processing_class.eos_token_id]
                        processed["attention_mask"] = processed["attention_mask"] + [1]
                    return processed

                dataset = dataset.map(
                    tokenize,
                    fn_kwargs={"processing_class": processing_class, "dataset_text_field": args.dataset_text_field},
                    **map_kwargs,
                )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"
                dataset = dataset.select_columns("input_ids")
                dataset = dataset.map(
                    pack_examples, batched=True, fn_kwargs={"seq_length": args.max_length}, **map_kwargs
                )
            elif args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"

                def truncate(example, max_length):
                    return {key: example[key][:max_length] for key in ["input_ids", "attention_mask"]}

                dataset = dataset.map(
                    truncate,
                    fn_kwargs={"max_length": args.max_length},
                    **map_kwargs,
                )
                # print('ds[0] : ', dataset[0])

            # For Liger kernel, ensure only input_ids is present
            if args.use_liger_kernel:
                dataset = dataset.select_columns("input_ids")

        return dataset

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "eval" if self.control.should_evaluate else "train"
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                num_tokens_in_batch = (
                    self.accelerator.gather_for_metrics(torch.tensor(inputs["position_ids"].size(1))).sum().item()
                )
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="SFT",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
    # +     
    def get_train_dataloader(self) -> DataLoader:
        """ for cot sft"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        #if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        #    train_dataset = self._remove_unused_columns(train_dataset, description="training")
        #else:
        #    data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": partial(train_collate_fn,
                                  args=self.args, tokenizer=self.processing_class),
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        #data_collator = self.data_collator

        #if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
        #    eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        #else:
        #    data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": partial(eval_collate_fn,
                                  args=self.args,
                                  tokenizer=self.processing_class),
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader

        return self.accelerator.prepare(eval_dataloader)
    
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
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
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

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        #print("before compute loss")
                        #print('shape: ', inputs['input_ids'].shape,
                        #      inputs['attention_mask'].shape,
                        #      inputs['labels'].shape)
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
    
# my sft gen:
    def generate(self):
        num_ret = self.args.num_ret
        dataloader = self.get_eval_dataloader()
        self.gen_loop(dataloader,
                      description = "Generate analysis data",
                      num_ret = num_ret)
    
    def gen_loop(
        self,
        dataloader: DataLoader,
            description: str,
            num_ret = 1,
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
            top_k=0.0,
            top_p=1,
            do_sample=True,
            use_cache=True,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,    
        )        
        #+end_src 
        # Main evaluation loop
        ret_list = []
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
                queries = torch.repeat_interleave(
                    inputs["input_ids"].to(device),
                    repeats=num_ret,
                    dim=0)
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
                    completed_tensors = pad_across_processes(
                        query_responses,
                        dim=1,
                        pad_index=self.processing_class.pad_token_id,
                        pad_first=False)
                    model_input_ids = completed_tensors
                    model_attention_mask = (
                        completed_tensors != self.processing_class.pad_token_id)
                    model_output = self.model(
                        input_ids=model_input_ids,
                        attention_mask=model_attention_mask)
                    logitss = model_output.logits
                    print("model output logits:", logitss)
                    
                    print("qr logshape:", model_input_ids.shape, logitss.shape)

                    
                    all_query_responses = self.accelerator.gather(model_input_ids)
                    all_logitss = self.accelerator.gather(logitss)

                    print("all qr log shape:",
                          all_query_responses.shape,
                          all_logitss.shape)
                    
                    label_list = [
                        floatify(a)
                        for a in text_dict['answer_value']
                        for i in range(num_ret)
                    ]
                    all_label_list = gather_object(label_list)

                    for qr, logits, label in zip(all_query_responses,
                                          all_logitss, all_label_list):
                        qr_text = self.processing_class.decode(
                            qr,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True).strip()
                        extract_value_list  =  extract_pred_values(qr_text,
                                                                   self.args)
                        pred = extract_value_list[0]

                        token_seq_list = batch_decode_to_tokens(
                            self.processing_class, [qr])
                        token_seq  = token_seq_list[0]
                        print("in for logits: ", logits.shape)
                        print("in for qr.shape: ", qr.shape)
                        index_logits = torch.gather(input=logits[:-1, :],
                                   dim=-1,
                                   index=qr[1:].unsqueeze(-1)
                                   ).squeeze(-1)
                        instance = {
                            'qr': qr.cpu().numpy().tolist(),
                            'qr_text': qr_text,
                            'pred': pred,
                            'gt': label,
                            'index_logits': index_logits.to(
                                torch.float).cpu().numpy().tolist(),
                            'token_seq': token_seq
                        }
                        ret_list.append(instance)

        if self.accelerator.is_main_process:
            output_path = Path(self.args.gen_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(output_path), 'w', encoding='utf-8') as w:
                for instance in ret_list:
                    ss = json.dumps(instance, ensure_ascii=False)
                    w.write(ss + '\n')


