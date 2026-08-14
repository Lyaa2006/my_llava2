import copy
import json
import logging
import os
import pathlib
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import transformers
from PIL import Image, ImageFile
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
)
from llava.mm_utils import tokenizer_image_token
from llava.model import *
from llava.train.hidesc_description_utils import build_description_key_mask
from llava.train.hidesc_trainer import HiDESCLoRATrainer
from llava.train.train import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    preprocess,
    preprocess_multimodal,
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
)

from peft import LoraConfig, TaskType, get_peft_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

local_rank = None


def rank0_print(*args):
    if local_rank in (None, -1, 0):
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    previous_task_model_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_vit_pos_embedding: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    memory_data_path: Optional[str] = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    description_prompt: str = field(
        default=(
            "Describe the image using visual evidence: objects, attributes, shapes, "
            "colors, textures, scene context, visible text, and spatial relations."
        )
    )
    description_cache_dir: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=512)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")
    group_by_modality_length: bool = field(default=False)

    enable_description_cl: bool = field(default=False)
    description_max_tokens: int = field(default=32)
    description_focus_weight: float = field(default=0.2)
    description_energy_weight: float = field(default=1e-4)
    description_energy_margin: float = field(default=30.0)

    # InternVL decoder layer bands from the HiDESC stage-1 schedule.
    b1_low_layer: int = field(default=12)
    b1_high_layer: int = field(default=14)
    b1_center_layer: int = field(default=14)
    b2_low_layer: int = field(default=27)
    b2_high_layer: int = field(default=29)
    b2_center_layer: int = field(default=28)

    align_band_eta: float = field(default=0.5)
    struct_band_eta: float = field(default=0.35)
    struct_band_energy_rho: float = field(default=1.0)
    loss_band_ema_gamma: float = field(default=0.9)
    loss_band_position_eps: float = field(default=0.05)
    align_loss_weight: float = field(default=0.01)
    standard_ce_weight: float = field(default=1.0)


def _load_reference_state(cache_dir, cache_key):
    if not cache_dir or not cache_key:
        return None
    path = os.path.join(cache_dir, f"{cache_key}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def _build_single_turn_prompt(message):
    conversation = conversation_lib.default_conversation.copy()
    conversation.messages = []
    conversation.append_message(conversation.roles[0], message)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def _build_description_text(text, data_args):
    source = [[{"from": "human", "value": f"{DEFAULT_IMAGE_TOKEN}\n{text}".strip()}]]
    source = preprocess_multimodal(source, data_args)
    return source[0][0]["value"]


def _expand2square(image, background_color):
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    result = Image.new(image.mode, (size, size), background_color)
    result.paste(image, ((size - width) // 2, (size - height) // 2))
    return result


class HiDESCLazySupervisedDataset(Dataset):
    def __init__(self, data_path, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.list_data_dict = json.load(open(data_path, "r"))

        if data_args.memory_data_path:
            memory = json.load(open(data_args.memory_data_path, "r"))
            offset = len(self.list_data_dict)
            for index, sample in enumerate(memory):
                sample["_description_cache_key"] = f"memory_{index:08d}"
            self.list_data_dict.extend(memory)
        for index, sample in enumerate(self.list_data_dict):
            sample.setdefault("_description_cache_key", f"train_{index:08d}")

        rank0_print("Formatting inputs...Skip in lazy mode")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        lengths = []
        for sample in self.list_data_dict:
            length = sum(len(item["value"].split()) for item in sample["conversations"])
            lengths.append(length if "image" in sample else -length)
        return lengths

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[index]
        has_image = "image" in sample
        sources = copy.deepcopy([sample["conversations"]])
        data = {}

        if has_image:
            image_file = sample["image"]
            if isinstance(image_file, list):
                image_file = image_file[0]
            image = Image.open(
                os.path.join(self.data_args.image_folder, image_file)
            ).convert("RGB")
            if self.data_args.image_aspect_ratio == "pad":
                image = _expand2square(
                    image,
                    tuple(int(value * 255) for value in self.data_args.image_processor.image_mean),
                )
            data["image"] = self.data_args.image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            sources = preprocess_multimodal(sources, self.data_args)
        else:
            sources = copy.deepcopy(sources)

        processed = preprocess(sources, self.tokenizer, has_image=has_image)
        data["input_ids"] = processed["input_ids"][0]
        data["labels"] = processed["labels"][0]

        if not has_image and self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

        if has_image and self.data_args.use_description_data:
            description_text = _build_description_text(
                self.data_args.description_prompt, self.data_args
            )
            description_prompt = _build_single_turn_prompt(description_text)
            data["description_input_ids"] = tokenizer_image_token(
                description_prompt, self.tokenizer, return_tensors="pt"
            )
            cache_key = sample["_description_cache_key"]
            data["reference_description_states"] = _load_reference_state(
                self.data_args.description_cache_dir, cache_key
            )
        return data


class HiDESCDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )[:, : self.tokenizer.model_max_length]
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )[:, : self.tokenizer.model_max_length]
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        if "image" in instances[0]:
            images = [item["image"] for item in instances]
            batch["images"] = (
                torch.stack(images)
                if all(image.shape == images[0].shape for image in images)
                else images
            )

        if "description_input_ids" in instances[0]:
            description_ids = torch.nn.utils.rnn.pad_sequence(
                [item["description_input_ids"] for item in instances],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )[:, : self.tokenizer.model_max_length]
            description_mask = description_ids.ne(self.tokenizer.pad_token_id)
            batch["description_input_ids"] = description_ids
            batch["description_attention_mask"] = description_mask
            batch["description_key_mask"] = build_description_key_mask(
                description_ids, self.tokenizer, description_mask
            )

            references = [item.get("reference_description_states") for item in instances]
            available = torch.tensor(
                [reference is not None for reference in references], dtype=torch.bool
            )
            if any(reference is not None for reference in references):
                hidden_size = next(
                    reference.shape[-1]
                    for reference in references
                    if reference is not None
                )
                max_length = max(
                    reference.shape[0] if reference is not None else 0
                    for reference in references
                )
                reference_states = torch.zeros(
                    len(references), max_length, hidden_size, dtype=torch.float32
                )
                reference_mask = torch.zeros(
                    len(references), max_length, dtype=torch.bool
                )
                for index, reference in enumerate(references):
                    if reference is None:
                        continue
                    length = reference.shape[0]
                    reference_states[index, :length] = reference.float()
                    reference_mask[index, :length] = True
                batch["reference_description_states"] = reference_states
                batch["reference_description_mask"] = reference_mask
                batch["reference_description_available"] = available
        return batch


def make_hidesc_data_module(tokenizer, data_args):
    data_args.use_description_data = data_args.is_multimodal and bool(
        data_args.enable_description_cl
    )
    dataset = HiDESCLazySupervisedDataset(
        data_args.data_path, tokenizer, data_args
    )
    return {
        "train_dataset": dataset,
        "eval_dataset": None,
        "data_collator": HiDESCDataCollator(tokenizer),
    }


def _load_previous_lora(model, checkpoint_dir):
    if not checkpoint_dir:
        return
    from peft.utils import WEIGHTS_NAME, set_peft_model_state_dict

    non_lora_path = os.path.join(checkpoint_dir, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        state = torch.load(non_lora_path, map_location="cpu")
        state = {
            key[11:] if key.startswith("base_model.") else key: value
            for key, value in state.items()
        }
        model.base_model.model.load_state_dict(state, strict=False)
    adapter_path = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if os.path.exists(adapter_path):
        adapter_state = torch.load(adapter_path, map_location="cpu")
        set_peft_model_state_dict(model, adapter_state, adapter_name="default")
    rank0_print(f"Loaded previous LoRA checkpoint from {checkpoint_dir}")


def train(attn_implementation=None):
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else torch.bfloat16
        if training_args.bf16
        else torch.float32
    )

    load_args = {}
    if training_args.bits in (4, 8):
        from transformers import BitsAndBytesConfig

        load_args = {
            "device_map": {"": training_args.device},
            "load_in_4bit": training_args.bits == 4,
            "load_in_8bit": training_args.bits == 8,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            ),
        }

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        **load_args,
    )
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in (4, 8):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        rank0_print("Added ordinary PEFT LoRA adapters (no MoE/prototype routing).")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if model_args.version == "v0" and tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            {"pad_token": "[PAD]"}, tokenizer, model
        )
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args, fsdp=training_args.fsdp
        )
        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer)

    data_args.enable_description_cl = training_args.enable_description_cl
    _load_previous_lora(model, model_args.previous_task_model_path)
    data_module = make_hidesc_data_module(tokenizer, data_args)
    trainer = HiDESCLoRATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    rank0_print(
        "InternVL HiDESC bands: "
        f"B1=[{training_args.b1_low_layer},{training_args.b1_high_layer}] "
        f"(center={training_args.b1_center_layer}), "
        f"B2=[{training_args.b2_low_layer},{training_args.b2_high_layer}] "
        f"(center={training_args.b2_center_layer})"
    )
    rank0_print(
        "Trainable parameters:",
        [name for name, parameter in model.named_parameters() if parameter.requires_grad],
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank in (0, -1):
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer, training_args.output_dir)

    if training_args.local_rank in (0, -1):
        subprocess.run(
            f"find {training_args.output_dir} -maxdepth 1 -type d "
            "-name 'checkpoint-*' -exec rm -rf {} +",
            shell=True,
            check=False,
        )


if __name__ == "__main__":
    train()
