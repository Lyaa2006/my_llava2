#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
from pathlib import Path
import json
import math
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from CoIN.peft.tuners import CoINMOELoraLinear


def _resolve_json_path(path, base_dir=None):
    if path is None:
        return None
    candidate = os.path.expanduser(path)
    if os.path.isabs(candidate):
        return candidate
    search_roots = []
    if base_dir is not None:
        search_roots.append(base_dir)
    search_roots.append(str(PROJECT_ROOT))
    search_roots.append(str(PROJECT_ROOT.parent))
    for root in search_roots:
        resolved = os.path.abspath(os.path.join(root, candidate))
        if os.path.exists(resolved):
            return resolved
    return os.path.abspath(os.path.join(search_roots[0], candidate))


def _load_json_dict(path, label, base_dir=None):
    if path is None:
        return None
    resolved = _resolve_json_path(path, base_dir=base_dir)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{label} must be a JSON object: {resolved}")
    return data, resolved


def _read_adapter_config(checkpoint_dir):
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        return {}
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _infer_effective_num_task(checkpoint_dir, declared_num_task):
    declared_num_task = max(1, int(declared_num_task))
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return declared_num_task

    adapter_config = _read_adapter_config(checkpoint_dir)
    cur_task = adapter_config.get("cur_task")
    if cur_task is not None:
        try:
            return max(1, min(int(cur_task) + 1, declared_num_task))
        except (TypeError, ValueError):
            pass

    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
    match = re.search(r"Task(\d+)", checkpoint_name)
    if match:
        return max(1, min(int(match.group(1)), declared_num_task))

    return declared_num_task


def _extract_first_expert_num(model):
    for module in model.modules():
        if isinstance(module, CoINMOELoraLinear):
            return int(module.expert_num)
    return None


def _build_task_centered_prior(expert_num, active_experts, eval_task_id, sigma):
    prior = torch.zeros(expert_num, dtype=torch.float32)
    active_experts = max(1, min(int(active_experts), int(expert_num)))
    center = max(0, min(int(eval_task_id) - 1, active_experts - 1))
    if active_experts == 1:
        prior[0] = 1.0
        return prior

    sigma = max(float(sigma), 1e-3)
    positions = torch.arange(active_experts, dtype=torch.float32)
    logits = -((positions - float(center)) ** 2) / (2.0 * sigma * sigma)
    prior[:active_experts] = torch.softmax(logits, dim=0)
    return prior


def _apply_eval_band_prior(model, checkpoint_dir, band_prior_config_path, eval_task_id):
    if band_prior_config_path is None or eval_task_id is None:
        return None

    config_pair = _load_json_dict(
        band_prior_config_path,
        "band prior config",
        base_dir=str(PROJECT_ROOT),
    )
    if config_pair is None:
        return None
    band_config, band_config_path = config_pair
    if not bool(band_config.get("enable", True)):
        return None

    schedule_pair = _load_json_dict(
        band_config.get("stage1_band_schedule_path"),
        "stage1 band schedule",
        base_dir=os.path.dirname(band_config_path),
    )
    if schedule_pair is None:
        raise ValueError("Band prior config requires stage1_band_schedule_path.")
    schedule_config, schedule_path = schedule_pair
    alpha_by_layer = {
        int(layer): float(alpha)
        for layer, alpha in schedule_config.get("alpha_by_layer", {}).items()
    }
    if not alpha_by_layer:
        raise ValueError(f"Stage1 band schedule has no alpha_by_layer entries: {schedule_path}")

    expert_num = _extract_first_expert_num(model)
    if expert_num is None:
        raise RuntimeError("Could not locate CoINMOELoraLinear modules to inject eval band prior.")
    effective_num_task = _infer_effective_num_task(checkpoint_dir, expert_num)
    active_policy = str(band_config.get("active_expert_policy", "checkpoint_stage")).lower()
    if active_policy == "all":
        active_experts = expert_num
    else:
        active_experts = min(expert_num, effective_num_task)

    prior = _build_task_centered_prior(
        expert_num=expert_num,
        active_experts=active_experts,
        eval_task_id=int(eval_task_id),
        sigma=float(band_config.get("expert_prior_sigma", 0.75)),
    )
    global_lambda = max(0.0, min(1.0, float(band_config.get("global_lambda", 0.35))))

    applied_layers = []
    module_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, CoINMOELoraLinear):
            continue
        module.eval_band_prior = prior.clone()
        module.eval_band_mix = 0.0
        module.eval_band_layer = None
        layer_match = re.search(r"\.layers\.(\d+)\.", name)
        if layer_match is None:
            continue
        layer_number = int(layer_match.group(1)) + 1
        alpha = alpha_by_layer.get(layer_number, 0.0)
        module.eval_band_mix = global_lambda * float(alpha)
        module.eval_band_layer = layer_number
        if module.eval_band_mix > 0.0:
            applied_layers.append(layer_number)
        module_count += 1

    summary = {
        "config_path": band_config_path,
        "schedule_path": schedule_path,
        "eval_task_id": int(eval_task_id),
        "expert_num": expert_num,
        "active_experts": active_experts,
        "effective_num_task": effective_num_task,
        "global_lambda": global_lambda,
        "prior": [round(float(x), 6) for x in prior.tolist()],
        "band_layers": sorted(set(applied_layers)),
        "module_count": module_count,
    }
    model._eval_band_prior_summary = summary
    print(f"Applied eval band prior: {summary}")
    return summary


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    band_prior_config_path=None,
    eval_task_id=None,
):
    kwargs = {"device_map": device_map}
    checkpoint_dir = os.path.abspath(os.path.expanduser(model_path))

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower() or 'intern' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from CoIN.peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from CoIN.peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower() or 'intern' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor
        _apply_eval_band_prior(
            model,
            checkpoint_dir=checkpoint_dir,
            band_prior_config_path=band_prior_config_path,
            eval_task_id=eval_task_id,
        )

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
