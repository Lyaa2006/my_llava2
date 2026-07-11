import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager, nullcontext

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)
from typing import List, Optional

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    @contextmanager
    def _temporary_anchor_update(self, model, enabled: bool):
        wrapped = getattr(model, "module", model)
        had_attr = hasattr(wrapped, "disable_anchor_update")
        prev_value = getattr(wrapped, "disable_anchor_update", None)
        try:
            wrapped.disable_anchor_update = not bool(enabled)
            yield
        finally:
            if had_attr:
                wrapped.disable_anchor_update = prev_value
            else:
                try:
                    delattr(wrapped, "disable_anchor_update")
                except AttributeError:
                    pass

    @contextmanager
    def _temporary_modules_eval(self, model, module_types):
        module_types = tuple(module_types)
        toggled = []
        for module in model.modules():
            if isinstance(module, module_types):
                toggled.append((module, module.training))
                module.training = False
        try:
            yield
        finally:
            for module, prev_training in toggled:
                module.training = prev_training

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _pad_description_sequences(self, sequences, dtype=None):
        max_len = max(seq.shape[0] for seq in sequences)
        hidden_size = sequences[0].shape[-1]
        device = sequences[0].device
        if dtype is None:
            dtype = sequences[0].dtype
        padded = torch.zeros((len(sequences), max_len, hidden_size), dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)
        for idx, seq in enumerate(sequences):
            seq_len = seq.shape[0]
            padded[idx, :seq_len] = seq.to(dtype=dtype)
            mask[idx, :seq_len] = True
        return padded, mask

    def _pad_description_key_masks(self, masks, max_len, device):
        if masks is None:
            return None
        padded = torch.zeros((len(masks), max_len), dtype=torch.bool, device=device)
        for idx, mask in enumerate(masks):
            seq_len = min(mask.shape[0], max_len)
            padded[idx, :seq_len] = mask[:seq_len].to(device=device, dtype=torch.bool)
        return padded

    def _masked_mean_pool(self, hidden_states, mask):
        if hidden_states.shape[1] != mask.shape[1]:
            shared_seq_len = min(hidden_states.shape[1], mask.shape[1])
            padding_side = getattr(self.model.config, "tokenizer_padding_side", "right")
            if padding_side == "left":
                hidden_states = hidden_states[:, -shared_seq_len:]
                mask = mask[:, -shared_seq_len:]
            else:
                hidden_states = hidden_states[:, :shared_seq_len]
                mask = mask[:, :shared_seq_len]
        mask = mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _unwrap_model(self, model):
        return getattr(model, "module", model)

    def _get_transformer_layers(self, model):
        wrapped = self._unwrap_model(model)
        candidates = [
            wrapped,
            getattr(wrapped, "model", None),
            getattr(getattr(wrapped, "base_model", None), "model", None),
            getattr(getattr(getattr(wrapped, "base_model", None), "model", None), "model", None),
        ]
        if hasattr(wrapped, "get_model"):
            candidates.append(wrapped.get_model())
        for candidate in candidates:
            if candidate is None:
                continue
            if hasattr(candidate, "layers"):
                return candidate.layers
            inner_model = getattr(candidate, "model", None)
            if inner_model is not None and hasattr(inner_model, "layers"):
                return inner_model.layers
        raise ValueError("Could not locate transformer layers for stage hidden-state capture.")

    def _resolve_stage_layers(self, model):
        layers = self._get_transformer_layers(model)
        num_layers = len(layers)

        def resolve(layer_idx, fallback=None):
            if layer_idx is None:
                layer_idx = fallback
            if layer_idx is None:
                raise ValueError("Stage layer index is required.")
            layer_idx = int(layer_idx)
            if layer_idx < 0:
                layer_idx += num_layers
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(f"Resolved stage layer {layer_idx} is outside [0, {num_layers}).")
            return layer_idx

        fallback_layer = getattr(self.args, "description_hidden_layer", -2)
        return {
            "early": resolve(getattr(self.args, "description_early_layer", None), fallback=fallback_layer),
        }

    def _capture_stage_hidden_states(self, model, target_layers):
        layer_modules = self._get_transformer_layers(model)
        captured = {}
        hooks = []

        for stage_name, layer_idx in target_layers.items():
            def hook_fn(_module, _inputs, output, current_stage=stage_name):
                if isinstance(output, tuple):
                    captured[current_stage] = output[0]
                else:
                    captured[current_stage] = output

            hooks.append(layer_modules[layer_idx].register_forward_hook(hook_fn))

        return captured, hooks

    def _run_with_stage_hooks(
        self,
        model,
        forward_kwargs,
        target_layers,
        *,
        no_grad=False,
        disable_anchor_update=False,
        disable_dropout=False,
    ):
        captured, hooks = self._capture_stage_hidden_states(model, target_layers)
        grad_context = torch.no_grad if no_grad else nullcontext
        dropout_context = (
            self._temporary_modules_eval(model, (nn.Dropout,))
            if disable_dropout
            else nullcontext()
        )
        anchor_context = (
            self._temporary_anchor_update(model, enabled=False)
            if disable_anchor_update
            else nullcontext()
        )
        try:
            with grad_context(), dropout_context, anchor_context:
                outputs = model(**forward_kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        missing = [stage for stage in target_layers if stage not in captured]
        if missing:
            raise RuntimeError(f"Failed to capture hidden states for stages: {missing}")
        return outputs, captured

    def _slice_hidden_with_mask(self, hidden_states, mask, max_tokens=None):
        if hidden_states.shape[1] != mask.shape[1]:
            shared_seq_len = min(hidden_states.shape[1], mask.shape[1])
            padding_side = getattr(self.model.config, "tokenizer_padding_side", "right")
            if padding_side == "left":
                hidden_states = hidden_states[:, -shared_seq_len:]
                mask = mask[:, -shared_seq_len:]
            else:
                hidden_states = hidden_states[:, :shared_seq_len]
                mask = mask[:, :shared_seq_len]
        if max_tokens is not None and hidden_states.shape[1] > max_tokens:
            padding_side = getattr(self.model.config, "tokenizer_padding_side", "right")
            if padding_side == "left":
                hidden_states = hidden_states[:, -max_tokens:]
                mask = mask[:, -max_tokens:]
            else:
                hidden_states = hidden_states[:, :max_tokens]
                mask = mask[:, :max_tokens]
        return hidden_states, mask.bool()

    def _build_effective_key_mask(self, valid_mask, key_mask):
        if key_mask is None:
            return valid_mask
        effective_key_mask = valid_mask & key_mask
        has_key = effective_key_mask.any(dim=1, keepdim=True)
        return torch.where(has_key, effective_key_mask, valid_mask)

    def _pool_description_stage(self, hidden_states, attention_mask, key_mask):
        hidden_states, valid_mask = self._slice_hidden_with_mask(
            hidden_states,
            attention_mask,
            max_tokens=getattr(self.args, "description_max_tokens", None),
        )
        if key_mask is not None:
            _, key_mask = self._slice_hidden_with_mask(
                hidden_states,
                key_mask,
                max_tokens=getattr(self.args, "description_max_tokens", None),
            )
        effective_key_mask = self._build_effective_key_mask(valid_mask, key_mask)
        pooled = self._masked_mean_pool(hidden_states.float(), effective_key_mask)
        return pooled, hidden_states, valid_mask, effective_key_mask

    def _align_description_reference(self, current_hidden, current_mask, reference_hidden, reference_mask, key_mask):
        shared_seq_len = min(current_hidden.shape[1], reference_hidden.shape[1], current_mask.shape[1], reference_mask.shape[1])
        padding_side = getattr(self.model.config, "tokenizer_padding_side", "right")
        if padding_side == "left":
            current_hidden = current_hidden[:, -shared_seq_len:]
            current_mask = current_mask[:, -shared_seq_len:]
            reference_hidden = reference_hidden[:, -shared_seq_len:]
            reference_mask = reference_mask[:, -shared_seq_len:]
            if key_mask is not None:
                key_mask = key_mask[:, -shared_seq_len:]
        else:
            current_hidden = current_hidden[:, :shared_seq_len]
            current_mask = current_mask[:, :shared_seq_len]
            reference_hidden = reference_hidden[:, :shared_seq_len]
            reference_mask = reference_mask[:, :shared_seq_len]
            if key_mask is not None:
                key_mask = key_mask[:, :shared_seq_len]
        valid_mask = current_mask.bool() & reference_mask.bool()
        effective_key_mask = self._build_effective_key_mask(valid_mask, key_mask)
        return current_hidden, reference_hidden, valid_mask, effective_key_mask

    def _compute_stage_delta(self, current_states, reference_states):
        return current_states.float() - reference_states.detach().float()

    def _compute_focus_loss(self, current_hidden, reference_hidden, valid_mask, key_mask):
        delta = current_hidden.float() - reference_hidden.detach().float()
        token_energy = delta.norm(dim=-1)
        key_mask = valid_mask & key_mask
        has_key = key_mask.any(dim=1, keepdim=True)
        key_mask = torch.where(has_key, key_mask, valid_mask)
        non_key_mask = valid_mask & (~key_mask)

        key_denom = key_mask.sum(dim=1).clamp_min(1).float()
        non_key_denom = non_key_mask.sum(dim=1).clamp_min(1).float()
        key_energy = (token_energy * key_mask.float()).sum(dim=1) / key_denom
        non_key_energy = (token_energy * non_key_mask.float()).sum(dim=1) / non_key_denom

        alpha = float(getattr(self.args, "description_focus_alpha", 0.4))
        focus_loss = F.relu(non_key_energy - alpha * key_energy).mean()
        return focus_loss, key_energy.mean(), non_key_energy.mean()

    def _compute_energy_loss(self, delta_desc, margin):
        desc_norm = delta_desc.float().norm(dim=-1)
        energy_loss = F.relu(desc_norm - float(margin)).pow(2).mean()
        return energy_loss, desc_norm.mean()

    def _get_aux_loss_scale(self):
        warmup_ratio = float(getattr(self.args, "description_loss_warmup_ratio", 0.0) or 0.0)
        if warmup_ratio <= 0.0:
            return 1.0
        max_steps = int(getattr(self.state, "max_steps", 0) or getattr(self.args, "max_steps", 0) or 0)
        if max_steps <= 0:
            return 1.0
        warmup_steps = max(1, int(max_steps * warmup_ratio))
        return min(1.0, float(self.state.global_step + 1) / float(warmup_steps))

    def _maybe_log_description_losses(
        self,
        total_loss,
        standard_loss,
        focus_loss,
        energy_loss,
        aux_scale,
        delta_desc_early_norm,
        key_energy,
        non_key_energy,
    ):
        logging_steps = max(1, int(getattr(self.args, "logging_steps", 1) or 1))
        global_step = int(getattr(self.state, "global_step", 0))
        if global_step % logging_steps != 0:
            return
        if getattr(self, "_last_description_loss_log_step", None) == global_step:
            return
        self._last_description_loss_log_step = global_step

        focus_weight = float(getattr(self.args, "description_focus_weight", 0.0))
        energy_weight = float(getattr(self.args, "description_energy_weight", 0.0))
        ce_weight = float(getattr(self.args, "standard_ce_weight", 1.0))

        self.log({
            "loss/total": total_loss.detach().float().item(),
            "loss/standard_ce": standard_loss.detach().float().item(),
            "loss/description_focus": focus_loss.detach().float().item(),
            "loss/description_energy": energy_loss.detach().float().item(),
            "loss_weighted/standard_ce": (standard_loss.detach().float() * ce_weight).item(),
            "loss_weighted/description_focus": (
                focus_loss.detach().float() * focus_weight * aux_scale
            ).item(),
            "loss_weighted/description_energy": (
                energy_loss.detach().float() * energy_weight * aux_scale
            ).item(),
            "description/aux_scale": float(aux_scale),
            "description/delta_desc_early_norm": delta_desc_early_norm.detach().float().item(),
            "description/key_energy": key_energy.detach().float().item(),
            "description/non_key_energy": non_key_energy.detach().float().item(),
            "config/description_focus_weight": focus_weight,
            "config/description_energy_weight": energy_weight,
            "config/standard_ce_weight": ce_weight,
        })

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not getattr(self.args, "enable_description_cl", False) or "description_input_ids" not in inputs:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        if "reference_description_states" not in inputs or "reference_description_mask" not in inputs:
            raise ValueError("Description continual-learning training requires cached reference description states.")

        stage_layers = self._resolve_stage_layers(model)
        standard_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            images=inputs.get("images"),
            return_dict=True,
            use_cache=False,
        )
        standard_loss = standard_outputs.loss

        desc_forward_kwargs = {
            "input_ids": inputs["description_input_ids"],
            "attention_mask": inputs["description_attention_mask"],
            "images": inputs.get("images"),
            "return_dict": True,
            "use_cache": False,
        }

        _, desc_cur_hidden = self._run_with_stage_hooks(
            model,
            desc_forward_kwargs,
            stage_layers,
            no_grad=False,
            disable_anchor_update=True,
            disable_dropout=False,
        )

        description_key_mask = inputs.get("description_key_mask")
        _, desc_cur_early_hidden, desc_cur_mask, desc_key_early = self._pool_description_stage(
            desc_cur_hidden["early"],
            inputs["description_attention_mask"],
            description_key_mask,
        )
        reference_description_states = inputs["reference_description_states"].to(desc_cur_early_hidden.device)
        reference_description_mask = inputs["reference_description_mask"].to(desc_cur_early_hidden.device)
        desc_cur_early_hidden, desc_ref_early_hidden, desc_valid_early, desc_key_early = self._align_description_reference(
            desc_cur_early_hidden.float(),
            desc_cur_mask,
            reference_description_states.float(),
            reference_description_mask,
            desc_key_early,
        )
        desc_cur_early = self._masked_mean_pool(desc_cur_early_hidden, desc_key_early)
        desc_ref_early = self._masked_mean_pool(desc_ref_early_hidden, desc_key_early)
        delta_desc_early = self._compute_stage_delta(desc_cur_early, desc_ref_early)
        focus_loss, key_energy, non_key_energy = self._compute_focus_loss(
            desc_cur_early_hidden,
            desc_ref_early_hidden,
            desc_valid_early,
            desc_key_early,
        )
        energy_loss, delta_desc_early_norm = self._compute_energy_loss(
            delta_desc_early,
            getattr(self.args, "description_energy_margin", 1.0),
        )

        aux_scale = self._get_aux_loss_scale()
        total_loss = self.args.standard_ce_weight * standard_loss
        total_loss = total_loss + aux_scale * (
            self.args.description_focus_weight * focus_loss
            + self.args.description_energy_weight * energy_loss
        )

        self._maybe_log_description_losses(
            total_loss=total_loss,
            standard_loss=standard_loss,
            focus_loss=focus_loss,
            energy_loss=energy_loss,
            aux_scale=aux_scale,
            delta_desc_early_norm=delta_desc_early_norm,
            key_energy=key_energy,
            non_key_energy=non_key_energy,
        )

        if return_outputs:
            return total_loss, {
                "standard_outputs": standard_outputs,
                "standard_loss": standard_loss.detach(),
                "description_focus_loss": focus_loss.detach(),
                "description_energy_loss": energy_loss.detach(),
            }
        return total_loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
