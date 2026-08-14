import logging
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F

from llava.train.description_utils import select_expanded_description_tokens
from llava.train.llava_trainer import LLaVATrainer


class HiDESCLLaVATrainer(LLaVATrainer):
    """MoELoRA trainer with HiDESC train-only losses."""

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
        raise ValueError("Could not locate InternVL decoder layers for HiDESC loss bands.")

    def _resolve_loss_band(self, model, low_layer, high_layer, name):
        num_layers = len(self._get_transformer_layers(model))
        low_layer = int(low_layer)
        high_layer = int(high_layer)
        if low_layer < 1 or high_layer > num_layers or low_layer > high_layer:
            raise ValueError(
                f"{name} must be within InternVL decoder layers [1, {num_layers}], "
                f"got [{low_layer}, {high_layer}]."
            )
        return list(range(low_layer, high_layer + 1))

    @contextmanager
    def _temporary_anchor_update(self, model, enabled: bool):
        # MoELoRA has no HiDESC prototype state; this is only a compatibility hook.
        yield

    def _prepare_multimodal_inputs(self, model, input_key, inputs, labels=None):
        wrapped = self._unwrap_model(model)
        attention_key = "attention_mask" if input_key == "input_ids" else f"{input_key}_attention_mask"
        return wrapped.prepare_inputs_labels_for_multimodal(
            inputs[input_key],
            inputs[attention_key],
            None,
            labels,
            inputs.get("images"),
            return_token_masks=True,
        )

    def _masked_mean_pool(self, hidden_states, mask):
        shared_len = min(hidden_states.shape[1], mask.shape[1])
        hidden_states = hidden_states[:, :shared_len]
        mask = mask[:, :shared_len].unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _compute_align_loss(self, hidden_states, image_mask, text_mask):
        zero = hidden_states.new_zeros(())
        if image_mask is None or text_mask is None:
            return zero, zero, zero
        shared_len = min(hidden_states.shape[1], image_mask.shape[1], text_mask.shape[1])
        hidden_states = hidden_states[:, :shared_len].float()
        image_mask = image_mask[:, :shared_len].bool()
        text_mask = text_mask[:, :shared_len].bool()
        valid = image_mask.any(dim=1) & text_mask.any(dim=1)
        if not torch.any(valid):
            return zero, zero, zero
        image = F.normalize(self._masked_mean_pool(hidden_states, image_mask)[valid], dim=-1)
        text = F.normalize(self._masked_mean_pool(hidden_states, text_mask)[valid], dim=-1)
        cosine = (image * text).sum(dim=-1).clamp(-1.0, 1.0)
        return (1.0 - cosine).mean(), valid.float().sum(), cosine.mean()

    def _band_weights(self, losses, layers, state_name, eta):
        values = torch.stack([loss.detach().float() for loss in losses])
        state = getattr(self, state_name, {})
        gamma = float(getattr(self.args, "loss_band_ema_gamma", 0.9))
        updated = []
        for layer, value in zip(layers, values):
            previous = state.get(layer, float(value.item()))
            current = gamma * previous + (1.0 - gamma) * float(value.item())
            state[layer] = current
            updated.append(current)
        setattr(self, state_name, state)
        ema = values.new_tensor(updated)
        normalized = (ema - ema.mean()) / ema.std(unbiased=False).clamp_min(1e-6)
        if len(layers) == 1:
            prior = values.new_ones(1)
        else:
            ramp = torch.linspace(0.0, 1.0, len(layers), device=values.device)
            eps = float(getattr(self.args, "loss_band_position_eps", 0.05))
            prior = eps + (1.0 - eps) * ramp
        return F.softmax(torch.log(prior.clamp_min(1e-6)) + float(eta) * normalized, dim=0)

    def _extract_description_states(self, model, inputs, layers):
        (
            _,
            attention_mask,
            past_key_values,
            inputs_embeds,
            _,
            _,
            text_mask,
        ) = self._prepare_multimodal_inputs(model, "description_input_ids", inputs)
        outputs = model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        sequences = {}
        key_masks = {}
        for layer in layers:
            selected, selected_keys = select_expanded_description_tokens(
                outputs.hidden_states[layer],
                text_mask,
                inputs["description_input_ids"],
                inputs["description_attention_mask"],
                inputs.get("description_key_mask"),
                int(self.args.description_max_tokens),
            )
            sequences[layer] = selected
            key_masks[layer] = selected_keys
        return sequences, key_masks

    def _pad_sequences(self, sequences, dtype=None):
        max_len = max(item.shape[0] for item in sequences)
        hidden = sequences[0].shape[-1]
        dtype = dtype or sequences[0].dtype
        device = sequences[0].device
        padded = torch.zeros(len(sequences), max_len, hidden, dtype=dtype, device=device)
        mask = torch.zeros(len(sequences), max_len, dtype=torch.bool, device=device)
        for idx, sequence in enumerate(sequences):
            padded[idx, : sequence.shape[0]] = sequence.to(dtype=dtype)
            mask[idx, : sequence.shape[0]] = True
        return padded, mask

    def _pad_key_masks(self, masks, max_len, device):
        if masks is None:
            return None
        padded = torch.zeros(len(masks), max_len, dtype=torch.bool, device=device)
        for idx, mask in enumerate(masks):
            if mask is not None:
                padded[idx, : min(max_len, mask.shape[0])] = mask[:max_len].to(device)
        return padded

    def _rms_normalize(self, hidden):
        rms = hidden.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()
        return hidden.float() / rms

    def _focus_loss(self, current, reference, valid, key_mask):
        change = (self._rms_normalize(current) - self._rms_normalize(reference.detach())).norm(dim=-1)
        change = change * valid.float()
        effective_key = valid & key_mask if key_mask is not None else valid
        if key_mask is not None:
            has_key = effective_key.any(dim=1, keepdim=True)
            effective_key = torch.where(has_key, effective_key, valid)
        total = change.sum(dim=1)
        active = total > 1e-6
        mass = (change / total.clamp_min(1e-6).unsqueeze(-1) * effective_key.float()).sum(dim=1)
        per_sample = -torch.log(mass.clamp_min(1e-6))
        return per_sample[active].mean() if torch.any(active) else change.new_zeros(())

    def _energy_loss(self, current, reference, valid):
        change = (self._rms_normalize(current) - self._rms_normalize(reference.detach())).norm(dim=-1)
        counts = valid.sum(dim=1)
        active = counts > 0
        mean_change = (change * valid.float()).sum(dim=1) / counts.clamp_min(1).float()
        margin = float(getattr(self.args, "description_energy_margin", 5.0))
        loss = F.relu(mean_change - margin).pow(2)
        return loss[active].mean() if torch.any(active) else change.new_zeros(())

    def _compute_struct_loss(self, states, key_masks, reference, reference_mask, available):
        layers = sorted(states)
        current, current_mask = self._pad_sequences(states[layers[0]], dtype=reference.dtype)
        reference = reference.to(current.device)
        reference_mask = reference_mask.to(current.device)
        shared_len = min(current.shape[1], reference.shape[1])
        valid = current_mask[:, :shared_len] & reference_mask[:, :shared_len]
        valid &= available.to(current.device).bool()[:, None]
        key_mask = self._pad_key_masks(key_masks[layers[0]], shared_len, current.device)
        focus_losses = []
        energy_losses = []
        for layer in layers:
            layer_states, _ = self._pad_sequences(states[layer], dtype=reference.dtype)
            layer_states = layer_states[:, :shared_len]
            focus_losses.append(self._focus_loss(layer_states, reference[:, :shared_len], valid, key_mask))
            energy_losses.append(self._energy_loss(layer_states, reference[:, :shared_len], valid))
        difficulty = [
            focus + float(getattr(self.args, "struct_band_energy_rho", 1.0)) * energy
            for focus, energy in zip(focus_losses, energy_losses)
        ]
        weights = self._band_weights(
            difficulty,
            layers,
            "_struct_band_ema_state",
            getattr(self.args, "struct_band_eta", 0.35),
        )
        focus = sum(weight * loss for weight, loss in zip(weights, focus_losses))
        energy = sum(weight * loss for weight, loss in zip(weights, energy_losses))
        return focus, energy, weights, focus_losses, energy_losses

    def _log_losses(self, values):
        logging_steps = max(1, int(getattr(self.args, "logging_steps", 1) or 1))
        step = int(getattr(self.state, "global_step", 0))
        if step % logging_steps == 0 and getattr(self, "_last_hidesc_log_step", None) != step:
            self._last_hidesc_log_step = step
            self.log({key: value.detach().float().item() if torch.is_tensor(value) else value for key, value in values.items()})

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not getattr(self.args, "enable_description_cl", False):
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        if "description_input_ids" not in inputs:
            raise ValueError("HiDESC train loss requires description_input_ids in the batch.")

        b1 = self._resolve_loss_band(model, self.args.b1_low_layer, self.args.b1_high_layer, "InternVL B1")
        b2 = self._resolve_loss_band(model, self.args.b2_low_layer, self.args.b2_high_layer, "InternVL B2")

        (
            _,
            answer_attention,
            answer_past,
            answer_embeds,
            answer_labels,
            image_mask,
            text_mask,
        ) = self._prepare_multimodal_inputs(model, "input_ids", inputs, inputs["labels"])
        answer_outputs = model(
            input_ids=None,
            attention_mask=answer_attention,
            past_key_values=answer_past,
            inputs_embeds=answer_embeds,
            labels=answer_labels,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        align_losses = [
            self._compute_align_loss(answer_outputs.hidden_states[layer], image_mask, text_mask)[0]
            for layer in b1
        ]
        align_weights = self._band_weights(
            align_losses,
            b1,
            "_align_band_ema_state",
            getattr(self.args, "align_band_eta", 0.5),
        )
        align_loss = sum(weight * loss for weight, loss in zip(align_weights, align_losses))

        description_states, description_key_masks = self._extract_description_states(model, inputs, b2)
        reference = inputs.get("reference_description_states")
        reference_mask = inputs.get("reference_description_mask")
        if reference is None or reference_mask is None:
            raise ValueError(
                "HiDESC reference states are missing. Run "
                "`extract_description_cache.sh` first and pass the same "
                "`description_cache_dir` to training."
            )
        available = inputs.get(
            "reference_description_available",
            torch.ones(reference.shape[0], dtype=torch.bool, device=reference.device),
        )
        focus_loss, energy_loss, struct_weights, focus_by_layer, energy_by_layer = self._compute_struct_loss(
            description_states,
            description_key_masks,
            reference,
            reference_mask,
            available,
        )
        total_loss = (
            float(getattr(self.args, "standard_ce_weight", 1.0)) * answer_outputs.loss
            + float(getattr(self.args, "align_loss_weight", 0.005)) * align_loss
            + float(getattr(self.args, "description_focus_weight", 0.05)) * focus_loss
            + float(getattr(self.args, "description_energy_weight", 0.0005)) * energy_loss
        )
        self._log_losses({
            "loss/total": total_loss,
            "loss/ce": answer_outputs.loss,
            "loss/align": align_loss,
            "loss/focus": focus_loss,
            "loss/energy": energy_loss,
            "loss/weighted_align": align_loss * float(getattr(self.args, "align_loss_weight", 0.005)),
            "loss/weighted_focus": focus_loss * float(getattr(self.args, "description_focus_weight", 0.05)),
            "loss/weighted_energy": energy_loss * float(getattr(self.args, "description_energy_weight", 0.0005)),
        })
        if return_outputs:
            return total_loss, {
                "standard_outputs": answer_outputs,
                "loss_ce": answer_outputs.loss.detach(),
                "loss_align": align_loss.detach(),
                "loss_focus": focus_loss.detach(),
                "loss_energy": energy_loss.detach(),
            }
        return total_loss
