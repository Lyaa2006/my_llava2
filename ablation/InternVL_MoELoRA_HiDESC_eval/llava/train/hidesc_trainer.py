from typing import Optional

import torch
import torch.nn.functional as F

from llava.train.hidesc_description_utils import (
    pad_description_sequences,
    select_expanded_description_tokens,
)
from llava.train.llava_trainer import LLaVATrainer


class HiDESCLoRATrainer(LLaVATrainer):
    """LoRA-FT trainer with HiDESC train losses only.

    This class deliberately contains no prototype, anchor, or routing state.
    """

    def _unwrap_model(self, model):
        return getattr(model, "module", model)

    def _get_transformer_layers(self, model):
        wrapped = self._unwrap_model(model)
        candidates = [
            wrapped,
            getattr(wrapped, "model", None),
            getattr(getattr(wrapped, "base_model", None), "model", None),
            getattr(
                getattr(getattr(wrapped, "base_model", None), "model", None),
                "model",
                None,
            ),
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
        raise ValueError("Could not locate decoder layers for HiDESC loss bands.")

    def _resolve_loss_band(self, model, low_layer, high_layer, name):
        num_layers = len(self._get_transformer_layers(model))
        low_layer, high_layer = int(low_layer), int(high_layer)
        if low_layer < 1 or high_layer > num_layers or low_layer > high_layer:
            raise ValueError(
                f"{name} must be within decoder layers [1, {num_layers}], "
                f"got [{low_layer}, {high_layer}]."
            )
        return list(range(low_layer, high_layer + 1))

    def _prepare_multimodal_inputs(self, model, input_ids, attention_mask, labels, images):
        wrapped = self._unwrap_model(model)
        return wrapped.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            None,
            labels,
            images,
            return_token_masks=True,
        )

    def _masked_mean_pool(self, hidden_states, mask):
        length = min(hidden_states.shape[1], mask.shape[1])
        hidden_states = hidden_states[:, :length]
        mask = mask[:, :length].unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _compute_align_loss_for_hidden(self, hidden_states, image_mask, text_mask):
        zero = hidden_states.new_zeros(())
        if image_mask is None or text_mask is None:
            return zero, zero, zero
        length = min(hidden_states.shape[1], image_mask.shape[1], text_mask.shape[1])
        hidden_states = hidden_states[:, :length].float()
        image_mask = image_mask[:, :length].bool()
        text_mask = text_mask[:, :length].bool()
        valid = image_mask.any(dim=1) & text_mask.any(dim=1)
        if not torch.any(valid):
            return zero, zero, zero
        image_pool = F.normalize(
            self._masked_mean_pool(hidden_states, image_mask)[valid], dim=-1
        )
        text_pool = F.normalize(
            self._masked_mean_pool(hidden_states, text_mask)[valid], dim=-1
        )
        cosine = (image_pool * text_pool).sum(dim=-1).clamp(-1.0, 1.0)
        return (1.0 - cosine).mean(), valid.float().sum(), cosine.mean()

    def _compute_band_weights(self, losses, band_layers, state_name, eta):
        values = torch.stack([loss.detach().float() for loss in losses])
        state = getattr(self, state_name, {})
        gamma = float(getattr(self.args, "loss_band_ema_gamma", 0.9))
        updated = []
        for layer, value in zip(band_layers, values):
            previous = state.get(layer, float(value.item()))
            current = gamma * previous + (1.0 - gamma) * float(value.item())
            state[layer] = current
            updated.append(current)
        setattr(self, state_name, state)
        ema = values.new_tensor(updated)
        normalized = (ema - ema.mean()) / ema.std(unbiased=False).clamp_min(1e-6)
        if len(band_layers) == 1:
            prior = values.new_ones(1)
        else:
            ramp = torch.linspace(0.0, 1.0, len(band_layers), device=values.device)
            eps = float(getattr(self.args, "loss_band_position_eps", 0.05))
            prior = eps + (1.0 - eps) * ramp
        return F.softmax(torch.log(prior.clamp_min(1e-6)) + float(eta) * normalized, dim=0)

    def _compute_align_band_loss(self, hidden_states, image_mask, text_mask, layers):
        losses, cosines = [], []
        valid_samples = None
        for layer in layers:
            loss, valid, cosine = self._compute_align_loss_for_hidden(
                hidden_states[layer], image_mask, text_mask
            )
            losses.append(loss)
            cosines.append(cosine)
            valid_samples = valid if valid_samples is None else valid_samples
        weights = self._compute_band_weights(
            losses, layers, "_align_band_ema_state", self.args.align_band_eta
        )
        return (
            sum(weight * loss for weight, loss in zip(weights, losses)),
            valid_samples,
            sum(weight * cosine for weight, cosine in zip(weights, cosines)),
            weights,
            losses,
        )

    def _extract_description_states(self, model, inputs, layers):
        (
            _,
            description_attention_mask,
            _,
            description_embeds,
            _,
            _,
            expanded_text_mask,
        ) = self._prepare_multimodal_inputs(
            model,
            inputs["description_input_ids"],
            inputs["description_attention_mask"],
            None,
            inputs.get("images"),
        )
        outputs = model(
            input_ids=None,
            attention_mask=description_attention_mask,
            inputs_embeds=description_embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        states_by_layer = {}
        keys_by_layer = {}
        for layer in layers:
            states_by_layer[layer], keys_by_layer[layer] = (
                select_expanded_description_tokens(
                    outputs.hidden_states[layer],
                    expanded_text_mask,
                    inputs["description_input_ids"],
                    inputs["description_attention_mask"],
                    inputs.get("description_key_mask"),
                    int(self.args.description_max_tokens),
                )
            )
        return states_by_layer, keys_by_layer[layers[0]]

    def _pad_key_masks(self, masks, max_len, device):
        if masks is None:
            return None
        padded = torch.zeros(len(masks), max_len, dtype=torch.bool, device=device)
        for index, mask in enumerate(masks):
            length = min(max_len, mask.shape[0])
            padded[index, :length] = mask[:length].to(device=device)
        return padded

    def _rms_normalize(self, hidden_states):
        rms = hidden_states.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()
        return hidden_states.float() / rms

    def _effective_key_mask(self, valid_mask, key_mask):
        if key_mask is None:
            return valid_mask
        selected = valid_mask & key_mask
        return torch.where(selected.any(dim=1, keepdim=True), selected, valid_mask)

    def _focus_loss(self, current, reference, valid_mask, key_mask):
        change = (
            self._rms_normalize(current) - self._rms_normalize(reference.detach())
        ).norm(dim=-1) * valid_mask.float()
        key_mask = self._effective_key_mask(valid_mask, key_mask)
        total = change.sum(dim=1)
        active = total > 1e-6
        distribution = change / total.clamp_min(1e-6).unsqueeze(-1)
        key_mass = (distribution * key_mask.float()).sum(dim=1)
        focus = -torch.log(key_mass.clamp_min(1e-6))
        focus = focus[active].mean() if torch.any(active) else change.new_zeros(())
        return focus, key_mass[active].mean() if torch.any(active) else change.new_zeros(())

    def _energy_loss(self, current, reference, valid_mask):
        change = (
            self._rms_normalize(current) - self._rms_normalize(reference.detach())
        ).norm(dim=-1) * valid_mask.float()
        counts = valid_mask.sum(dim=1)
        active = counts > 0
        mean_change = change.sum(dim=1) / counts.clamp_min(1).float()
        margin = float(getattr(self.args, "description_energy_margin", 30.0))
        loss = F.relu(mean_change - margin).pow(2)
        if not torch.any(active):
            return change.new_zeros(())
        return loss[active].mean()

    def _compute_struct_loss(
        self,
        states_by_layer,
        key_masks,
        reference_states,
        reference_mask,
        reference_available,
    ):
        layers = sorted(states_by_layer)
        current, current_mask = pad_description_sequences(states_by_layer[layers[0]])
        device = current.device
        reference_states = reference_states.to(device=device, dtype=current.dtype)
        reference_mask = reference_mask.to(device=device, dtype=torch.bool)
        length = min(current.shape[1], reference_states.shape[1])
        current_mask = current_mask[:, :length]
        reference_states = reference_states[:, :length]
        reference_mask = reference_mask[:, :length]
        valid = current_mask & reference_mask & reference_available[:, None].to(device)
        key_mask = self._pad_key_masks(key_masks, length, device)

        focus_losses, energy_losses = [], []
        for layer in layers:
            layer_states, _ = pad_description_sequences(
                states_by_layer[layer], dtype=current.dtype
            )
            layer_states = layer_states[:, :length]
            focus, _ = self._focus_loss(
                layer_states, reference_states, valid, key_mask
            )
            energy = self._energy_loss(layer_states, reference_states, valid)
            focus_losses.append(focus)
            energy_losses.append(energy)
        difficulty = [
            focus + self.args.struct_band_energy_rho * energy
            for focus, energy in zip(focus_losses, energy_losses)
        ]
        weights = self._compute_band_weights(
            difficulty, layers, "_struct_band_ema_state", self.args.struct_band_eta
        )
        focus = sum(weight * loss for weight, loss in zip(weights, focus_losses))
        energy = sum(weight * loss for weight, loss in zip(weights, energy_losses))
        struct = (
            self.args.description_focus_weight * focus
            + self.args.description_energy_weight * energy
        )
        return struct, focus, energy, weights, valid.float().sum()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (
            not getattr(self.args, "enable_description_cl", False)
            or "description_input_ids" not in inputs
        ):
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        b1_layers = self._resolve_loss_band(
            model, self.args.b1_low_layer, self.args.b1_high_layer, "InternVL B1"
        )
        b2_layers = self._resolve_loss_band(
            model, self.args.b2_low_layer, self.args.b2_high_layer, "InternVL B2"
        )
        (
            answer_ids,
            answer_attention,
            answer_past,
            answer_embeds,
            answer_labels,
            image_mask,
            text_mask,
        ) = self._prepare_multimodal_inputs(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["labels"],
            inputs.get("images"),
        )
        outputs = model(
            input_ids=answer_ids,
            attention_mask=answer_attention,
            past_key_values=answer_past,
            inputs_embeds=answer_embeds,
            labels=answer_labels,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        align_loss, valid_samples, cosine, _, _ = self._compute_align_band_loss(
            outputs.hidden_states, image_mask, text_mask, b1_layers
        )

        states_by_layer, key_masks = self._extract_description_states(
            model, inputs, b2_layers
        )
        reference_states = inputs.get("reference_description_states")
        reference_mask = inputs.get("reference_description_mask")
        if reference_states is None or reference_mask is None:
            struct = focus = energy = outputs.loss.new_zeros(())
            valid_tokens = outputs.loss.new_zeros(())
        else:
            available = inputs.get(
                "reference_description_available",
                torch.ones(
                    reference_states.shape[0],
                    dtype=torch.bool,
                    device=outputs.loss.device,
                ),
            ).to(device=outputs.loss.device, dtype=torch.bool)
            struct, focus, energy, _, valid_tokens = self._compute_struct_loss(
                states_by_layer,
                key_masks,
                reference_states,
                reference_mask,
                available,
            )

        total = (
            self.args.standard_ce_weight * outputs.loss
            + struct
            + self.args.align_loss_weight * align_loss
        )
        if self.state.global_step % max(1, int(self.args.logging_steps or 1)) == 0:
            self.log(
                {
                    "loss/ce": outputs.loss.detach().float().item(),
                    "loss/b1_align": align_loss.detach().float().item(),
                    "loss/b2_focus": focus.detach().float().item(),
                    "loss/b2_energy": energy.detach().float().item(),
                    "loss/b2_struct": struct.detach().float().item(),
                    "loss/total": total.detach().float().item(),
                    "hidesc/b1_valid_samples": valid_samples.detach().float().item(),
                    "hidesc/b1_cosine": cosine.detach().float().item(),
                    "hidesc/b2_valid_tokens": valid_tokens.detach().float().item(),
                }
            )
        if return_outputs:
            return total, {
                "standard_outputs": outputs,
                "loss_ce": outputs.loss.detach(),
                "loss_align": align_loss.detach(),
                "loss_focus": focus.detach(),
                "loss_energy": energy.detach(),
                "loss_struct": struct.detach(),
            }
        return total
