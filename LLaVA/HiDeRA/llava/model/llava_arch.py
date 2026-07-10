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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import inspect

from .multimodal_encoder.builder import build_vision_tower, build_text_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from HiDeRA.peft.tuners import HiDeMOELoraModel


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        
        if hasattr(config, "mm_text_tower"):
            self.text_tower = build_text_tower(config, delay_load=True)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_text_tower(self):
        text_tower = getattr(self, 'text_tower', None)
        if type(text_tower) is list:
            text_tower = text_tower[0]
        return text_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'),strict = False)

    def initialize_text_modules(self, model_args, fsdp=None):
        text_tower = model_args.text_tower

        if self.get_text_tower() is None:
            text_tower = build_text_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.text_tower = [text_tower]
            else:
                self.text_tower = text_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                text_tower = self.text_tower[0]
            else:
                text_tower = self.text_tower
            text_tower.load_model()


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_text_tower(self):
        return self.get_model().get_text_tower()

    def encode_images(self, images):
        clip_image_features, image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return clip_image_features.to(self.device), image_features.to(self.device)

    def _get_relation_config(self):
        return getattr(
            self,
            "relation_routing_config",
            {
                "routing_image_weight": 0.5,
                "routing_text_weight": 0.5,
                "routing_history_weight": 0.15,
                "routing_temperature": 0.1,
                "routing_top_k": 2,
                "routing_min_similarity": -1.0,
                "routing_prior_momentum": 0.8,
                "bootstrap_steps": 100,
                "role_top_k": 2,
                "role_birth_threshold": 0.3,
                "routing_early_layers": 8,
                "routing_middle_layers": 16,
                "routing_late_top_k": 2,
                "routing_role_prior_weight": 0.0,
                "routing_self_weight": 1.0,
            },
        )

    def _safe_normalize(self, tensor):
        if tensor.ndim > 1:
            tensor = tensor.squeeze(0)
        return F.normalize(tensor.float(), dim=0)

    def _get_active_role_count(self):
        return int(self.active_role_count.detach().float().item())

    def _set_active_role_count(self, count):
        self.active_role_count.data[0] = float(max(0, count))

    def _ensure_progressive_state_for_task(self):
        if getattr(self, "_progressive_task_id", None) != int(self.cur_task):
            self.reset_progressive_state()

    def _compose_relation_logits(self, image_scores, text_scores, history_scores):
        config = self._get_relation_config()
        return (
            float(config["routing_image_weight"]) * image_scores
            + float(config["routing_text_weight"]) * text_scores
            + float(config["routing_history_weight"]) * history_scores
        )

    def _build_sparse_relation_weights(self, logits, top_k):
        if logits.numel() == 0:
            return logits
        config = self._get_relation_config()
        top_k = int(top_k)
        if top_k <= 0:
            top_k = logits.numel()
        top_k = min(top_k, logits.numel())
        if top_k < logits.numel():
            top_values, top_indices = torch.topk(logits, k=top_k)
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits.scatter_(0, top_indices, top_values)
        else:
            masked_logits = logits
        temperature = max(float(config["routing_temperature"]), 1e-6)
        return F.softmax(masked_logits / temperature, dim=0)

    def _mask_relation_logits(self, relation_logits):
        min_similarity = float(self._get_relation_config()["routing_min_similarity"])
        if min_similarity <= -1.0:
            return relation_logits
        masked_relation_logits = torch.where(
            relation_logits >= min_similarity,
            relation_logits,
            torch.full_like(relation_logits, float("-inf")),
        )
        if not torch.isfinite(masked_relation_logits).any():
            return relation_logits
        return masked_relation_logits

    def _get_completed_task_count(self):
        if self.training:
            return max(0, int(self.cur_task))
        return max(1, int(self.expert_num))

    def _get_layer_stage(self, layer_idx, total_layers):
        config = self._get_relation_config()
        early_layers = min(max(0, int(config["routing_early_layers"])), total_layers)
        middle_layers = min(max(0, int(config["routing_middle_layers"])), max(0, total_layers - early_layers))
        if layer_idx < early_layers:
            return "early"
        if layer_idx < early_layers + middle_layers:
            return "middle"
        return "late"

    def _apply_relation_weights_to_experts(self, route_plan):
        proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        for layer_idx, layer in enumerate(self.model.layers):
            stage = self._get_layer_stage(layer_idx, len(self.model.layers))
            active_weights = route_plan[stage]
            for proj_name in proj_names:
                if proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    proj_layer = getattr(layer.self_attn, proj_name)
                else:
                    proj_layer = getattr(layer.mlp, proj_name)
                if hasattr(proj_layer, "expert_weight"):
                    proj_layer.task_fuse_weight = active_weights
                    proj_layer.expert_weight = active_weights
                    proj_layer.progressive_stage = stage

    def _get_task_anchor(self, task_id):
        return (
            self._safe_normalize(self.image_anchors[task_id].detach()),
            self._safe_normalize(self.text_anchors[task_id].detach()),
        )

    def _score_tasks(self, task_indices, image_anchor, text_anchor, device):
        if len(task_indices) == 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        image_bank = torch.stack(
            [self._safe_normalize(self.image_anchors[idx].detach()).to(device) for idx in task_indices],
            dim=0,
        )
        text_bank = torch.stack(
            [self._safe_normalize(self.text_anchors[idx].detach()).to(device) for idx in task_indices],
            dim=0,
        )
        image_scores = torch.matmul(image_bank, image_anchor.to(device))
        text_scores = torch.matmul(text_bank, text_anchor.to(device))
        history_scores = self.expert_usage_prior[task_indices].detach().float().to(device)
        logits = self._compose_relation_logits(image_scores, text_scores, history_scores)
        return self._mask_relation_logits(logits)

    def _score_roles(self, image_anchor, text_anchor, device):
        active_roles = self._get_active_role_count()
        if active_roles == 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        role_image_bank = torch.stack(
            [self._safe_normalize(self.role_image_prototypes[idx].detach()).to(device) for idx in range(active_roles)],
            dim=0,
        )
        role_text_bank = torch.stack(
            [self._safe_normalize(self.role_text_prototypes[idx].detach()).to(device) for idx in range(active_roles)],
            dim=0,
        )
        role_scores = (
            float(self._get_relation_config()["routing_image_weight"]) * torch.matmul(role_image_bank, image_anchor.to(device))
            + float(self._get_relation_config()["routing_text_weight"]) * torch.matmul(role_text_bank, text_anchor.to(device))
            + float(self._get_relation_config()["routing_role_prior_weight"]) * self.role_usage_prior[:active_roles].detach().float().to(device)
        )
        return self._mask_relation_logits(role_scores)

    def _blend_with_current_task(self, history_weights, active_experts):
        task_id = min(max(int(self.cur_task), 0), max(0, active_experts - 1))
        combined = torch.zeros(active_experts, device=history_weights.device, dtype=torch.float32)
        if history_weights.numel() > 0:
            combined[:history_weights.numel()] = history_weights
        combined[task_id] += float(self._get_relation_config()["routing_self_weight"])
        if float(combined.sum().item()) <= 0.0:
            combined[task_id] = 1.0
        return combined / combined.sum()

    def _role_member_tasks(self, role_id, active_experts, exclude_task_id=None):
        memberships = self.task_role_membership[:active_experts, role_id].detach().float()
        member_tasks = [idx for idx in range(active_experts) if float(memberships[idx].item()) > 0.0]
        if exclude_task_id is not None:
            member_tasks = [idx for idx in member_tasks if idx != exclude_task_id]
        return member_tasks

    def ensure_role_bank_initialized(self, completed_task_count):
        completed_task_count = min(int(completed_task_count), self.max_task_slots)
        if completed_task_count <= 0 or self._get_active_role_count() > 0:
            return
        existing_membership = self.task_role_membership[:completed_task_count].detach().abs().sum().item()
        if existing_membership > 0:
            active_roles = int((self.role_task_count.detach() > 0).sum().item())
            self._set_active_role_count(active_roles)
            return

        for task_id in range(completed_task_count):
            image_anchor, text_anchor = self._get_task_anchor(task_id)
            role_scores = self._score_roles(image_anchor, text_anchor, image_anchor.device)
            if role_scores.numel() == 0 or float(role_scores.max().item()) < float(self._get_relation_config()["role_birth_threshold"]):
                role_id = self._get_active_role_count()
                if role_id >= self.max_role_slots:
                    break
                self.role_image_prototypes[role_id].data.copy_(image_anchor.unsqueeze(0).to(self.role_image_prototypes[role_id].dtype))
                self.role_text_prototypes[role_id].data.copy_(text_anchor.unsqueeze(0).to(self.role_text_prototypes[role_id].dtype))
                self.role_task_count.data[role_id] = 1.0
                self.role_usage_prior.data[role_id] = 1.0
                self.task_role_membership.data[task_id].zero_()
                self.task_role_membership.data[task_id, role_id] = 1.0
                self._set_active_role_count(role_id + 1)
                continue

            role_id = int(torch.argmax(role_scores).item())
            count = float(self.role_task_count[role_id].detach().item())
            updated_count = count + 1.0
            new_image_proto = (
                count * self._safe_normalize(self.role_image_prototypes[role_id].detach())
                + image_anchor
            ) / updated_count
            new_text_proto = (
                count * self._safe_normalize(self.role_text_prototypes[role_id].detach())
                + text_anchor
            ) / updated_count
            self.role_image_prototypes[role_id].data.copy_(new_image_proto.unsqueeze(0).to(self.role_image_prototypes[role_id].dtype))
            self.role_text_prototypes[role_id].data.copy_(new_text_proto.unsqueeze(0).to(self.role_text_prototypes[role_id].dtype))
            self.role_task_count.data[role_id] = updated_count
            self.role_usage_prior.data[role_id] = max(self.role_usage_prior[role_id].detach().item(), 1.0)
            self.task_role_membership.data[task_id].zero_()
            self.task_role_membership.data[task_id, role_id] = 1.0

    def _assign_roles_for_current_task(self, image_anchor, text_anchor):
        role_scores = self._score_roles(image_anchor, text_anchor, image_anchor.device)
        if role_scores.numel() == 0:
            return torch.empty(0, device=image_anchor.device), [], True
        max_role_score = float(role_scores.max().item())
        if max_role_score < float(self._get_relation_config()["role_birth_threshold"]):
            return torch.empty(0, device=image_anchor.device), [], True

        top_m = min(int(self._get_relation_config()["role_top_k"]), role_scores.numel())
        top_values, top_indices = torch.topk(role_scores, k=top_m)
        membership = F.softmax(top_values / max(float(self._get_relation_config()["routing_temperature"]), 1e-6), dim=0)
        return membership, top_indices.tolist(), False

    def _build_progressive_route_plan(self, active_experts, image_anchor, text_anchor, role_membership, candidate_roles):
        device = image_anchor.device
        task_id = min(max(int(self.cur_task), 0), max(0, active_experts - 1))
        if active_experts <= 1 or len(candidate_roles) == 0:
            identity = torch.zeros(active_experts, device=device, dtype=torch.float32)
            identity[task_id] = 1.0
            return {"early": identity, "middle": identity, "late": identity, "candidate_experts": [task_id]}

        early_history = torch.zeros(active_experts, device=device, dtype=torch.float32)
        middle_history = torch.zeros(active_experts, device=device, dtype=torch.float32)
        late_history = torch.zeros(active_experts, device=device, dtype=torch.float32)
        candidate_experts = set()

        for local_idx, role_id in enumerate(candidate_roles):
            role_weight = float(role_membership[local_idx].item())
            member_tasks = self._role_member_tasks(role_id, active_experts, exclude_task_id=task_id)
            if len(member_tasks) == 0:
                continue
            uniform_weight = role_weight / float(len(member_tasks))
            for member_task in member_tasks:
                early_history[member_task] += uniform_weight

            task_logits = self._score_tasks(member_tasks, image_anchor, text_anchor, device)
            if task_logits.numel() == 0:
                continue
            local_weights = F.softmax(task_logits / max(float(self._get_relation_config()["routing_temperature"]), 1e-6), dim=0)
            for member_idx, member_task in enumerate(member_tasks):
                middle_history[member_task] += role_weight * local_weights[member_idx]

            best_local_idx = int(torch.argmax(task_logits).item())
            candidate_experts.add(member_tasks[best_local_idx])

        if len(candidate_experts) == 0:
            candidate_experts = {task_id}
        candidate_experts = sorted(candidate_experts)
        history_candidates = [idx for idx in candidate_experts if idx != task_id]
        if len(history_candidates) > 0:
            candidate_logits = self._score_tasks(history_candidates, image_anchor, text_anchor, device)
            late_weights_local = self._build_sparse_relation_weights(
                candidate_logits,
                top_k=min(int(self._get_relation_config()["routing_late_top_k"]), len(history_candidates)),
            )
            for local_idx, member_task in enumerate(history_candidates):
                late_history[member_task] = late_weights_local[local_idx]

        early = self._blend_with_current_task(early_history, active_experts)
        middle = self._blend_with_current_task(middle_history, active_experts)
        late = self._blend_with_current_task(late_history, active_experts)
        return {
            "early": early,
            "middle": middle,
            "late": late,
            "candidate_experts": candidate_experts,
        }

    def _update_relation_statistics(self, task_id, relation_weights):
        if relation_weights.numel() == 0:
            return
        momentum = float(self._get_relation_config()["routing_prior_momentum"])
        previous_scores = self.task_relation_scores[task_id, :relation_weights.numel()].detach().float()
        updated_scores = momentum * previous_scores + (1.0 - momentum) * relation_weights.detach().float()
        self.task_relation_scores.data[task_id, :relation_weights.numel()] = updated_scores.to(
            self.task_relation_scores.dtype
        )

        previous_prior = self.expert_usage_prior[:relation_weights.numel()].detach().float()
        updated_prior = momentum * previous_prior + (1.0 - momentum) * relation_weights.detach().float()
        self.expert_usage_prior.data[:relation_weights.numel()] = updated_prior.to(self.expert_usage_prior.dtype)

    def _update_task_anchors(self, task_id, current_image_features, current_text_features):
        image_sum = self.image_anchors[task_id].detach().float().squeeze(0) * self.image_boundary[task_id].detach().float()
        image_sum = image_sum + current_image_features.float().sum(dim=0)
        text_sum = self.text_anchors[task_id].detach().float().squeeze(0) * self.text_boundary[task_id].detach().float()
        text_sum = text_sum + current_text_features.float().sum(dim=0)

        self.image_boundary[task_id].data += current_image_features.shape[0]
        self.text_boundary[task_id].data += current_text_features.shape[0]
        self.image_anchors[task_id].data.copy_(
            (image_sum / self.image_boundary[task_id].detach().float()).unsqueeze(0).to(self.image_anchors[task_id].dtype)
        )
        self.text_anchors[task_id].data.copy_(
            (text_sum / self.text_boundary[task_id].detach().float()).unsqueeze(0).to(self.text_anchors[task_id].dtype)
        )

    def _update_bootstrap_statistics(self, current_image_features, current_text_features):
        if self._bootstrap_image_sum is None:
            self._bootstrap_image_sum = current_image_features.detach().float().sum(dim=0)
            self._bootstrap_text_sum = current_text_features.detach().float().sum(dim=0)
        else:
            self._bootstrap_image_sum = self._bootstrap_image_sum + current_image_features.detach().float().sum(dim=0)
            self._bootstrap_text_sum = self._bootstrap_text_sum + current_text_features.detach().float().sum(dim=0)
        self._bootstrap_sample_count += int(current_image_features.shape[0])
        self._bootstrap_seen_steps += 1

    def _bootstrap_finished(self):
        return self._bootstrap_seen_steps >= int(self._get_relation_config()["bootstrap_steps"])

    def _finalize_current_task_role_memory_impl(self):
        task_id = min(max(int(self.cur_task), 0), self.max_task_slots - 1)
        if task_id > 0:
            self.ensure_role_bank_initialized(task_id)
        task_image_anchor, task_text_anchor = self._get_task_anchor(task_id)

        if task_id == 0 and self._get_active_role_count() == 0:
            self.role_image_prototypes[0].data.copy_(task_image_anchor.unsqueeze(0).to(self.role_image_prototypes[0].dtype))
            self.role_text_prototypes[0].data.copy_(task_text_anchor.unsqueeze(0).to(self.role_text_prototypes[0].dtype))
            self.role_task_count.data[0] = 1.0
            self.role_usage_prior.data[0] = 1.0
            self.task_role_membership.data[task_id].zero_()
            self.task_role_membership.data[task_id, 0] = 1.0
            self._set_active_role_count(1)
            return

        membership = getattr(self, "_pending_role_membership", None)
        candidate_roles = getattr(self, "_pending_candidate_roles", [])
        if membership is None or len(candidate_roles) == 0 or getattr(self, "_pending_role_birth", False):
            role_id = self._get_active_role_count()
            if role_id >= self.max_role_slots:
                role_id = self.max_role_slots - 1
            self.role_image_prototypes[role_id].data.copy_(task_image_anchor.unsqueeze(0).to(self.role_image_prototypes[role_id].dtype))
            self.role_text_prototypes[role_id].data.copy_(task_text_anchor.unsqueeze(0).to(self.role_text_prototypes[role_id].dtype))
            self.role_task_count.data[role_id] = max(1.0, float(self.role_task_count[role_id].detach().item()))
            self.role_usage_prior.data[role_id] = 1.0
            self.task_role_membership.data[task_id].zero_()
            self.task_role_membership.data[task_id, role_id] = 1.0
            self._set_active_role_count(max(self._get_active_role_count(), role_id + 1))
            return

        momentum = float(self._get_relation_config()["routing_prior_momentum"])
        self.task_role_membership.data[task_id].zero_()
        for local_idx, role_id in enumerate(candidate_roles):
            weight = float(membership[local_idx].detach().item())
            if weight <= 0.0:
                continue
            count = float(self.role_task_count[role_id].detach().item())
            updated_count = count + weight
            updated_image = (
                count * self._safe_normalize(self.role_image_prototypes[role_id].detach())
                + weight * task_image_anchor
            ) / max(updated_count, 1e-6)
            updated_text = (
                count * self._safe_normalize(self.role_text_prototypes[role_id].detach())
                + weight * task_text_anchor
            ) / max(updated_count, 1e-6)
            self.role_image_prototypes[role_id].data.copy_(updated_image.unsqueeze(0).to(self.role_image_prototypes[role_id].dtype))
            self.role_text_prototypes[role_id].data.copy_(updated_text.unsqueeze(0).to(self.role_text_prototypes[role_id].dtype))
            self.role_task_count.data[role_id] = updated_count
            self.role_usage_prior.data[role_id] = momentum * self.role_usage_prior[role_id].detach().float() + (1.0 - momentum) * weight
            self.task_role_membership.data[task_id, role_id] = weight

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        
        vision_tower = self.get_vision_tower()
        if hasattr(self, "reset_relation_losses"):
            self.reset_relation_losses(device=input_ids.device if input_ids is not None else self.device)

        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_guide_features, image_features = self.encode_images(images)

        assert image_features.shape[1] == 576, 'vision tower not a withprojection version.'
        text_tower = self.get_text_tower()

        # with torch.no_grad():
        #     # image_guide_features: bs, 4096
        #     image_guide_features = image_features[:,0]
        
        input_pad = np.where(input_ids.cpu().detach().numpy()!=-200,input_ids.cpu().detach().numpy(),self.tokenizer.pad_token_id)
        decoded_inputs = self.tokenizer.batch_decode(input_pad, skip_special_tokens=True)
        decoded_hidden_inputs = ['\n'.join(decode_input.split('\n')[1:]) for decode_input in decoded_inputs]
        decoded_clip_inputs = [decode_input.split(' ASSISTANT')[0] for decode_input in decoded_hidden_inputs]

        clip_text_inputs = self.clip_tokenizer(
                decoded_clip_inputs,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )

        # text_guide_features: bs, 768
        text_guide_features = text_tower(clip_text_inputs)

        self._ensure_progressive_state_for_task()
        if self.training:
            current_image_features = image_guide_features  # [batch_size, feature_dim]
            current_text_features = text_guide_features  # [batch_size, feature_dim]
            task_id = self.cur_task
            current_image_summary = self._safe_normalize(current_image_features.float().mean(dim=0))
            current_text_summary = self._safe_normalize(current_text_features.float().mean(dim=0))

            if task_id > 0:
                self.ensure_role_bank_initialized(task_id)
            self._update_bootstrap_statistics(current_image_features, current_text_features)

            if task_id == 0 or not self._bootstrap_finished():
                route_plan = {
                    "early": torch.eye(max(1, task_id + 1), device=current_image_summary.device, dtype=torch.float32)[task_id],
                    "middle": torch.eye(max(1, task_id + 1), device=current_image_summary.device, dtype=torch.float32)[task_id],
                    "late": torch.eye(max(1, task_id + 1), device=current_image_summary.device, dtype=torch.float32)[task_id],
                    "candidate_experts": [task_id],
                }
            else:
                if not getattr(self, "_role_induction_complete", False):
                    bootstrap_image_anchor = self._safe_normalize(self._bootstrap_image_sum / max(self._bootstrap_sample_count, 1))
                    bootstrap_text_anchor = self._safe_normalize(self._bootstrap_text_sum / max(self._bootstrap_sample_count, 1))
                    membership, candidate_roles, role_birth = self._assign_roles_for_current_task(
                        bootstrap_image_anchor,
                        bootstrap_text_anchor,
                    )
                    self._pending_role_membership = membership.detach() if membership is not None else None
                    self._pending_candidate_roles = list(candidate_roles)
                    self._pending_role_birth = bool(role_birth)
                    self._role_induction_complete = True

                route_plan = self._build_progressive_route_plan(
                    active_experts=max(1, task_id + 1),
                    image_anchor=current_image_summary,
                    text_anchor=current_text_summary,
                    role_membership=self._pending_role_membership if self._pending_role_membership is not None else torch.empty(0, device=current_image_summary.device),
                    candidate_roles=self._pending_candidate_roles,
                )
                self._pending_candidate_experts = list(route_plan["candidate_experts"])
                self._update_relation_statistics(task_id, route_plan["late"])
                if task_id > 0:
                    historical_image = torch.stack(
                        [self._safe_normalize(self.image_anchors[idx].detach()) for idx in range(task_id + 1)],
                        dim=0,
                    ).to(current_image_summary.device)
                    historical_text = torch.stack(
                        [self._safe_normalize(self.text_anchors[idx].detach()) for idx in range(task_id + 1)],
                        dim=0,
                    ).to(current_text_summary.device)
                    target_image = torch.matmul(route_plan["late"].unsqueeze(0), historical_image).squeeze(0)
                    target_text = torch.matmul(route_plan["late"].unsqueeze(0), historical_text).squeeze(0)
                    self.transfer_aux_loss = (
                        1.0 - F.cosine_similarity(current_image_summary.unsqueeze(0), target_image.unsqueeze(0), dim=-1)
                    ).mean()
                    self.transfer_aux_loss += (
                        1.0 - F.cosine_similarity(current_text_summary.unsqueeze(0), target_text.unsqueeze(0), dim=-1)
                    ).mean()
                    history_weights = route_plan["late"][:task_id]
                    if history_weights.numel() > 0:
                        self.routing_aux_loss = -(history_weights * history_weights.clamp_min(1e-8).log()).sum()

            self._apply_relation_weights_to_experts(route_plan)
            self._update_task_anchors(task_id, current_image_features, current_text_features)
        else:
            active_experts = max(1, min(self.expert_num, len(self.image_anchors)))
            self.ensure_role_bank_initialized(active_experts)
            current_image_summary = self._safe_normalize(image_guide_features.float().mean(dim=0))
            current_text_summary = self._safe_normalize(text_guide_features.float().mean(dim=0))
            task_membership = self.task_role_membership[self.cur_task].detach().float()
            candidate_roles = torch.nonzero(task_membership > 0, as_tuple=False).flatten().tolist()
            membership = task_membership[candidate_roles]
            if len(candidate_roles) == 0:
                membership, candidate_roles, _ = self._assign_roles_for_current_task(
                    current_image_summary,
                    current_text_summary,
                )
            route_plan = self._build_progressive_route_plan(
                active_experts,
                current_image_summary,
                current_text_summary,
                membership if isinstance(membership, torch.Tensor) else torch.empty(0, device=current_image_summary.device),
                candidate_roles,
            )
            self._apply_relation_weights_to_experts(route_plan)


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
