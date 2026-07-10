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

from HiDESC.peft.tuners import HiDeMOELoraModel
from collections import deque


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
                "routing_min_similarity": -1.0,
                "routing_prior_momentum": 0.8,
                "role_top_k": 2,
                "role_birth_threshold": 0.70,
                "role_assignment_top_k": 1,
                "role_assignment_min_similarity": 0.75,
                "role_assignment_margin": 0.10,
                "role_member_top_k": 2,
                "routing_role_prior_weight": 0.1,
                "routing_role_member_weight": 0.5,
                "routing_role_size_penalty": 0.20,
                "routing_early_layers": 16,
                "routing_early_mode": "uniform",
                "routing_middle_layers": 13,
                "routing_middle_temperature": 0.5,
                "routing_middle_role_gamma": 1.5,
                "routing_middle_role_margin_low": 0.10,
                "routing_middle_role_margin_high": 0.30,
                "routing_middle_intra_margin_low": 0.05,
                "routing_middle_intra_margin_high": 0.20,
                "routing_late_layers": 3,
                "routing_late_top_k": 2,
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

    def _get_task_anchor(self, task_id):
        return (
            self._safe_normalize(self.image_anchors[task_id].detach()),
            self._safe_normalize(self.text_anchors[task_id].detach()),
        )

    def _compose_relation_logits(self, image_scores, text_scores, history_scores):
        config = self._get_relation_config()
        return (
            float(config["routing_image_weight"]) * image_scores
            + float(config["routing_text_weight"]) * text_scores
            + float(config["routing_history_weight"]) * history_scores
        )

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

    def _build_sparse_relation_weights(self, logits, top_k):
        if logits.numel() == 0:
            return logits
        top_k = min(max(int(top_k), 1), logits.numel())
        if top_k < logits.numel():
            top_values, top_indices = torch.topk(logits, k=top_k)
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits.scatter_(0, top_indices, top_values)
        else:
            masked_logits = logits
        temperature = max(float(self._get_relation_config()["routing_temperature"]), 1e-6)
        return F.softmax(masked_logits / temperature, dim=0)

    def _normalize_route_weights(self, weights):
        weights = weights.clamp_min(0.0)
        if float(weights.sum().item()) <= 0.0:
            weights = torch.ones_like(weights)
        return weights / weights.sum()

    def _build_uniform_route_weights(self, active_experts, device):
        if active_experts <= 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        return torch.full(
            (active_experts,),
            1.0 / float(active_experts),
            device=device,
            dtype=torch.float32,
        )

    def _summarize_guide_features(self, guide_features):
        if guide_features.ndim == 1:
            summary = guide_features
        else:
            summary = guide_features.float().mean(dim=0)
        return self._safe_normalize(summary)

    def _build_dense_relation_weights(self, logits):
        if logits.numel() == 0:
            return logits
        finite_mask = torch.isfinite(logits)
        if not finite_mask.any():
            return self._build_uniform_route_weights(logits.numel(), logits.device)
        safe_logits = torch.where(
            finite_mask,
            logits,
            torch.full_like(logits, float("-inf")),
        )
        temperature = max(float(self._get_relation_config()["routing_temperature"]), 1e-6)
        weights = F.softmax(safe_logits / temperature, dim=0)
        return self._normalize_route_weights(weights)

    def _build_dense_relation_weights_with_temperature(self, logits, temperature):
        if logits.numel() == 0:
            return logits
        finite_mask = torch.isfinite(logits)
        if not finite_mask.any():
            return self._build_uniform_route_weights(logits.numel(), logits.device)
        safe_logits = torch.where(
            finite_mask,
            logits,
            torch.full_like(logits, float("-inf")),
        )
        weights = F.softmax(safe_logits / max(float(temperature), 1e-6), dim=0)
        return self._normalize_route_weights(weights)

    def _compute_adaptive_blend(self, margin, lower, upper):
        lower = float(lower)
        upper = float(upper)
        if upper <= lower:
            return 1.0 if float(margin) >= upper else 0.0
        blend = (float(margin) - lower) / (upper - lower)
        return float(max(0.0, min(1.0, blend)))

    def _compute_shared_task_scores(self, image_guide_features, text_guide_features, active_experts):
        device = image_guide_features.device
        if active_experts <= 0:
            return torch.empty(0, device=device, dtype=torch.float32)

        image_bank = torch.stack(
            [self._safe_normalize(self.image_anchors[idx].detach()).to(device) for idx in range(active_experts)],
            dim=0,
        )
        text_bank = torch.stack(
            [self._safe_normalize(self.text_anchors[idx].detach()).to(device) for idx in range(active_experts)],
            dim=0,
        )

        normalized_image_features = F.normalize(image_guide_features.float(), dim=-1)
        normalized_text_features = F.normalize(text_guide_features.float(), dim=-1)

        image_scores = torch.matmul(normalized_image_features, image_bank.T).max(dim=0).values
        text_scores = torch.matmul(normalized_text_features, text_bank.T).max(dim=0).values

        return self._mask_relation_logits(
            float(self._get_relation_config()["routing_image_weight"]) * image_scores
            + float(self._get_relation_config()["routing_text_weight"]) * text_scores
        )

    def _role_member_tasks(self, role_id, active_experts):
        memberships = self.task_role_membership[:active_experts, role_id].detach().float()
        return [idx for idx in range(active_experts) if float(memberships[idx].item()) > 0.0]

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

    def _score_roles(self, active_experts, expert_logits, image_anchor, text_anchor, device, return_details=False):
        active_roles = self._get_active_role_count()
        if active_roles == 0:
            empty = torch.empty(0, device=device, dtype=torch.float32)
            if return_details:
                return empty, empty, empty
            return empty

        role_image_bank = torch.stack(
            [self._safe_normalize(self.role_image_prototypes[idx].detach()).to(device) for idx in range(active_roles)],
            dim=0,
        )
        role_text_bank = torch.stack(
            [self._safe_normalize(self.role_text_prototypes[idx].detach()).to(device) for idx in range(active_roles)],
            dim=0,
        )
        prototype_scores = (
            float(self._get_relation_config()["routing_image_weight"]) * torch.matmul(role_image_bank, image_anchor.to(device))
            + float(self._get_relation_config()["routing_text_weight"]) * torch.matmul(role_text_bank, text_anchor.to(device))
        )

        role_scores = []
        member_top_k = int(self._get_relation_config()["role_member_top_k"])
        member_weight = float(self._get_relation_config()["routing_role_member_weight"])
        prior_weight = float(self._get_relation_config()["routing_role_prior_weight"])
        size_penalty = float(self._get_relation_config().get("routing_role_size_penalty", 0.0))
        role_priors = self.role_usage_prior[:active_roles].detach().float().to(device)
        role_sizes = self.role_task_count[:active_roles].detach().float().to(device)
        member_scores = []
        for role_id in range(active_roles):
            member_tasks = self._role_member_tasks(role_id, active_experts)
            if len(member_tasks) == 0:
                member_score = torch.tensor(0.0, device=device, dtype=torch.float32)
            else:
                top_k = min(max(member_top_k, 1), len(member_tasks))
                member_values = torch.topk(expert_logits[member_tasks], k=top_k).values
                member_score = member_values.mean()
            member_scores.append(member_score)
            role_scores.append(
                prototype_scores[role_id]
                + member_weight * member_score
                + prior_weight * role_priors[role_id]
                - size_penalty * torch.log1p(role_sizes[role_id].clamp_min(0.0))
            )
        masked_role_scores = self._mask_relation_logits(torch.stack(role_scores, dim=0))
        if return_details:
            return masked_role_scores, prototype_scores, torch.stack(member_scores, dim=0)
        return masked_role_scores

    def _build_role_weight_plan(self, active_experts, expert_logits, image_anchor, text_anchor, use_all_roles=False, role_pool_key="role_top_k"):
        role_scores = self._score_roles(
            active_experts,
            expert_logits,
            image_anchor,
            text_anchor,
            image_anchor.device,
        )
        if role_scores.numel() == 0:
            return role_scores, []
        if use_all_roles:
            role_weights = self._build_dense_relation_weights(role_scores)
        else:
            role_pool = min(
                max(1, int(self._get_relation_config().get(role_pool_key, self._get_relation_config()["role_top_k"]))),
                role_scores.numel(),
            )
            role_weights = self._build_sparse_relation_weights(role_scores, role_pool)
        candidate_roles = torch.nonzero(role_weights > 0.0, as_tuple=False).flatten().tolist()
        return role_weights, candidate_roles

    def _build_early_route_weights(self, active_experts, task_scores, role_weights):
        if role_weights.numel() == 0:
            return self._fallback_expert_weights(task_scores, active_experts)

        device = task_scores.device
        weights = torch.zeros(active_experts, device=device, dtype=torch.float32)
        early_mode = str(self._get_relation_config().get("routing_early_mode", "uniform")).lower()

        for role_id, role_weight in enumerate(role_weights):
            role_weight = float(role_weight.detach().item())
            if role_weight <= 0.0:
                continue
            member_tasks = self._role_member_tasks(role_id, active_experts)
            if len(member_tasks) == 0:
                continue

            if early_mode == "task_softmax_within_role":
                member_weights = self._build_dense_relation_weights(task_scores[member_tasks])
            else:
                member_weights = torch.full(
                    (len(member_tasks),),
                    1.0 / float(len(member_tasks)),
                    device=device,
                    dtype=torch.float32,
                )

            for local_idx, task_id in enumerate(member_tasks):
                weights[task_id] += role_weight * member_weights[local_idx]

        if float(weights.sum().item()) <= 0.0:
            return self._fallback_expert_weights(task_scores, active_experts)
        return self._normalize_route_weights(weights)

    def _build_middle_route_weights(self, active_experts, task_scores, role_weights):
        if role_weights.numel() == 0:
            return self._fallback_expert_weights(task_scores, active_experts)

        config = self._get_relation_config()
        device = task_scores.device
        middle_temperature = float(config.get("routing_middle_temperature", 0.5))
        role_gamma = float(config.get("routing_middle_role_gamma", 1.35))
        role_margin_low = float(config.get("routing_middle_role_margin_low", 0.05))
        role_margin_high = float(config.get("routing_middle_role_margin_high", 0.25))
        intra_margin_low = float(config.get("routing_middle_intra_margin_low", 0.02))
        intra_margin_high = float(config.get("routing_middle_intra_margin_high", 0.12))

        sorted_role_weights, _ = torch.sort(role_weights, descending=True)
        top1_role_weight = float(sorted_role_weights[0].item()) if sorted_role_weights.numel() > 0 else 1.0
        top2_role_weight = float(sorted_role_weights[1].item()) if sorted_role_weights.numel() > 1 else 0.0
        role_margin = top1_role_weight - top2_role_weight
        beta = self._compute_adaptive_blend(role_margin, role_margin_low, role_margin_high)

        sharpened_role_weights = self._normalize_route_weights(role_weights.pow(role_gamma))
        middle_role_weights = self._normalize_route_weights(
            (1.0 - beta) * role_weights + beta * sharpened_role_weights
        )

        weights = torch.zeros(active_experts, device=device, dtype=torch.float32)
        for role_id, role_weight in enumerate(middle_role_weights):
            role_weight_value = float(role_weight.detach().item())
            if role_weight_value <= 0.0:
                continue
            member_tasks = self._role_member_tasks(role_id, active_experts)
            if len(member_tasks) == 0:
                continue

            if len(member_tasks) == 1:
                local_weights = torch.ones(1, device=device, dtype=torch.float32)
            else:
                local_scores = task_scores[member_tasks]
                local_prob = self._build_dense_relation_weights_with_temperature(
                    local_scores,
                    middle_temperature,
                )
                local_uniform = torch.full(
                    (len(member_tasks),),
                    1.0 / float(len(member_tasks)),
                    device=device,
                    dtype=torch.float32,
                )
                local_top_values = torch.topk(local_scores, k=min(2, local_scores.numel())).values
                local_margin = float(local_top_values[0].item()) - float(local_top_values[1].item())
                alpha = self._compute_adaptive_blend(
                    local_margin,
                    intra_margin_low,
                    intra_margin_high,
                )
                local_weights = (1.0 - alpha) * local_uniform + alpha * local_prob
                local_weights = self._normalize_route_weights(local_weights)

            for local_idx, task_id in enumerate(member_tasks):
                weights[task_id] += role_weight_value * local_weights[local_idx]

        if float(weights.sum().item()) <= 0.0:
            return self._fallback_expert_weights(task_scores, active_experts)
        return self._normalize_route_weights(weights)

    def _assign_roles_from_sample(self, active_experts, expert_logits, image_anchor, text_anchor):
        role_scores, prototype_scores, _ = self._score_roles(
            active_experts,
            expert_logits,
            image_anchor,
            text_anchor,
            image_anchor.device,
            return_details=True,
        )
        if role_scores.numel() == 0:
            return torch.empty(0, device=image_anchor.device), [], True
        top_value, top_index = torch.topk(role_scores, k=1)
        best_score = float(top_value[0].item())
        best_role_id = int(top_index[0].item())
        best_prototype_score = float(prototype_scores[best_role_id].item())
        birth_threshold = float(self._get_relation_config()["role_birth_threshold"])
        prototype_threshold = float(self._get_relation_config().get("role_assignment_min_similarity", birth_threshold))
        score_margin = float(self._get_relation_config().get("role_assignment_margin", 0.0))

        if best_score < birth_threshold or best_prototype_score < prototype_threshold:
            return torch.empty(0, device=image_anchor.device), [], True

        if role_scores.numel() > 1 and score_margin > 0.0:
            top2_values = torch.topk(role_scores, k=2).values
            if (
                float((top2_values[0] - top2_values[1]).item()) < score_margin
                and best_prototype_score < prototype_threshold + score_margin
            ):
                return torch.empty(0, device=image_anchor.device), [], True

        top_m = min(
            max(1, int(self._get_relation_config().get("role_assignment_top_k", 1))),
            role_scores.numel(),
        )
        top_values, top_indices = torch.topk(role_scores, k=top_m)
        if top_m == 1:
            membership = torch.ones(1, device=image_anchor.device, dtype=torch.float32)
        else:
            membership = F.softmax(
                top_values / max(float(self._get_relation_config()["routing_temperature"]), 1e-6),
                dim=0,
            )
        return membership, top_indices.tolist(), False

    def _fallback_expert_weights(self, expert_logits, active_experts, sparse_top_k=None):
        if expert_logits.numel() == 0:
            identity = torch.zeros(active_experts, device=self.device, dtype=torch.float32)
            identity[0] = 1.0
            return identity
        if sparse_top_k is None:
            weights = F.softmax(
                expert_logits / max(float(self._get_relation_config()["routing_temperature"]), 1e-6),
                dim=0,
            )
        else:
            weights = self._build_sparse_relation_weights(expert_logits, sparse_top_k)
        return self._normalize_route_weights(weights)

    def _scatter_local_weights(self, active_experts, member_tasks, local_weights, device):
        weights = torch.zeros(active_experts, device=device, dtype=torch.float32)
        for local_idx, task_id in enumerate(member_tasks):
            weights[task_id] = local_weights[local_idx]
        return self._normalize_route_weights(weights)

    def _build_progressive_route_plan(self, active_experts, task_scores, image_summary, text_summary):
        device = task_scores.device
        if active_experts <= 1 or task_scores.numel() <= 1:
            identity = torch.zeros(active_experts, device=device, dtype=torch.float32)
            identity[0] = 1.0
            return {"early": identity, "middle": identity, "late": identity, "candidate_experts": [0]}

        role_weights, _ = self._build_role_weight_plan(
            active_experts,
            task_scores,
            image_summary,
            text_summary,
            use_all_roles=True,
        )
        early = self._build_early_route_weights(active_experts, task_scores, role_weights)
        middle = self._build_middle_route_weights(active_experts, task_scores, role_weights)
        top_k = max(1, min(int(self._get_relation_config()["routing_late_top_k"]), active_experts))
        late = self._build_sparse_relation_weights(task_scores, top_k)
        candidate_experts = torch.topk(late, k=top_k).indices.tolist()
        return {
            "early": early,
            "middle": middle,
            "late": late,
            "candidate_experts": candidate_experts,
        }

    def _get_layer_stage(self, layer_idx, total_layers):
        config = self._get_relation_config()
        late_layers = min(max(1, int(config.get("routing_late_layers", 1))), total_layers)
        available_before_late = max(0, total_layers - late_layers)
        early_layers = min(max(0, int(config["routing_early_layers"])), available_before_late)
        middle_layers = min(
            max(0, int(config["routing_middle_layers"])),
            max(0, available_before_late - early_layers),
        )
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

    def _commit_task_to_roles(self, task_id, task_image_anchor, task_text_anchor, membership, candidate_roles, role_birth):
        if task_id < 0 or task_id >= self.max_task_slots:
            return
        if self._get_active_role_count() == 0 or role_birth or len(candidate_roles) == 0:
            role_id = min(self._get_active_role_count(), self.max_role_slots - 1)
            self.role_image_prototypes[role_id].data.copy_(task_image_anchor.unsqueeze(0).to(self.role_image_prototypes[role_id].dtype))
            self.role_text_prototypes[role_id].data.copy_(task_text_anchor.unsqueeze(0).to(self.role_text_prototypes[role_id].dtype))
            self.role_task_count.data[role_id] = max(1.0, float(self.role_task_count[role_id].detach().item()))
            self.role_usage_prior.data[role_id] = max(1.0, float(self.role_usage_prior[role_id].detach().item()))
            self.task_role_membership.data[task_id].zero_()
            self.task_role_membership.data[task_id, role_id] = 1.0
            self._set_active_role_count(max(self._get_active_role_count(), role_id + 1))
        else:
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
        self.expert_usage_prior.data[task_id] = max(1.0, float(self.expert_usage_prior[task_id].detach().item()))

    def ensure_role_bank_initialized(self, completed_task_count):
        completed_task_count = min(int(completed_task_count), self.max_task_slots)
        if completed_task_count <= 0:
            return
        if self._get_active_role_count() > 0:
            return
        existing_membership = self.task_role_membership[:completed_task_count].detach().abs().sum().item()
        if existing_membership > 0:
            active_roles = int((self.role_task_count.detach() > 0).sum().item())
            if active_roles <= 0:
                active_roles = int((self.task_role_membership[:completed_task_count].detach().abs().sum(dim=0) > 0).sum().item())
            self._set_active_role_count(active_roles)
            return

        for task_id in range(completed_task_count):
            task_image_anchor, task_text_anchor = self._get_task_anchor(task_id)
            if task_id == 0 and self._get_active_role_count() == 0:
                self._commit_task_to_roles(task_id, task_image_anchor, task_text_anchor, None, [], True)
                continue
            expert_logits = self._score_tasks(
                list(range(task_id)),
                task_image_anchor,
                task_text_anchor,
                task_image_anchor.device,
            )
            membership, candidate_roles, role_birth = self._assign_roles_from_sample(
                task_id,
                expert_logits,
                task_image_anchor,
                task_text_anchor,
            )
            self._commit_task_to_roles(task_id, task_image_anchor, task_text_anchor, membership, candidate_roles, role_birth)

    def _finalize_current_task_role_memory_impl(self):
        task_id = min(max(int(self.cur_task), 0), self.max_task_slots - 1)
        if task_id > 0:
            self.ensure_role_bank_initialized(task_id)
        task_image_anchor, task_text_anchor = self._get_task_anchor(task_id)
        if task_id == 0 and self._get_active_role_count() == 0:
            self._commit_task_to_roles(task_id, task_image_anchor, task_text_anchor, None, [], True)
            return
        expert_logits = self._score_tasks(
            list(range(task_id)),
            task_image_anchor,
            task_text_anchor,
            task_image_anchor.device,
        )
        membership, candidate_roles, role_birth = self._assign_roles_from_sample(
            task_id,
            expert_logits,
            task_image_anchor,
            task_text_anchor,
        )
        self._commit_task_to_roles(task_id, task_image_anchor, task_text_anchor, membership, candidate_roles, role_birth)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        
        vision_tower = self.get_vision_tower()

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

        if self.training and not getattr(self, "disable_anchor_update", False):
            current_image_features = image_guide_features  # [batch_size, feature_dim]
            current_text_features = text_guide_features  # [batch_size, feature_dim]
            task_id = self.cur_task

            image_sum = self.image_anchors[task_id] * self.image_boundary[task_id] + current_image_features.sum(dim=0)
            text_sum = self.text_anchors[task_id] * self.text_boundary[task_id] + current_text_features.sum(dim=0)

            self.image_boundary[task_id].data += current_image_features.shape[0]
            self.text_boundary[task_id].data += current_text_features.shape[0]

            self.image_anchors[task_id] = image_sum / self.image_boundary[task_id]
            self.text_anchors[task_id] = text_sum / self.text_boundary[task_id]
        else:
            active_experts = max(1, min(int(self.expert_num), len(self.image_anchors)))
            self.ensure_role_bank_initialized(active_experts)
            image_summary = self._summarize_guide_features(image_guide_features)
            text_summary = self._summarize_guide_features(text_guide_features)
            task_scores = self._compute_shared_task_scores(
                image_guide_features=image_guide_features,
                text_guide_features=text_guide_features,
                active_experts=active_experts,
            )
            route_plan = self._build_progressive_route_plan(
                active_experts,
                task_scores,
                image_summary,
                text_summary,
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
