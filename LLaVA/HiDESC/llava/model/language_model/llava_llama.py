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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.training = False
        self.cur_task = 0
        self.expert_num = 8
        self.declared_expert_num = self.expert_num
        self.effective_expert_num = self.expert_num
        self.max_task_slots = 10
        self.max_role_slots = self.max_task_slots
        saved_relation_config = getattr(config, "relation_routing_config", None)
        if not isinstance(saved_relation_config, dict):
            saved_relation_config = {}
        spectral_low_bins = int(saved_relation_config.get("spectral_low_bins", 4))
        spectral_high_bins = int(saved_relation_config.get("spectral_high_bins", 4))
        configured_spectral_dim = getattr(config, "mm_spectral_feature_dim", None)
        self.spectral_image_dim = int(
            configured_spectral_dim
            or 768 * (spectral_low_bins + spectral_high_bins)
        )

        self.text_anchors = nn.ParameterList(
            [nn.Parameter(0.1 * torch.randn(1, 768)) for _ in range(self.max_task_slots)]
        )

        self.text_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1, dtype=torch.bfloat16)) for _ in range(self.max_task_slots)]
            )
        self.spectral_image_anchors = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, self.spectral_image_dim))
                for _ in range(self.max_task_slots)
            ]
        )
        self.spectral_image_boundary = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, dtype=torch.float32))
                for _ in range(self.max_task_slots)
            ]
        )

        self.expert_weight = [0., 0., 0., 0., 0., 0., 0., 0.]
        self.expert_usage_prior = nn.Parameter(torch.zeros(self.max_task_slots), requires_grad=False)
        self.role_spectral_prototypes = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(1, self.spectral_image_dim), requires_grad=False
                )
                for _ in range(self.max_role_slots)
            ]
        )
        self.role_text_prototypes = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 768), requires_grad=False) for _ in range(self.max_role_slots)]
        )
        self.role_task_count = nn.Parameter(torch.zeros(self.max_role_slots), requires_grad=False)
        self.role_usage_prior = nn.Parameter(torch.zeros(self.max_role_slots), requires_grad=False)
        self.task_role_membership = nn.Parameter(
            torch.zeros(self.max_task_slots, self.max_role_slots), requires_grad=False
        )
        self.active_role_count = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.relation_routing_config = {
            "use_spectral_image_routing": True,
            "use_spectral_role_prototype": True,
            "role_reset_on_strategy_change": False,
            "use_text_anchor_routing": True,
            "use_stage1_band_schedule_eval": True,
            "stage1_band_schedule_path": None,
            "eval_use_role_spectral_prototype": True,
            "eval_disable_role_image_prototype": True,
            "spectral_cutoff": 0.33,
            "spectral_low_bins": 4,
            "spectral_high_bins": 4,
            "spectral_image_weight": 0.5,
            "text_weight": 0.5,
            "history_weight": 0.15,
            "routing_image_weight": 0.5,
            "routing_text_weight": 0.5,
            "routing_history_weight": 0.15,
            "routing_temperature": 0.1,
            "routing_min_similarity": -1.0,
            "routing_prior_momentum": 0.8,
            "role_top_k": 2,
            "role_birth_threshold": 0.55,
            "role_assignment_strategy": "task_affinity_complete_link",
            "role_assignment_score_mode": "member_max",
            "role_assignment_top_k": 1,
            "role_assignment_min_similarity": 0.68,
            "role_assignment_margin": 0.10,
            "role_assignment_pair_weight": 0.25,
            "role_member_top_k": 2,
            "routing_role_prior_weight": 0.1,
            "routing_role_member_weight": 0.85,
            "routing_role_size_penalty": 0.20,
            "routing_role_score_mode": "member_max",
            "routing_role_prototype_weight": 0.15,
            "routing_early_layers": 16,
            "routing_early_mode": "task_softmax_within_role",
            "routing_early_uniform_mix": 0.25,
            "routing_middle_layers": 13,
            "routing_middle_temperature": 0.5,
            "routing_middle_role_gamma": 1.15,
            "routing_middle_role_uniform_mix": 0.10,
            "routing_middle_task_uniform_mix": 0.10,
            "routing_middle_role_margin_low": 0.10,
            "routing_middle_role_margin_high": 0.30,
            "routing_middle_intra_margin_low": 0.05,
            "routing_middle_intra_margin_high": 0.20,
            "routing_late_layers": 3,
            "routing_late_top_k": 2,
        }
        self.relation_routing_config.update(saved_relation_config)
        self.config.mm_spectral_feature_dim = self.spectral_image_dim
        self.config.relation_routing_config = dict(self.relation_routing_config)

    def set_cur_task(self, cur_task, expert_num):
        self.cur_task = cur_task
        self.expert_num = expert_num
        self.declared_expert_num = expert_num
        self.effective_expert_num = expert_num

        for name, param in self.text_anchors.named_parameters():
            param.requires_grad = True

        for name, param in self.spectral_image_anchors.named_parameters():
            param.requires_grad = True

    def set_boundary_for_save(self):
        for name, param in self.text_boundary.named_parameters():
            param.requires_grad = True

        for name, param in self.spectral_image_boundary.named_parameters():
            param.requires_grad = True

        for name, param in self.text_anchors.named_parameters():
            param.requires_grad = True

        for name, param in self.role_spectral_prototypes.named_parameters():
            param.requires_grad = True

        for name, param in self.role_text_prototypes.named_parameters():
            param.requires_grad = True

        self.expert_usage_prior.requires_grad = True
        self.role_task_count.requires_grad = True
        self.role_usage_prior.requires_grad = True
        self.task_role_membership.requires_grad = True
        self.active_role_count.requires_grad = True

    def get_model(self):
        return self.model

    def configure_relation_routing(self, **kwargs):
        for key, value in kwargs.items():
            if value is None:
                continue
            self.relation_routing_config[key] = value
        self.config.relation_routing_config = dict(self.relation_routing_config)

    def set_eval(self, num_task, eval_task_id=None, effective_num_task=None):
        self.declared_expert_num = int(num_task)
        if effective_num_task is None:
            effective_num_task = self.declared_expert_num
        self.effective_expert_num = max(1, min(int(effective_num_task), self.max_task_slots))
        self.expert_num = self.effective_expert_num
        if eval_task_id is None:
            self.cur_task = max(0, self.expert_num - 1)
        else:
            self.cur_task = min(max(int(eval_task_id), 0), max(0, self.expert_num - 1))

    def finalize_current_task_role_memory(self):
        if hasattr(self, "_finalize_current_task_role_memory_impl"):
            self._finalize_current_task_role_memory_impl()

    def set_clip_tokenizer(self, tokenizer):
        self.clip_tokenizer = tokenizer

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
