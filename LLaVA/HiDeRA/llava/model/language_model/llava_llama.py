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
        self.max_task_slots = 10
        self.max_role_slots = self.max_task_slots

        # Initialize anchors
        self.image_anchors = nn.ParameterList(
            [nn.Parameter(0.1 * torch.randn(1, 768)) for _ in range(self.max_task_slots)]
        )

        self.text_anchors = nn.ParameterList(
            [nn.Parameter(0.1 * torch.randn(1, 768)) for _ in range(self.max_task_slots)]
        )

        self.image_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1, dtype=torch.bfloat16)) for _ in range(self.max_task_slots)]
            )
        self.text_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1, dtype=torch.bfloat16)) for _ in range(self.max_task_slots)]
            )

        self.expert_weight = [0., 0., 0., 0., 0., 0., 0., 0.]
        self.task_relation_scores = nn.Parameter(
            torch.zeros(self.max_task_slots, self.max_task_slots), requires_grad=False
        )
        self.expert_usage_prior = nn.Parameter(torch.zeros(self.max_task_slots), requires_grad=False)
        self.role_image_prototypes = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 768), requires_grad=False) for _ in range(self.max_role_slots)]
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
        self.transfer_aux_loss = torch.zeros(())
        self.routing_aux_loss = torch.zeros(())
        self.cached_task_fuse_weights = None
        self.cached_task_fuse_task_id = None
        self.cached_task_fuse_expert_num = None
        self.relation_routing_config = {
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
        }
        self._progressive_task_id = None
        self._bootstrap_seen_steps = 0
        self._bootstrap_image_sum = None
        self._bootstrap_text_sum = None
        self._bootstrap_sample_count = 0
        self._pending_role_membership = None
        self._pending_candidate_roles = []
        self._pending_candidate_experts = []
        self._pending_role_birth = False
        self._role_induction_complete = False

    def set_cur_task(self, cur_task, expert_num):
        self.cur_task = cur_task
        self.expert_num = expert_num
        self.clear_relation_routing_cache()
        self.reset_relation_losses()
        self.reset_progressive_state()

        for name, param in self.image_anchors.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_anchors.named_parameters():
            param.requires_grad = True

    def set_boundary_for_save(self):
        for name, param in self.image_boundary.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_boundary.named_parameters():
            param.requires_grad = True

        for name, param in self.image_anchors.named_parameters():
            param.requires_grad = True
        
        for name, param in self.text_anchors.named_parameters():
            param.requires_grad = True

    def configure_relation_routing(self, **kwargs):
        for key, value in kwargs.items():
            if value is None:
                continue
            self.relation_routing_config[key] = value
        self.clear_relation_routing_cache()

    def clear_relation_routing_cache(self):
        self.cached_task_fuse_weights = None
        self.cached_task_fuse_task_id = None
        self.cached_task_fuse_expert_num = None

    def reset_progressive_state(self):
        self._progressive_task_id = int(self.cur_task)
        self._bootstrap_seen_steps = 0
        self._bootstrap_image_sum = None
        self._bootstrap_text_sum = None
        self._bootstrap_sample_count = 0
        self._pending_role_membership = None
        self._pending_candidate_roles = []
        self._pending_candidate_experts = []
        self._pending_role_birth = False
        self._role_induction_complete = False

    def reset_relation_losses(self, device=None):
        if device is None:
            device = self.lm_head.weight.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        self.transfer_aux_loss = zero
        self.routing_aux_loss = zero

    def get_model(self):
        return self.model

    def set_eval(self, num_task, eval_task_id=None):
        self.expert_num = int(num_task)
        if eval_task_id is None:
            self.cur_task = max(0, self.expert_num - 1)
        else:
            self.cur_task = min(max(int(eval_task_id), 0), max(0, self.expert_num - 1))
        self.clear_relation_routing_cache()
        self.reset_relation_losses()
        self.reset_progressive_state()
        if hasattr(self, "ensure_role_bank_initialized"):
            self.ensure_role_bank_initialized(self.expert_num)

    def set_clip_tokenizer(self, tokenizer):
        self.clip_tokenizer = tokenizer

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def finalize_current_task_role_memory(self):
        if hasattr(self, "_finalize_current_task_role_memory_impl"):
            self._finalize_current_task_role_memory_impl()

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
