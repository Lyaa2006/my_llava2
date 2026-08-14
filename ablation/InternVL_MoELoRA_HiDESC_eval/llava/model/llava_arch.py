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

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter


        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            print("Loading mm_projector weights...")
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            has_vit_pos_embedding = False
            for key in mm_projector_weights.keys():
                if "position_embedding" in key:
                    has_vit_pos_embedding = True
                    break
            if has_vit_pos_embedding:
                print("Loading vision_tower.embeddings.position_embedding weights...")
                missing_keys, unexpected_keys = self.vision_tower.vision_tower.embeddings.load_state_dict(
                    get_w(mm_projector_weights, 'vision_tower.vision_tower.embeddings'), strict=False)
                if len(missing_keys) > 0:
                    print("missing keys:\n", missing_keys)
                if len(unexpected_keys) > 0:
                    print("unexpected_keys:\n", unexpected_keys)




class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        return_token_masks=False,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            if return_token_masks:
                text_token_mask = attention_mask.bool() if attention_mask is not None else None
                return input_ids, attention_mask, past_key_values, None, labels, None, text_token_mask
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_image_token_masks = []
        new_text_token_masks = []
        new_attention_masks = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx] if cur_image_idx < len(image_features) else None
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([
                    cur_input_embeds_1,
                    cur_input_embeds_2,
                ], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_image_token_masks.append(torch.zeros(cur_input_embeds.shape[0], dtype=torch.bool, device=cur_input_embeds.device))
                cur_attention = (
                    attention_mask[batch_idx].bool()
                    if attention_mask is not None
                    else torch.ones(cur_input_embeds.shape[0], dtype=torch.bool, device=cur_input_embeds.device)
                )
                new_text_token_masks.append(cur_attention.to(device=cur_input_embeds.device))
                new_attention_masks.append(cur_attention.to(device=cur_input_embeds.device))
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                if cur_image_features is not None:
                    cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_image_mask_parts = []
            cur_text_mask_parts = []
            cur_attention = (
                attention_mask[batch_idx]
                if attention_mask is not None
                else torch.ones(cur_input_ids.shape[0], dtype=torch.bool, device=cur_input_ids.device)
            )
            cur_attention_parts = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    text_prefix = self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach()
                    image_start = self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start])
                    text_prefix_attention = cur_attention[:image_token_start-1]
                    image_start_attention = cur_attention[image_token_start-1:image_token_start]
                    image_attention = torch.ones(
                        cur_image_features.shape[0],
                        dtype=cur_attention.dtype,
                        device=cur_attention.device,
                    )
                    image_end_attention = cur_attention[image_token_start+1:image_token_start+2]
                    cur_new_input_embeds.append(text_prefix)
                    cur_new_input_embeds.append(image_start)
                    cur_new_input_embeds.append(cur_image_features)
                    image_end = self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2])
                    cur_new_input_embeds.append(image_end)
                    cur_image_mask_parts.extend([
                        torch.zeros(text_prefix.shape[0], dtype=torch.bool, device=text_prefix.device),
                        torch.zeros(image_start.shape[0], dtype=torch.bool, device=image_start.device),
                        torch.ones(cur_image_features.shape[0], dtype=torch.bool, device=cur_image_features.device),
                        torch.zeros(image_end.shape[0], dtype=torch.bool, device=image_end.device),
                    ])
                    cur_text_mask_parts.extend([
                        torch.ones(text_prefix.shape[0], dtype=torch.bool, device=text_prefix.device),
                        torch.ones(image_start.shape[0], dtype=torch.bool, device=image_start.device),
                        torch.zeros(cur_image_features.shape[0], dtype=torch.bool, device=cur_image_features.device),
                        torch.ones(image_end.shape[0], dtype=torch.bool, device=image_end.device),
                    ])
                    cur_attention_parts.extend([
                        text_prefix_attention,
                        image_start_attention,
                        image_attention,
                        image_end_attention,
                    ])
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    text_prefix = self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    text_prefix_attention = cur_attention[:image_token_start]
                    image_attention = torch.ones(
                        cur_image_features.shape[0],
                        dtype=cur_attention.dtype,
                        device=cur_attention.device,
                    )
                    cur_new_input_embeds.append(text_prefix)
                    cur_new_input_embeds.append(cur_image_features)
                    cur_image_mask_parts.extend([
                        torch.zeros(text_prefix.shape[0], dtype=torch.bool, device=text_prefix.device),
                        torch.ones(cur_image_features.shape[0], dtype=torch.bool, device=cur_image_features.device),
                    ])
                    cur_text_mask_parts.extend([
                        torch.ones(text_prefix.shape[0], dtype=torch.bool, device=text_prefix.device),
                        torch.zeros(cur_image_features.shape[0], dtype=torch.bool, device=cur_image_features.device),
                    ])
                    cur_attention_parts.extend([text_prefix_attention, image_attention])
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                    cur_attention = cur_attention[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                    cur_attention = cur_attention[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    tail = self.get_model().embed_tokens(cur_input_ids).detach()
                else:
                    tail = self.get_model().embed_tokens(cur_input_ids)
                cur_new_input_embeds.append(tail)
                cur_image_mask_parts.append(torch.zeros(tail.shape[0], dtype=torch.bool, device=tail.device))
                cur_text_mask_parts.append(torch.ones(tail.shape[0], dtype=torch.bool, device=tail.device))
                cur_attention_parts.append(cur_attention)
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            cur_attention = torch.cat(cur_attention_parts, dim=0).to(device=cur_new_input_embeds.device)
            new_input_embeds.append(cur_new_input_embeds)
            new_image_token_masks.append(torch.cat(cur_image_mask_parts, dim=0).to(cur_new_input_embeds.device))
            new_text_token_masks.append(
                torch.cat(cur_text_mask_parts, dim=0).to(cur_new_input_embeds.device)
                & cur_attention.bool()
            )
            new_attention_masks.append(cur_attention)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        max_len = max(x.shape[0] for x in new_input_embeds)
        padding_side = getattr(self.config, "tokenizer_padding_side", "right")
        hidden_size = new_input_embeds[0].shape[-1]
        padded_embeds = []
        padded_image_masks = []
        padded_text_masks = []
        padded_attention = []
        padded_labels = [] if labels is not None else None

        for index, (cur_embeds, image_mask, text_mask) in enumerate(
            zip(new_input_embeds, new_image_token_masks, new_text_token_masks)
        ):
            cur_len = cur_embeds.shape[0]
            pad_len = max_len - cur_len
            if padding_side == "left":
                padded_embeds.append(torch.cat((
                    torch.zeros((pad_len, hidden_size), dtype=cur_embeds.dtype, device=cur_embeds.device),
                    cur_embeds,
                ), dim=0))
                padded_image_masks.append(torch.cat((
                    torch.zeros(pad_len, dtype=torch.bool, device=image_mask.device),
                    image_mask,
                ), dim=0))
                padded_text_masks.append(torch.cat((
                    torch.zeros(pad_len, dtype=torch.bool, device=text_mask.device),
                    text_mask,
                ), dim=0))
                padded_attention.append(torch.cat((
                    torch.zeros(pad_len, dtype=attention_mask.dtype, device=attention_mask.device)
                    if attention_mask is not None else torch.zeros(pad_len, dtype=torch.bool, device=cur_embeds.device),
                    new_attention_masks[index].to(
                        dtype=attention_mask.dtype if attention_mask is not None else torch.bool,
                        device=cur_embeds.device,
                    ),
                ), dim=0))
                if labels is not None:
                    padded_labels.append(torch.cat((
                        torch.full((pad_len,), IGNORE_INDEX, dtype=new_labels[index].dtype, device=new_labels[index].device),
                        new_labels[index],
                    ), dim=0))
            else:
                padded_embeds.append(torch.cat((
                    cur_embeds,
                    torch.zeros((pad_len, hidden_size), dtype=cur_embeds.dtype, device=cur_embeds.device),
                ), dim=0))
                padded_image_masks.append(torch.cat((
                    image_mask,
                    torch.zeros(pad_len, dtype=torch.bool, device=image_mask.device),
                ), dim=0))
                padded_text_masks.append(torch.cat((
                    text_mask,
                    torch.zeros(pad_len, dtype=torch.bool, device=text_mask.device),
                ), dim=0))
                padded_attention.append(torch.cat((
                    new_attention_masks[index].to(
                        dtype=attention_mask.dtype if attention_mask is not None else torch.bool,
                        device=cur_embeds.device,
                    ),
                    torch.zeros(pad_len, dtype=attention_mask.dtype, device=attention_mask.device)
                    if attention_mask is not None else torch.zeros(pad_len, dtype=torch.bool, device=cur_embeds.device),
                ), dim=0))
                if labels is not None:
                    padded_labels.append(torch.cat((
                        new_labels[index],
                        torch.full((pad_len,), IGNORE_INDEX, dtype=new_labels[index].dtype, device=new_labels[index].device),
                    ), dim=0))

        new_input_embeds = torch.stack(padded_embeds, dim=0)
        new_image_token_masks = torch.stack(padded_image_masks, dim=0)
        new_text_token_masks = torch.stack(padded_text_masks, dim=0)
        if labels is not None:
            new_labels = torch.stack(padded_labels, dim=0)
        if attention_mask is not None:
            attention_mask = torch.stack(padded_attention, dim=0)

        if return_token_masks:
            return (
                None,
                attention_mask,
                past_key_values,
                new_input_embeds,
                new_labels,
                new_image_token_masks,
                new_text_token_masks,
            )
        return None, attention_mask, past_key_values, new_input_embeds, new_labels

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
