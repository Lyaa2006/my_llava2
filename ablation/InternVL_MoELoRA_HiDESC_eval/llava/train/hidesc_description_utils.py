from typing import List, Optional, Tuple

import torch

from llava.constants import IMAGE_TOKEN_INDEX


DESCRIPTION_KEY_TERMS = (
    "object",
    "objects",
    "attribute",
    "attributes",
    "shape",
    "shapes",
    "color",
    "colors",
    "texture",
    "textures",
    "scene",
    "text",
    "spatial",
    "relation",
    "relations",
)


def select_expanded_description_tokens(
    hidden_states: torch.Tensor,
    expanded_text_mask: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    key_mask: Optional[torch.Tensor],
    max_tokens: int,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Select the text states after the image sentinel expands to image patches."""
    attention_mask = (
        torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask is None
        else attention_mask.bool()
    )
    key_mask = key_mask.bool() if key_mask is not None else None

    selected_states = []
    selected_key_masks = [] if key_mask is not None else None
    for batch_idx in range(hidden_states.shape[0]):
        raw_valid = attention_mask[batch_idx]
        raw_ids = input_ids[batch_idx][raw_valid]
        raw_key_mask = key_mask[batch_idx][raw_valid] if key_mask is not None else None
        raw_text_key_mask = (
            raw_key_mask[raw_ids != IMAGE_TOKEN_INDEX]
            if raw_key_mask is not None
            else None
        )

        text_positions = torch.nonzero(
            expanded_text_mask[batch_idx].bool(), as_tuple=False
        ).flatten()
        if raw_text_key_mask is not None:
            if text_positions.numel() > raw_text_key_mask.numel():
                raise ValueError(
                    "Expanded text mask contains more positions than the raw input."
                )
            expanded_key_mask = torch.zeros(
                expanded_text_mask.shape[1],
                dtype=torch.bool,
                device=hidden_states.device,
            )
            expanded_key_mask[text_positions] = raw_text_key_mask[
                : text_positions.numel()
            ].to(device=hidden_states.device)
        else:
            expanded_key_mask = None

        selected_positions = text_positions[-max_tokens:]
        selected_states.append(hidden_states[batch_idx, selected_positions])
        if selected_key_masks is not None:
            selected_key_masks.append(expanded_key_mask[selected_positions])

    return selected_states, selected_key_masks


def build_description_key_mask(
    input_ids: torch.Tensor,
    tokenizer,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mark description tokens that carry visual attributes or relations."""
    key_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    special_token_ids = set(tokenizer.all_special_ids)
    vocab_size = len(tokenizer)
    for batch_idx in range(input_ids.shape[0]):
        valid_len = int(attention_mask[batch_idx].long().sum().item())
        for token_idx in range(valid_len):
            token_id = int(input_ids[batch_idx, token_idx].item())
            if token_id < 0 or token_id >= vocab_size or token_id in special_token_ids:
                continue
            try:
                token_text = tokenizer.decode(
                    [token_id], skip_special_tokens=True
                ).strip().lower()
            except OverflowError:
                continue
            token_text = "".join(ch for ch in token_text if ch.isalnum())
            if token_text and any(
                term in token_text or token_text in term
                for term in DESCRIPTION_KEY_TERMS
            ):
                key_mask[batch_idx, token_idx] = True
        if not torch.any(key_mask[batch_idx, :valid_len]):
            key_mask[batch_idx, :valid_len] = attention_mask[
                batch_idx, :valid_len
            ].bool()
    return key_mask


def pad_description_sequences(
    sequences: List[torch.Tensor],
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Description sequence list must not be empty.")
    max_len = max(sequence.shape[0] for sequence in sequences)
    hidden_size = sequences[0].shape[-1]
    device = sequences[0].device
    dtype = dtype or sequences[0].dtype
    padded = torch.zeros(
        len(sequences), max_len, hidden_size, dtype=dtype, device=device
    )
    mask = torch.zeros(len(sequences), max_len, dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        length = sequence.shape[0]
        padded[index, :length] = sequence.to(dtype=dtype)
        mask[index, :length] = True
    return padded, mask
