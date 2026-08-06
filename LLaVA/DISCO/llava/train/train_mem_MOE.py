# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

if torch.cuda.is_available():
    try:
        replace_llama_attn_with_flash_attn()
    except RuntimeError:
        pass

from llava.train.train_MOE import train

if __name__ == "__main__":
    train()
