# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
# from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# replace_llama_attn_with_flash_attn()
import os
import sys

CURRENT_METHOD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CURRENT_METHOD_ROOT not in sys.path:
    sys.path.insert(0, CURRENT_METHOD_ROOT)

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")  # attn_implementation="flash_attention_2"
