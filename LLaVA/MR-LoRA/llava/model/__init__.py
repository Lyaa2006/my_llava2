from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig

try:
    from .language_model.llava_mpt import LlavaMPTForCausalLM, LlavaMPTConfig
except ImportError:
    # MPT support depends on older Transformers internals. Keep LLaMA routes usable.
    LlavaMPTForCausalLM = None
    LlavaMPTConfig = None
