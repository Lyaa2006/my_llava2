from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig

# The UCIT baseline uses the LLaMA backbone.  MPT support is optional, and its
# legacy Transformers-private imports are incompatible with newer versions;
# do not make importing the LLaMA path depend on MPT support.
try:
    from .language_model.llava_mpt import LlavaMPTForCausalLM, LlavaMPTConfig
except ImportError:
    LlavaMPTForCausalLM, LlavaMPTConfig = None, None
