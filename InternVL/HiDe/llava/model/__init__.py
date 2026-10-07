from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig

# MPT is a legacy optional backend.  The InternVL/HiDe evaluation path uses
# the LLaMA backend, while newer transformers versions no longer expose the
# old ``MptConfig`` symbols.  Keep the LLaMA/evaluation imports usable when
# that optional backend is unavailable.
try:
    from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
except ImportError:
    LlavaMptForCausalLM = None
    LlavaMptConfig = None
