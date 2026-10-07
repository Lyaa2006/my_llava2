"""Disable optional bitsandbytes discovery for unquantized inference smoke.

The remote legacy environment has a CUDA-11 bitsandbytes wheel but only CUDA
12 sparse libraries.  HiDESC smoke uses full-precision LoRA inference, so
bitsandbytes is not needed; hiding only package discovery lets peft/accelerate
import without changing the installed environment.  This file is activated
only when its directory is explicitly prepended to PYTHONPATH by the smoke
launcher.
"""

import importlib.util


_original_find_spec = importlib.util.find_spec


def _find_spec_without_bitsandbytes(name, *args, **kwargs):
    if name == "bitsandbytes" or name.startswith("bitsandbytes."):
        return None
    return _original_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _find_spec_without_bitsandbytes
