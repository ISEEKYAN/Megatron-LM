"""Vendored standalone Megatron-FSDP implementation.

Imports stay lazy so checkpoint helpers can reuse ``utils`` without loading the
CUDA-facing wrapper at package import time.
"""

__all__: list[str] = []
