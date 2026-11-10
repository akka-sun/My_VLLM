"""Symmetric per-output-channel W8A16 quantization.

CUDA dispatches to a fused Triton kernel; CPU is a numerical reference path.
Weights are quantized after tensor-parallel slicing, from floating checkpoints.
"""
import torch
import torch.nn.functional as F


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not weight.is_floating_point() or weight.ndim != 2:
        raise ValueError('INT8 loading requires a floating-point weight matrix')
    values = weight.float()
    maximum = values.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
    quantized = (values / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def int8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
    if x.is_cuda:
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError('W8A16 CUDA activations must be float16 or bfloat16')
        from nanovllm.layers.quantization_triton import linear
        return linear(x, weight, scale, bias)
    # Test/reference path; never used by the CUDA inference engine.
    restored = (weight.float() * scale).to(x.dtype)
    return F.linear(x, restored, bias.to(x.dtype) if bias is not None else None)
