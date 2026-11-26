"""Fixed symmetric INT4 residuals, group32, with deployed FP16 scales.

Encoding is detached. Training replays gradients through the base codebook only.
Signed codes are held as int8 during training; this module makes no packed-memory
or serving-efficiency claim.
"""
import torch


@torch.no_grad()
def encode_residual(actual, base, group_size=32):
    if actual.shape != base.shape or actual.shape[-1] % group_size:
        raise ValueError("Residual tensors need equal shapes and complete channel groups")
    error = (actual.float() - base.float()).reshape(*actual.shape[:-1], -1, group_size)
    scale = (error.abs().amax(-1, keepdim=True) / 7).to(torch.float16)
    # Keep very small nonzero groups representable and define all-zero groups.
    scale = scale.clamp_min(torch.finfo(torch.float16).tiny)
    if not torch.isfinite(scale).all():
        raise ValueError("Residual scale exceeds the FP16 format")
    codes = torch.round(error / scale.float()).clamp(-7, 7).to(torch.int8)
    return codes.reshape_as(actual), scale


def decode_residual(encoded):
    codes, scale = encoded
    grouped = codes.reshape(*scale.shape[:-1], -1)
    return (grouped.float() * scale.float()).reshape_as(codes)
