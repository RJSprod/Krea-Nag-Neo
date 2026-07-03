from __future__ import annotations

from typing import Any

import torch

from .logging import debug

EPS = 1e-7


def normalized_attention_guidance(z_positive: torch.Tensor, z_negative: torch.Tensor, scale: float, tau: float, alpha: float) -> torch.Tensor:
    original_dtype = z_positive.dtype
    pos = z_positive.float()
    neg = z_negative.float().to(device=pos.device)
    guided = pos * scale - neg * (scale - 1.0)
    norm_pos = pos.abs().sum(dim=-1, keepdim=True)
    norm_guided = guided.abs().sum(dim=-1, keepdim=True)
    ratio = norm_guided / (norm_pos + EPS)
    normalized = guided * torch.minimum(ratio, torch.tensor(float(tau), device=ratio.device, dtype=ratio.dtype)) / (ratio + EPS)
    out = normalized * alpha + pos * (1.0 - alpha)
    return out.to(dtype=original_dtype)


def _infer_image_slice(x: torch.Tensor, state: Any) -> slice:
    if state.image_token_slices:
        start, end = state.image_token_slices[0]
        return slice(start, end)
    text_len = max(state.text_lengths) if state.text_lengths else 0
    if text_len and text_len < x.shape[1]:
        return slice(text_len, x.shape[1])
    return slice(0, x.shape[1])


def apply_nag_to_attention_output(attn_out: torch.Tensor, state: Any) -> torch.Tensor:
    if not getattr(state, "active", False) or attn_out.ndim < 3 or attn_out.shape[0] < 2:
        return attn_out
    img_slice = _infer_image_slice(attn_out, state)
    pos_index = 0 if state.branch_layout == "turbo_cfg_disabled" else min(1, attn_out.shape[0] - 2)
    neg_index = attn_out.shape[0] - 1
    result = attn_out.clone()
    debug(state.debug, f"applying NAG: shape={tuple(attn_out.shape)} pos={pos_index} neg={neg_index} image_slice={img_slice.start}:{img_slice.stop} dtype={attn_out.dtype}")
    result[pos_index, img_slice, :] = normalized_attention_guidance(
        attn_out[pos_index, img_slice, :],
        attn_out[neg_index, img_slice, :],
        state.nag_scale,
        state.nag_tau,
        state.nag_alpha,
    )
    return result


def wrap_forward(module: Any, original_forward: Any, state: Any):
    def forward(*args: Any, **kwargs: Any):
        out = original_forward(*args, **kwargs)
        if torch.is_tensor(out):
            return apply_nag_to_attention_output(out, state)
        if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
            return (apply_nag_to_attention_output(out[0], state), *out[1:])
        return out
    return forward
