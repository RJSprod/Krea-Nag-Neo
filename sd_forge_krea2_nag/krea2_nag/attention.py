from __future__ import annotations

from typing import Any

import torch

from .logging import debug

EPS = 1e-7


def normalized_attention_guidance(z_positive: torch.Tensor, z_negative: torch.Tensor, scale: float, tau: float, alpha: float) -> torch.Tensor:
    """Low-VRAM NAG blend matching Wan2GP's attention-space implementation."""
    original_dtype = z_positive.dtype
    pos = z_positive.float()
    neg = z_negative.float().to(device=pos.device)

    # Wan2GP computes this in-place as:
    #   x_neg.mul_(1 - nag_scale); x_neg.add_(x_pos, alpha=nag_scale)
    guided = neg.mul(1.0 - scale).add(pos, alpha=scale)

    norm_pos = torch.norm(pos, p=1, dim=-1, keepdim=True)
    norm_guided = torch.norm(guided, p=1, dim=-1, keepdim=True)
    ratio = torch.nan_to_num(norm_guided / (norm_pos + EPS), 10.0)
    factor = norm_pos * float(tau) / (norm_guided + EPS)
    guided = torch.where(ratio > float(tau), guided * factor, guided)

    out = guided.mul(float(alpha)).add(pos, alpha=1.0 - float(alpha))
    return out.to(dtype=original_dtype)


def _context_tensor(state: Any, like: torch.Tensor) -> torch.Tensor | None:
    context = getattr(state, "nag_context", None)
    if context is None or not torch.is_tensor(context) or context.ndim != like.ndim:
        return None
    if context.shape[-1] != like.shape[-1]:
        return None
    return context.to(device=like.device, dtype=like.dtype)


def _infer_text_len(x: torch.Tensor, state: Any, context: torch.Tensor | None = None) -> int:
    if getattr(state, "text_lengths", None):
        text_len = max(int(v) for v in state.text_lengths if int(v) >= 0)
        if 0 < text_len < x.shape[1]:
            return text_len
    if context is not None:
        return min(int(context.shape[1]), max(0, x.shape[1] - 1))
    return 0


def _infer_image_slice(x: torch.Tensor, state: Any) -> slice:
    if state.image_token_slices:
        start, end = state.image_token_slices[0]
        return slice(start, end)
    text_len = _infer_text_len(x, state, _context_tensor(state, x))
    if text_len and text_len < x.shape[1]:
        return slice(text_len, x.shape[1])
    return slice(0, x.shape[1])


def _build_positive_negative_batch(x: torch.Tensor, state: Any) -> tuple[torch.Tensor, int] | None:
    """Create Wan2GP-style [positive, nag-negative] attention batch for Krea2.

    Krea2 is a single-stream DiT: text and image tokens are concatenated before
    attention.  Wan2GP's NAG runs attention once with positive text and once with
    negative text, then blends the positive image-token features.  This helper
    mirrors that layout by replacing the leading text-token slice with the
    encoded NAG negative context while keeping the same image tokens.
    """
    if not getattr(state, "active", False) or x.ndim != 3:
        return None
    context = _context_tensor(state, x)
    if context is None:
        return None
    text_len = _infer_text_len(x, state, context)
    if text_len <= 0 or text_len >= x.shape[1]:
        return None

    batch = x.shape[0]
    if context.shape[0] == 1 and batch != 1:
        context = context.expand(batch, -1, -1)
    elif context.shape[0] != batch:
        context = context[:batch]
        if context.shape[0] != batch:
            return None

    neg = x.clone()
    copy_len = min(text_len, context.shape[1])
    neg[:, :copy_len, :] = context[:, :copy_len, :]
    if copy_len < text_len:
        neg[:, copy_len:text_len, :] = 0
    return torch.cat([x, neg], dim=0), batch


def apply_nag_to_attention_output(attn_out: torch.Tensor, state: Any, positive_batch_size: int | None = None) -> torch.Tensor:
    if not getattr(state, "active", False) or attn_out.ndim < 3 or attn_out.shape[0] < 2:
        return attn_out
    if positive_batch_size is None:
        return attn_out
    pos_batch = int(positive_batch_size or 1)
    if attn_out.shape[0] < pos_batch * 2:
        return attn_out
    img_slice = _infer_image_slice(attn_out, state)
    result = attn_out.clone()
    debug(state.debug, f"applying NAG: shape={tuple(attn_out.shape)} pos_batch={pos_batch} image_slice={img_slice.start}:{img_slice.stop} dtype={attn_out.dtype}")
    result[:pos_batch, img_slice, :] = normalized_attention_guidance(
        attn_out[:pos_batch, img_slice, :],
        attn_out[pos_batch:pos_batch * 2, img_slice, :],
        state.nag_scale,
        state.nag_tau,
        state.nag_alpha,
    )
    return result[:pos_batch]


def _replace_first_tensor(args: tuple[Any, ...], replacement: torch.Tensor) -> tuple[Any, ...]:
    for idx, arg in enumerate(args):
        if torch.is_tensor(arg) and arg.ndim == 3:
            return (*args[:idx], replacement, *args[idx + 1:])
    return args


def wrap_forward(module: Any, original_forward: Any, state: Any):
    def forward(*args: Any, **kwargs: Any):
        positive_batch_size: int | None = None
        call_args = args
        for arg in args:
            if torch.is_tensor(arg) and arg.ndim == 3:
                expanded = _build_positive_negative_batch(arg, state)
                if expanded is not None:
                    x, positive_batch_size = expanded
                    call_args = _replace_first_tensor(args, x)
                    debug(state.debug, f"expanded Krea2 attention batch for Wan2GP-style NAG: {tuple(arg.shape)} -> {tuple(x.shape)}")
                break

        out = original_forward(*call_args, **kwargs)
        if torch.is_tensor(out):
            return apply_nag_to_attention_output(out, state, positive_batch_size)
        if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
            return (apply_nag_to_attention_output(out[0], state, positive_batch_size), *out[1:])
        return out
    return forward
