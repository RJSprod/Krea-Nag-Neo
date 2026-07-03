from __future__ import annotations

from typing import Any

from .attention import wrap_forward
from .logging import debug, warn

_ORIGINALS: dict[int, tuple[Any, Any]] = {}

ATTN_NAME_MARKERS = ("attn", "attention")
REQUIRED_ATTR_SETS = (("qkv", "wo"), ("to_q", "to_k", "to_v"), ("q", "k", "v"))


def _looks_like_attention(name: str, module: Any) -> bool:
    lname = name.lower()
    if not any(marker in lname or marker in module.__class__.__name__.lower() for marker in ATTN_NAME_MARKERS):
        return False
    if not callable(getattr(module, "forward", None)):
        return False
    return any(all(hasattr(module, attr) for attr in attrs) for attrs in REQUIRED_ATTR_SETS) or "attention" in module.__class__.__name__.lower()


def iter_attention_modules(model: Any):
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        yield from ((name, module) for name, module in named_modules() if _looks_like_attention(name, module))
        return
    for name in ("attention", "attn"):
        module = getattr(model, name, None)
        if module is not None and _looks_like_attention(name, module):
            yield name, module


def install_krea2_attention_patches(model: Any, state: Any) -> int:
    count = 0
    for name, module in iter_attention_modules(model):
        key = id(module)
        if key in _ORIGINALS:
            continue
        original = module.forward
        _ORIGINALS[key] = (module, original)
        module.forward = wrap_forward(module, original, state)
        count += 1
        debug(state.debug, f"patched attention module: {name} ({module.__class__.__name__})")
    state.patched_modules = count
    if count == 0:
        warn("no compatible Krea2 attention modules were found; NAG will be inactive")
    return count


def restore_krea2_attention_patches(model: Any | None = None) -> int:
    restored = 0
    for key, (module, original) in list(_ORIGINALS.items()):
        module.forward = original
        _ORIGINALS.pop(key, None)
        restored += 1
    return restored
