from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .attention import wrap_forward
from .logging import debug, warn

_ORIGINALS: dict[int, tuple[Any, Any]] = {}

ATTN_NAME_MARKERS = ("attn", "attention")
ATTN_CLASS_MARKERS = ("attention", "selfattention", "jointattention")
REQUIRED_ATTR_SETS = (
    ("qkv", "wo"),  # official Krea2/mmdit style
    ("qkv", "o"),
    ("qkv", "proj"),
    ("to_q", "to_k", "to_v"),
    ("q", "k", "v"),
)
CONTAINER_ATTRS = (
    "model",
    "diffusion_model",
    "inner_model",
    "inner",
    "forge_objects",
    "unet",
    "transformer",
    "model_base",
    "base_model",
)
BLOCK_ATTRS = ("blocks", "transformer_blocks", "double_blocks", "single_blocks", "joint_blocks")
MAX_FALLBACK_DEPTH = 8


def _has_attention_projections(module: Any) -> bool:
    return any(all(hasattr(module, attr) for attr in attrs) for attrs in REQUIRED_ATTR_SETS)


def _looks_like_attention(name: str, module: Any) -> bool:
    if not callable(getattr(module, "forward", None)):
        return False
    lname = name.lower()
    cls_name = module.__class__.__name__.lower()
    has_name_hint = any(marker in lname for marker in ATTN_NAME_MARKERS) or any(marker in cls_name for marker in ATTN_CLASS_MARKERS)
    has_projection_hint = _has_attention_projections(module)
    # Forge Neo/Krea2 wrappers sometimes expose official qkv/wo attention modules
    # under block-local names that do not contain "attn".  Projection structure is
    # the stronger compatibility signal, so accept it even without a name hint.
    return has_projection_hint or (has_name_hint and "attention" in cls_name)


def _iter_named_modules(model: Any) -> Iterable[tuple[str, Any]]:
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        yield from named_modules()


def _iter_fallback_modules(root: Any) -> Iterable[tuple[str, Any]]:
    seen: set[int] = set()

    def walk(obj: Any, path: str, depth: int):
        if obj is None or depth > MAX_FALLBACK_DEPTH:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        yield path, obj

        children: list[tuple[str, Any]] = []
        for attr in (*CONTAINER_ATTRS, *BLOCK_ATTRS, "attention", "attn"):
            try:
                child = getattr(obj, attr, None)
            except Exception:
                child = None
            if child is not None:
                children.append((f"{path}.{attr}" if path else attr, child))
        if isinstance(obj, dict):
            children.extend((f"{path}.{key}" if path else str(key), value) for key, value in obj.items())
        elif isinstance(obj, (list, tuple)):
            children.extend((f"{path}.{idx}" if path else str(idx), value) for idx, value in enumerate(obj))
        else:
            for key, value in getattr(obj, "__dict__", {}).items():
                if key.startswith("_") or callable(value):
                    continue
                children.append((f"{path}.{key}" if path else key, value))

        for child_path, child in children:
            yield from walk(child, child_path, depth + 1)

    yield from walk(root, "", 0)


def iter_attention_modules(model: Any, state: Any | None = None):
    seen: set[int] = set()
    candidates: list[str] = []

    for source_name, source_iter in (("named_modules", _iter_named_modules(model)), ("fallback", _iter_fallback_modules(model))):
        for name, module in source_iter:
            key = id(module)
            if key in seen:
                continue
            seen.add(key)
            if _looks_like_attention(name, module):
                debug(getattr(state, "debug", False), f"compatible attention candidate via {source_name}: {name or '<root>'} ({module.__class__.__name__})")
                yield name or "<root>", module
            elif getattr(state, "debug", False) and callable(getattr(module, "forward", None)) and (_has_attention_projections(module) or "att" in module.__class__.__name__.lower()):
                candidates.append(f"{name or '<root>'} ({module.__class__.__name__})")
    if candidates:
        debug(True, "unpatched attention-like candidates: " + "; ".join(candidates[:20]))


def install_krea2_attention_patches(model: Any, state: Any) -> int:
    count = 0
    for name, module in iter_attention_modules(model, state):
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
        warn("no compatible Krea2 attention modules were found; NAG will be inactive; enable debug logging to inspect discovery")
    return count


def restore_krea2_attention_patches(model: Any | None = None) -> int:
    restored = 0
    for key, (module, original) in list(_ORIGINALS.items()):
        module.forward = original
        _ORIGINALS.pop(key, None)
        restored += 1
    return restored
