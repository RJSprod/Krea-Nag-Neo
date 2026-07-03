from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KREA_MARKERS = ("krea", "krea2", "single stream dit", "singlestreamdit", "singlemmdit", "qwen3vlconditioner")
PATH_MARKERS = ("krea-2", "krea2", "krea_2", "krea-2-turbo", "krea-2-raw", "turbo.safetensors", "raw.safetensors")


@dataclass
class Krea2Detection:
    is_krea2: bool
    mode: str = "unknown"
    reason: str = ""
    class_name: str = ""
    checkpoint: str = ""
    dtype: str = "unknown"
    quantization: str = "unknown"
    markers: list[str] = field(default_factory=list)


def _safe_str(value: Any) -> str:
    try:
        return str(value or "")
    except Exception:
        return ""


def _collect_model_text(model: Any, p: Any | None) -> tuple[str, str, str, str]:
    names = [model.__class__.__name__, model.__class__.__module__]
    for attr in ("model", "diffusion_model", "forge_objects", "conditioner", "text_encoder"):
        obj = getattr(model, attr, None)
        if obj is not None:
            names.extend([obj.__class__.__name__, obj.__class__.__module__])
    checkpoint_bits = []
    for obj in (p, model, getattr(model, "sd_checkpoint_info", None), getattr(model, "checkpoint_info", None)):
        if obj is None:
            continue
        for attr in ("sd_model_checkpoint", "filename", "name", "title", "model_name", "model_path", "checkpoint", "repo_id"):
            checkpoint_bits.append(_safe_str(getattr(obj, attr, "")))
    dtype = _safe_str(getattr(model, "dtype", "") or getattr(getattr(model, "model", None), "dtype", "")) or "unknown"
    quant = " ".join(_safe_str(getattr(model, attr, "")) for attr in ("quantization", "weight_dtype", "load_device")) or "unknown"
    return " ".join(names), " ".join(checkpoint_bits), dtype, quant


def detect_krea2(model: Any, p: Any | None = None, requested_mode: str = "auto") -> Krea2Detection:
    if requested_mode == "force_disable":
        return Krea2Detection(False, reason="forced disabled")
    model_text, checkpoint_text, dtype, quant = _collect_model_text(model, p)
    haystack = f"{model_text} {checkpoint_text}".lower().replace("_", "-")
    markers = [m for m in KREA_MARKERS + PATH_MARKERS if m.lower().replace("_", "-") in haystack]
    has_shape_hint = any(hasattr(model, attr) for attr in ("blocks", "transformer_blocks", "double_blocks", "single_blocks"))
    is_krea2 = bool(markers) or has_shape_hint and "qwen" in haystack
    mode = "unknown"
    if "turbo" in haystack or requested_mode == "krea2_turbo":
        mode = "turbo"
    elif "raw" in haystack or requested_mode == "krea2_raw":
        mode = "raw"
    elif requested_mode in ("krea2_turbo", "krea2_raw"):
        mode = requested_mode.replace("krea2_", "")
        is_krea2 = True
    return Krea2Detection(is_krea2, mode, "matched markers" if is_krea2 else "no Krea2 markers", model_text, checkpoint_text, dtype, quant, markers)
