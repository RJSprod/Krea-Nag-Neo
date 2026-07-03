from __future__ import annotations

from typing import Any

from .logging import debug, warn


def build_prompt_batch(prompt: str, p: Any) -> list[str]:
    batch_size = int(getattr(p, "batch_size", 1) or 1) * int(getattr(p, "n_iter", 1) or 1)
    return [prompt] * max(1, batch_size)


def encode_krea2_nag_negative(p: Any, model: Any, prompt: str, debug_enabled: bool = False) -> tuple[Any, Any, list[str]]:
    prompts = build_prompt_batch(prompt, p)
    if not prompt.strip():
        return None, None, prompts
    for owner in (model, getattr(model, "conditioner", None), getattr(model, "forge_objects", None)):
        if owner is None:
            continue
        for method_name in ("encode_prompts", "encode_prompt", "get_learned_conditioning", "encode_text"):
            method = getattr(owner, method_name, None)
            if callable(method):
                try:
                    encoded = method(prompts)
                    debug(debug_enabled, f"encoded NAG negative prompt via {owner.__class__.__name__}.{method_name}")
                    if isinstance(encoded, tuple):
                        return encoded[0], encoded[1] if len(encoded) > 1 else None, prompts
                    return encoded, None, prompts
                except TypeError:
                    continue
                except Exception as exc:
                    warn(f"negative prompt encoding failed via {method_name}: {exc}")
                    return None, None, prompts
    warn("could not locate Forge/Krea2 text encoder; attention patches may be installed but no negative conditioning was encoded")
    return None, None, prompts
