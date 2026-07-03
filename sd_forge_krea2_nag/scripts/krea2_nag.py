from __future__ import annotations

import os
import sys

EXTENSION_ROOT = os.path.dirname(os.path.dirname(__file__))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

try:
    import gradio as gr
    from modules import scripts
except Exception:  # allows syntax checks outside Forge Neo
    gr = None
    class _BaseScript:  # type: ignore
        pass
    class scripts:  # type: ignore
        Script = _BaseScript

from krea2_nag.conditioning import encode_krea2_nag_negative
from krea2_nag.detect import detect_krea2
from krea2_nag.logging import debug, info, warn
from krea2_nag.patcher import install_krea2_attention_patches, restore_krea2_attention_patches
from krea2_nag.state import Krea2NagState
from krea2_nag.version import ADAPTER_VERSION


class Krea2NagScript(scripts.Script):
    def title(self):
        return "Krea2 NAG"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        if gr is None:
            return []
        with gr.Accordion("Krea2 Normalized Attention Guidance", open=False):
            enabled = gr.Checkbox(label="Enable Krea2 NAG", value=False)
            negative = gr.Textbox(label="NAG negative prompt", lines=3, placeholder="text, watermark, blurry, low quality, deformed hands", value="")
            nag_scale = gr.Slider(label="nag_scale", minimum=1.0, maximum=15.0, step=0.1, value=5.0)
            nag_tau = gr.Slider(label="nag_tau", minimum=0.1, maximum=5.0, step=0.05, value=2.5)
            nag_alpha = gr.Slider(label="nag_alpha", minimum=0.0, maximum=1.0, step=0.01, value=0.125)
            timestep_end = gr.Slider(label="Apply NAG until t <=", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
            mode = gr.Dropdown(label="Compatibility mode", choices=["auto", "krea2_turbo", "krea2_raw", "force_disable"], value="auto")
            debug_logging = gr.Checkbox(label="Debug logging", value=True)
        return [enabled, negative, nag_scale, nag_tau, nag_alpha, timestep_end, mode, debug_logging]

    def process_before_every_sampling(self, p, enabled, negative, nag_scale, nag_tau, nag_alpha, timestep_end, mode, debug_logging, *args):
        restore_krea2_attention_patches(getattr(p, "sd_model", None))
        if not enabled:
            return
        model = getattr(p, "sd_model", None)
        if model is None:
            warn("no active sd_model found; skipping")
            return
        detection = detect_krea2(model, p, mode)
        debug(debug_logging, f"detection={detection}")
        if not detection.is_krea2:
            debug(debug_logging, "active model is not detected as Krea2; no-op")
            return
        if not str(negative or "").strip() or nag_scale <= 1.0:
            debug(debug_logging, "empty NAG negative prompt or scale <= 1; no-op")
            return
        nag_context, nag_mask, prompt_batch = encode_krea2_nag_negative(p, model, negative, debug_logging)
        state = Krea2NagState(
            enabled=True,
            mode=detection.mode,
            nag_negative_prompt=prompt_batch,
            nag_scale=float(nag_scale),
            nag_tau=float(nag_tau),
            nag_alpha=float(nag_alpha),
            nag_timestep_end=float(timestep_end),
            positive_batch_size=int(getattr(p, "batch_size", 1) or 1),
            branch_layout="turbo_cfg_disabled" if float(getattr(p, "cfg_scale", 0.0) or 0.0) == 0.0 else "cfg_with_nag_negative",
            debug=bool(debug_logging),
            detection=detection,
            nag_context=nag_context,
            nag_mask=nag_mask,
        )
        count = install_krea2_attention_patches(model, state)
        params = getattr(p, "extra_generation_params", None)
        if isinstance(params, dict):
            params.update({
                "Krea2 NAG": "enabled",
                "NAG negative prompt": negative,
                "nag_scale": nag_scale,
                "nag_tau": nag_tau,
                "nag_alpha": nag_alpha,
                "nag_timestep_end": timestep_end,
                "Krea2 NAG mode detected": detection.mode,
                "Krea2 NAG adapter version": ADAPTER_VERSION,
            })
        info(f"enabled for {detection.mode}; patched {count} attention modules; dtype={detection.dtype}; quant={detection.quantization}")

    def postprocess(self, p, processed, *args):
        restored = restore_krea2_attention_patches(getattr(p, "sd_model", None))
        if restored:
            info(f"restored {restored} attention modules")
