# Krea2 NAG for Forge Neo

This extension implements Krea2-family Normalized Attention Guidance (NAG) for Forge Neo.
Initial support targets Krea2 Turbo bf16, Krea2 Turbo fp8_scaled, Krea2 Raw bf16,
and Krea2 LoRA workflows.

## Install

Copy `sd_forge_krea2_nag` into your Forge Neo `extensions/` directory and restart Forge Neo.
The **Krea2 Normalized Attention Guidance** section appears in txt2img.

## Controls

- **Enable Krea2 NAG**: enables the extension for the current txt2img request.
- **NAG negative prompt**: the attention-space negative prompt. Empty prompts are a no-op.
- **nag_scale**: guidance strength. `1.0` is effectively no guidance.
- **nag_tau**: L1 normalization cap.
- **nag_alpha**: blend from original positive attention features to normalized guided features.
- **Apply NAG until t <=**: reserved timestep cutoff control for Forge sampler integration.
- **Compatibility mode**: `auto`, `krea2_turbo`, `krea2_raw`, or `force_disable`.
- **Debug logging**: prints model detection, patching, dtype, and attention tensor diagnostics.

## Warnings

nvfp4, mxfp8, int8, GGUF, SageAttention, FlashAttention, Spectrum, TeaCache,
WaveSpeed, and full `torch.compile` paths are experimental or unsupported until tested.
Disable other attention-guidance extensions when using Krea2 NAG.

## Notes

NAG is applied inside compatible Krea2 attention modules, not as normal CFG at the
final denoiser output. The extension patches the active model instance after load
and restores patched attention forwards after generation.
