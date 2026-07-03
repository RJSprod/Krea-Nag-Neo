import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sd_forge_krea2_nag"))
sys.modules.setdefault("torch", types.SimpleNamespace(Tensor=object, is_tensor=lambda value: False))

from krea2_nag.patcher import install_krea2_attention_patches, restore_krea2_attention_patches
from krea2_nag.state import Krea2NagState


class Projection:
    pass


class OfficialKreaAttention:
    def __init__(self):
        self.qkv = Projection()
        self.wo = Projection()

    def forward(self, x):
        return x


class ForgeWrapper:
    def __init__(self):
        block = types.SimpleNamespace(attn=OfficialKreaAttention())
        diffusion_model = types.SimpleNamespace(blocks=[block])
        self._modules = {"diffusion_model": diffusion_model}


class FusedAttention:
    def __init__(self):
        self.to_qkv = Projection()
        self.proj_out = Projection()

    def forward(self, x):
        return x


def _state():
    return Krea2NagState(enabled=True, nag_negative_prompt=["bad"], nag_scale=2.0)


def test_finds_modules_stored_under_private_modules_dict():
    wrapper = ForgeWrapper()
    try:
        assert install_krea2_attention_patches(wrapper, _state()) == 1
    finally:
        restore_krea2_attention_patches(wrapper)


def test_finds_forge_fused_qkv_projection_names():
    root = types.SimpleNamespace(blocks=[types.SimpleNamespace(mixer=FusedAttention())])
    try:
        assert install_krea2_attention_patches(root, _state()) == 1
    finally:
        restore_krea2_attention_patches(root)
