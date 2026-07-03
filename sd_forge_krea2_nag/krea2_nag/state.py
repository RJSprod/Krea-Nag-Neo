from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Krea2NagState:
    enabled: bool = False
    mode: str = "unknown"
    nag_negative_prompt: list[str] = field(default_factory=list)
    nag_scale: float = 5.0
    nag_tau: float = 2.5
    nag_alpha: float = 0.125
    nag_timestep_end: float | None = None
    current_step: int | None = None
    current_timestep: float | None = None
    current_sigma: float | None = None
    positive_batch_size: int = 1
    branch_layout: str = "unknown"
    text_lengths: list[int] = field(default_factory=list)
    image_token_slices: list[tuple[int, int]] = field(default_factory=list)
    debug: bool = False
    detection: Any = None
    nag_context: Any = None
    nag_mask: Any = None
    patched_modules: int = 0

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.nag_negative_prompt) and self.nag_scale > 1.0
