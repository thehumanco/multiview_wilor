"""Minimal, dependency-free LoRA for fine-tuning WiLoR's RefineNet without forgetting.

The env has no ``peft``/``loralib``, so this is a small self-contained adapter. ``LoRALinear``
wraps a (frozen) ``nn.Linear`` and adds a trainable low-rank delta ``B @ A`` scaled by
``alpha/rank``. ``B`` is zero-initialized, so at step 0 the wrapped layer is EXACTLY the
pretrained linear (identity adapter) — this keeps the "step 0 == pretrained WiLoR" property the
zero-gated fusion already gives.

A "relatively high" rank approaches full-rank finetuning of the head while still freezing the
pretrained weights, so the base capability is preserved (no catastrophic forgetting) and only the
small delta moves.
"""
import math
from typing import List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """``base(x) + scaling * (x A^T) B^T`` with ``base`` frozen and ``A,B`` trainable.

    rank is clamped to ``min(in, out)`` so tiny heads (e.g. the 3-d cam head) degrade
    gracefully to (near) full rank. ``B`` zero-init => delta == 0 at init.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)  # frozen pretrained weight (+ bias)

        in_f, out_f = base.in_features, base.out_features
        r = max(1, min(rank, in_f, out_f))
        self.rank = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B stays 0 => identity at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_A), self.lora_B)
        return self.base(x) + self.scaling * delta


def inject_lora_linear(module: nn.Module, rank: int, alpha: float) -> List[str]:
    """Recursively replace every ``nn.Linear`` under ``module`` with a ``LoRALinear``.

    Non-linear submodules (the RefineNet deconv's Conv2d/BatchNorm feature extractor) are left
    untouched and stay frozen by the caller, so only the low-rank head deltas train. Returns the
    dotted names of the layers that were wrapped (for logging/sanity).
    """
    wrapped: List[str] = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            wrapped.append(name)
        else:
            wrapped.extend(f"{name}.{w}" for w in inject_lora_linear(child, rank, alpha))
    return wrapped
