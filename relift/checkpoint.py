"""Checkpoint files.

Release format (``format`` = 1): ``state_dict`` (EMA weights), ``config`` (constructor arguments of ``Corrector``),
``step``, ``train_args`` (the training command's arguments, for the record) and ``source_md5``.  The research-time
training checkpoint (``model``, ``ema``, ``opt``, ``args``, ``step``, ``elapsed``) is accepted too; its EMA weights
are used, as in the paper.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import torch

from .model import IN_FEATS, Corrector
from .records.model import CheckpointInfo, ModelConfig, ReleaseCheckpoint

FORMAT = 1


def md5(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _config_from_train_args(a: Any) -> ModelConfig:
    assert a["mode"] == "regress_clean", f"not a corrector checkpoint (mode {a['mode']})"
    return dict(
        dim=a["dim"],
        depth=a["depth"],
        heads=a["dim"] // 64,
        in_feats=IN_FEATS,
        chunk=a.get("chunk", 4),
    )


def export(src: str | Path, dst: str | Path) -> ReleaseCheckpoint:
    """Training checkpoint -> release checkpoint (EMA weights only).  Returns the release dict."""
    ck = torch.load(src, map_location="cpu")
    out: ReleaseCheckpoint = dict(
        format=FORMAT,
        state_dict=ck["ema"],
        config=_config_from_train_args(ck["args"]),
        step=int(ck["step"]),
        train_args=dict(ck["args"]),
        source_md5=md5(src),
        num_parameters=int(sum(v.numel() for v in ck["ema"].values())),
    )
    torch.save(out, dst)
    return out


def load(path: str | Path, device: str | torch.device = "cuda") -> tuple[Corrector, CheckpointInfo]:
    """-> (model in eval mode on ``device``, info dict with ``step``, ``config``, ``format``)."""
    ck = torch.load(path, map_location=device)
    cfg: ModelConfig
    sd: dict[str, torch.Tensor]
    step: int
    if "format" in ck:
        cfg, sd, step = cast(ModelConfig, ck["config"]), ck["state_dict"], ck["step"]
    else:  # research-time training checkpoint
        cfg, sd, step = _config_from_train_args(ck["args"]), ck["ema"], ck["step"]
    model = Corrector(**cfg).to(device)
    model.load_state_dict(sd)
    return model.eval(), dict(step=int(step), config=cfg, format=ck.get("format", 0))
