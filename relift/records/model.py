"""Training, checkpoint, provider and evaluation record contracts."""

from typing import Any, Literal, NotRequired, TypedDict

import numpy as np
import torch
from PIL.Image import Image

type Array = np.ndarray[Any, Any]


class ModelConfig(TypedDict, total=False):
    dim: int
    depth: int
    heads: int
    in_feats: int
    chunk: int


class CheckpointInfo(TypedDict):
    step: int
    config: ModelConfig
    format: int


class ReleaseCheckpoint(TypedDict):
    format: int
    state_dict: dict[str, torch.Tensor]
    config: ModelConfig
    step: int
    train_args: dict[str, Any]
    source_md5: str
    num_parameters: int


class TrainingBatch(TypedDict):
    X: torch.Tensor
    Y: torch.Tensor
    obs: torch.Tensor
    ray: torch.Tensor
    origin: torch.Tensor
    dyn: torch.Tensor
    vis: torch.Tensor


class LossMetrics(TypedDict):
    pos: torch.Tensor
    vel: torch.Tensor


class Geometry(TypedDict):
    depth: Array
    poses: Array
    intrinsics: Array


class DA3Geometry(Geometry):
    images: Array


class RC4Geometry(Geometry):
    model_hw: Array
    valid_depth: float
    check_px: float
    check_rel: float


class VGGTGeometry(Geometry):
    depth_conf: Array
    internal_hw: list[int]


class WorldTrackClip(TypedDict):
    frames: list[Image]
    tracks_uv: Array
    intrinsics: Array
    world: Array
    visibility: Array
    name: str


class VisibilityStats(TypedDict):
    name: str
    mean: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    frac_gt_085: float
    frac_gt_05: float
    hist: list[float]


class SubsetMeans(TypedDict):
    all: float
    dyn: float
    clips: int


class PointOdysseyWindows(TypedDict):
    X: list[Array]
    vis: list[Array]
    origin: list[Array]
    dyn: list[Array]
    step: list[int]
    start: list[int]


type AlignmentMode = Literal["sim3", "global"]
type SubsetName = Literal["adt", "pstudio", "po", "ds"]


class AlignmentScore(TypedDict):
    apd: float
    epe: float
    occ: float
    vis: float
    dyn: NotRequired[float]


ClipScore = TypedDict(
    "ClipScore",
    {
        "n": int,
        "dyn": int,
        "occ_cells": int,
        "T": int,
        "sim3": NotRequired[AlignmentScore],
        "global": NotRequired[AlignmentScore],
    },
)


class MeanCells(TypedDict):
    all: float
    dyn: float


class EvaluationRow(TypedDict):
    variant: str
    subsets: list[SubsetName]
    missing: NotRequired[list[SubsetName]]
    mean: NotRequired[MeanCells]
    adt: NotRequired[SubsetMeans]
    pstudio: NotRequired[SubsetMeans]
    po: NotRequired[SubsetMeans]
    ds: NotRequired[SubsetMeans]


class MachineInfo(TypedDict):
    host: str
    date: str
    torch: NotRequired[str]
    gpu: NotRequired[str | None]
