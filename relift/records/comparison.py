"""Wire-compatible numerical comparison and parity-report schemas."""

from typing import Literal, NotRequired, TypedDict, TypeIs


class ErrorReport(TypedDict):
    vector_count: int
    nonfinite_vectors: int
    bitwise_equal: bool
    max_normalized_vector_error: float | None
    mean_normalized_vector_error: float | None
    median_normalized_error_upper_bound: float | None
    p99_normalized_error_upper_bound: float | None
    quantile_method: str


class ArrayComparison(TypedDict):
    reference_shape: list[int]
    candidate_shape: list[int]
    reference_dtype: str
    candidate_dtype: str
    equal: bool
    max_absolute_error: float | None


class ImageComparison(TypedDict):
    equal: bool
    reference_shape: list[int]
    candidate_shape: list[int]


class ChunkedArrayComparison(TypedDict):
    equal: bool
    max_absolute_error: float | None
    shape: list[int]
    dtype: str


class PreparedUnavailable(TypedDict):
    compared: Literal[False]
    reason: str


class PreparedArrays(TypedDict):
    compared: Literal[True]
    arrays: dict[str, ArrayComparison | ChunkedArrayComparison]


type PreparedComparison = PreparedUnavailable | PreparedArrays


class ErrorGroups(TypedDict):
    ordinary_xyz: ErrorReport
    repeated_xyz: ErrorReport
    raw_lift: ErrorReport


class SharedArrays(TypedDict):
    poses: ArrayComparison
    intrinsics: ArrayComparison
    images: ImageComparison


class PixelGrouping(TypedDict, total=False):
    seed: int
    group: int
    batch: int


class PixelPolicy(TypedDict):
    source: str
    explicit: bool
    shape: list[int]
    manifest_sha256: str | None
    grouping: PixelGrouping


class ComparisonTrace(TypedDict, total=False):
    """Provenance added by the campaign validator, absent from standalone reports."""

    variant: str
    phase: str
    trial: str
    trial_sha256: str
    control_trial: str
    checker_sha256: str


class ShapeMismatch(ComparisonTrace):
    passed: Literal[False]
    reason: Literal["shape mismatch"]
    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]


class TrajectoryComparison(ComparisonTrace):
    passed: bool
    reference: str
    candidate: str
    shape: list[int]
    frames: int
    pixels: int
    compared_cells: int
    repeated_pixel_count: int
    repeated_pixel_ids: list[int]
    normalization_scale: float
    normalization_source: str
    tolerance: float
    errors: ErrorGroups
    changed_observation_cells: int
    shared_arrays: SharedArrays
    changed_dtypes: list[str]
    policy: str
    pixel_policy: PixelPolicy
    prepared_outputs: NotRequired[dict[str, PreparedComparison]]


type ComparisonReport = ShapeMismatch | TrajectoryComparison


class ProviderParity(ComparisonTrace):
    state_equal: bool | None
    inputs_equal: bool
    artifacts_verified: bool
    artifacts_equal: bool
    passed: bool


class ParityReport(TypedDict):
    passed: bool
    reports: list[ComparisonReport | ProviderParity]


def is_trajectory_comparison(report: ComparisonReport) -> TypeIs[TrajectoryComparison]:
    """Narrow the two wire-compatible results produced by compare_offload.compare."""
    return "errors" in report
