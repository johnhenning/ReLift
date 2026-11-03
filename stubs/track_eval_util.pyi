from collections.abc import Callable
from typing import Any

tqdm: Callable[..., Any]
PIXEL_TO_FIXED_METRIC_THRESH: dict[str, float]

def compute_average_pts_within_thresh(*args: Any, **kwargs: Any) -> Any: ...
