"""Compatibility paths for upstream modules installed by relift-frontends."""
import sys
from importlib.resources import files


def _installed_path(name):
    try:
        path = files("relift_frontends").joinpath(name)
    except ModuleNotFoundError as exc:
        raise ImportError("Front-end dependencies are missing; run `uv sync --locked`") from exc
    if not path.is_dir():
        raise ImportError(f"Installed front-end package is missing {name}; rerun `uv sync --locked --reinstall-package relift-frontends`")
    return str(path)


def front_ends():
    """AllTracker uses absolute imports of its private nets/utils packages."""
    path = _installed_path("alltracker")
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def rc4_eval():
    """Installed 4RC scoring/clip modules, with the RANSAC progress bar silenced."""
    path = _installed_path("rc4_eval")
    if path not in sys.path:
        sys.path.insert(0, path)
    import tapvid3d_utils as TE
    import track_eval_util as U
    U.tqdm = lambda x, *a, **k: x
    return TE, U
