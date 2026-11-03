"""The installed wheel must work without upstream source checkouts on sys.path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_frontends_import_from_installed_wheel(tmp_path: Path) -> None:
    code = """
from importlib.metadata import version
from importlib.resources import files
import json
from relift.vendor import front_ends, rc4_eval
front_ends()
from nets.alltracker import Net
from depth_anything_3.api import DepthAnything3
from relift.providers.cowtracker import _install_sdpa_attention
_install_sdpa_attention()
from cowtracker import CoWTrackerWindowed
from vggt_omega.models import VGGTOmega
from arc.models.arc import Arc
TE, U = rc4_eval()
import track_eval_dense
root = files('relift_frontends')
manifest = json.loads(root.joinpath('upstreams.json').read_text())
for entry in manifest:
    assert root.joinpath('licenses', entry['name'], 'LICENSE').is_file()
assert version('relift-frontends') == '0.1.0'
print(json.dumps(manifest))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = json.loads(
        (Path(__file__).resolve().parents[1] / "packages/frontends/upstreams.json").read_text()
    )
    # Upstream imports can print informational messages before the manifest.
    assert json.loads(result.stdout.splitlines()[-1]) == expected
