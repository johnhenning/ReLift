"""Build a normal wheel from immutable upstream archives without vendoring source in Git."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def unpack_selected(archive, destination, packages):
    """Copy regular members only, retaining upstream package contents and nested licences."""
    found = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe upstream archive path: {member.name}")
            if not member.isfile() or len(path.parts) < 2:
                continue
            relative = PurePosixPath(*path.parts[1:])
            for source, target in packages.items():
                if relative == PurePosixPath(source) or PurePosixPath(source) in relative.parents:
                    output = destination / target / relative.relative_to(source)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, output.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    found.add(source)
    if missing := set(packages) - found:
        raise ValueError(f"Upstream archive is missing packages: {sorted(missing)}")


class UpstreamBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        self.staging = tempfile.TemporaryDirectory(prefix="relift-frontends-")
        staging = Path(self.staging.name)
        manifest = Path(self.root) / "upstreams.json"
        entries = json.loads(manifest.read_text())
        for entry in entries:
            archive = staging / (entry["name"] + ".tar.gz")
            url = f"https://codeload.github.com/{entry['repo']}/tar.gz/{entry['revision']}"
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as dst:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(block)
                    dst.write(block)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"SHA256 mismatch for {entry['name']} at {entry['revision']}")
            packages = dict(entry["packages"])
            # Retain each upstream's root licence alongside its installed source.
            packages[entry.get("license_file", "LICENSE")] = f"relift_frontends/licenses/{entry['name']}/LICENSE"
            unpack_selected(archive, staging / "wheel", packages)
        for name in ("alltracker", "rc4_eval"):
            (staging / "wheel/relift_frontends" / name / "__init__.py").touch()
        shutil.copyfile(manifest, staging / "wheel/relift_frontends/upstreams.json")
        for path in (staging / "wheel").iterdir():
            build_data["force_include"][str(path)] = path.name

    def finalize(self, version, build_data, artifact_path):
        self.staging.cleanup()
