"""Reconstruction must use committed bytes and preserve the live working tree."""
import hashlib
import json
import subprocess

import pytest

from scripts.freeze_benchmark_baseline import freeze_revision


def test_snapshot_uses_committed_revision_and_refuses_overwrite(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()
    git("init", "-q")
    source = repository / "implementation.py"
    source.write_text("committed\n")
    git("add", "implementation.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "Reference")
    commit = git("rev-parse", "HEAD")
    source.write_text("uncommitted\n")
    snapshot = freeze_revision(tmp_path / "snapshot", commit, repository)
    assert (snapshot / "implementation.py").read_text() == "committed\n"
    assert source.read_text() == "uncommitted\n"
    manifest = json.loads((snapshot / "snapshot.json").read_text())
    assert manifest["git_revision"] == commit
    assert manifest["files"]["implementation.py"] == hashlib.sha256(b"committed\n").hexdigest()
    with pytest.raises(FileExistsError):
        freeze_revision(snapshot, commit, repository)
