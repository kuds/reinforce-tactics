"""scripts/cloud/vertex_train.py syncs train_bootstrap's run directory, not only models/ & co.

The wrapper's periodic and final syncs covered models/, checkpoints/,
tensorboard/ and logs/, but train_bootstrap.py writes every run under
benchmarks/bootstrap/<run_id>/. A bootstrap job killed before its own final
upload (preemption, cancel, the grace period running out) lost the whole run.
"""

import functools
import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from reinforcetactics.cloud import storage
from reinforcetactics.cloud.storage import DEFAULT_OUTPUT_DIRS

REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeClient:
    """Stands in for google.cloud.storage.Client; records blob names."""

    def __init__(self):
        self.uploaded: list[str] = []

    def bucket(self, name):
        return SimpleNamespace(
            blob=lambda blob_name: SimpleNamespace(upload_from_filename=lambda path: self.uploaded.append(blob_name))
        )


def _load_script(monkeypatch, relative: str):
    # Both scripts prepend the repo root to sys.path at import; train_bootstrap
    # also setdefaults two env vars. monkeypatch restores all of it.
    monkeypatch.setattr(sys, "path", [*sys.path])
    monkeypatch.setenv("SDL_VIDEODRIVER", os.environ.get("SDL_VIDEODRIVER", "dummy"))
    monkeypatch.setenv("MPLBACKEND", os.environ.get("MPLBACKEND", "Agg"))
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(f"{path.stem}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def vertex_train(monkeypatch):
    return _load_script(monkeypatch, "scripts/cloud/vertex_train.py")


def test_default_sync_dirs_include_the_bootstrap_runs(vertex_train):
    dirs = vertex_train.resolve_sync_dirs({})
    assert {name: dirs[name] for name in DEFAULT_OUTPUT_DIRS} == {name: name for name in DEFAULT_OUTPUT_DIRS}
    # train_bootstrap's default output_dir is benchmarks/bootstrap/<run_id>.
    assert dirs["benchmarks/bootstrap"] == ""


def test_gcs_sync_dirs_adds_to_the_defaults(vertex_train):
    dirs = vertex_train.resolve_sync_dirs({"GCS_SYNC_DIRS": " runs/sweep/ , results=eval/out/ ,, "})
    assert dirs["runs/sweep"] == "runs/sweep"
    assert dirs["results"] == "eval/out"
    assert dirs["models"] == "models"
    assert dirs["benchmarks/bootstrap"] == ""


def test_periodic_sync_writes_the_objects_the_final_upload_writes(vertex_train, monkeypatch, tmp_path):
    """The wrapper's copy of a run lands exactly where train_bootstrap's own upload puts it.

    Otherwise a finished run would be stored twice, and a preempted one would
    sit somewhere other than where the docs say to fetch it from.
    """
    run_dir = Path("benchmarks") / "bootstrap" / "20260926_120000"
    for rel in ("checkpoints/stage_1.zip", "stage_1/eval_results.jsonl", "final_model.zip"):
        (tmp_path / run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / run_dir / rel).write_bytes(b"x")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "ppo_final.zip").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    base = "gs://bucket/jobs/job1"

    periodic = _FakeClient()
    monkeypatch.setattr(vertex_train, "sync_directories", functools.partial(storage.sync_directories, client=periodic))
    vertex_train._sync(base, None, threading.Lock(), {}, vertex_train.resolve_sync_dirs({}))

    run_objects = {
        "jobs/job1/20260926_120000/checkpoints/stage_1.zip",
        "jobs/job1/20260926_120000/stage_1/eval_results.jsonl",
        "jobs/job1/20260926_120000/final_model.zip",
    }
    assert sorted(periodic.uploaded) == sorted(run_objects | {"jobs/job1/models/ppo_final.zip"})

    # train_bootstrap's final upload, resolving the same base from the env the
    # submit script sets.
    train_bootstrap = _load_script(monkeypatch, "scripts/train/train_bootstrap.py")
    final = _FakeClient()
    monkeypatch.setattr(storage, "upload_tree", functools.partial(storage.upload_tree, client=final))
    monkeypatch.setenv("GCS_OUTPUT_URI", base)
    train_bootstrap._maybe_upload(run_dir, SimpleNamespace(gcs_output=None, no_gcs=False))
    assert set(final.uploaded) == run_objects
