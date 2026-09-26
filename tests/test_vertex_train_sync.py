"""scripts/cloud/vertex_train.py syncs train_bootstrap's run directory, not only models/ & co.

The wrapper's periodic and final syncs covered models/, checkpoints/,
tensorboard/ and logs/, but train_bootstrap.py writes every run under
benchmarks/bootstrap/<run_id>/. A bootstrap job killed before its own final
upload (preemption, cancel, the grace period running out) lost the whole run.
"""

import functools
import importlib.util
import os
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from reinforcetactics.cloud import storage
from reinforcetactics.cloud.storage import DEFAULT_OUTPUT_DIRS, WRAPPER_SYNC_ENV

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


def test_gcs_sync_dirs_prefixes_are_normalised(vertex_train):
    """``.`` means the base itself; it used to become a literal "." in every object name."""
    dirs = vertex_train.resolve_sync_dirs({"GCS_SYNC_DIRS": ".,flat=,dotted=.,out=./eval/../runs/,up=../elsewhere"})
    assert dirs["."] == ""  # jobs/<name>/models/..., not jobs/<name>/./models/...
    assert dirs["flat"] == ""
    assert dirs["dotted"] == ""
    assert dirs["out"] == "runs"
    assert "up" not in dirs  # would write above the job's base


def test_overlapping_sync_dirs_upload_to_every_destination(vertex_train, monkeypatch, tmp_path, caplog):
    """GCS_SYNC_DIRS=benchmarks overlaps the default benchmarks/bootstrap entry.

    The manifest was keyed by local path only, so whichever entry uploaded a
    file first marked it done for the other, and the documented
    ``<base>/benchmarks/...`` copy never appeared.
    """
    run_file = tmp_path / "benchmarks" / "bootstrap" / "run1" / "a.zip"
    run_file.parent.mkdir(parents=True)
    run_file.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("WARNING", logger="vertex_train"):
        sync_dirs = vertex_train.resolve_sync_dirs({"GCS_SYNC_DIRS": "benchmarks"})
    assert "benchmarks/bootstrap is inside benchmarks" in caplog.text

    client = _FakeClient()
    monkeypatch.setattr(vertex_train, "sync_directories", functools.partial(storage.sync_directories, client=client))
    manifest: dict = {}
    vertex_train._sync("gs://bucket/jobs/j", None, threading.Lock(), manifest, sync_dirs)
    assert sorted(client.uploaded) == ["jobs/j/benchmarks/bootstrap/run1/a.zip", "jobs/j/run1/a.zip"]

    # Both copies are recorded: the next pass uploads nothing.
    again = _FakeClient()
    monkeypatch.setattr(vertex_train, "sync_directories", functools.partial(storage.sync_directories, client=again))
    vertex_train._sync("gs://bucket/jobs/j", None, threading.Lock(), manifest, sync_dirs)
    assert again.uploaded == []


@pytest.fixture
def restore_signal_handlers():
    """vertex_train.main() installs process-wide SIGTERM/SIGINT forwarders."""
    saved = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    yield
    for signum, handler in saved.items():
        signal.signal(signum, handler)


def test_bootstrap_leaves_its_upload_to_the_wrappers_final_sync(vertex_train, monkeypatch, tmp_path, restore_signal_handlers):
    """The run directory is uploaded once, by the wrapper, not re-sent in full by the child first.

    On a SIGTERM the child's upload_tree re-uploaded every file of the run
    (checkpoints the periodic sync already stored included) inside Vertex's
    grace period, before the wrapper's changed-files-only final sync could
    start.
    """
    base = "gs://bucket/jobs/job1"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GCS_OUTPUT_URI", base)
    monkeypatch.setenv("GCS_SYNC_INTERVAL", "0")
    monkeypatch.delenv("GCS_SYNC_DIRS", raising=False)
    final_syncs: list[str] = []
    monkeypatch.setattr(vertex_train, "sync_directories", lambda base_uri, **kwargs: final_syncs.append(base_uri) or {})
    child = "import os, pathlib; pathlib.Path('child_env').write_text(os.environ.get('GCS_WRAPPER_SYNC', ''))"
    monkeypatch.setattr(sys, "argv", ["vertex_train.py", sys.executable, "-c", child])

    assert vertex_train.main() == 0
    assert final_syncs == [base]
    child_env = {WRAPPER_SYNC_ENV: (tmp_path / "child_env").read_text()}

    run_dir = Path("benchmarks") / "bootstrap" / "20260926_120000"
    assert storage.synced_by_wrapper(run_dir, f"{base}/20260926_120000", child_env)
    assert storage.synced_by_wrapper(Path("models") / "sweep", f"{base}/models/sweep", child_env)
    # Anything the wrapper would put elsewhere, or not sync at all, stays the child's job.
    assert not storage.synced_by_wrapper(run_dir, f"{base}/benchmarks/bootstrap/20260926_120000", child_env)
    assert not storage.synced_by_wrapper(run_dir, "gs://other-bucket/x/20260926_120000", child_env)
    assert not storage.synced_by_wrapper(Path("runs") / "20260926_120000", f"{base}/20260926_120000", child_env)
    assert not storage.synced_by_wrapper(run_dir, f"{base}/20260926_120000", {})
    assert not storage.synced_by_wrapper(run_dir, f"{base}/20260926_120000", {WRAPPER_SYNC_ENV: "not json"})

    train_bootstrap = _load_script(monkeypatch, "scripts/train/train_bootstrap.py")
    uploads: list[str] = []
    monkeypatch.setattr(storage, "upload_tree", lambda local_dir, dest_uri, **kwargs: uploads.append(dest_uri) or 1)
    monkeypatch.setenv(WRAPPER_SYNC_ENV, child_env[WRAPPER_SYNC_ENV])
    train_bootstrap._maybe_upload(run_dir, SimpleNamespace(gcs_output=None, no_gcs=False))
    assert uploads == []
    # An explicit destination the wrapper does not cover is still uploaded by the script.
    train_bootstrap._maybe_upload(run_dir, SimpleNamespace(gcs_output="gs://other-bucket/x", no_gcs=False))
    assert uploads == ["gs://other-bucket/x/20260926_120000"]


def test_without_a_sync_target_the_child_is_promised_nothing(vertex_train, monkeypatch, tmp_path, restore_signal_handlers):
    monkeypatch.chdir(tmp_path)
    for name in ("GCS_OUTPUT_URI", "AIP_MODEL_DIR"):
        monkeypatch.delenv(name, raising=False)
    # Even one inherited from an outer process: nothing here will upload the run.
    monkeypatch.setenv(WRAPPER_SYNC_ENV, storage.wrapper_sync_env("gs://bucket/jobs/j", {"benchmarks/bootstrap": ""}))
    child = "import os, pathlib; pathlib.Path('child_env').write_text(repr(os.environ.get('GCS_WRAPPER_SYNC')))"
    monkeypatch.setattr(sys, "argv", ["vertex_train.py", sys.executable, "-c", child])
    assert vertex_train.main() == 0
    assert (tmp_path / "child_env").read_text() == "None"


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
    monkeypatch.delenv(WRAPPER_SYNC_ENV, raising=False)  # run outside the wrapper
    train_bootstrap._maybe_upload(run_dir, SimpleNamespace(gcs_output=None, no_gcs=False))
    assert set(final.uploaded) == run_objects
