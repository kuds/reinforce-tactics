"""scripts/train/train_bootstrap.py: exit codes, and the SIGTERM path that uploads the run.

Two bugs from the review (rltrain-3, prior-10):

* Vertex sends SIGTERM on cancel and preemption, and vertex_train.py forwards
  it to this script. Python's default SIGTERM action ends the process without
  running ``finally`` blocks, so ``main``'s ``finally: _maybe_upload(...)``
  never ran and the whole run directory was lost.
* A stalled curriculum exited 0, so schedulers recorded it as a success.

The curriculum is stubbed out, so no training happens. The in-process tests
cover the control flow; the subprocess test sends a real SIGTERM from outside
and checks the process exit status, which is what Vertex and vertex_train.py see.
"""

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "train" / "train_bootstrap.py"
CONFIG = REPO_ROOT / "configs" / "ppo" / "bootstrap.yaml"


def _argv(output_dir: Path) -> list[str]:
    return [
        "--config",
        str(CONFIG),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--no-gcs",
        "--skip-plots",
        "--skip-videos",
        "--sanity-episodes",
        "0",
    ]


class _SigtermNotHandled(Exception):
    """Raised by the test's stand-in SIGTERM handler, i.e. main() installed none."""


@pytest.fixture
def train_bootstrap(monkeypatch):
    """Import the script as a module without leaking its import-time side effects."""
    # It setdefaults these at import and prepends the repo root to sys.path;
    # monkeypatch puts all three back afterwards.
    monkeypatch.setenv("SDL_VIDEODRIVER", os.environ.get("SDL_VIDEODRIVER", "dummy"))
    monkeypatch.setenv("MPLBACKEND", os.environ.get("MPLBACKEND", "Agg"))
    monkeypatch.setattr(sys, "path", [*sys.path])
    spec = importlib.util.spec_from_file_location("train_bootstrap_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sigterm_guard():
    """main() installs a process-wide SIGTERM handler; contain it to the test.

    Without main's handler, the stand-in turns ``raise_signal(SIGTERM)`` into
    an exception, so the test fails instead of killing the pytest process.
    """

    def _stand_in(signum, frame):
        raise _SigtermNotHandled

    original = signal.signal(signal.SIGTERM, _stand_in)
    yield
    signal.signal(signal.SIGTERM, original)


@pytest.fixture
def run_main(train_bootstrap, monkeypatch, tmp_path, sigterm_guard):
    """Run main() with ``curriculum`` in place of run_curriculum; record uploads."""
    import reinforcetactics.rl.bootstrap as bootstrap

    uploads: list[Path] = []
    monkeypatch.setattr(train_bootstrap, "_maybe_upload", lambda output_dir, args: uploads.append(output_dir))
    output_dir = tmp_path / "run"

    def run(curriculum):
        monkeypatch.setattr(bootstrap, "run_curriculum", curriculum)
        return train_bootstrap.main(_argv(output_dir))

    return run, uploads, output_dir


def test_completed_curriculum_exits_0_and_uploads(run_main):
    run, uploads, output_dir = run_main
    assert run(lambda cfg, output_dir: {"history": [], "final_model_path": None}) == 0
    assert uploads == [output_dir]


def test_stalled_curriculum_exits_3_and_still_uploads(run_main):
    from reinforcetactics.rl.bootstrap import CurriculumStalled

    run, uploads, output_dir = run_main

    def stall(cfg, output_dir):
        raise CurriculumStalled("stage_1", achieved_win_rate=0.3, threshold=0.6, timesteps=1000)

    assert run(stall) == 3
    assert uploads == [output_dir]


def test_failure_propagates_after_uploading(run_main):
    """An exception exits non-zero (the interpreter's 1) with the upload done first."""
    run, uploads, output_dir = run_main

    def crash(cfg, output_dir):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run(crash)
    assert uploads == [output_dir]


def test_sigterm_becomes_exit_143_after_the_upload(run_main, train_bootstrap, monkeypatch):
    run, uploads, output_dir = run_main

    def upload_while_signalled(output_dir, args):
        # A second SIGTERM (vertex_train forwards every one it gets) must not
        # abort the upload the first one is waiting for.
        signal.raise_signal(signal.SIGTERM)
        uploads.append(output_dir)

    monkeypatch.setattr(train_bootstrap, "_maybe_upload", upload_while_signalled)

    def preempted(cfg, output_dir):
        signal.raise_signal(signal.SIGTERM)
        pytest.fail("SIGTERM did not interrupt training")

    with pytest.raises(SystemExit) as excinfo:
        run(preempted)
    assert excinfo.value.code == 143
    assert uploads == [output_dir]


def test_sigterm_during_the_final_upload_lets_it_finish(run_main, train_bootstrap, monkeypatch):
    """A cancel that lands while a finished run uploads must not cut the upload short."""
    run, uploads, output_dir = run_main

    def upload_interrupted_by_cancel(output_dir, args):
        signal.raise_signal(signal.SIGTERM)
        uploads.append(output_dir)

    monkeypatch.setattr(train_bootstrap, "_maybe_upload", upload_interrupted_by_cancel)
    assert run(lambda cfg, output_dir: {"history": [], "final_model_path": None}) == 0
    assert uploads == [output_dir]


def test_sigterm_handler_does_not_write_through_sys_stdout(train_bootstrap, monkeypatch, sigterm_guard):
    """The signal can land inside a print; re-entering stdout then raises RuntimeError, not SystemExit."""

    class _StdoutMidWrite:
        def write(self, text):
            raise RuntimeError("reentrant call inside <_io.BufferedWriter name='<stdout>'>")

        flush = write

    monkeypatch.setattr(sys, "stdout", _StdoutMidWrite())
    with pytest.raises(SystemExit) as excinfo:
        train_bootstrap._exit_on_sigterm(signal.SIGTERM, None)
    assert excinfo.value.code == 143


# The child replaces the curriculum with a stand-in that writes a file and then
# waits to be terminated, and replaces upload_tree (called by the real
# _maybe_upload) with one that records what it would have uploaded.
_CHILD = """
import importlib.util, sys, time
from pathlib import Path

script, workdir, argv = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
spec = importlib.util.spec_from_file_location("train_bootstrap", script)
train_bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_bootstrap)

import reinforcetactics.cloud.storage as storage
import reinforcetactics.rl.bootstrap as bootstrap

def curriculum(cfg, output_dir):
    stage = Path(output_dir) / "stage_1"
    stage.mkdir()
    (stage / "eval_results.jsonl").write_text("{}\\n")
    (workdir / "training").touch()
    time.sleep(120)
    raise RuntimeError("SIGTERM never arrived")

def upload_tree(local_dir, dest_uri, **kwargs):
    files = sorted(p.relative_to(local_dir).as_posix() for p in Path(local_dir).rglob("*") if p.is_file())
    (workdir / "uploaded").write_text("\\n".join([dest_uri, *files]))
    return len(files)

bootstrap.run_curriculum = curriculum
storage.upload_tree = upload_tree
sys.exit(train_bootstrap.main(argv))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_real_sigterm_uploads_the_run_and_exits_143(tmp_path):
    argv = _argv(tmp_path / "run")
    argv[argv.index("--no-gcs")] = "--gcs-output=gs://bucket/jobs/job1"
    env = {k: v for k, v in os.environ.items() if k not in ("GCS_OUTPUT_URI", "AIP_MODEL_DIR", "GCS_WRAPPER_SYNC")}
    env.update(PYTHONPATH=str(REPO_ROOT), SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", MPLBACKEND="Agg")
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(SCRIPT), str(tmp_path), *argv],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 120
        while not (tmp_path / "training").exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                proc.kill()
                pytest.fail(f"child never started training:\n{proc.communicate()[0]}")
            time.sleep(0.05)
        proc.send_signal(signal.SIGTERM)
        output = proc.communicate(timeout=60)[0]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    assert proc.returncode == 143, f"exit status {proc.returncode}; output:\n{output}"
    uploaded = (tmp_path / "uploaded").read_text().splitlines()
    assert uploaded[0] == "gs://bucket/jobs/job1/run"
    assert "stage_1/eval_results.jsonl" in uploaded[1:]
