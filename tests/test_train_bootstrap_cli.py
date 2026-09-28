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
# The child runs in a scratch directory, where the config's relative map paths
# do not resolve; resolving pad_to_size reads them (and is part of the stubbed
# curriculum's job anyway).
bootstrap.resolve_config = lambda cfg: cfg
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


# ---------------------------------------------------------------------------
# --seed, --resume-if-exists, --check-only (the per-seed command of run_seeds.py)
# ---------------------------------------------------------------------------

_COMMON = ["--device", "cpu", "--no-gcs", "--skip-plots", "--skip-videos", "--sanity-episodes", "0"]


@pytest.fixture
def fake_curriculum(train_bootstrap, monkeypatch, tmp_path, sigterm_guard):
    """run_curriculum replaced by a recorder; a small valid config to run."""
    import yaml

    import reinforcetactics.rl.bootstrap as bootstrap

    calls: list[dict] = []

    def fake_run(cfg, output_dir, **kwargs):
        calls.append({"cfg": cfg, "output_dir": Path(output_dir), **kwargs})
        return {"history": [], "final_model_path": None}

    monkeypatch.setattr(bootstrap, "run_curriculum", fake_run)
    monkeypatch.setattr(train_bootstrap, "_maybe_upload", lambda output_dir, args: None)
    data = {
        "env": {"n_envs": 1, "use_subprocess": False},
        "curriculum": {
            "stages": [{"name": "s", "map_file": "maps/1v1/beginner.csv", "opponent": "noop", "max_timesteps": 100}]
        },
    }
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    return calls, config


def _status(run: Path, status: str, final_model: bool = False) -> None:
    import json

    payload: dict[str, object] = {"status": status}
    if status == "curriculum_stalled":
        payload.update(stalled_stage="s", peak_win_rate=0.4, threshold=0.9, retries_used=1)
    (run / "run_status.json").write_text(json.dumps(payload))
    if final_model:
        (run / "final_model.zip").write_bytes(b"zip")


class TestResumeIfExists:
    def test_decision_table(self, train_bootstrap, fake_curriculum, tmp_path, capsys):
        calls, config = fake_curriculum
        out = tmp_path / "run"
        argv = ["--config", str(config), "--output-dir", str(out), "--resume-if-exists", "--seed", "42", *_COMMON]

        # 5. Nothing there: a fresh start (and a record to resume from).
        assert train_bootstrap.main(argv) == 0
        assert "resume" not in calls[-1] and calls[-1]["cfg"].seed == 42
        assert (out / "resolved_config.yaml").is_file()
        # 3. A record but no run_status.json: interrupted, so the same command resumes it.
        assert train_bootstrap.main(argv) == 0
        assert calls[-1]["resume"] is True and calls[-1]["output_dir"] == out
        # 1. Finished: nothing is trained or post-processed.
        _status(out, "completed_curriculum", final_model=True)
        n = len(calls)
        assert train_bootstrap.main(argv) == 0 and len(calls) == n
        assert "already complete" in capsys.readouterr().out
        # ...but a finished record without final_model.zip resumes (which rebuilds it).
        (out / "final_model.zip").unlink()
        assert train_bootstrap.main(argv) == 0 and calls[-1]["resume"] is True
        # 2. Stalled: a result, reported with exit 3 and never resumed.
        _status(out, "curriculum_stalled")
        n = len(calls)
        assert train_bootstrap.main(argv) == 3 and len(calls) == n
        assert "stalled at stage 's'" in capsys.readouterr().out

    def test_foreign_output_is_refused_and_logs_are_not_output(self, train_bootstrap, fake_curriculum, tmp_path):
        calls, config = fake_curriculum
        foreign = tmp_path / "foreign"
        (foreign / "stage_a").mkdir(parents=True)
        (foreign / "stage_a" / "eval_results.jsonl").write_text("{}\n")
        argv = ["--config", str(config), "--resume-if-exists", *_COMMON]
        with pytest.raises(SystemExit, match="not a train_bootstrap.py run dir"):
            train_bootstrap.main([*argv, "--output-dir", str(foreign)])
        other = tmp_path / "csv_only"
        other.mkdir()
        (other / "bootstrap_results.csv").write_text("stage\n")
        with pytest.raises(SystemExit, match="bootstrap_results.csv"):
            train_bootstrap.main([*argv, "--output-dir", str(other)])
        assert calls == []
        launched = tmp_path / "launched"
        (launched / "logs").mkdir(parents=True)
        (launched / "logs" / "train.20260928T120000Z.log").write_text("# launcher session\n")
        assert train_bootstrap.main([*argv, "--output-dir", str(launched)]) == 0
        assert "resume" not in calls[-1]

    def test_usage(self, train_bootstrap, fake_curriculum, tmp_path):
        _, config = fake_curriculum
        with pytest.raises(SystemExit, match="needs --output-dir"):
            train_bootstrap.main(["--config", str(config), "--resume-if-exists", *_COMMON])
        with pytest.raises(SystemExit, match="drop --resume"):
            train_bootstrap.main(["--resume", str(tmp_path), "--output-dir", str(tmp_path), "--resume-if-exists", *_COMMON])

    def test_a_different_config_is_refused_unless_forced(self, train_bootstrap, fake_curriculum, tmp_path):
        calls, config = fake_curriculum
        out = tmp_path / "run"
        base = ["--config", str(config), "--output-dir", str(out), "--resume-if-exists", *_COMMON]
        assert train_bootstrap.main([*base, "--seed", "42"]) == 0
        with pytest.raises(SystemExit, match="seed"):
            train_bootstrap.main([*base, "--seed", "43"])
        assert train_bootstrap.main([*base, "--seed", "43", "--force"]) == 0
        assert calls[-1]["resume"] is True and calls[-1]["force"] is True and calls[-1]["cfg"].seed == 43
        # --force on a fresh start is a no-op, not an error.
        fresh = ["--config", str(config), "--output-dir", str(tmp_path / "new"), "--resume-if-exists", "--force", *_COMMON]
        assert train_bootstrap.main(fresh) == 0

    def test_build_bc_is_ignored_when_resuming(self, train_bootstrap, fake_curriculum, tmp_path, capsys):
        calls, config = fake_curriculum
        out = tmp_path / "run"
        argv = ["--config", str(config), "--output-dir", str(out), "--resume-if-exists", *_COMMON]
        assert train_bootstrap.main(argv) == 0
        assert train_bootstrap.main([*argv, "--build-bc"]) == 0
        assert calls[-1]["resume"] is True and "--build-bc ignored" in capsys.readouterr().out


class TestSeedAndCheckOnly:
    def test_seed_is_set_seed(self, train_bootstrap, fake_curriculum, tmp_path):
        calls, config = fake_curriculum
        assert (
            train_bootstrap.main(["--config", str(config), "--output-dir", str(tmp_path / "a"), "--seed", "1042", *_COMMON])
            == 0
        )
        assert (
            train_bootstrap.main(
                ["--config", str(config), "--output-dir", str(tmp_path / "b"), "--set", "seed=1042", *_COMMON]
            )
            == 0
        )
        assert calls[0]["cfg"].seed == calls[1]["cfg"].seed == 1042
        a = (tmp_path / "a" / "resolved_config.yaml").read_text()
        assert a == (tmp_path / "b" / "resolved_config.yaml").read_text() and "seed: 1042" in a
        with pytest.raises(SystemExit, match="--seed and --set seed"):
            train_bootstrap.main(["--config", str(config), "--seed", "1", "--set", "seed=2", *_COMMON])

    def test_check_only_writes_nothing(self, train_bootstrap, fake_curriculum, tmp_path, capsys):
        calls, config = fake_curriculum
        out = tmp_path / "run"
        assert (
            train_bootstrap.main(["--config", str(config), "--output-dir", str(out), "--check-only", "--strict", *_COMMON])
            == 0
        )
        assert not out.exists() and calls == []
        assert "Config OK" in capsys.readouterr().out
        with pytest.raises(KeyError, match="Unknown config key"):
            train_bootstrap.main(["--config", str(config), "--check-only", "--set", "curriculum.stages[s].bogus=1", *_COMMON])

    def test_torch_threads(self, train_bootstrap, fake_curriculum, tmp_path):
        import torch

        config = str(fake_curriculum[1])
        before = torch.get_num_threads()
        try:
            assert (
                train_bootstrap.main(
                    ["--config", config, "--output-dir", str(tmp_path / "r"), "--torch-threads", "1", *_COMMON]
                )
                == 0
            )
            assert torch.get_num_threads() == 1
        finally:
            torch.set_num_threads(before)
        with pytest.raises(SystemExit, match="--torch-threads"):
            train_bootstrap.main(["--config", config, "--torch-threads", "0", *_COMMON])


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_sigterm_mid_stage_then_the_identical_command_completes(tmp_path):
    """The per-seed command of run_seeds.py, killed mid-stage and re-run unchanged, finishes the run."""
    import json

    from tests.test_curriculum_recovery import _tiny_config

    config = _tiny_config(tmp_path, 3_200)
    out = tmp_path / "run"
    env = {k: v for k, v in os.environ.items() if k not in ("GCS_OUTPUT_URI", "AIP_MODEL_DIR", "GCS_WRAPPER_SYNC")}
    env.update(PYTHONPATH=str(REPO_ROOT), SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", MPLBACKEND="Agg")
    command = [
        sys.executable,
        str(SCRIPT),
        "--config",
        str(config),
        "--seed",
        "7",
        "--output-dir",
        str(out),
        "--resume-if-exists",
        "--strict",
        *_COMMON,
    ]
    proc = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    jsonl = out / "s2" / "eval_results.jsonl"
    try:
        deadline = time.monotonic() + 300
        while not (jsonl.exists() and len(jsonl.read_text().splitlines()) >= 3):
            if proc.poll() is not None or time.monotonic() > deadline:
                proc.kill()
                pytest.fail(f"never reached stage 2:\n{proc.communicate()[0][-4000:]}")
            time.sleep(0.05)
        proc.send_signal(signal.SIGTERM)
        first = proc.communicate(timeout=120)[0]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 143, first[-4000:]
    assert not (out / "run_status.json").exists()

    resumed = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert resumed.returncode == 0, resumed.stdout[-4000:] + resumed.stderr[-4000:]
    assert "interrupted run found; resuming it" in resumed.stdout
    status = json.loads((out / "run_status.json").read_text())
    assert status["status"] == "completed_curriculum" and status["resume_count"] == 1
    # Once more: the run is finished, so nothing runs.
    again = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300, check=False)
    assert again.returncode == 0 and "already complete" in again.stdout
