"""The seed launcher: reinforcetactics/experiments/seed_runs.py and scripts/train/run_seeds.py.

The scheduler tests run the real launcher with a stub in place of
train_bootstrap.py (``--train-script``): the stub records how it was called and
then completes, stalls, fails, is interrupted, or waits for a signal, as a
per-seed plan says. The Vertex tests replace the command runner and the GCS
reader; nothing touches the network.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from reinforcetactics.experiments import seed_runs as sr
from reinforcetactics.rl.config import TrainingConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "configs" / "ppo" / "bootstrap.yaml"
GROUP = "20260928_120000_t"


@pytest.fixture
def run_seeds(monkeypatch):
    monkeypatch.setattr(sys, "path", [*sys.path])
    spec = importlib.util.spec_from_file_location("run_seeds_under_test", REPO_ROOT / "scripts" / "train" / "run_seeds.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_config(path: Path, **eval_overrides) -> Path:
    data = {
        "seed": 0,
        "env": {"n_envs": 2, "use_subprocess": False, "action_space_type": "flat_discrete", "max_flat_actions": 64},
        "eval": {"eval_freq": 64, "n_eval_episodes": 4, **eval_overrides},
        "curriculum": {
            "stages": [
                {
                    "name": "s1",
                    "map_file": "maps/1v1/beginner.csv",
                    "opponent": "noop",
                    "promotion_win_rate": 0.0,
                    "patience": 1,
                    "max_timesteps": 128,
                }
            ]
        },
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Names and seeds
# ---------------------------------------------------------------------------


class TestNames:
    def test_group_and_run_names(self):
        group = sr.make_group_id("val-3", datetime(2026, 9, 28, 12, 0, 0))
        assert group == "20260928_120000_val-3"
        assert sr.parse_group_id(group) == ("20260928_120000", "val-3")
        assert sr.run_dir_name(group, 1042) == "20260928_120000_val-3_s1042"
        assert sr.parse_run_dir_name("20260928_120000_val-3_s1042") == (group, 1042)
        assert sr.parse_run_dir_name("20260601_172412") is None
        assert sr.vertex_job_name(group, 42) == "rt-val-3-20260928-120000-s42"
        assert sr.group_manifest_path("root", group) == Path("root/_groups") / group / "seed_group.json"

    @pytest.mark.parametrize("tag", ["", "Val", "-x", "a_b", "a" * 32, "a b"])
    def test_bad_tags(self, tag):
        with pytest.raises(ValueError, match="tag"):
            sr.make_group_id(tag)

    def test_bad_group_and_seed(self):
        with pytest.raises(ValueError, match="group"):
            sr.run_dir_name("val", 1)
        with pytest.raises(ValueError, match="seed"):
            sr.run_dir_name(GROUP, -1)

    def test_seeds(self):
        assert sr.parse_seeds(" 42, 1042 ,2042") == [42, 1042, 2042]
        assert sr.seeds_from_stride(42, 3) == [42, 1042, 2042]
        assert sr.seeds_from_stride(7, 2, stride=5) == [7, 12]
        for bad in ("42,42", "", "a,b", "-1"):
            with pytest.raises(ValueError):
                sr.parse_seeds(bad)
        with pytest.raises(ValueError):
            sr.seeds_from_stride(42, 0)
        with pytest.raises(ValueError):
            sr.seeds_from_stride(42, 2, stride=0)


class TestSeedCollisions:
    def test_the_defaults_match_the_config_dataclasses(self):
        cfg = TrainingConfig()
        assert sr._DEFAULTS == {
            "n_envs": cfg.env.n_envs,
            "n_eval_episodes": cfg.eval.n_eval_episodes,
            "eval_freq": cfg.eval.eval_freq,
            "seed_offset": cfg.eval.seed_offset,
            "max_retries": cfg.curriculum.max_retries,
            "max_timesteps": 1_000_000,
        }

    def test_consecutive_seeds_share_training_and_eval_streams(self):
        cfg = load_config(BOOTSTRAP)
        found = sr.seed_collisions([42, 43, 44], cfg)
        assert any("42 and 43: the training envs share 7 of 8" in f for f in found)
        assert any("42 and 43: the gate evals share 59 of 60" in f for f in found)
        assert any("42 and 44: the training envs share 6 of 8" in f for f in found)

    def test_between_n_envs_and_n_eval(self):
        found = sr.seed_collisions([42, 92], load_config(BOOTSTRAP))
        assert found == [
            "seeds 42 and 92: the gate evals share 10 of 60 episode seeds per seat, "
            "policy-sampling streams included (difference 50 < n_eval_episodes 60)"
        ]

    def test_default_seeds_do_not_collide_with_bootstrap_yaml(self):
        cfg = load_config(BOOTSTRAP)
        assert sr.seed_collisions([42, 1042, 2042], cfg) == []
        assert sr.seed_collisions(sr.seeds_from_stride(cfg.seed, 3), cfg.to_dict()) == []

    def test_resampled_eval_sets_collide_on_the_1000_residue(self):
        cfg = load_config(BOOTSTRAP).to_dict()
        cfg["eval"]["resample_eval_seeds"] = True
        found = sr.seed_collisions([42, 1042], cfg)
        assert len(found) == 1 and "d mod 1000 = 0" in found[0]
        assert sr.seed_collisions([42, 542], cfg) == []  # 542 - 42 = 500: in [60, 940]
        assert sr.seed_collisions([42, 1112], cfg) == []  # residue 70: in [60, 940]
        assert sr.seed_collisions([42, 1092], cfg)  # residue 50 < 60

    def test_training_envs_meeting_another_seeds_eval_set(self):
        cfg = {"env": {"n_envs": 4}, "eval": {"n_eval_episodes": 10, "seed_offset": 1_000_000}}
        found = sr.seed_collisions([0, 1_000_005], cfg)
        assert found == ["seeds 0 and 1000005: the training envs of seed 1000005 reuse eval episode seeds of seed 0"]
        assert sr.seed_collisions([0], {"env": {"n_envs": 4}, "eval": {"seed_offset": 2}})


# ---------------------------------------------------------------------------
# Run states, CPUs, the child command
# ---------------------------------------------------------------------------


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestClassifyRun:
    def test_states(self, tmp_path):
        assert sr.classify_run(tmp_path / "missing") == sr.PENDING
        run = tmp_path / "run"
        _write(run / "logs" / "train.log")
        assert sr.classify_run(run) == sr.PENDING
        _write(run / "resolved_config.yaml", "seed: 1\n")
        assert sr.classify_run(run) == sr.INTERRUPTED
        _write(run / "run_status.json", json.dumps({"status": "completed_curriculum"}))
        assert sr.classify_run(run) == sr.INTERRUPTED  # no final_model.zip yet: --resume-if-exists rebuilds it
        _write(run / "final_model.zip")
        assert sr.classify_run(run) == sr.COMPLETED
        _write(run / "run_status.json", json.dumps({"status": "curriculum_stalled"}))
        assert sr.classify_run(run) == sr.STALLED
        _write(run / "run_status.json", "{torn")
        assert sr.classify_run(run) == sr.INTERRUPTED

    def test_states_after_an_exit_and_the_launcher_exit_code(self, tmp_path):
        assert sr.state_after_exit(tmp_path, 143) == sr.INTERRUPTED
        assert sr.state_after_exit(tmp_path, 1) == sr.FAILED
        assert sr.state_after_exit(tmp_path, 0) == sr.FAILED  # exit 0 without a finished record
        # A hard kill (kill -9, the OOM killer) or a hangup leaves a resumable run, not a failed one.
        assert sr.state_after_exit(tmp_path, 137) == sr.INTERRUPTED
        assert sr.state_after_exit(tmp_path, 129) == sr.INTERRUPTED
        assert sr.launcher_exit_code({1: sr.COMPLETED, 2: sr.COMPLETED}) == 0
        assert sr.launcher_exit_code({1: sr.COMPLETED, 2: sr.STALLED}) == 3
        assert sr.launcher_exit_code({1: sr.STALLED, 2: sr.FAILED}) == 1
        assert sr.launcher_exit_code({1: sr.STALLED, 2: sr.INTERRUPTED}, codes={2: 130}) == 130
        assert sr.launcher_exit_code({1: sr.STALLED, 2: sr.INTERRUPTED}, codes={2: 137}) == 143
        assert sr.launcher_exit_code({1: sr.COMPLETED, 2: sr.RUNNING}) == 1
        assert sr.launcher_exit_code({1: sr.COMPLETED}, signalled=signal.SIGTERM) == 143

    @pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "win32", reason="flock")
    def test_a_locked_run_dir_is_running_not_interrupted(self, tmp_path):
        import fcntl

        run = tmp_path / "run"
        _write(run / "resolved_config.yaml", "seed: 1\n")
        record = {"last_state": "running", "sessions": [{"pid": 1, "ended_at": None}]}
        assert sr.run_lock_held(run) is None  # no lock file: the record decides (pid 1 is not this run)
        assert sr.seed_state(run, record) == sr.INTERRUPTED
        # Another process (another open file description) holds the lock: the trainer is alive.
        fd = os.open(run / sr.RUN_LOCK, os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert sr.run_lock_held(run) is True
            assert sr.seed_state(run, {"last_state": "interrupted"}) == sr.RUNNING
            assert sr.seed_state(run, {"last_state": "failed"}) == sr.RUNNING
            with pytest.raises(sr.RunLockedError, match="in use by another process"):
                sr.acquire_run_lock(run, attempts=2, wait=0.01)
        finally:
            os.close(fd)
        assert sr.run_lock_held(run) is False
        assert sr.seed_state(run, {"last_state": "interrupted"}) == sr.INTERRUPTED
        # This process takes it (and taking it again reuses the same lock); a finished run is never "running".
        held = sr.acquire_run_lock(run)
        assert held is not None and sr.acquire_run_lock(run) == held
        assert sr.lock_holder(run)["pid"] == os.getpid()
        _write(run / "run_status.json", json.dumps({"status": "curriculum_stalled"}))
        assert sr.seed_state(run, {}) == sr.STALLED
        sr.release_run_lock(run)
        assert sr.run_lock_held(run) is False

    def test_a_live_unlocked_child_is_found_by_its_pid(self, tmp_path):
        """A trainer that takes no lock: the launcher's record, its pid alive and naming the run dir."""
        run = tmp_path / "20260928_120000_t_s7"
        _write(run / "resolved_config.yaml", "seed: 7\n")
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "--output-dir", str(run)])
        try:
            session = {"pid": child.pid, "host": sr._hostname(), "ended_at": None}
            assert sr.seed_state(run, {"last_state": "running", "sessions": [session]}) == sr.RUNNING
            # Another host's pid, a session that ended, or a record not left running say nothing.
            assert sr.seed_state(run, {"last_state": "running", "sessions": [{**session, "host": "elsewhere"}]}) == (
                sr.INTERRUPTED
            )
            assert sr.seed_state(run, {"last_state": "running", "sessions": [{**session, "ended_at": "x"}]}) == (
                sr.INTERRUPTED
            )
            assert sr.seed_state(run, {"last_state": "interrupted", "sessions": [session]}) == sr.INTERRUPTED
            if Path("/proc").is_dir():  # a pid alive but running something else
                other = tmp_path / "20260928_120000_t_s8"
                _write(other / "resolved_config.yaml", "seed: 8\n")
                assert sr.seed_state(other, {"last_state": "running", "sessions": [session]}) == sr.INTERRUPTED
        finally:
            child.kill()
            child.wait()
        assert sr.seed_state(run, {"last_state": "running", "sessions": [session]}) == sr.INTERRUPTED

    def test_vertex_seed_states(self, tmp_path):
        run = tmp_path / "run"
        assert sr.seed_state(run, {"last_state": "submitted"}) == sr.SUBMITTED  # not fetched: not "pending"
        _write(run / "resolved_config.yaml", "seed: 1\n")
        assert sr.seed_state(run, {"last_state": "submitted"}) == sr.SUBMITTED  # a fetched copy of a live run
        _write(run / "run_status.json", json.dumps({"status": "completed_curriculum"}))
        _write(run / "final_model.zip")
        assert sr.seed_state(run, {"last_state": "submitted"}) == sr.COMPLETED

    def test_a_vertex_seed_finished_in_gcs_keeps_that_state_until_fetched(self, tmp_path):
        """A continuation found the run finished or stalled in GCS; the run dir here is missing or an old copy."""
        job = "projects/p/locations/us-central1/customJobs/101"
        missing = tmp_path / "missing"
        partial = tmp_path / "partial"
        _write(partial / "resolved_config.yaml", "seed: 1\n")  # fetched while it still ran
        for run in (missing, partial):
            for state in (sr.COMPLETED, sr.STALLED):
                assert sr.seed_state(run, {"last_state": state, "job_resource": job}) == state
        # A local record says nothing about a run dir that is gone: the run dir decides.
        assert sr.seed_state(missing, {"last_state": sr.COMPLETED}) == sr.PENDING
        assert sr.seed_state(partial, {"last_state": sr.STALLED}) == sr.INTERRUPTED
        # The fetched run dir, once finished, is the verdict.
        _write(partial / "run_status.json", json.dumps({"status": "curriculum_stalled"}))
        assert sr.seed_state(partial, {"last_state": sr.COMPLETED, "job_resource": job}) == sr.STALLED

    def test_failed_job_exit_status(self):
        message = (
            "The replica workerpool0-0 exited with a non-zero status of 143. To find out more about why your job "
            "exited please check the logs: https://console.cloud.google.com/logs/viewer?project=1"
        )
        assert sr.job_exit_status(message) == 143
        assert sr.job_exit_status("Job exceeded the timeout") is None
        fake = SimpleNamespace(returncode=0, stdout=f"JOB_STATE_FAILED\t{message}\n", stderr="")
        assert sr.describe_job("projects/p/locations/r/customJobs/1", runner=lambda *a, **k: fake) == (
            "JOB_STATE_FAILED",
            message,
        )


class TestCpus:
    def test_split(self):
        assert sr.split_cpus(range(10), 3) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert sr.split_cpus([3, 1, 2, 0], 2) == [[0, 1], [2, 3]]
        with pytest.raises(ValueError):
            sr.split_cpus([0, 1], 3)
        assert sr.parse_cpu_sets("0-9, 10-19,20") == [list(range(10)), list(range(10, 20)), [20]]
        with pytest.raises(ValueError):
            sr.parse_cpu_sets("3-1")
        assert sr.format_cpu_set([4, 5, 6]) == "4-6" and sr.format_cpu_set([1, 3]) == "1,3"

    def test_needed_and_warnings(self):
        cfg = load_config(BOOTSTRAP)
        # bootstrap.yaml runs its 8 eval envs in worker processes
        assert (cfg.eval.n_eval_envs, cfg.eval.eval_use_subprocess) == (8, True)
        assert sr.cpus_needed(cfg) == cfg.env.n_envs + 9
        data = cfg.to_dict()
        data["eval"].update(eval_use_subprocess=False)
        assert sr.cpus_needed(data) == cfg.env.n_envs + 1
        warnings = sr.cpu_warnings([list(range(17)), list(range(17, 25))], cfg)
        assert len(warnings) == 1 and warnings[0].startswith("CPU set 1 (8 CPU(s))")


class TestChildCommand:
    def test_command_and_env(self):
        cmd = sr.build_child_command(
            "c.yaml", 42, "runs/g_s42", "cuda", ["--strict", "--build-bc"], torch_threads=10, python="py"
        )
        assert cmd == [
            "py",
            str(REPO_ROOT / sr.TRAIN_SCRIPT),
            "--config",
            "c.yaml",
            "--seed",
            "42",
            "--output-dir",
            "runs/g_s42",
            "--resume-if-exists",
            "--device",
            "cuda",
            "--strict",
            "--build-bc",
            "--bc-seed",
            "42",
            "--torch-threads",
            "10",
        ]
        env = sr.build_child_env([0, 1], "1", base={"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0,1"})
        assert env == {
            "PATH": "/bin",
            "CUDA_VISIBLE_DEVICES": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "SDL_VIDEODRIVER": "dummy",
            "MPLBACKEND": "Agg",
        }
        assert sr.build_child_env(None, "", base={})["CUDA_VISIBLE_DEVICES"] == ""
        assert "CUDA_VISIBLE_DEVICES" not in sr.build_child_env(None, None, base={})

    @pytest.mark.parametrize(
        "args",
        [
            ["--config", "x"],
            ["--output-dir=d"],
            ["--resume", "d"],
            ["--resume-if-exists"],
            ["--seed", "1"],
            ["--set", "seed=1"],
            ["--set=seed=1"],
            ["--bc-seed", "3"],
        ],
    )
    def test_reserved_passthrough(self, args):
        with pytest.raises(ValueError):
            sr.build_child_command("c.yaml", 1, "d", None, ["--strict", *args])
        sr.check_passthrough(["--set", "eval.n_eval_episodes=4", "--set=env.max_steps=10", "--strict"])


class TestManifest:
    def test_round_trip_digest_and_mismatches(self, tmp_path):
        cfg = load_config(BOOTSTRAP)
        digest = sr.config_digest(cfg)
        other_seed = cfg.to_dict()
        other_seed["seed"], other_seed["ppo"]["device"] = 7, "cuda"
        assert sr.config_digest(other_seed) == digest
        changed = cfg.to_dict()
        changed["eval"]["n_eval_episodes"] = 5
        assert sr.config_digest(changed) != digest
        manifest = sr.new_group_manifest(
            group=GROUP,
            root="r",
            backend="local",
            config="c.yaml",
            digest=digest,
            seeds=[42, 1042],
            train_args=["--strict"],
            git={"commit": "a" * 40, "short": "a" * 7, "dirty": False},
        )
        path = tmp_path / "g" / "seed_group.json"
        sr.write_group(path, manifest)
        assert sr.read_group(path) == manifest and not list(path.parent.glob("*.partial"))
        assert manifest["runs"]["1042"]["run_dir"] == f"{GROUP}_s1042" and manifest["tag"] == "t"
        assert sr.manifest_mismatches(manifest, digest=digest, seeds=[42, 1042], train_args=["--strict"]) == []
        problems = sr.manifest_mismatches(manifest, digest="0" * 64, seeds=[42], train_args=[])
        assert len(problems) == 3


# ---------------------------------------------------------------------------
# The local scheduler, with a stub train_bootstrap.py
# ---------------------------------------------------------------------------

STUB = r"""
import json, os, signal, sys, time
from pathlib import Path

args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1] if name in args else None

seed = int(opt("--seed"))
out = Path(opt("--output-dir"))
calls = out.parent / f"calls_{seed}.jsonl"
record = {
    "argv": args, "cwd": os.getcwd(), "omp": os.environ.get("OMP_NUM_THREADS"),
    "cuda": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
}
with calls.open("a") as fh:
    fh.write(json.dumps(record) + "\n")
n = len(calls.read_text().splitlines())
plan = json.loads(Path(os.environ["STUB_PLAN"]).read_text())[str(seed)]
action = plan[min(n, len(plan)) - 1]
out.mkdir(parents=True, exist_ok=True)
print(f"stub seed {seed} call {n}: {action}", flush=True)
if action != "fail":
    (out / "resolved_config.yaml").write_text(f"seed: {seed}\n")
if action == "complete":
    (out / "run_status.json").write_text(json.dumps({"status": "completed_curriculum"}))
    (out / "final_model.zip").write_bytes(b"zip")
    sys.exit(0)
if action == "stall":
    (out / "run_status.json").write_text(json.dumps({"status": "curriculum_stalled", "stalled_stage": "s1"}))
    sys.exit(3)
if action == "fail":
    sys.exit(1)
if action == "interrupt":
    sys.exit(143)
def on_signal(signum, frame):
    (out / "got_signal").write_text(str(signum))
    sys.exit(128 + signum)
if action == "wait":
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
else:  # "ignore": only SIGKILL ends it
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
(out / "waiting").write_text("1")
time.sleep(120)
sys.exit(1)
"""


@pytest.fixture
def stub(tmp_path, monkeypatch):
    script = tmp_path / "stub_train.py"
    script.write_text(STUB)
    plan_path = tmp_path / "plan.json"
    monkeypatch.setenv("STUB_PLAN", str(plan_path))

    def set_plan(plan: dict[int, list[str]]) -> None:
        plan_path.write_text(json.dumps({str(k): v for k, v in plan.items()}))

    return script, set_plan


def _calls(root: Path, seed: int) -> list[dict]:
    path = root / f"calls_{seed}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class TestLocalScheduler:
    def test_parallel_sessions_exit_codes_and_reruns(self, run_seeds, stub, tmp_path, capsys):
        script, set_plan = stub
        config = _tiny_config(tmp_path / "tiny.yaml")
        root = tmp_path / "root"
        set_plan({0: ["complete"], 1000: ["stall"], 2000: ["fail", "interrupt", "complete"]})
        base = ["--config", str(config), "--group", GROUP, "--root", str(root), "--train-script", str(script)]
        argv = [
            *base,
            "--seeds",
            "0,1000,2000",
            "--parallel",
            "2",
            "--no-pin",
            "--device",
            "cpu",
            "--",
            "--strict",
            "--skip-videos",
        ]
        assert run_seeds.main(argv) == 1  # seed 2000 failed

        manifest = sr.read_group(sr.group_manifest_path(root, GROUP))
        assert manifest["seeds"] == [0, 1000, 2000] and manifest["train_args"] == ["--strict", "--skip-videos"]
        states = {s: manifest["runs"][str(s)]["last_state"] for s in (0, 1000, 2000)}
        assert states == {0: "completed", 1000: "stalled", 2000: "failed"}
        for seed, code in ((0, 0), (1000, 3), (2000, 1)):
            (session,) = manifest["runs"][str(seed)]["sessions"]
            assert session["exit_code"] == code and session["started_at"] and session["ended_at"] and session["pid"]
            assert session["cpu_set"] is None and session["gpu"] == ""
            log = Path(session["log"])
            assert log.parent == root / f"{GROUP}_s{seed}" / "logs" and log.name.startswith("train.")
            assert f"stub seed {seed} call 1" in log.read_text()
        (call,) = _calls(root, 0)
        assert call["argv"][:6] == [
            "--config",
            str(config.resolve()),
            "--seed",
            "0",
            "--output-dir",
            str((root / f"{GROUP}_s0").resolve()),
        ]
        assert call["argv"][6:] == ["--resume-if-exists", "--device", "cpu", "--strict", "--skip-videos"]
        assert call["cwd"] == str(sr.REPO_ROOT) and call["omp"] == "1" and call["cuda"] == ""

        # Re-run: finished seeds are left alone, the failed one is not retried.
        assert run_seeds.main(base) == 1
        assert [len(_calls(root, s)) for s in (0, 1000, 2000)] == [1, 1, 1]
        assert "--retry-failed" in capsys.readouterr().out
        # --retry-failed: the seed is interrupted this time (exit 143).
        assert run_seeds.main([*base, "--retry-failed"]) == 143
        assert sr.read_group(sr.group_manifest_path(root, GROUP))["runs"]["2000"]["last_state"] == "interrupted"
        # An interrupted seed is relaunched without asking; now every seed is done.
        assert run_seeds.main(base) == 3
        assert [len(_calls(root, s)) for s in (0, 1000, 2000)] == [1, 1, 3]
        manifest = sr.read_group(sr.group_manifest_path(root, GROUP))
        assert [s["exit_code"] for s in manifest["runs"]["2000"]["sessions"]] == [1, 143, 0]

        assert run_seeds.main(["status", "--group", GROUP, "--root", str(root)]) == 0
        out = capsys.readouterr().out
        assert "completed" in out and "stalled" in out and "last exit" in out

    @pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="CPU pinning is Linux-only")
    def test_pinned_slots_and_torch_threads(self, run_seeds, stub, tmp_path):
        script, set_plan = stub
        cpu = sr.available_cpus()[0]
        set_plan({0: ["complete"], 1000: ["complete"]})
        root = tmp_path / "root"
        argv = [
            "--config",
            str(_tiny_config(tmp_path / "tiny.yaml")),
            "--seeds",
            "0,1000",
            "--tag",
            "pin",
            "--root",
            str(root),
            "--train-script",
            str(script),
            "--parallel",
            "2",
            "--cpu-sets",
            f"{cpu},{cpu}",
            "--gpus",
            "3",
        ]
        assert run_seeds.main(argv) == 0
        for seed in (0, 1000):
            (call,) = _calls(root, seed)
            assert call["affinity"] == [cpu] and call["cuda"] == "3"
            assert call["argv"][-2:] == ["--torch-threads", "1"] and "cuda" in call["argv"]

    def test_relaunch_refusals(self, run_seeds, stub, tmp_path, capsys):
        script, set_plan = stub
        set_plan({0: ["complete"], 1000: ["complete"]})
        config = _tiny_config(tmp_path / "tiny.yaml")
        root = tmp_path / "root"
        base = ["--group", GROUP, "--root", str(root), "--train-script", str(script)]
        assert run_seeds.main(["--config", str(config), "--seeds", "0,1000", *base]) == 0
        changed = _tiny_config(tmp_path / "changed.yaml", n_eval_episodes=6)
        assert run_seeds.main(["--config", str(changed), *base]) == 2
        assert "config digest" in capsys.readouterr().err
        assert run_seeds.main([*base, "--seeds", "0"]) == 2
        assert run_seeds.main([*base, "--", "--strict"]) == 2
        assert run_seeds.main(["--config", str(changed), *base, "--force"]) == 0
        manifest = sr.read_group(sr.group_manifest_path(root, GROUP))
        assert manifest["history"][0]["event"] == "forced relaunch" and manifest["config"] == str(changed)
        # Usage errors.
        assert run_seeds.main(["--config", str(config), "--seeds", "0"]) == 2  # no tag
        assert run_seeds.main(["--config", str(config), "--tag", "x", "--seeds", "0,1"]) == 2  # overlapping seeds
        assert "share random streams" in capsys.readouterr().err
        assert run_seeds.main(["--config", str(config), "--tag", "x", "--seeds", "0", "--", "--seed", "3"]) == 2
        assert run_seeds.main(["--config", str(config), "--tag", "x", "--seeds", "0", "--dry-run"]) == 0

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
    @pytest.mark.parametrize(
        "sent, action, expected_child, expected_launcher",
        [("SIGTERM", "wait", 143, 143), ("SIGTERM", "ignore", 137, 143), ("SIGHUP", "wait", 143, 129)],
    )
    def test_sigterm_is_forwarded_and_the_launcher_exits_143(
        self, stub, tmp_path, sent, action, expected_child, expected_launcher
    ):
        """SIGTERM reaches the child; a hangup (the SSH session went away) reaches it as SIGTERM, so it checkpoints."""
        script, set_plan = stub
        set_plan({0: [action]})
        root = tmp_path / "root"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train" / "run_seeds.py"),
            "--config",
            str(_tiny_config(tmp_path / "t.yaml")),
            "--seeds",
            "0",
            "--group",
            GROUP,
            "--root",
            str(root),
            "--train-script",
            str(script),
            "--kill-timeout",
            "2",
        ]
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        waiting = root / f"{GROUP}_s0" / "waiting"
        try:
            deadline = time.monotonic() + 120
            while not waiting.exists():
                if proc.poll() is not None or time.monotonic() > deadline:
                    proc.kill()
                    pytest.fail(f"the child never started:\n{proc.communicate()[0]}")
                time.sleep(0.05)
            proc.send_signal(getattr(signal, sent))
            output = proc.communicate(timeout=60)[0]
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        assert proc.returncode == expected_launcher, output
        manifest = sr.read_group(sr.group_manifest_path(root, GROUP))
        (session,) = manifest["runs"]["0"]["sessions"]
        assert session["exit_code"] == expected_child and session["host"]
        # Handled (143) or killed after the timeout (137): either way the run resumes.
        assert manifest["runs"]["0"]["last_state"] == "interrupted"
        if action == "wait":
            assert (root / f"{GROUP}_s0" / "got_signal").read_text() == str(int(signal.SIGTERM))

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
    def test_a_launcher_under_nohup_trains_on_through_a_hangup(self, stub, tmp_path):
        """nohup leaves SIGHUP ignored; the launcher keeps it so, and only a real stop (SIGTERM) stops the group."""
        script, set_plan = stub
        set_plan({0: ["wait"]})
        root = tmp_path / "root"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train" / "run_seeds.py"),
            "--config",
            str(_tiny_config(tmp_path / "t.yaml")),
            "--seeds",
            "0",
            "--group",
            GROUP,
            "--root",
            str(root),
            "--train-script",
            str(script),
            "--kill-timeout",
            "2",
        ]
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN),  # what nohup does
        )
        run_dir = root / f"{GROUP}_s0"
        try:
            deadline = time.monotonic() + 120
            while not (run_dir / "waiting").exists():
                if proc.poll() is not None or time.monotonic() > deadline:
                    proc.kill()
                    pytest.fail(f"the child never started:\n{proc.communicate()[0]}")
                time.sleep(0.05)
            proc.send_signal(signal.SIGHUP)
            time.sleep(1.5)  # several of the launcher's 0.2 s polls
            assert proc.poll() is None and not (run_dir / "got_signal").exists()
            manifest = sr.read_group(sr.group_manifest_path(root, GROUP))
            assert manifest["runs"]["0"]["sessions"][-1]["exit_code"] is None
            proc.send_signal(signal.SIGTERM)
            output = proc.communicate(timeout=60)[0]
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        assert proc.returncode == 143, output
        assert (run_dir / "got_signal").read_text() == str(int(signal.SIGTERM))
        manifest = sr.read_group(sr.group_manifest_path(root, GROUP))
        (session,) = manifest["runs"]["0"]["sessions"]
        assert session["exit_code"] == 143 and manifest["runs"]["0"]["last_state"] == "interrupted"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
    def test_children_of_a_killed_launcher_are_running_not_relaunched(self, run_seeds, stub, tmp_path, capsys):
        """SIGKILL the launcher of a --parallel group: its children train on; status and a relaunch see them."""
        script, set_plan = stub
        set_plan({0: ["wait"], 1000: ["wait"]})
        root = tmp_path / "root"
        config = _tiny_config(tmp_path / "t.yaml")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train" / "run_seeds.py"),
            "--config",
            str(config),
            "--seeds",
            "0,1000",
            "--group",
            GROUP,
            "--root",
            str(root),
            "--train-script",
            str(script),
            "--parallel",
            "2",
            "--no-pin",
        ]
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pids: list[int] = []
        try:
            deadline = time.monotonic() + 120
            while not all((root / f"{GROUP}_s{s}" / "waiting").exists() for s in (0, 1000)):
                assert proc.poll() is None and time.monotonic() < deadline, "the children never started"
                time.sleep(0.05)
            proc.kill()
            proc.wait()
            manifest = sr.read_group(sr.group_manifest_path(root, GROUP))
            pids = [manifest["runs"][str(s)]["sessions"][-1]["pid"] for s in (0, 1000)]
            for pid in pids:
                os.kill(pid, 0)  # still alive, with nobody recording them
            assert run_seeds.main(["status", "--group", GROUP, "--root", str(root)]) == 0
            out = capsys.readouterr().out
            assert out.count("running") == 2 and "interrupted" not in out
            assert run_seeds.main(["--group", GROUP, "--root", str(root), "--train-script", str(script), "--dry-run"]) == 0
            out = capsys.readouterr().out
            assert "would run" not in out and "nothing to run" in out and "still running" in out
        finally:
            if proc.poll() is None:
                proc.kill()
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

    def test_continuing_reuses_the_scheduling_and_a_relaunch_without_group_is_refused(self, run_seeds, stub, tmp_path, capsys):
        script, set_plan = stub
        set_plan({0: ["interrupt", "complete"], 1000: ["complete"]})
        config = _tiny_config(tmp_path / "tiny.yaml")
        root = tmp_path / "root"
        launch = [
            "--config",
            str(config),
            "--seeds",
            "0,1000",
            "--tag",
            "val",
            "--root",
            str(root),
            "--train-script",
            str(script),
            "--parallel",
            "2",
            "--no-pin",
            "--device",
            "cpu",
            "--",
            "--strict",
        ]
        assert run_seeds.main(launch) == 143  # seed 0 interrupted
        (group_dir,) = (root / sr.GROUPS_DIR).iterdir()
        group = group_dir.name
        manifest = sr.read_group(group_dir / sr.GROUP_MANIFEST)
        assert manifest["launch"] == {
            "parallel": 2,
            "cpu_sets": None,
            "no_pin": True,
            "gpus": None,
            "device": "cpu",
            "tee": None,
        }
        capsys.readouterr()
        # The runbook's old advice, re-running the identical launch command, would start over in a new group.
        assert run_seeds.main(launch) == 2
        err = capsys.readouterr().err
        assert f"--group {group}" in err and f"unfinished group {group}: s0 interrupted" in err
        assert len(list((root / sr.GROUPS_DIR).iterdir())) == 1
        assert run_seeds.main([*launch[:-2], "--new-group", "--dry-run", *launch[-2:]]) == 0
        # --group alone continues with the group's config, seeds, args and scheduling.
        assert run_seeds.main(["--group", group, "--root", str(root), "--train-script", str(script)]) == 0
        out = capsys.readouterr().out
        assert "--parallel 2" in out and "the group's scheduling: --parallel 2 --no-pin --device cpu" in out
        calls = _calls(root, 0)
        assert len(calls) == 2 and calls[-1]["argv"][6:] == ["--resume-if-exists", "--device", "cpu", "--strict"]
        # A finished group does not block a new one; --new-group starts one beside an unfinished one.
        set_plan({0: ["complete"], 1000: ["complete"]})
        assert run_seeds.main([*launch[:-2], "--dry-run"]) == 0
        assert run_seeds.main(["--group", group, "--root", str(root), "--parallel", "1"]) == 0
        manifest = sr.read_group(group_dir / sr.GROUP_MANIFEST)
        assert manifest["launch"]["parallel"] == 1
        assert manifest["history"][-1]["event"] == "scheduling changed"
        assert manifest["history"][-1]["changes"] == {"parallel": [2, 1]}


# ---------------------------------------------------------------------------
# Vertex: one job per seed through submit_vertex_job.sh
# ---------------------------------------------------------------------------


class _FakeGcloud:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.state = "JOB_STATE_RUNNING"
        self.message = ""  # the job's error.message
        self.next_id = 100

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        if cmd[:4] == ["gcloud", "ai", "custom-jobs", "describe"]:
            fields = "\t".join(v for v in (self.state, self.message) if v) if "error.message" in cmd[5] else self.state
            return SimpleNamespace(returncode=0, stdout=f"{fields}\n", stderr="")
        self.next_id += 1
        out = f"Job config: ...\nJOB_RESOURCE=projects/p/locations/us-central1/customJobs/{self.next_id}\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    def submissions(self):
        return [(cmd, kw) for cmd, kw in self.calls if cmd[0] == "bash"]


class TestVertex:
    @pytest.fixture
    def vertex(self, run_seeds, monkeypatch, tmp_path):
        fake = _FakeGcloud()
        gcs: dict[str, object] = {"state": None}

        def gcs_state(uri, run):
            if isinstance(gcs["state"], Exception):
                raise gcs["state"]
            return gcs["state"]

        monkeypatch.setattr(run_seeds, "RUNNER", fake)
        monkeypatch.setattr(run_seeds, "GCS_STATE", gcs_state)
        monkeypatch.setenv("BUCKET", "gs://bkt/")
        monkeypatch.setenv("TAG", "abc1234")
        monkeypatch.delenv("IMAGE_URI", raising=False)
        root = tmp_path / "root"
        argv = [
            "--backend",
            "vertex",
            "--config",
            str(BOOTSTRAP),
            "--seeds",
            "42,1042",
            "--group",
            GROUP,
            "--root",
            str(root),
            "--",
            "--strict",
            "--skip-videos",
        ]
        return fake, gcs, root, argv

    def test_dry_run_changes_nothing(self, run_seeds, vertex, capsys):
        fake, _, root, argv = vertex
        assert run_seeds.main([*argv[:-3], "--dry-run", *argv[-3:]]) == 0
        assert fake.calls == [] and not root.exists()
        out = capsys.readouterr().out
        assert f"JOB_NAME=rt-t-20260928-120000-s42 OUTPUT_URI=gs://bkt/jobs/{GROUP}" in out

    def test_submission_spec_and_no_double_submit(self, run_seeds, vertex):
        fake, gcs, root, argv = vertex
        assert run_seeds.main(argv) == 0
        subs = fake.submissions()
        assert len(subs) == 2
        cmd, kw = subs[0]
        run = f"{GROUP}_s42"
        assert cmd == [
            "bash",
            "scripts/cloud/submit_vertex_job.sh",
            "python3",
            "scripts/train/train_bootstrap.py",
            "--config",
            "configs/ppo/bootstrap.yaml",
            "--seed",
            "42",
            "--output-dir",
            f"benchmarks/bootstrap/{run}",
            "--resume-if-exists",
            "--device",
            "cuda",
            "--strict",
            "--skip-videos",
        ]
        env = kw["env"]
        assert env["JOB_NAME"] == "rt-t-20260928-120000-s42"
        assert env["OUTPUT_URI"] == f"gs://bkt/jobs/{GROUP}"
        assert env["RESTORE_DIRS"] == f"benchmarks/bootstrap/{run}={run}"
        assert env["BUCKET"] == "bkt" and env["TAG"] == "abc1234" and kw["cwd"] == str(sr.REPO_ROOT)
        manifest = sr.read_group(sr.group_manifest_path(root, GROUP))
        assert manifest["runs"]["42"]["job_resource"] == "projects/p/locations/us-central1/customJobs/101"
        assert manifest["vertex"]["output_uri"] == f"gs://bkt/jobs/{GROUP}" and manifest["backend"] == "vertex"

        # Running jobs are never resubmitted.
        fake.calls.clear()
        assert run_seeds.main(argv) == 0
        assert fake.submissions() == []
        describe = [c for c, _ in fake.calls]
        assert describe[0][:5] == [
            "gcloud",
            "ai",
            "custom-jobs",
            "describe",
            "projects/p/locations/us-central1/customJobs/101",
        ]
        assert "--region=us-central1" in describe[0]

        # A finished job whose run completed stays; one that was interrupted mid-run (preempted: SIGTERM,
        # exit 143) is resubmitted with the same spec.
        fake.state = "JOB_STATE_FAILED"
        gcs["state"] = "completed"
        assert run_seeds.main(argv) == 0 and fake.submissions() == []
        gcs["state"] = None
        fake.message = "The replica workerpool0-0 exited with a non-zero status of 143. To find out more ..."
        assert run_seeds.main(argv) == 0
        assert [c for c, _ in fake.submissions()] == [cmd, subs[1][0]]
        # One that failed (exit 1, or no exit status: a timeout) would fail again: --retry-failed, as locally.
        fake.calls.clear()
        fake.message = "The replica workerpool0-0 exited with a non-zero status of 1. To find out more ..."
        assert run_seeds.main(argv) == 1 and fake.submissions() == []
        assert sr.read_group(sr.group_manifest_path(root, GROUP))["runs"]["42"]["last_state"] == "failed"
        assert run_seeds.main([*argv[:-3], "--retry-failed", *argv[-3:]]) == 0 and len(fake.submissions()) == 2
        # Without google-cloud-storage, only --only-seeds resubmits.
        fake.calls.clear()
        gcs["state"] = ImportError("no google-cloud-storage")
        assert run_seeds.main(argv) == 0 and fake.submissions() == []
        assert run_seeds.main([*argv[:-3], "--only-seeds", "1042", *argv[-3:]]) == 0
        assert [c[7] for c, _ in fake.submissions()] == ["1042"]

    def test_continuing_keeps_the_bucket_image_and_machine(self, run_seeds, vertex, monkeypatch, capsys):
        fake, _, root, argv = vertex
        monkeypatch.setenv("MACHINE_TYPE", "n1-standard-16")
        monkeypatch.setenv("ACCELERATOR_TYPE", "NVIDIA_TESLA_T4")
        monkeypatch.setenv("TAG", sr.git_info()["short"] or "abc1234")
        assert run_seeds.main(argv) == 0
        first = fake.submissions()[0][1]["env"]
        assert first["MACHINE_TYPE"] == "n1-standard-16" and first["ACCELERATOR_TYPE"] == "NVIDIA_TESLA_T4"
        head = sr.git_info()["commit"]
        if head:  # the tag names this checkout's commit: the job records it
            assert first["RT_GIT_COMMIT"] == head
        manifest = sr.read_group(sr.group_manifest_path(root, GROUP))
        session = manifest["runs"]["42"]["sessions"][0]
        assert session["image"] == f"TAG={first['TAG']}" and session["machine"]["MACHINE_TYPE"] == "n1-standard-16"
        assert manifest["vertex"]["machine"] == {"MACHINE_TYPE": "n1-standard-16", "ACCELERATOR_TYPE": "NVIDIA_TESLA_T4"}

        # --group alone, in a fresh shell (no BUCKET / TAG / MACHINE_TYPE): the group's settings.
        for key in ("BUCKET", "TAG", "MACHINE_TYPE", "ACCELERATOR_TYPE"):
            monkeypatch.delenv(key)
        fake.state, fake.message = "JOB_STATE_CANCELLED", ""
        fake.calls.clear()
        assert run_seeds.main(["--group", GROUP, "--root", str(root)]) == 0
        env = fake.submissions()[0][1]["env"]
        assert env["BUCKET"] == "bkt" and env["TAG"] == first["TAG"] and env["MACHINE_TYPE"] == "n1-standard-16"

        # Another bucket or image is refused (a new bucket restores nothing: every seed would start over).
        fake.calls.clear()
        monkeypatch.setenv("BUCKET", "other-bucket")
        assert run_seeds.main(argv) == 2 and fake.submissions() == []
        assert "bucket other-bucket != the group's bkt" in capsys.readouterr().err
        monkeypatch.setenv("BUCKET", "bkt")
        monkeypatch.setenv("TAG", "def5678")
        assert run_seeds.main(argv) == 2 and fake.submissions() == []
        assert run_seeds.main([*argv[:-3], "--force", *argv[-3:]]) == 0
        manifest = sr.read_group(sr.group_manifest_path(root, GROUP))
        assert manifest["history"][-1]["event"] == "forced Vertex change"
        assert manifest["vertex"]["image_uri"] == "TAG=def5678"

    def test_status_of_submitted_and_fetched_seeds(self, run_seeds, vertex, capsys):
        fake, _, root, argv = vertex
        assert run_seeds.main(argv) == 0
        capsys.readouterr()
        assert run_seeds.main(["status", "--group", GROUP, "--root", str(root), "--jobs"]) == 0
        out = capsys.readouterr().out
        assert out.count("submitted") == 2 and "pending" not in out and "JOB_STATE_RUNNING" in out
        # A fetched finished run (final_model.zip is fetched by default) reads as completed.
        run = root / f"{GROUP}_s42"
        _write(run / "resolved_config.yaml", "seed: 42\n")
        _write(run / "run_status.json", json.dumps({"status": "completed_curriculum"}))
        _write(run / "final_model.zip")
        assert run_seeds.main(["status", "--group", GROUP, "--root", str(root)]) == 0
        out = capsys.readouterr().out
        assert "completed" in out and out.count("submitted") == 1
        assert not re.search(run_seeds.FETCH_EXCLUDE, "final_model.zip")
        assert re.search(run_seeds.FETCH_EXCLUDE, "starter_simple/best_model.zip")
        assert re.search(run_seeds.FETCH_EXCLUDE, "starter_simple/latest.zip")

    def test_seeds_found_finished_in_gcs_are_finished_before_fetch(self, run_seeds, vertex, monkeypatch, capsys):
        """A continuation reads run_status.json in GCS: status, and a fresh launch, see those seeds as finished."""
        fake, _, root, argv = vertex
        assert run_seeds.main(argv) == 0
        fake.state, fake.message = "JOB_STATE_SUCCEEDED", ""
        monkeypatch.setattr(run_seeds, "GCS_STATE", lambda uri, run: "completed" if run.endswith("_s42") else "stalled")
        fake.calls.clear()
        assert run_seeds.main(argv) == 0 and fake.submissions() == []
        manifest = sr.read_group(sr.group_manifest_path(root, GROUP))
        assert [manifest["runs"][s]["last_state"] for s in ("42", "1042")] == ["completed", "stalled"]
        assert not (root / f"{GROUP}_s42").exists()  # nothing fetched
        capsys.readouterr()
        assert run_seeds.main(["status", "--group", GROUP, "--root", str(root)]) == 0
        out = capsys.readouterr().out
        assert "completed" in out and "stalled" in out and "pending" not in out and "submitted" not in out
        # The group is finished: a new launch with the same tag and config is not sent back to it.
        fresh = [a for a in argv if a not in ("--group", GROUP)]
        fresh[fresh.index("--root") : fresh.index("--root")] = ["--tag", "t"]
        assert run_seeds.main([*fresh[:-3], "--dry-run", *fresh[-3:]]) == 0
        assert "unfinished group" not in capsys.readouterr().err

    def test_needs_bucket_and_an_image(self, run_seeds, vertex, monkeypatch):
        _, _, _, argv = vertex
        monkeypatch.delenv("TAG")
        assert run_seeds.main(argv) == 2
        monkeypatch.setenv("TAG", "abc1234")
        monkeypatch.delenv("BUCKET")
        assert run_seeds.main(argv) == 2

    def test_fetch_downloads_each_run(self, run_seeds, vertex, monkeypatch):
        fake, _, root, argv = vertex
        assert run_seeds.main(argv) == 0
        seen = []

        def download_tree(src, local, *, exclude=None, **kw):
            seen.append((src, Path(local), exclude.pattern))
            return 2

        monkeypatch.setattr("reinforcetactics.cloud.storage.download_tree", download_tree)
        assert run_seeds.main(["fetch", "--group", GROUP, "--root", str(root)]) == 0
        assert seen[0][0] == f"gs://bkt/jobs/{GROUP}/{GROUP}_s42" and seen[0][1] == root / f"{GROUP}_s42"
        assert r"\.zip$" in seen[0][2]
        assert run_seeds.main(["fetch", "--group", GROUP, "--root", str(root), "--include-checkpoints"]) == 0
        assert r"\.zip$" not in seen[-1][2]


@pytest.mark.skipif(sys.platform == "win32", reason="bash")
def test_submit_script_passes_the_group_settings_to_the_job(tmp_path):
    """submit_vertex_job.sh with a fake gcloud on PATH: OUTPUT_URI, RESTORE_DIRS, the restart flag, JOB_RESOURCE."""
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    capture = tmp_path / "job.yaml"
    gcloud = fakebin / "gcloud"
    gcloud.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == config ]]; then echo proj; exit 0; fi\n'
        'for a in "$@"; do case "$a" in --config=*) cp "${a#--config=}" "' + str(capture) + '";; esac; done\n'
        'echo "CustomJob [projects/123/locations/us-central1/customJobs/456] is submitted successfully." >&2\n'
    )
    gcloud.chmod(0o755)
    inherited = {
        k: v for k, v in os.environ.items() if k not in ("PROJECT_ID", "REGION", "IMAGE_URI", "SYNC_DIRS", "SERVICE_ACCOUNT")
    }
    env = {
        **inherited,
        "PATH": f"{fakebin}:{os.environ['PATH']}",
        "BUCKET": "bkt",
        "JOB_NAME": "rt-t-20260928-120000-s42",
        "OUTPUT_URI": "gs://bkt/jobs/20260928_120000_t",
        "RESTORE_DIRS": "benchmarks/bootstrap/20260928_120000_t_s42=20260928_120000_t_s42",
        "RESTART_ON_WORKER_RESTART": "1",
        "TAG": "abc1234",
        "RT_GIT_COMMIT": "abc1234" + "0" * 33,
    }
    out = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "cloud" / "submit_vertex_job.sh"), "python3", "x.py", "--seed", "42"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    assert "JOB_RESOURCE=projects/123/locations/us-central1/customJobs/456" in out.stdout.splitlines()
    spec = yaml.safe_load(capture.read_text())
    assert spec["baseOutputDirectory"]["outputUriPrefix"] == "gs://bkt/jobs/20260928_120000_t"
    assert spec["scheduling"] == {"restartJobOnWorkerRestart": True}
    container = spec["workerPoolSpecs"][0]["containerSpec"]
    env_vars = {e["name"]: e["value"] for e in container["env"]}
    assert env_vars["GCS_OUTPUT_URI"] == "gs://bkt/jobs/20260928_120000_t"
    assert env_vars["GCS_RESTORE_DIRS"] == env["RESTORE_DIRS"]
    assert env_vars["RT_GIT_COMMIT"] == env["RT_GIT_COMMIT"]
    assert container["args"] == ["python3", "x.py", "--seed", "42"]
    assert container["imageUri"] == "us-central1-docker.pkg.dev/proj/reinforce-tactics/rl-trainer:abc1234"


def test_run_records_take_the_commit_from_the_job_env_without_a_checkout(monkeypatch):
    """The training image has no .git: a Vertex job's records name the commit run_seeds.py passed (RT_GIT_COMMIT)."""
    from reinforcetactics.utils import run_config

    def no_checkout(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "rev-parse")

    monkeypatch.setattr(run_config.subprocess, "check_output", no_checkout)
    monkeypatch.delenv("RT_GIT_COMMIT", raising=False)
    assert run_config._git_meta() == {"commit": None, "short": None, "dirty": None}
    monkeypatch.setenv("RT_GIT_COMMIT", "d7bd68498ec0b512049e279b6c6f8279b6e0d8ab")
    meta = run_config._git_meta()
    assert meta["commit"] == "d7bd68498ec0b512049e279b6c6f8279b6e0d8ab" and meta["short"] == "d7bd684"
    assert meta["dirty"] is None and meta["source"] == "RT_GIT_COMMIT"


def test_a_teed_child_writes_to_its_log_file_not_through_the_launcher(tmp_path):
    """With tee on, the child used to write to a pipe the launcher read, so a killed launcher
    broke the child at its next print; it now writes to its log file, which the console follows."""
    child = tmp_path / "child.py"
    child.write_text(
        "import os, stat, sys\n"
        "print('stdout is a regular file:', stat.S_ISREG(os.fstat(1).st_mode), flush=True)\n"
        "for i in range(3):\n"
        "    print(f'line {i}', flush=True)\n"
        "sys.stdout.write('last line without a newline')\n"
    )
    console = io.StringIO()

    def make_spec(seed: int, slot: int) -> sr.ChildSpec:
        return sr.ChildSpec(
            seed=seed,
            run_dir=tmp_path / f"s{seed}",
            cmd=[sys.executable, str(child)],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            cwd=tmp_path,
        )

    result = sr.run_local([7], make_spec, tee=True, console=console, handle_signals=False, poll_interval=0.05)

    assert result.exit_codes == {7: 0}
    shown = console.getvalue()
    assert "stdout is a regular file: True" in shown
    assert "line 0\nline 1\nline 2\n" in shown and shown.endswith("last line without a newline\n")
    (log,) = (tmp_path / "s7" / "logs").glob("train.*")
    assert "line 2" in log.read_text() and "last line without a newline" in log.read_text()
