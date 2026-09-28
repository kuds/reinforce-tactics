"""The seed launcher: reinforcetactics/experiments/seed_runs.py and scripts/train/run_seeds.py.

The scheduler tests run the real launcher with a stub in place of
train_bootstrap.py (``--train-script``): the stub records how it was called and
then completes, stalls, fails, is interrupted, or waits for a signal, as a
per-seed plan says. The Vertex tests replace the command runner and the GCS
reader; nothing touches the network.
"""

from __future__ import annotations

import importlib.util
import json
import os
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
        assert sr.launcher_exit_code({1: sr.COMPLETED, 2: sr.COMPLETED}) == 0
        assert sr.launcher_exit_code({1: sr.COMPLETED, 2: sr.STALLED}) == 3
        assert sr.launcher_exit_code({1: sr.STALLED, 2: sr.FAILED}) == 1
        assert sr.launcher_exit_code({1: sr.STALLED, 2: sr.INTERRUPTED}, codes={2: 130}) == 130
        assert sr.launcher_exit_code({1: sr.COMPLETED}, signalled=signal.SIGTERM) == 143


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
    @pytest.mark.parametrize("action, expected_child", [("wait", 143), ("ignore", 137)])
    def test_sigterm_is_forwarded_and_the_launcher_exits_143(self, stub, tmp_path, action, expected_child):
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
            proc.send_signal(signal.SIGTERM)
            output = proc.communicate(timeout=60)[0]
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        assert proc.returncode == 143, output
        manifest = sr.read_group(sr.group_manifest_path(root, GROUP))
        (session,) = manifest["runs"]["0"]["sessions"]
        assert session["exit_code"] == expected_child
        if action == "wait":
            assert (root / f"{GROUP}_s0" / "got_signal").read_text() == str(int(signal.SIGTERM))
            assert manifest["runs"]["0"]["last_state"] == "interrupted"


# ---------------------------------------------------------------------------
# Vertex: one job per seed through submit_vertex_job.sh
# ---------------------------------------------------------------------------


class _FakeGcloud:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.state = "JOB_STATE_RUNNING"
        self.next_id = 100

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        if cmd[:4] == ["gcloud", "ai", "custom-jobs", "describe"]:
            return SimpleNamespace(returncode=0, stdout=f"{self.state}\n", stderr="")
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

        # A finished job whose run completed stays; one that died mid-run is resubmitted with the same spec.
        fake.state = "JOB_STATE_FAILED"
        gcs["state"] = "completed"
        assert run_seeds.main(argv) == 0 and fake.submissions() == []
        gcs["state"] = None
        assert run_seeds.main(argv) == 0
        assert [c for c, _ in fake.submissions()] == [cmd, subs[1][0]]
        # Without google-cloud-storage, only --only-seeds resubmits.
        fake.calls.clear()
        gcs["state"] = ImportError("no google-cloud-storage")
        assert run_seeds.main(argv) == 0 and fake.submissions() == []
        assert run_seeds.main([*argv[:-3], "--only-seeds", "1042", *argv[-3:]]) == 0
        assert [c[7] for c, _ in fake.submissions()] == ["1042"]

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
    assert container["args"] == ["python3", "x.py", "--seed", "42"]
    assert container["imageUri"] == "us-central1-docker.pkg.dev/proj/reinforce-tactics/rl-trainer:abc1234"
