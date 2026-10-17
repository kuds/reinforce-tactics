"""Execute the notebook's real persistence/evaluation flow with a tiny CPU run."""

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
import torch
from sb3_contrib import MaskablePPO

from reinforcetactics.rl.callbacks import PeriodicEvalCallback, TrainingMetricsCallback
from reinforcetactics.rl.masking import make_maskable_env, make_maskable_vec_env

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "ppo_training.ipynb"


def _execute_cell(identifier, namespace, notebook=NOTEBOOK):
    cells = json.loads(notebook.read_text())["cells"]
    cell = next(c for c in cells if c.get("id") == identifier)
    exec(compile("".join(cell["source"]), f"{notebook}:{identifier}", "exec"), namespace)


def _namespace(tmp_path):
    import time

    return {
        "json": json,
        "time": time,
        "pd": pd,
        "Path": Path,
        "MaskablePPO": MaskablePPO,
        "TrainingMetricsCallback": TrainingMetricsCallback,
        "PeriodicEvalCallback": PeriodicEvalCallback,
        "make_maskable_env": make_maskable_env,
        "make_maskable_vec_env": make_maskable_vec_env,
        "BENCHMARK_DIR": tmp_path,
        "MAP_FILE": str(REPO_ROOT / "maps" / "1v1" / "starter.csv"),
        "OPPONENT": "random",
        "MAX_STEPS": 8,
        "MAX_TURNS": 2,
        "N_ENVS": 2,
        "SEED": 42,
        "ENABLED_UNITS": ["W", "M", "A"],
        "ACTION_SPACE": "flat_discrete",
        "TOTAL_TIMESTEPS": 17,
        "EVAL_FREQ": 16,
        "N_EVAL_EPISODES": 1,
        "EVAL_SEATS": [1, 2],
        "EVAL_BOTH_MODES": True,
        "N_FINAL_EVAL_EPISODES": 1,
        "FINAL_EVAL_SEED": 1_000_042,
        "REPLAY_SEED": 2_000_042,
        "MAX_TRACES_PER_EVAL": 1,
        "REWARD_CONFIG": {},
        "PPO_CONFIG": {"n_steps": 8, "batch_size": 8, "n_epochs": 1, "gamma": 0.99},
        "DEVICE": "cpu",
        "USE_GOOGLE_DRIVE": False,
        "RUN_ID": "smoke-test",
    }


def test_notebook_persists_before_videos_and_evaluates_saved_final_weights(tmp_path):
    ns = _namespace(tmp_path)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _execute_cell("create-envs", ns)
        _execute_cell("create-model", ns)
        _execute_cell("8dd0d8092fe74a7c96281538738b07e2", ns)
        initial = json.loads((tmp_path / "config.json").read_text())
        assert initial["extra"]["status"] == "initialized"
        _execute_cell("train-loop", ns)
        completed = json.loads((tmp_path / "config.json").read_text())
        assert completed["extra"]["status"] == "completed"
        assert completed["extra"]["actual_timesteps"] == 32
        assert (tmp_path / "training_metrics.csv").read_text().count("\n") >= 3
        assert (tmp_path / "benchmark_results.csv").is_file()
        development = [json.loads(line) for line in (tmp_path / "eval_results.jsonl").read_text().splitlines()]
        assert development[-1]["timesteps"] == 32
        assert len(development[-1]["episode_records"]) == 2
        assert development[-1]["win_rate_stochastic"] is not None
        assert development[-1]["win_rate_greedy"] is not None
        summary = json.loads((tmp_path / "benchmark_results.json").read_text())
        assert len(summary["results"]) == len(development)
        _execute_cell("heldout-eval", ns)
        rows = [json.loads(line) for line in (tmp_path / "heldout_results.jsonl").read_text().splitlines()]
        assert {(row["checkpoint"], row["deterministic"]) for row in rows} == {
            (name, deterministic) for name in ("best", "final") for deterministic in (True, False)
        }
        assert {row["eval_seed"] for row in rows} == {ns["FINAL_EVAL_SEED"]}
        assert all(set(row["by_seat"]) == {"1", "2"} for row in rows)
        final_rows = [row for row in rows if row["checkpoint"] == "final"]
        assert all(row["checkpoint_timesteps"] == completed["extra"]["actual_timesteps"] for row in final_rows)
        assert all(rec["seed"] == ns["FINAL_EVAL_SEED"] for row in rows for rec in row["episode_records"])
        assert all(len(list(path.glob("*.jsonl"))) <= 1 for path in (tmp_path / "traces").glob("eval_*"))
    finally:
        for key in ("vec_env", "eval_env"):
            if key in ns:
                ns[key].close()
        torch.set_num_threads(previous_threads)


def test_notebook_keeps_partial_results_on_training_failure(tmp_path):
    ns = _namespace(tmp_path)
    ns.update(eval_env=SimpleNamespace(), model=SimpleNamespace(num_timesteps=0))
    _execute_cell("8dd0d8092fe74a7c96281538738b07e2", ns)

    def fail_training(**kwargs):
        ns["model"].num_timesteps = 12
        ns["eval_callback"].results.append({"timesteps": 12, "win_rate": 0.0})
        raise RuntimeError("simulated training failure")

    ns["model"].learn = fail_training
    with pytest.raises(RuntimeError, match="simulated training failure"):
        _execute_cell("train-loop", ns)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["extra"]["status"] == "interrupted"
    assert config["extra"]["actual_timesteps"] == 12
    summary = json.loads((tmp_path / "benchmark_results.json").read_text())
    assert summary["results"] == [{"timesteps": 12, "win_rate": 0.0}]


def test_bootstrap_sanity_uses_stage_gate_override_instead_of_raw_win_rate(monkeypatch, capsys):
    from reinforcetactics.rl import bootstrap
    from reinforcetactics.rl.config import CurriculumConfig, CurriculumStage, EnvConfig, EvalConfig, PPOConfig

    stage = CurriculumStage(name="test", promotion_win_rate=0.9, promotion_criterion="wilson")
    cfg = SimpleNamespace(
        seed=42,
        curriculum=CurriculumConfig(stages=[stage], promotion_criterion="point"),
        env=EnvConfig(),
        ppo=PPOConfig(),
        eval=EvalConfig(eval_both_modes=False),
    )
    closed = []
    monkeypatch.setattr(
        bootstrap, "make_stage_env", lambda *args, **kwargs: SimpleNamespace(close=lambda: closed.append(True))
    )
    monkeypatch.setattr(MaskablePPO, "load", lambda *args, **kwargs: object())
    metrics = {"episodes": 100, "wins": 92, "losses": 8, "draws": 0, "win_rate": 0.92, "avg_reward": 1.0}
    namespace = {
        "cfg": cfg,
        "result": {"history": [{"stage": "test"}], "final_model_path": "unused.zip"},
        "MaskablePPO": MaskablePPO,
        "evaluate_model": lambda *args, **kwargs: metrics,
    }
    _execute_cell(
        "d4a0fda0de0a48d2b313fde4e730c733",
        namespace,
        REPO_ROOT / "notebooks" / "ppo_bootstrap.ipynb",
    )
    assert namespace["sanity_statistic"] < stage.promotion_win_rate < metrics["win_rate"]
    assert "WARNING: held-out gate score" in capsys.readouterr().out
    assert closed == [True]
