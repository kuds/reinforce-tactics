"""End-to-end smoke runs of the training entry points on a tiny config.

Runs ``scripts/train/train_bootstrap.py`` and ``scripts/train/train_self_play.py``
as a user would (a subprocess, ``--config``, ``--strict``) for a few hundred
timesteps on one env with a tiny network, and checks each finishes and leaves
its artifacts. They exercise the whole config -> env -> trainer path the
config and plumbing changes touch (typed loading, the ignored-field check,
``resolve_config``, the per-stage record, the self-play env kwargs and the
opponent-pool knob). About 6-7 s each, so they stay in the default run.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP = "maps/1v1/beginner.csv"


def _run(argv: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("GCS_OUTPUT_URI", "AIP_MODEL_DIR", "GCS_WRAPPER_SYNC")}
    env.update(PYTHONPATH=str(REPO_ROOT), SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", MPLBACKEND="Agg")
    # cwd: the configs name maps relative to the repository root.
    return subprocess.run(
        [sys.executable, *argv], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600, check=False
    )


def test_train_bootstrap_runs_a_tiny_curriculum(tmp_path):
    config = tmp_path / "tiny_bootstrap.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "env": {
                    "n_envs": 1,
                    "use_subprocess": False,
                    "action_space_type": "flat_discrete",
                    "max_flat_actions": 64,
                    "max_steps": 40,
                    "max_turns": 5,
                    "enabled_units": ["W"],
                    "fog_of_war": True,
                },
                "ppo": {"n_steps": 64, "batch_size": 32, "n_epochs": 1, "policy_kwargs": {"net_arch": [16]}},
                "eval": {"eval_freq": 64, "n_eval_episodes": 1},
                "curriculum": {
                    "stages": [
                        {
                            "name": "tiny",
                            "map_file": MAP,
                            "opponent": "noop",
                            # Promote at the first eval after 128 stage steps.
                            "promotion_win_rate": 0.0,
                            "patience": 1,
                            "min_timesteps_before_promotion": 128,
                            "max_timesteps": 320,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "run"
    proc = _run(
        [
            "scripts/train/train_bootstrap.py", "--config", str(config), "--output-dir", str(out), "--device", "cpu",
            "--no-gcs", "--skip-plots", "--skip-videos", "--sanity-episodes", "1", "--strict",
        ],
        tmp_path,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]

    assert (out / "final_model.zip").exists()
    assert json.loads((out / "run_status.json").read_text())["status"] == "completed_curriculum"
    resolved = yaml.safe_load((out / "resolved_config.yaml").read_text())
    assert resolved["env"]["flat_action_version"] == 2
    stage_record = json.loads((out / "tiny" / "config.json").read_text())
    assert stage_record["env_config"]["fog_of_war"] is True
    assert stage_record["extra"]["n_eval_episodes"] == 1


def test_train_self_play_runs_a_tiny_session(tmp_path):
    config = tmp_path / "tiny_self_play.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "algorithm": "self_play",
                "total_timesteps": 192,
                "seed": 0,
                "env": {
                    "map_file": MAP,
                    "n_envs": 1,
                    "use_subprocess": False,
                    "action_space_type": "flat_discrete",
                    "max_flat_actions": 64,
                    "max_steps": 32,
                    "fog_of_war": True,
                    "engine_overrides": {"starting_gold": 300},
                },
                "ppo": {"n_steps": 64, "batch_size": 32, "n_epochs": 1, "device": "cpu", "policy_kwargs": {"net_arch": [16]}},
                "self_play": {
                    "use_opponent_pool": True,
                    "opponent_update_freq": 64,
                    "add_to_pool_freq": 64,
                    "min_win_rate_for_pool": 0.0,
                    "latest_opponent_prob": 0.5,
                },
                "eval": {"eval_freq": 64, "n_eval_episodes": 1, "checkpoint_freq": 128},
                "logging": {"log_dir": str(tmp_path / "logs")},
            }
        ),
        encoding="utf-8",
    )
    proc = _run(["scripts/train/train_self_play.py", "--config", str(config), "--strict", "--no-progress-bar"], tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]

    (log_dir,) = list((tmp_path / "logs").iterdir())
    assert (log_dir / "final_model.zip").exists()
    record = json.loads((log_dir / "config.json").read_text())
    assert record["fog_of_war"] is True
    assert record["engine_overrides"] == {"starting_gold": 300}
    assert record["latest_opponent_prob"] == 0.5
