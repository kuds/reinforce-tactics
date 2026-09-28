"""A seed makes a run reproducible, and different seeds make different runs (tiny real curriculum, CPU).

The seed replication of the §2.1 validation run rests on both: re-running a
seed must give the same numbers (so a seed's result is a property of the
seed), and seeds 42 / 1042 / 2042 must not be copies of one another.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reinforcetactics.rl.bootstrap import run_curriculum
from reinforcetactics.rl.config import config_from_dict

# Written per row by the eval callback; not part of what a seed determines.
_TIMING = ("wall_time", "eval_seconds")


def _run(tmp_path: Path, seed: int, name: str) -> tuple[list[dict[str, Any]], str]:
    data: dict[str, Any] = {
        "seed": seed,
        "env": {
            "n_envs": 1,
            "use_subprocess": False,
            "action_space_type": "flat_discrete",
            "max_flat_actions": 64,
            "max_steps": 40,
            "max_turns": 5,
            "enabled_units": ["W"],
        },
        "ppo": {"n_steps": 64, "batch_size": 32, "n_epochs": 1, "policy_kwargs": {"net_arch": [16]}, "device": "cpu"},
        "eval": {"eval_freq": 64, "n_eval_episodes": 3, "checkpoint_freq": 64},
        "curriculum": {
            "stages": [
                {
                    "name": "s1",
                    "map_file": "maps/1v1/beginner.csv",
                    "opponent": "random",
                    "promotion_win_rate": 0.0,
                    "patience": 1,
                    "min_timesteps_before_promotion": 128,
                    "max_timesteps": 192,
                }
            ]
        },
    }
    out = tmp_path / name
    run_curriculum(config_from_dict(data), out)
    rows = json.loads((out / "s1" / "eval_results.json").read_text())
    for row in rows:
        for key in _TIMING:
            assert key in row
            row.pop(key)
        row.pop("traces", None)
    return rows, (out / "train_metrics.csv").read_text()


def test_same_seed_same_run_and_different_seeds_differ(tmp_path):
    first, first_metrics = _run(tmp_path, 0, "a")
    again, again_metrics = _run(tmp_path, 0, "b")
    other, _ = _run(tmp_path, 1000, "c")
    assert len(first) >= 2
    # The same seed: identical eval rows (both modes) and training diagnostics.
    assert again == first
    assert again_metrics == first_metrics
    # Another seed: another eval problem set and other games.
    assert [r["eval_seed"] for r in other] == [r["eval_seed"] + 1000 for r in first]
    assert [(r["rewards"], r["lengths"]) for r in other] != [(r["rewards"], r["lengths"]) for r in first]
