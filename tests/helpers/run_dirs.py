"""Writers of synthetic bootstrap run directories, new and legacy layouts (tests of the seed aggregator).

The layouts mirror what ``reinforcetactics/rl/bootstrap.py`` and
``scripts/train/train_bootstrap.py`` write today (``resolved_config.yaml``,
per-row gate mode and ``other_mode``, stage step bounds, retries,
``run_status.json`` / ``run_manifest.json``), and what the 2026-06 archive
holds (greedy rows without ``deterministic``, the source YAML in the run
root, a 15-column ``bootstrap_results.csv``, no run status).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

GIT = {"commit": "abc1234" + "0" * 33, "short": "abc1234", "dirty": False}
LEGACY_GIT = {"commit": "078313e" + "0" * 33, "short": "078313e", "dirty": False}
MAP = "maps/1v1/beginner.csv"
COMPONENTS = ("action", "shaping_delta", "invalid_penalty", "terminal")
# Per-episode reward components by outcome (their sum is the episode return).
PER_EPISODE = {
    "wins": {"action": 10.0, "shaping_delta": -2.0, "invalid_penalty": 0.0, "terminal": 50.0},
    "draws": {"action": 6.0, "shaping_delta": -1.0, "invalid_penalty": -0.5, "terminal": -10.0},
    "losses": {"action": 2.0, "shaping_delta": -1.0, "invalid_penalty": 0.0, "terminal": -50.0},
}
LEGACY_CSV_COLUMNS = (
    "stage",
    "map_file",
    "opponent",
    "timesteps",
    "win_rate",
    "avg_reward",
    "std_reward",
    "avg_length",
    "std_length",
    "avg_turns",
    "std_turns",
    "wins",
    "losses",
    "draws",
    "episodes",
)


def _counts(wins: int, draws: int, losses: int) -> dict[str, Any]:
    n = wins + draws + losses
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "episodes": n,
        "win_rate": wins / n,
        "draw_rate": draws / n,
        "loss_rate": losses / n,
    }


def eval_row(
    timesteps: int,
    wins: int,
    draws: int,
    losses: int,
    *,
    mode: str = "stochastic",
    other: tuple[int, int, int] | None = None,
    attempt: int = 0,
    eval_seed: int = 1_000_042,
    wall_time: float | None = None,
    eval_seconds: float | None = None,
    captures: tuple[float, float, float] = (1.0, 0.5, 0.0),
    opponent_captures: tuple[int, int] = (1, 0),
    truncated: int = 0,
    flat_truncated_rate: float = 0.0,
    per_episode: dict[str, dict[str, float]] | None = None,
    end_turns: int = 20,
) -> dict[str, Any]:
    """One eval row. ``mode`` "stochastic" / "greedy" is the gate mode of a new row; "legacy" is an archive row.

    ``captures`` are per episode (tower, building, hq); ``truncated`` draws
    ended at max_steps, the other draws at max_turns. A new row counts
    ``end_turns`` agent end_turn actions per episode (``action_counts``).
    """
    per = per_episode or PER_EPISODE
    outcomes = ["wins"] * wins + ["draws"] * draws + ["losses"] * losses
    rewards = [sum(per[o].values()) for o in outcomes]
    n = len(outcomes)
    comps = {c: sum(per[o][c] for o in outcomes) for c in COMPONENTS}
    row: dict[str, Any] = {
        **_counts(wins, draws, losses),
        "avg_reward": sum(rewards) / n,
        "std_reward": 0.0,
        "avg_length": 100.0,
        "std_length": 0.0,
        "avg_turns": 20.0,
        "std_turns": 0.0,
        "rewards": rewards,
        "lengths": [100] * n,
        "turns": [20] * n,
        "end_reasons": {
            "hq_capture": 0,
            "elimination": wins + losses,
            "max_turns_draw": draws - truncated,
            "max_steps_truncate": truncated,
        },
        "captures_by_type": {k: round(v * n) for k, v in zip(("tower", "building", "hq"), captures, strict=True)},
        "reward_components": comps,
        "timesteps": timesteps,
        "eval_seed": eval_seed,
    }
    if mode == "legacy":
        return row
    row.update(
        outcomes=outcomes,
        seats=[1] * n,
        by_seat={"1": _counts(wins, draws, losses)},
        opponent_captures={"neutral": opponent_captures[0] * n, "owned": opponent_captures[1] * n},
        reward_components_by_outcome={
            o: {"episodes": outcomes.count(o), **{c: per[o][c] * outcomes.count(o) for c in COMPONENTS}}
            for o in ("wins", "draws", "losses")
        },
        reward_components_abs={c: sum(abs(per[o][c]) for o in outcomes) for c in COMPONENTS},
        action_counts={"end_turn": end_turns * n},
        flat_truncated_rate=flat_truncated_rate,
        deterministic=mode == "greedy",
        attempt=attempt,
        gate_win_rate=wins / n,
        stage_steps=0,
        best_eligible=True,
    )
    if wall_time is not None:
        row["wall_time"] = wall_time
    if eval_seconds is not None:
        row["eval_seconds"] = eval_seconds
    if other is not None:
        o = _counts(*other)
        row["other_mode"] = {"deterministic": mode != "greedy", "gate_win_rate": o["win_rate"], **o, "avg_reward": 0.0}
        row["win_rate_stochastic" if mode == "greedy" else "win_rate_greedy"] = o["win_rate"]
    row["win_rate_greedy" if mode == "greedy" else "win_rate_stochastic"] = row["win_rate"]
    return row


def stage(
    name: str,
    rows: list[dict[str, Any]],
    *,
    promoted: bool | None,
    opponent: str = "random",
    opponent_kwargs: dict[str, Any] | None = None,
    map_file: str = MAP,
    promotion_win_rate: float = 0.7,
    patience: int = 2,
    max_timesteps: int = 1000,
    start: int | None = None,
    end: int | None = None,
    retries: int = 0,
    created_at: str | None = None,
    reward_config: dict[str, float] | None = None,
) -> dict[str, Any]:
    """A stage of a synthetic run (``promoted`` None: the stage the run was in when it stopped)."""
    return {
        "name": name,
        "rows": rows,
        "promoted": promoted,
        "opponent": opponent,
        "opponent_kwargs": opponent_kwargs,
        "map_file": map_file,
        "promotion_win_rate": promotion_win_rate,
        "patience": patience,
        "max_timesteps": max_timesteps,
        "start": start,
        "end": end,
        "retries": retries,
        "created_at": created_at,
        "reward_config": reward_config,
    }


def make_config(
    stages: list[dict[str, Any]],
    *,
    seed: int = 42,
    legacy: bool = False,
    reward_config: dict[str, float] | None = None,
    eval_freq: int = 100,
    n_eval_episodes: int = 10,
) -> dict[str, Any]:
    """A raw config for ``stages`` (a resolved_config.yaml, or with ``legacy`` an archived source YAML)."""
    cfg: dict[str, Any] = {
        "algorithm": "maskable_ppo",
        "total_timesteps": 1000,
        "seed": seed,
        "env": {
            "n_envs": 8,
            "max_steps": 3000 if legacy else 2000,
            "max_turns": 30,
            "max_flat_actions": 512,
            "max_actions_per_turn": None if legacy else 40,
            "action_space_type": "flat_discrete",
            "reward_config": dict(reward_config or {"win": 50.0, "draw": -10.0, "turn_penalty": 0.0 if legacy else -0.5}),
            "engine_overrides": None,
        },
        "ppo": {"learning_rate": 0.0003, "n_steps": 2048, "gamma": 0.99, "device": "auto"},
        "eval": {"eval_freq": eval_freq, "n_eval_episodes": n_eval_episodes, "seed_offset": 1_000_000},
        "logging": {"log_dir": "./logs"},
        "curriculum": {
            "stages": [
                {
                    k: v
                    for k, v in {
                        "name": s["name"],
                        "map_file": s["map_file"],
                        "opponent": s["opponent"],
                        "opponent_kwargs": s["opponent_kwargs"],
                        "promotion_win_rate": s["promotion_win_rate"],
                        "patience": s["patience"],
                        "max_timesteps": s["max_timesteps"],
                        "reward_config": s["reward_config"],
                    }.items()
                    if v is not None or not legacy
                }
                for s in stages
            ]
        },
    }
    if not legacy:
        cfg["eval"].update(
            eval_deterministic=False, eval_both_modes=True, n_eval_envs=1, eval_seats=[1], resample_eval_seeds=False
        )
        cfg["curriculum"].update(max_retries=1, promotion_criterion="point")
    else:
        cfg["eval"]["resample_eval_seeds"] = True
    return cfg


def _stage_config_json(s: dict[str, Any], cfg: dict[str, Any], *, legacy: bool, seed: int, eval_freq: int) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "stage_name": s["name"],
        "promotion_win_rate": s["promotion_win_rate"],
        "patience": s["patience"],
        "max_timesteps": s["max_timesteps"],
        "n_eval_episodes": cfg["eval"]["n_eval_episodes"],
        "eval_freq": eval_freq,
        "promoted": bool(s["promoted"]),
        "best_win_rate": max((r["win_rate"] for r in s["rows"]), default=None),
    }
    if not legacy:
        extra.update(
            retries_used=s["retries"],
            attempts=[],
            stage_start_timesteps=s["start"],
            stage_end_timesteps=s["end"],
            peak_win_rate=max((r["win_rate"] for r in s["rows"]), default=None),
            eval_deterministic=False,
        )
    reward = dict(cfg["env"]["reward_config"])
    reward.update(s.get("reward_config") or {})
    return {
        "run_type": "ppo_bootstrap",
        "map_file": s["map_file"],
        "opponent": s["opponent"],
        "seed": seed,
        "env_config": {
            "map_file": s["map_file"],
            "max_turns": cfg["env"]["max_turns"],
            "opponent_kwargs": s["opponent_kwargs"],
            "reward_config": reward,
        },
        "extra": extra,
        "meta": {"created_at": s["created_at"], "git": LEGACY_GIT if legacy else GIT},
    }


def write_run(
    root: Path,
    name: str,
    stages: list[dict[str, Any]],
    *,
    seed: int = 42,
    status: str = "completed",
    legacy: bool = False,
    config: dict[str, Any] | None = None,
    resume_count: int = 0,
    metadata_write_failures: int = 0,
    source_yaml: str = "v52a_maxturn_scaled_draw.yaml",
) -> Path:
    """Write a run directory; ``status`` completed / stalled / interrupted (legacy runs are always aborted)."""
    run = Path(root) / name
    run.mkdir(parents=True, exist_ok=True)
    cfg = config or make_config(stages, seed=seed, legacy=legacy)
    eval_freq = int(cfg["eval"]["eval_freq"])
    if legacy:
        (run / source_yaml).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    else:
        (run / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (run / "bootstrap.yaml").write_text("# source copy\n", encoding="utf-8")
    finished = [s for s in stages if s["promoted"] is not None]
    for s in stages:
        stage_dir = run / s["name"]
        stage_dir.mkdir(exist_ok=True)
        if s["promoted"] is None:
            with (stage_dir / "eval_results.jsonl").open("w", encoding="utf-8") as fh:
                for r in s["rows"]:
                    fh.write(json.dumps(r) + "\n")
            continue
        (stage_dir / "config.json").write_text(
            json.dumps(_stage_config_json(s, cfg, legacy=legacy, seed=seed, eval_freq=eval_freq), indent=2), encoding="utf-8"
        )
        (stage_dir / "eval_results.json").write_text(json.dumps(s["rows"], indent=2), encoding="utf-8")
    columns = LEGACY_CSV_COLUMNS if legacy else (*LEGACY_CSV_COLUMNS, "deterministic", "attempt")
    with (run / "bootstrap_results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore", lineterminator="\r\n")
        writer.writeheader()
        for s in finished:
            for r in s["rows"]:
                writer.writerow({**r, "stage": s["name"], "map_file": s["map_file"], "opponent": s["opponent"]})
    if legacy:
        return run
    manifest: dict[str, Any] = {
        "version": 1,
        "stages": [s["name"] for s in stages],
        "completed": [{"stage": s["name"], "promoted": s["promoted"], "summary": {}} for s in finished],
        "current": None,
        "resume_count": resume_count,
        "metadata_write_failures": metadata_write_failures,
    }
    in_progress = next((s for s in stages if s["promoted"] is None), None)
    if status == "interrupted" and in_progress is not None:
        manifest["current"] = {
            "stage": in_progress["name"],
            "attempt": 0,
            "latest_timesteps": in_progress["rows"][-1]["timesteps"] if in_progress["rows"] else None,
        }
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    common = {"resume_count": resume_count, "metadata_write_failures": metadata_write_failures}
    if status == "completed":
        (run / "run_status.json").write_text(json.dumps({"status": "completed_curriculum", **common}), encoding="utf-8")
        (run / "final_model.zip").write_bytes(b"zip")
    elif status == "stalled":
        stalled = next(s for s in stages if s["promoted"] is False)
        payload = {
            "status": "curriculum_stalled",
            "stalled_stage": stalled["name"],
            "trained_timesteps": stalled["end"] - stalled["start"],
            "retries_used": stalled["retries"],
            **common,
        }
        (run / "run_status.json").write_text(json.dumps(payload), encoding="utf-8")
    return run
