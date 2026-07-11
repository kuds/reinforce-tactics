#!/usr/bin/env python3
"""A/B harness: legacy 6-head worker vs. AlphaStar-style autoregressive worker.

Trains two ``FeudalRLAgent`` runs from the same seed and identical
hyperparameters, with the only difference being ``--autoregressive-worker``.
Both runs use the same eval opponent, eval cadence, and total budget.
Prints a side-by-side comparison of evaluation win rate / mean reward /
goal-reached rate per checkpoint.

This is the harness for ROADMAP Phase 3.7 — pair its output with
``docs/feudal_rl_review.md`` to settle whether AR should be the default.

Example
-------
    python scripts/ab_feudal_ar.py \\
        --map maps/1v1/beginner.csv \\
        --total-timesteps 100000 --n-steps 1024 --eval-freq 10000 \\
        --eval-opponent random --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from reinforcetactics.rl.feudal_rl import FeudalRLAgent
from reinforcetactics.rl.gym_env import StrategyGameEnv


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env(map_file: str, opponent: str, max_steps: int, max_turns: int, enabled_units: list[str], seed: int):
    env = StrategyGameEnv(
        map_file=map_file,
        opponent=opponent,
        render_mode=None,
        max_steps=max_steps,
        max_turns=max_turns,
        enabled_units=enabled_units,
    )
    env.reset(seed=seed)
    return env


def _train_one(args, *, autoregressive: bool, eval_steps: list[int]) -> list[dict]:
    """Train one variant and return the eval-checkpoint history.

    Both variants run with identical seeds + hyperparameters so any
    divergence is attributable to the worker head choice.
    """
    label = "AR" if autoregressive else "6head"
    print(f"\n=== Training {label} worker ===")

    _seed_everything(args.seed)
    env = _make_env(args.map, args.opponent, args.max_steps, args.max_turns, args.enabled_units, args.seed)
    eval_env = _make_env(args.map, args.eval_opponent, args.max_steps, args.max_turns, args.enabled_units, args.seed + 10_000)

    agent = FeudalRLAgent(
        observation_space=env.observation_space,
        grid_width=env.grid_width,
        grid_height=env.grid_height,
        agent_player=getattr(env, "agent_player", 1),
        device=args.device,
        autoregressive_worker=autoregressive,
    )
    agent.manager_horizon = args.manager_horizon
    agent.setup_training(learning_rate=args.learning_rate)
    agent.reset_goal()

    history: list[dict] = []
    total_timesteps = 0
    last_eval = 0
    n_updates = args.total_timesteps // args.n_steps
    start = time.time()

    for update_idx in range(n_updates):
        buf = agent.collect_rollout(
            env,
            n_steps=args.n_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            worker_reward_alpha=args.worker_reward_alpha,
            reward_scale=args.reward_scale,
        )
        agent.update(
            buf,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
        )
        total_timesteps += args.n_steps

        if total_timesteps - last_eval >= args.eval_freq or update_idx == n_updates - 1:
            last_eval = total_timesteps
            res = agent.evaluate(eval_env, n_episodes=args.n_eval_episodes)
            res["timesteps"] = total_timesteps
            res["goal_reached_rate"] = float(buf.w_reached_goal.mean()) if hasattr(buf, "w_reached_goal") else 0.0
            res["worker_intrinsic_mean"] = float(buf.w_intrinsic.mean()) if hasattr(buf, "w_intrinsic") else 0.0
            history.append(res)
            print(
                f"  [{label}] step {total_timesteps:>7,}  win_rate={res['win_rate']:.2f} "
                f"reward={res['mean_reward']:+.1f} goal_reached={res['goal_reached_rate']:.2f}"
            )
            eval_steps.append(total_timesteps)

    elapsed = time.time() - start
    print(f"=== {label} done in {elapsed:.1f}s ({total_timesteps / elapsed:.0f} steps/s) ===")
    return history


def _print_comparison(legacy: list[dict], ar: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("A/B comparison: legacy 6-head vs autoregressive worker")
    print("=" * 78)
    print(f"{'Step':>10}  {'6head WR':>10}  {'AR WR':>10}  {'WR Δ':>8}  {'6head R':>12}  {'AR R':>12}  {'Goal% Δ':>10}")
    print("-" * 78)
    for a, b in zip(legacy, ar):
        if a["timesteps"] != b["timesteps"]:
            continue  # Skip if eval cadence drifted
        wr_delta = b["win_rate"] - a["win_rate"]
        goal_delta = b["goal_reached_rate"] - a["goal_reached_rate"]
        print(
            f"{a['timesteps']:>10,}  {a['win_rate']:>10.2f}  {b['win_rate']:>10.2f}  "
            f"{wr_delta:>+8.2f}  {a['mean_reward']:>+12.1f}  {b['mean_reward']:>+12.1f}  {goal_delta:>+10.2f}"
        )
    print("=" * 78)
    if legacy and ar:
        final_wr_delta = ar[-1]["win_rate"] - legacy[-1]["win_rate"]
        verdict = "AR wins" if final_wr_delta > 0.05 else ("6head wins" if final_wr_delta < -0.05 else "tie")
        print(f"Verdict (final win-rate Δ ≥ 0.05): {verdict}  (Δ={final_wr_delta:+.2f})")


def main():
    parser = argparse.ArgumentParser(description="A/B compare legacy vs AR worker for feudal RL")
    parser.add_argument("--map", type=str, default="maps/1v1/beginner.csv")
    parser.add_argument("--opponent", type=str, default="random")
    parser.add_argument("--eval-opponent", type=str, default="random")
    parser.add_argument("--enabled-units", nargs="+", default=["W", "M", "A"])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--manager-horizon", type=int, default=10)
    parser.add_argument("--worker-reward-alpha", type=float, default=0.5)
    parser.add_argument("--reward-scale", type=float, default=0.001)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON dump of both histories")
    args = parser.parse_args()

    eval_steps_legacy: list[int] = []
    eval_steps_ar: list[int] = []
    legacy_hist = _train_one(args, autoregressive=False, eval_steps=eval_steps_legacy)
    ar_hist = _train_one(args, autoregressive=True, eval_steps=eval_steps_ar)

    _print_comparison(legacy_hist, ar_hist)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"legacy": legacy_hist, "ar": ar_hist, "args": vars(args)}, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
