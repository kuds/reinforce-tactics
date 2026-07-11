#!/usr/bin/env python3
"""Build a BC warm-start checkpoint for the curriculum bootstrap.

Pipeline:
  1. Collect demonstrations from scripted bot rollouts on beginner.csv with
     the production [W,M,C,A,K] roster (see configs/imitation/
     bc_beginner_warmstart.yaml for the scenario mix).
  2. Behavior-clone a fresh MaskablePPO policy on those demonstrations
     (masked cross-entropy on (obs, action, per-dim mask) tuples). Value
     head is left untouched -- PPO will fit it during curriculum.
  3. Save the resulting MaskablePPO checkpoint to disk.

The output checkpoint is what ``warm_start_path`` in
``v33_production_bc_warmstart.yaml`` loads at curriculum start. Exact
space match is required (curriculum env must use the same observation /
action shapes as the template env this script builds), which the v33
config is set up to provide:
  * action_space_type: multi_discrete (BC infra requirement)
  * enabled_units: [W, M, C, A, K]
  * map_file: maps/1v1/beginner.csv (v33's first stage map; size 6x6)
  * NO pad_to_size (v33 is a beginner-only truncated curriculum so all
    stages share 6x6 dims; multi-size BC needs pad_to_size threaded
    through collect_demonstrations -- deferred until BC is validated)

Usage:
    python scripts/build_bc_warmstart.py
    python scripts/build_bc_warmstart.py --epochs 10 --output models/bc_v33.zip
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenarios",
        default="configs/imitation/bc_beginner_warmstart.yaml",
        help="YAML describing the demonstration scenario mix.",
    )
    parser.add_argument(
        "--curriculum-config",
        default="configs/ppo/bootstrap_sweep/v33_production_bc_warmstart.yaml",
        help=(
            "Curriculum YAML the BC checkpoint will be loaded into. The "
            "script reads ppo.policy_kwargs from this config so the BC "
            "model is built with the same features extractor and net_arch "
            "as the downstream curriculum -- otherwise set_parameters "
            "(exact_match=True) fails with a state_dict mismatch."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Where to write the BC checkpoint. Defaults to "
            "benchmarks/bc/<timestamp>/bc_warmstart.zip -- the timestamp lets "
            "multiple BC builds coexist for comparison."
        ),
    )
    parser.add_argument("--epochs", type=int, default=8, help="BC training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="BC minibatch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="BC Adam LR.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for demo collection.")
    parser.add_argument(
        "--map-file",
        default="maps/1v1/beginner.csv",
        help="Map for the template env (must match scenarios + v33's stages).",
    )
    parser.add_argument(
        "--enabled-units",
        nargs="+",
        default=["W", "M", "C", "A", "K"],
        help="Roster for the template env (must match scenarios + v33).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=75,
        help="max_turns on the template env (matches v33's beginner stages).",
    )
    parser.add_argument(
        "--end-turn-weight",
        type=float,
        default=None,
        help=(
            "Per-sample loss weight for end_turn demonstrations during BC. "
            "Default (unset) auto-balances to n_non_end/n_end so the "
            "aggregate end_turn gradient matches all other action_types "
            "combined -- corrects the ~10:1 demo imbalance that produces "
            "the never-end-turn attractor at eval time. Pass 1.0 to "
            "disable; pass a larger float (e.g. 20.0) to over-emphasise."
        ),
    )
    return parser.parse_args()


def _resolve_output_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path("benchmarks") / "bc" / ts / "bc_warmstart.zip"


def main() -> int:
    args = parse_args()

    try:
        from sb3_contrib import MaskablePPO  # noqa: F401  (import-time check)
    except ImportError:
        print("Error: sb3-contrib is required. Install with: pip install sb3-contrib", file=sys.stderr)
        return 1

    from reinforcetactics.rl import (
        load_scenarios_from_yaml,
        make_maskable_env,
        make_warm_started_model,
    )
    from reinforcetactics.rl.bootstrap import _resolve_policy_kwargs
    from reinforcetactics.rl.config import load_config

    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios_path = Path(args.scenarios)
    if not scenarios_path.is_file():
        print(f"Error: scenarios file not found: {scenarios_path}", file=sys.stderr)
        return 1
    scenarios = load_scenarios_from_yaml(str(scenarios_path))

    # Pull policy_kwargs from the curriculum config the BC checkpoint will
    # feed into. The downstream load uses set_parameters(exact_match=True),
    # so the BC MaskablePPO must be built with the same features extractor
    # class and net_arch as the curriculum's model. Without this the BC ckpt
    # has SB3 defaults (CombinedExtractor + [64, 64] MLP) and the load fails
    # with "Missing keys ... features_extractor.cnn.*" and a 64-vs-256
    # size mismatch on the MLP heads.
    curriculum_config_path = Path(args.curriculum_config)
    if not curriculum_config_path.is_file():
        print(f"Error: curriculum config not found: {curriculum_config_path}", file=sys.stderr)
        return 1
    curriculum_cfg = load_config(str(curriculum_config_path))
    policy_kwargs = _resolve_policy_kwargs(curriculum_cfg.ppo.policy_kwargs)

    # Template env: the policy network shapes are built around this env's
    # spaces, so it must match the curriculum env v33 will use. Action space
    # is multi_discrete (BC infra requirement); enabled_units / map / max_turns
    # are set explicitly to mirror v33's beginner stages.
    template_env = make_maskable_env(
        map_file=args.map_file,
        opponent="medium",  # arbitrary -- template env never trains, only provides shapes
        action_space_type="multi_discrete",
        enabled_units=args.enabled_units,
        max_turns=args.max_turns,
    )

    print("=" * 64)
    print("BC WARM-START BUILD")
    print("=" * 64)
    print(f"  scenarios: {scenarios_path} ({len(scenarios)} entries)")
    total_eps = sum(int(s.n_episodes * s.weight) for s in scenarios)
    print(f"  total weighted episodes: ~{total_eps}")
    print(
        f"  template env: map={args.map_file} units={args.enabled_units} "
        f"max_turns={args.max_turns} action_space=multi_discrete"
    )
    print(f"  curriculum config (policy_kwargs source): {curriculum_config_path}")
    if policy_kwargs:
        fe_class = policy_kwargs.get("features_extractor_class")
        fe_name = getattr(fe_class, "__name__", str(fe_class)) if fe_class else "(SB3 default)"
        print(f"    features_extractor: {fe_name}")
        print(f"    net_arch: {policy_kwargs.get('net_arch')}")
    print(f"  BC: {args.epochs} epochs, batch={args.batch_size}, lr={args.learning_rate}")
    if args.end_turn_weight is None:
        print("  end_turn_weight: auto (n_non_end / n_end)")
    else:
        print(f"  end_turn_weight: {args.end_turn_weight}")
    print(f"  output: {output_path}")
    print("=" * 64)

    ppo_kwargs: dict[str, Any] = {"verbose": 0}
    if policy_kwargs:
        ppo_kwargs["policy_kwargs"] = policy_kwargs

    t0 = time.time()
    model, dataset, bc_stats = make_warm_started_model(
        env=template_env,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        ppo_kwargs=ppo_kwargs,
        scenarios=scenarios,
        end_turn_weight=args.end_turn_weight,
    )
    elapsed = time.time() - t0

    print("\n--- BC summary ---")
    print(f"  demonstrations collected: {len(dataset):,}")
    print(f"  wall time: {elapsed:.1f}s")
    if bc_stats:
        final = bc_stats[-1]
        print(f"  final loss: {final.loss:.4f}")
        print(f"  action_type accuracy: {final.accuracy_action_type:.3f}")
        print(f"  full-action accuracy: {final.accuracy_full:.3f}")
    print()

    model.save(str(output_path))
    print(f"BC checkpoint written: {output_path}")
    print()
    print("Next step: set this path in v33's warm_start_path and run the")
    print("production curriculum:")
    print(f"  warm_start_path: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
