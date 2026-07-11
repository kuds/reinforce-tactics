#!/usr/bin/env python3
"""Headless CLI for the curriculum-bootstrap pipeline (the ``ppo_bootstrap`` notebook).

This mirrors ``notebooks/ppo_bootstrap.ipynb`` as a single command so the full
run — optional BC warm-start, the curriculum train (``run_curriculum``), all the
diagnostic charts, the final sanity eval, and per-stage replay videos — can run
unattended in Docker / on Vertex AI instead of in Colab.

Everything is written under ``--output-dir`` (default
``benchmarks/bootstrap/<timestamp>``):

    <output-dir>/
      bootstrap.yaml            # snapshot of the config that ran
      <stage>/                  # per-stage tensorboard, checkpoints, eval json
      checkpoints/              # flattened <stage>.zip / <stage>_best.zip
      charts/                   # every PNG (curriculum + per-stage + bc)
      videos/                   # <stage>.mp4 replay per stage
      bootstrap_results.csv
      final_model.zip

When a GCS destination is configured (``--gcs-output gs://...`` or, on Vertex,
the ``GCS_OUTPUT_URI`` / ``AIP_MODEL_DIR`` env), the whole output directory is
uploaded there at the end — including on a stall — so the charts and videos
survive the ephemeral job.

Examples:
    python3 scripts/train/train_bootstrap.py --config configs/ppo/bootstrap.yaml --device cuda
    python3 scripts/train/train_bootstrap.py --config configs/ppo/bootstrap.yaml \\
        --build-bc --gcs-output gs://my-bucket/bootstrap
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Force headless rendering BEFORE anything imports pygame or matplotlib. SDL and
# matplotlib both pick their backend at import time, so these must be set first.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

# Make the package importable when run as a script from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the curriculum-bootstrap pipeline headlessly (CLI mirror of ppo_bootstrap.ipynb).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default="configs/ppo/bootstrap.yaml", help="Path to the bootstrap YAML config")
    p.add_argument("--output-dir", type=str, default=None, help="Output dir (default: benchmarks/bootstrap/<timestamp>)")
    p.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, or auto")
    p.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="Override a config value (dotted key), e.g. --set env.enabled_units='[W,M,C,A,K]'. Repeatable.",
    )

    # Behaviour-cloning warm-start (notebook section 3c-3e). Off by default;
    # requires an action_space_type=multi_discrete config and BC scenarios.
    p.add_argument("--build-bc", action="store_true", help="Build a BC warm-start before the curriculum")
    p.add_argument(
        "--bc-scenarios", type=str, default="configs/imitation/bc_beginner_warmstart.yaml", help="BC scenarios YAML"
    )
    p.add_argument("--bc-epochs", type=int, default=60, help="BC training epochs")
    p.add_argument("--bc-batch-size", type=int, default=128, help="BC batch size")
    p.add_argument("--bc-learning-rate", type=float, default=3e-4, help="BC learning rate")
    p.add_argument("--bc-seed", type=int, default=42, help="BC seed")
    p.add_argument("--bc-end-turn-weight", type=float, default=30.0, help="BC end_turn demo loss weight")

    # Post-training artifacts.
    p.add_argument("--skip-plots", action="store_true", help="Skip chart generation")
    p.add_argument("--skip-videos", action="store_true", help="Skip per-stage replay videos")
    p.add_argument("--sanity-episodes", type=int, default=100, help="Final sanity-eval episodes (0 to skip)")

    # GCS persistence.
    p.add_argument(
        "--gcs-output",
        type=str,
        default=None,
        help="gs:// base to upload the output dir to (default: GCS_OUTPUT_URI / AIP_MODEL_DIR env)",
    )
    p.add_argument("--no-gcs", action="store_true", help="Never upload to GCS even if an env URI is set")
    return p


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _apply_set_overrides(cfg, set_items):
    """Apply ``--set key=value`` overrides (values parsed as YAML scalars/lists)."""
    if not set_items:
        return cfg
    import yaml

    from reinforcetactics.rl.config import apply_overrides

    overrides = {}
    for item in set_items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item!r}")
        overrides[key.strip()] = yaml.safe_load(raw)
    return apply_overrides(cfg, overrides)


def _print_stage_table(cfg) -> None:
    stages = cfg.curriculum.stages
    print(f"Config stages: {len(stages)} | enabled_units={cfg.env.enabled_units} | n_envs={cfg.env.n_envs}")
    for s in stages:
        print(f"  {s.name:<30s} opp={s.opponent:<10s} WR>={s.promotion_win_rate:>4.0%} budget={s.max_timesteps:>12,}")
    print(f"  total budget (worst case): {sum(s.max_timesteps for s in stages):,} env steps")


# ---------------------------------------------------------------------------
# Behaviour cloning warm-start (optional; notebook sections 3c-3e)
# ---------------------------------------------------------------------------


def _bc_build(cfg, output_dir: Path, args):
    """Build a BC warm-start checkpoint and point cfg.warm_start_path at it."""
    from reinforcetactics.rl import load_scenarios_from_yaml, make_maskable_env, make_warm_started_model
    from reinforcetactics.rl.bootstrap import _resolve_policy_kwargs

    if cfg.env.action_space_type != "multi_discrete":
        raise SystemExit(
            f"--build-bc requires action_space_type=multi_discrete; config has "
            f"'{cfg.env.action_space_type}'. Use e.g. configs/ppo/bootstrap_sweep/v33_production_bc_warmstart.yaml."
        )

    scenarios = load_scenarios_from_yaml(args.bc_scenarios)
    print(f"BC: {len(scenarios)} scenarios from {args.bc_scenarios}; training {args.bc_epochs} epochs")

    first_stage = cfg.curriculum.stages[0]
    template_env = make_maskable_env(
        map_file=first_stage.map_file,
        opponent="medium",  # arbitrary; only used for obs/action shape resolution
        action_space_type=cfg.env.action_space_type,
        enabled_units=cfg.env.enabled_units,
        max_turns=first_stage.resolve_max_turns(cfg.env),
    )

    # Forward the curriculum's policy_kwargs so the BC checkpoint state dict
    # matches what bootstrap.py loads via set_parameters(exact_match=True).
    bc_policy_kwargs = _resolve_policy_kwargs(cfg.ppo.policy_kwargs)
    ppo_kwargs = {"verbose": 0, **({"policy_kwargs": bc_policy_kwargs} if bc_policy_kwargs else {})}

    bc_model, bc_dataset, bc_stats = make_warm_started_model(
        env=template_env,
        n_epochs=args.bc_epochs,
        batch_size=args.bc_batch_size,
        learning_rate=args.bc_learning_rate,
        seed=args.bc_seed,
        ppo_kwargs=ppo_kwargs,
        scenarios=scenarios,
        end_turn_weight=args.bc_end_turn_weight,
    )

    checkpoint = output_dir / "bc_warmstart.zip"
    bc_model.save(str(checkpoint))
    cfg.warm_start_path = str(checkpoint)
    if bc_stats:
        f = bc_stats[-1]
        print(
            f"BC: done — demos={len(bc_dataset):,} final_loss={f.loss:.4f} "
            f"full_action_acc={f.accuracy_full:.3f}; warm_start_path -> {checkpoint}"
        )
    return bc_model, bc_dataset, bc_stats


def _bc_diagnostics(bc_dataset, bc_stats, charts_dir: Path, plt) -> None:
    """BC training-curve + demo-outcome charts (notebook section 3d)."""
    from reinforcetactics.rl.viz import plot_bc_demo_outcomes, plot_bc_training_curves

    bc_charts = charts_dir / "bc"
    bc_charts.mkdir(parents=True, exist_ok=True)
    if bc_stats:
        _close(plot_bc_training_curves(bc_stats, charts_dir=bc_charts), plt)
    if getattr(bc_dataset, "scenario_stats", None):
        _close(plot_bc_demo_outcomes(bc_dataset.scenario_stats, charts_dir=bc_charts), plt)
    print(f"BC: diagnostics charts -> {bc_charts}")


def _bc_sanity_eval(cfg, bc_model) -> None:
    """Greedy + sampling eval of the warm-start vs the bot ladder (notebook section 3e)."""
    from reinforcetactics.rl import evaluate_bc_against_bot_ladder

    for mode, det in (("greedy", True), ("sampling", False)):
        metrics = evaluate_bc_against_bot_ladder(bc_model, cfg, n_episodes=30, deterministic=det)
        for opp, m in metrics.items():
            print(f"  BC ({mode}) vs {opp:10s} WR={m['win_rate']:5.1%} reward={m['avg_reward']:+8.1f}")


# ---------------------------------------------------------------------------
# Curriculum post-processing (notebook sections 4b-9)
# ---------------------------------------------------------------------------


def _snapshot_stage_checkpoints(result, cfg, output_dir: Path):
    """Flatten per-stage checkpoints and build the stage_checkpoints map (section 4b)."""
    import shutil

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    stage_checkpoints = {}
    for h in result["history"]:
        stage_name = h["stage"]
        stage_dir = output_dir / stage_name
        flat_final = checkpoints_dir / f"{stage_name}.zip"
        flat_best = checkpoints_dir / f"{stage_name}_best.zip"
        if (stage_dir / "stage_final.zip").exists():
            shutil.copy2(stage_dir / "stage_final.zip", flat_final)
        if (stage_dir / "best_model.zip").exists():
            shutil.copy2(stage_dir / "best_model.zip", flat_best)
        stage_checkpoints[stage_name] = {
            "map_file": next(s.map_file for s in cfg.curriculum.stages if s.name == stage_name),
            "opponent": next(s.opponent for s in cfg.curriculum.stages if s.name == stage_name),
            "stage_final": str(flat_final) if flat_final.exists() else None,
            "best_model": str(flat_best) if flat_best.exists() else None,
            "promoted": h["promoted"],
            "best_win_rate": h["best_win_rate"],
        }
    print(f"Snapshotted {len(stage_checkpoints)} stage checkpoints -> {checkpoints_dir}")
    return stage_checkpoints


def _close(fig, plt) -> None:
    """Close a figure (or dict/list of figures) to free memory in long runs."""
    if fig is None:
        return
    figs = fig.values() if isinstance(fig, dict) else (fig if isinstance(fig, (list, tuple)) else [fig])
    for f in figs:
        if f is not None:
            plt.close(f)


def _curriculum_plots(result, cfg, charts_dir: Path, plt) -> None:
    """All curriculum + per-stage diagnostic charts (notebook section 5)."""
    from reinforcetactics.rl import viz

    history = result.get("history") or []
    if not history:
        print("Plots: skipped — no eval history.")
        return

    _close(viz.plot_curriculum_summary(history, cfg.curriculum.stages, charts_dir=charts_dir), plt)
    _close(viz.plot_curriculum_metrics(history, charts_dir=charts_dir), plt)
    _close(
        viz.plot_curriculum_composition_summary(history, charts_dir=charts_dir / "curriculum", final_evals_to_average=3),
        plt,
    )

    for h in history:
        if not h["results"]:
            continue
        stage_charts = charts_dir / h["stage"]
        stage_charts.mkdir(parents=True, exist_ok=True)
        _close(viz.plot_stage_diagnostics(h["results"], charts_dir=stage_charts, title_suffix=h["stage"]), plt)

    # Eval curves across the whole curriculum on the cumulative timestep axis.
    metrics_callback = result.get("metrics_callback")
    train_records = metrics_callback.records if metrics_callback is not None else None
    all_results = [r for h in history for r in h["results"]]
    stage_boundaries = [h["results"][-1]["timesteps"] for h in history[:-1] if h["results"]]
    if all_results:
        _close(
            viz.plot_eval_curves(
                all_results,
                train_records=train_records,
                opponent_label="curriculum",
                charts_dir=charts_dir,
                stage_boundaries=stage_boundaries,
            ),
            plt,
        )
    print(f"Plots: charts written under {charts_dir}")


def _final_sanity_eval(result, cfg, episodes: int) -> None:
    """Evaluate the final checkpoint on the last trained stage (notebook section 6)."""
    if episodes <= 0 or not result.get("final_model_path") or not result.get("history"):
        return
    from sb3_contrib import MaskablePPO

    from reinforcetactics.rl import make_maskable_env
    from reinforcetactics.rl.evaluation import evaluate_model

    last_stage_name = result["history"][-1]["stage"]
    stage = next(s for s in cfg.curriculum.stages if s.name == last_stage_name)
    env = make_maskable_env(
        map_file=stage.map_file,
        opponent=stage.opponent,
        max_steps=stage.resolve_max_steps(cfg.env),
        max_turns=stage.resolve_max_turns(cfg.env),
        reward_config=stage.resolve_reward_config(cfg.env),
        enabled_units=cfg.env.enabled_units,
        action_space_type=cfg.env.action_space_type,
        seed=cfg.seed + 9999,
        opponent_kwargs=stage.opponent_kwargs,
        pad_to_size=cfg.env.pad_to_size,
    )
    model = MaskablePPO.load(result["final_model_path"])
    metrics = evaluate_model(model, env, n_episodes=episodes, seed=cfg.seed + 9999)
    env.close()
    print(
        f"Sanity eval ({stage.name}, n={episodes}): WR={metrics['win_rate']:.1%} "
        f"reward={metrics['avg_reward']:+.1f} W/L/D={metrics['wins']}/{metrics['losses']}/{metrics['draws']}"
    )


def _record_videos(result, cfg, output_dir: Path, stage_checkpoints):
    """Record one replay .mp4 per stage (notebook section 8)."""
    from reinforcetactics.rl import record_curriculum_replays

    if not stage_checkpoints:
        return []
    videos_dir = output_dir / "videos"
    summary = record_curriculum_replays(cfg, stage_checkpoints, videos_dir)
    print(f"Saved {len(summary)} replay video(s) -> {videos_dir}")
    return summary


def _individual_game_stats(video_summary, charts_dir: Path, plt) -> None:
    """Per-stage individual-game stat panels from the recorded replays (section 9)."""
    if not video_summary:
        return
    from reinforcetactics.rl.viz import plot_individual_game_stats

    stats_dir = charts_dir / "individual_game_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for entry in video_summary:
        if entry.get("step_stats"):
            _close(plot_individual_game_stats(entry, charts_dir=stats_dir, title_suffix=entry["stage"]), plt)
    print(f"Individual-game stats -> {stats_dir}")


def _maybe_upload(output_dir: Path, args) -> None:
    """Upload the run directory to GCS when configured (final, runs even on stall)."""
    from reinforcetactics.cloud.storage import resolve_output_base, upload_tree

    base = args.gcs_output or (None if args.no_gcs else resolve_output_base())
    if not base:
        return
    dest = f"{base.rstrip('/')}/{output_dir.name}"
    print(f"Uploading {output_dir} -> {dest} ...")
    count = upload_tree(str(output_dir), dest)
    print(f"Uploaded {count} file(s) to {dest}" if count else f"No files uploaded to {dest}")


def main() -> int:
    args = build_parser().parse_args()

    # Heavy imports are deferred until after arg parsing so --help works without
    # torch / sb3 / the rest of the package installed.
    import matplotlib.pyplot as plt  # MPLBACKEND=Agg set above

    from reinforcetactics.rl.bootstrap import CurriculumStalled, run_curriculum
    from reinforcetactics.rl.config import load_config

    config_path = Path(args.config)
    cfg = load_config(config_path)
    cfg = _apply_set_overrides(cfg, args.set)
    cfg.ppo.device = _resolve_device(args.device)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("benchmarks") / "bootstrap" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    shutil.copy2(config_path, output_dir / config_path.name)

    print(f"\n🚀 Bootstrap run {run_id} on {cfg.ppo.device}")
    print(f"Output dir: {output_dir}")
    _print_stage_table(cfg)

    try:
        if args.build_bc:
            bc_model, bc_dataset, bc_stats = _bc_build(cfg, output_dir, args)
            if not args.skip_plots:
                _bc_diagnostics(bc_dataset, bc_stats, charts_dir, plt)
            _bc_sanity_eval(cfg, bc_model)

        try:
            result = run_curriculum(cfg, output_dir=output_dir)
        except CurriculumStalled as exc:
            print(f"\n⚠️  STALLED: {exc}")
            result = exc.partial_result()

        if result is not None:
            stage_checkpoints = _snapshot_stage_checkpoints(result, cfg, output_dir)
            if not args.skip_plots:
                _curriculum_plots(result, cfg, charts_dir, plt)
            _final_sanity_eval(result, cfg, args.sanity_episodes)
            video_summary = [] if args.skip_videos else _record_videos(result, cfg, output_dir, stage_checkpoints)
            if not args.skip_plots:
                _individual_game_stats(video_summary, charts_dir, plt)
            print(f"\n✅ Done. Final model: {result.get('final_model_path')}")
    finally:
        _maybe_upload(output_dir, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
