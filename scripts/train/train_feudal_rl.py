"""
Training script for Feudal RL agent on GCP.
Supports distributed training with multiple seeds.

Supports both regular PPO and MaskablePPO (sb3-contrib) for action masking.
Action masking significantly improves training efficiency by preventing
the agent from wasting samples on invalid actions.

Usage:
    # Train with action masking (recommended)
    python train_feudal_rl.py --mode flat --use-action-masking

    # Train without action masking (baseline)
    python train_feudal_rl.py --mode flat
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Local imports
from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.rl.masking import ActionMaskedEnv, make_maskable_env, make_maskable_vec_env


def make_env(rank: int, seed: int = 0, opponent: str = "bot", use_masking: bool = False):
    """
    Utility function for multiprocessed env.

    Args:
        rank: Index of the subprocess
        seed: Random seed
        opponent: Opponent type
        use_masking: Whether to wrap env for action masking
    """

    def _init():
        env = StrategyGameEnv(
            map_file=None,  # Random maps
            opponent=opponent,
            render_mode=None,
            max_steps=500,
        )
        env.reset(seed=seed + rank)
        if use_masking:
            env = ActionMaskedEnv(env)
        return env

    set_random_seed(seed)
    return _init


def train_flat_baseline(args):
    """Train flat PPO baseline for comparison."""
    use_masking = getattr(args, "use_action_masking", False)

    print("\n" + "=" * 60)
    if use_masking:
        print("Training Flat MaskablePPO (with Action Masking)")
    else:
        print("Training Flat PPO Baseline")
    print("=" * 60 + "\n")

    # Create output directories
    model_name = "maskable_ppo" if use_masking else "flat_ppo"
    log_dir = Path(args.log_dir) / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Create vectorized environments
    if use_masking:
        # Use masking-compatible vectorized environments
        env = make_maskable_vec_env(
            n_envs=args.n_envs, opponent=args.opponent, seed=args.seed, use_subprocess=(args.n_envs > 1)
        )
        # Create eval environment with masking
        eval_env = make_maskable_env(opponent=args.opponent, render_mode=None)
    else:
        # Standard environments without masking
        if args.n_envs > 1:
            env = SubprocVecEnv([make_env(i, args.seed, args.opponent, use_masking=False) for i in range(args.n_envs)])
        else:
            env = DummyVecEnv([make_env(0, args.seed, args.opponent, use_masking=False)])
        eval_env = StrategyGameEnv(opponent=args.opponent, render_mode=None)

    # Create model - use MaskablePPO if action masking is enabled
    if use_masking:
        try:
            from sb3_contrib import MaskablePPO

            print("Using MaskablePPO from sb3-contrib")
        except ImportError:
            raise ImportError("sb3-contrib is required for action masking. Install with: pip install sb3-contrib")

        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard"),
            device=args.device,
        )
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard"),
            device=args.device,
        )

    # Callbacks. With action masking, use MaskableEvalCallback so masks are
    # applied during evaluation — plain EvalCallback would let the policy
    # pick invalid actions at eval time.
    if use_masking:
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(log_dir / "best_model"),
            log_path=str(log_dir / "eval"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            use_masking=True,
        )
    else:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(log_dir / "best_model"),
            log_path=str(log_dir / "eval"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq, save_path=str(checkpoint_dir), name_prefix="flat_ppo"
    )

    # Train
    print(f"Training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, callback=[eval_callback, checkpoint_callback], progress_bar=True)

    # Save final model
    final_path = log_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\n✅ Training complete! Model saved to {final_path}")

    # Save training config
    config = vars(args)
    config_path = log_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return log_dir


def _make_training_state(
    total_timesteps,
    best_eval_reward,
    best_eval_win_rate,
    last_eval_step,
    last_ckpt_step,
    last_snapshot_step=0,
):
    """Pack the resumeable counters/best-metric state into a checkpoint dict.

    Including ``last_snapshot_step`` lets self-play resume cleanly: the
    snapshot cadence picks up where it left off rather than firing
    immediately on the first post-resume update.
    """
    return {
        "total_timesteps": int(total_timesteps),
        "best_eval_reward": float(best_eval_reward),
        "best_eval_win_rate": float(best_eval_win_rate),
        "last_eval_step": int(last_eval_step),
        "last_ckpt_step": int(last_ckpt_step),
        "last_snapshot_step": int(last_snapshot_step),
    }


def train_feudal_rl(args):
    """Train Feudal RL agent with Manager-Worker hierarchy."""
    import random

    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

    from reinforcetactics.rl.feudal_rl import FeudalRLAgent

    print("\n" + "=" * 60)
    print("Training Feudal RL Agent (Manager-Worker Hierarchy)")
    print("=" * 60 + "\n")

    # Seed everything that drives the feudal training path. SB3's
    # set_random_seed isn't used by collect_rollout / update, so we have to
    # seed torch, numpy, and python directly here.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directories
    log_dir = Path(args.log_dir) / f"feudal_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    (log_dir / "best_model").mkdir(exist_ok=True)

    # Create environments. ``--n-envs > 1`` activates vectorized rollouts via
    # ``FeudalRLAgent.collect_rollout_vec``; per-env state lives inside the
    # agent so the envs themselves can be plain StrategyGameEnv instances.
    env = StrategyGameEnv(map_file=None, opponent=args.opponent, render_mode=None, max_steps=args.max_steps)
    eval_env = StrategyGameEnv(map_file=None, opponent=args.opponent, render_mode=None, max_steps=args.max_steps)
    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 10_000)
    vec_envs: list = []
    if args.n_envs > 1:
        # Each env gets a distinct seed so rollouts aren't perfectly correlated.
        for i in range(args.n_envs):
            ev = StrategyGameEnv(map_file=None, opponent=args.opponent, render_mode=None, max_steps=args.max_steps)
            ev.reset(seed=args.seed + i + 1)
            vec_envs.append(ev)

    # Create agent
    agent = FeudalRLAgent(
        observation_space=env.observation_space,
        grid_width=env.grid_width,
        grid_height=env.grid_height,
        agent_player=getattr(env, "agent_player", 1),
        device=args.device,
        autoregressive_worker=args.autoregressive_worker,
    )
    agent.manager_horizon = args.manager_horizon

    # Setup training
    agent.setup_training(
        learning_rate=args.learning_rate,
        manager_lr_scale=args.manager_lr_scale,
        worker_lr_scale=args.worker_lr_scale,
    )

    # collect_rollout auto-initializes _last_obs on first call, so no need
    # to prime it here.
    agent.reset_goal()

    # Resume from a prior checkpoint if requested. We restore weights +
    # optimizer state via load_checkpoint, then pull timesteps and the best
    # eval metric out of the returned training_state blob so the run picks
    # up where it left off (eval cadence, best-model tracking, LR schedule).
    resume_state: Dict = {}
    if args.resume:
        print(f"Resuming from {args.resume}")
        resume_state = agent.load_checkpoint(args.resume) or {}
        if not resume_state:
            print("  (no training_state in checkpoint — starting counters from 0)")

    # ----- Self-play setup -----
    # When ``--opponent self``, the env's opponent is a snapshot of the agent
    # under training, refreshed every ``--opponent-snapshot-freq`` env steps.
    # A small rolling pool (``--opponent-pool-size`` snapshots) provides
    # diversity — each fresh episode samples one from the pool. Bootstrapping
    # uses RandomBot for the very first episode (no snapshot exists yet).
    self_play_enabled = args.opponent == "self"
    snapshot_dir = log_dir / "snapshots"
    snapshot_pool: list = []  # list of Path, most-recent first
    last_snapshot_step = int(resume_state.get("last_snapshot_step", 0))

    def _add_snapshot(timestep: int) -> None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snap = snapshot_dir / f"opp_{timestep}.pt"
        agent.save_checkpoint(str(snap))
        snapshot_pool.insert(0, snap)
        # Trim — drop the oldest beyond the pool budget. unlink failures are
        # fine to swallow (e.g. file already removed by another process).
        while len(snapshot_pool) > max(args.opponent_pool_size, 1):
            old = snapshot_pool.pop()
            try:
                old.unlink()
            except OSError:
                pass

    def _build_self_play_opponent(game_state, opponent_player):
        """Factory passed to env.set_self_play_opponent_factory.

        Returns a fresh Bot bound to the given game_state. Picks uniformly
        from the snapshot pool; if the pool is empty (first episode) falls
        back to RandomBot so the first agent rollout still has activity.
        """
        from reinforcetactics.game.bot import RandomBot  # pylint: disable=import-outside-toplevel
        from reinforcetactics.game.model_bot import ModelBot  # pylint: disable=import-outside-toplevel

        if not snapshot_pool:
            return RandomBot(game_state, player=opponent_player)
        snap = random.choice(snapshot_pool)
        try:
            return ModelBot(game_state, player=opponent_player, model_path=str(snap))
        except Exception as exc:  # pragma: no cover — defensive
            print(f"Failed to load self-play snapshot {snap}: {exc}; using RandomBot")
            return RandomBot(game_state, player=opponent_player)

    if self_play_enabled:
        env.set_self_play_opponent_factory(_build_self_play_opponent)
        # Take an initial snapshot so episode #1 already has a real opponent
        # (otherwise the first ``opponent_snapshot_freq`` steps would train
        # against RandomBot only, which is the same trap the bootstrap notebook
        # exists to escape).
        _add_snapshot(0)
        env.reset(seed=args.seed)
        # Eval keeps using the originally-requested eval opponent (typically
        # 'random' or 'bot') — we don't want eval scores to drift with the
        # self-play opponent strength.
        if args.eval_opponent != "self":
            eval_env_opp = args.eval_opponent
            # Recreate eval env so it tracks the requested fixed opponent.
            eval_env_new = StrategyGameEnv(map_file=None, opponent=eval_env_opp, render_mode=None, max_steps=args.max_steps)
            eval_env_new.reset(seed=args.seed + 10_000)
            eval_env = eval_env_new  # noqa: F841 — overrides outer eval_env
        print(f"Self-play enabled: snapshot every {args.opponent_snapshot_freq:,} steps, pool={args.opponent_pool_size}")

    writer = SummaryWriter(str(log_dir / "tensorboard"))
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb  # pylint: disable=import-outside-toplevel
        except ImportError:
            print("wandb requested but not installed — continuing with TensorBoard only")
            use_wandb = False
            wandb = None

    def _log(metrics: dict, step: int):
        for k, v in metrics.items():
            writer.add_scalar(k, v, step)
        if use_wandb:
            wandb.log(metrics, step=step)

    def _set_lr(progress_remaining: float) -> float:
        """Apply the chosen LR schedule across both optimizer's param groups.

        progress_remaining ∈ [1.0, 0.0]. Returns the scheduled multiplier
        (1.0 means full base LR) so it can be logged.
        """
        if args.lr_schedule == "linear":
            mult = max(progress_remaining, 0.0)
        else:  # constant
            mult = 1.0
        for opt in (agent.worker_optimizer, agent.manager_optimizer):
            for i, group in enumerate(opt.param_groups):
                group["lr"] = group.get("initial_lr", group["lr"]) * mult
                # Stash initial_lr on first call so we can keep applying mult.
                group.setdefault("initial_lr", group["lr"] / max(mult, 1e-8))
        return mult

    # Stash the initial LR on every param group so the schedule has a stable
    # base to multiply against. (Order matters: must run before update 0.)
    for opt in (agent.worker_optimizer, agent.manager_optimizer):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    num_updates = args.total_timesteps // args.n_steps
    total_timesteps = int(resume_state.get("total_timesteps", 0))
    best_eval_reward = float(resume_state.get("best_eval_reward", float("-inf")))
    best_eval_win_rate = float(resume_state.get("best_eval_win_rate", -1.0))
    last_eval_step = int(resume_state.get("last_eval_step", total_timesteps))
    last_ckpt_step = int(resume_state.get("last_ckpt_step", total_timesteps))
    start_update = total_timesteps // args.n_steps  # for progress display

    print(f"Manager horizon:        {args.manager_horizon}")
    print(f"Worker reward alpha:    {args.worker_reward_alpha}")
    print(f"Reward scale:           {args.reward_scale}")
    print(f"Autoregressive worker:  {args.autoregressive_worker}")
    print(f"LR schedule:            {args.lr_schedule}")
    print(f"Updates to run:         {num_updates}  (resuming at update {start_update})")
    print(f"Steps per update:       {args.n_steps}\n")

    for update_idx in range(start_update, num_updates):
        # Apply LR schedule based on progress through total budget.
        progress_remaining = 1.0 - (total_timesteps / max(args.total_timesteps, 1))
        lr_mult = _set_lr(progress_remaining)

        # Collect rollout. With n_envs > 1 use the vectorized path; the
        # merged buffer holds n_envs * n_steps worker transitions.
        if args.n_envs > 1:
            buf = agent.collect_rollout_vec(
                vec_envs,
                n_steps=args.n_steps,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                worker_reward_alpha=args.worker_reward_alpha,
                reward_scale=args.reward_scale,
            )
            total_timesteps += args.n_steps * args.n_envs
        else:
            buf = agent.collect_rollout(
                env,
                n_steps=args.n_steps,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                worker_reward_alpha=args.worker_reward_alpha,
                reward_scale=args.reward_scale,
            )
            total_timesteps += args.n_steps

        # PPO update
        losses = agent.update(
            buf,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
        )

        # Self-play: take a fresh snapshot of the agent every
        # ``opponent_snapshot_freq`` steps. Future episodes (after the next
        # env reset) sample from the updated pool via the registered factory.
        if self_play_enabled and total_timesteps - last_snapshot_step >= args.opponent_snapshot_freq:
            _add_snapshot(total_timesteps)
            last_snapshot_step = total_timesteps

        # Train metrics: losses + per-rollout summary diagnostics. Surface
        # the env-side reward_breakdown components so the same per-component
        # diagnostics the PPO notebook shows are visible in TensorBoard / W&B.
        metrics: dict = {f"train/{k}": v for k, v in losses.items()}
        metrics["train/manager_segments"] = len(buf.m_rewards)
        metrics["train/worker_mean_reward"] = float(buf.w_rewards.mean())
        metrics["train/episode_dones"] = float(buf.w_dones.sum())
        metrics["train/lr_mult"] = lr_mult
        # Intrinsic / extrinsic reward split: tells us whether the worker is
        # being driven mostly by the manager's goal shaping (high intrinsic)
        # or the env's terminal/per-step reward (high extrinsic). Helps
        # calibrate worker_reward_alpha.
        if hasattr(buf, "w_intrinsic"):
            metrics["train/worker_intrinsic_mean"] = float(buf.w_intrinsic.mean())
            metrics["train/worker_extrinsic_mean"] = float(buf.w_extrinsic.mean())
            # Goal-achievement rate: fraction of worker steps where the agent's
            # nearest unit sat on the manager's goal cell (the +5.0 bonus signal
            # in compute_intrinsic_reward). A low rate over time means goals
            # aren't being achieved — either too hard, too far, or the worker
            # isn't conditioning on them. Track to validate the hierarchy is
            # actually doing useful work.
            metrics["train/goal_reached_rate"] = float(buf.w_reached_goal.mean())
        for k, v in getattr(buf, "reward_breakdown", {}).items():
            metrics[f"train/reward_{k}"] = float(v)
        # End-reason histogram per rollout (one counter per reason).
        for reason in getattr(buf, "end_reasons", []):
            key = f"train/end_reason_{reason}"
            metrics[key] = metrics.get(key, 0.0) + 1.0
        _log(metrics, total_timesteps)

        # Progress logging
        if (update_idx + 1) % 10 == 0:
            reasons = getattr(buf, "end_reasons", [])
            reason_str = ",".join(reasons) if reasons else "—"
            print(
                f"[{total_timesteps:,}] w_policy={losses.get('worker_policy_loss', 0):.3f} "
                f"m_policy={losses.get('manager_policy_loss', 0):.3f} "
                f"w_entropy={losses.get('worker_entropy', 0):.3f} "
                f"w_grad={losses.get('worker_grad_norm', 0):.2f} "
                f"end={reason_str}"
            )

        # Periodic evaluation. Use a high-watermark (last_eval_step) instead
        # of modular arithmetic so the trigger is robust when n_steps doesn't
        # divide eval_freq cleanly (and won't repeat-fire if eval_freq < n_steps).
        if total_timesteps - last_eval_step >= args.eval_freq or update_idx == num_updates - 1:
            last_eval_step = total_timesteps
            eval_results = agent.evaluate(eval_env, n_episodes=args.n_eval_episodes)
            _log(
                {
                    "eval/mean_reward": eval_results["mean_reward"],
                    "eval/std_reward": eval_results["std_reward"],
                    "eval/win_rate": eval_results["win_rate"],
                },
                total_timesteps,
            )
            print(
                f"  EVAL [{total_timesteps:,}] reward={eval_results['mean_reward']:.1f} "
                f"win_rate={eval_results['win_rate']:.2f}"
            )

            # Best-model selection: prefer higher win_rate, break ties on
            # mean_reward. For win/loss-dominant envs win_rate is the more
            # meaningful score; reward fluctuates with shaping noise.
            wr = eval_results["win_rate"]
            mr = eval_results["mean_reward"]
            improved = (wr > best_eval_win_rate) or (wr == best_eval_win_rate and mr > best_eval_reward)
            if improved:
                best_eval_win_rate = wr
                best_eval_reward = mr
                agent.save_checkpoint(
                    str(log_dir / "best_model" / "best_feudal.pt"),
                    training_state=_make_training_state(
                        total_timesteps,
                        best_eval_reward,
                        best_eval_win_rate,
                        last_eval_step,
                        last_ckpt_step,
                        last_snapshot_step,
                    ),
                )
                print(f"     [BEST] win_rate={best_eval_win_rate:.2f} reward={best_eval_reward:.1f}")

        # Periodic checkpoint (also high-watermark gated).
        if total_timesteps - last_ckpt_step >= args.checkpoint_freq:
            last_ckpt_step = total_timesteps
            agent.save_checkpoint(
                str(checkpoint_dir / f"feudal_{total_timesteps}.pt"),
                training_state=_make_training_state(
                    total_timesteps, best_eval_reward, best_eval_win_rate, last_eval_step, last_ckpt_step, last_snapshot_step
                ),
            )

    # Save final model and config
    agent.save_checkpoint(
        str(log_dir / "final_model.pt"),
        training_state=_make_training_state(
            total_timesteps, best_eval_reward, best_eval_win_rate, last_eval_step, last_ckpt_step, last_snapshot_step
        ),
    )
    config = vars(args)
    config_path = log_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    writer.close()
    print(f"\nTraining complete! Model saved to {log_dir / 'final_model.pt'}")

    return log_dir


_ARG_TO_CONFIG_PATH = {
    "opponent": "env.opponent",
    "n_envs": "env.n_envs",
    "total_timesteps": "total_timesteps",
    "seed": "seed",
    "device": "ppo.device",
    "learning_rate": "ppo.learning_rate",
    "n_steps": "ppo.n_steps",
    "batch_size": "ppo.batch_size",
    "n_epochs": "ppo.n_epochs",
    "gamma": "ppo.gamma",
    "gae_lambda": "ppo.gae_lambda",
    "clip_range": "ppo.clip_range",
    "ent_coef": "ppo.ent_coef",
    "vf_coef": "ppo.vf_coef",
    "max_grad_norm": "ppo.max_grad_norm",
    "use_action_masking": "ppo.use_action_masking",
    "manager_horizon": "feudal.manager_horizon",
    "worker_reward_alpha": "feudal.worker_reward_alpha",
    "manager_lr_scale": "feudal.manager_lr_scale",
    "worker_lr_scale": "feudal.worker_lr_scale",
    "autoregressive_worker": "feudal.autoregressive_worker",
    "reward_scale": "feudal.reward_scale",
    "lr_schedule": "ppo.lr_schedule",
    "max_steps": "env.max_steps",
    "resume": "resume",
    "opponent_snapshot_freq": "self_play.snapshot_freq",
    "opponent_pool_size": "self_play.pool_size",
    "eval_opponent": "self_play.eval_opponent",
    "eval_freq": "eval.eval_freq",
    "n_eval_episodes": "eval.n_eval_episodes",
    "checkpoint_freq": "eval.checkpoint_freq",
    "log_dir": "logging.log_dir",
    "wandb": "logging.wandb",
    "wandb_project": "logging.wandb_project",
    "wandb_entity": "logging.wandb_entity",
}


def main():
    """Main entry point for training script."""
    # Pre-parse --config so YAML values become the parser's defaults; CLI
    # flags still override because argparse prefers user-supplied values.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON training config")
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train RL agents for Reinforce Tactics",
        parents=[pre_parser],
    )

    # Training mode
    parser.add_argument(
        "--mode", type=str, default="flat", choices=["flat", "feudal"], help="Training mode: flat baseline or feudal RL"
    )

    # Environment args
    parser.add_argument("--opponent", type=str, default="bot", choices=["bot", "random", "self"], help="Opponent type")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument(
        "--use-action-masking",
        action="store_true",
        help="Use MaskablePPO with action masking (recommended for faster training)",
    )

    # Training args
    parser.add_argument("--total-timesteps", type=int, default=10000000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, or auto")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.05, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")

    # Feudal RL hyperparameters
    parser.add_argument("--manager-horizon", type=int, default=10, help="Worker steps between manager goal updates")
    parser.add_argument(
        "--worker-reward-alpha",
        type=float,
        default=0.5,
        help="Weight of extrinsic reward in worker reward (0=intrinsic only, 1=extrinsic only)",
    )
    parser.add_argument(
        "--manager-lr-scale", type=float, default=1.0, help="Manager learning rate multiplier relative to base LR"
    )
    parser.add_argument(
        "--worker-lr-scale", type=float, default=1.0, help="Worker learning rate multiplier relative to base LR"
    )
    parser.add_argument(
        "--autoregressive-worker",
        action="store_true",
        help="Use AlphaStar-style autoregressive worker head with stage-conditional masks "
        "(requires env.structured_action_masks(); falls back to unmasked otherwise)",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to extrinsic rewards inside collect_rollout. "
        "Set to e.g. 0.001 when terminal magnitudes (±5000) blow up value targets.",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear"],
        help="LR schedule: constant or linear-anneal-to-zero across the total budget.",
    )
    parser.add_argument("--max-steps", type=int, default=500, help="Max env steps per episode")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from (restores weights, optimizers, and total_timesteps / best-metric counters).",
    )
    parser.add_argument(
        "--opponent-snapshot-freq",
        type=int,
        default=10000,
        help="Self-play only: snapshot the agent every N env steps; subsequent "
        "episodes sample opponents from the rolling pool.",
    )
    parser.add_argument(
        "--opponent-pool-size",
        type=int,
        default=5,
        help="Self-play only: max number of agent snapshots kept as opponents. Each episode samples one uniformly at reset.",
    )
    parser.add_argument(
        "--eval-opponent",
        type=str,
        default="random",
        help="Self-play only: opponent type to use for periodic evaluation. "
        "Defaults to 'random' so eval scores don't drift with training opponent strength.",
    )

    # Evaluation args
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Checkpoint save frequency")

    # Logging args
    parser.add_argument("--log-dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="reinforcetactics", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity name")

    if pre_args.config:
        from reinforcetactics.rl.config import config_to_argparse_defaults, load_config

        cfg = load_config(pre_args.config)
        parser.set_defaults(**config_to_argparse_defaults(cfg, _ARG_TO_CONFIG_PATH))

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🚀 Starting training on {args.device}")
    print(f"Mode: {args.mode}")
    print(f"Action masking: {'enabled (MaskablePPO)' if args.use_action_masking else 'disabled (standard PPO)'}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel envs: {args.n_envs}")

    # Initialize W&B if requested
    if args.wandb:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                name=f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            print("✅ Weights & Biases initialized")
        except ImportError:
            print("⚠️  wandb not installed, skipping W&B logging")

    # Train
    if args.mode == "flat":
        log_dir = train_flat_baseline(args)
    elif args.mode == "feudal":
        log_dir = train_feudal_rl(args)

    print("\n✅ Training complete!")
    print(f"Logs saved to: {log_dir}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
