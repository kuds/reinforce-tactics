"""
Training script for self-play RL agents.

Self-play training allows the agent to learn by playing against copies of itself,
enabling continuous improvement without requiring hand-crafted opponents.

Features:
- Fictitious self-play with opponent pool
- Periodic opponent updates (pushed to SubprocVecEnv workers too)
- Mixed training: part of the workers play a scripted bot, the rest self-play
- Win rate tracking and model selection
- Evaluation against a fixed scripted bot

Usage:
    # Basic self-play training
    python train_self_play.py --map-file maps/1v1/beginner.csv --action-space flat_discrete

    # With opponent pool
    python train_self_play.py --use-opponent-pool --pool-size 10

    # Mixed training (half the workers against SimpleBot, half self-play)
    python train_self_play.py --mode mixed --bot-ratio 0.5

    # From a config file (env / ppo / self_play / eval / logging sections are
    # honoured; fields this script does not read are reported, and are an
    # error with --strict)
    python train_self_play.py --config configs/self_play/self_play.yaml --strict

    # Resume from checkpoint
    python train_self_play.py --resume-from logs/self_play_xxx/checkpoints/model_100000.zip

Frequencies (``--opponent-update-freq``, ``--add-to-pool-freq``,
``--eval-freq``, ``--checkpoint-freq``) are in environment timesteps; they are
converted to callback calls (one call = ``n_envs`` timesteps) internally.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from reinforcetactics.game.bot_registry import accepted_names
from reinforcetactics.rl.callbacks import AtomicCheckpointCallback, SaveModelAtomicallyCallback, save_model_atomically
from reinforcetactics.rl.gym_env import (
    FLAT_ACTION_VERSIONS,
    flat_action_version_of,
    parse_agent_seat,
    resolve_flat_action_version,
    stamp_flat_action_version,
)
from reinforcetactics.rl.masking import make_maskable_env

# Local imports
from reinforcetactics.rl.self_play import (
    OpponentPool,
    SelfPlayCallback,
    check_worker_split,
    make_self_play_vec_env,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _enabled_units_list(value: Any) -> Any:
    """``--enabled-units`` (a comma-separated string) as the config's list."""
    if isinstance(value, str):
        return [u.strip() for u in value.split(",") if u.strip()]
    return value


def build_env_kwargs(args) -> dict[str, Any]:
    """StrategyGameEnv parameters shared by the training and evaluation envs.

    Every EnvConfig field self-play honours. ``engine_overrides``, the
    observation scales and ``fog_of_war`` used to be dropped between the
    config and the env, so a config's balance overlay or fog of war trained
    the default game without a word. ``flat_action_version`` is resolved
    here: an explicit value wins, else a ``--resume-from`` flat_discrete
    checkpoint keeps the table it was trained on, else the latest.
    """
    enabled_units = _enabled_units_list(args.enabled_units)
    pad_to_size = tuple(args.pad_to_size) if args.pad_to_size else None
    return {
        "map_file": args.map_file,
        "max_steps": args.max_steps,
        "max_turns": args.max_turns,
        "reward_config": args.reward_config,
        "enabled_units": enabled_units or None,
        "action_space_type": args.action_space_type,
        "max_flat_actions": args.max_flat_actions,
        "max_actions_per_turn": args.max_actions_per_turn,
        # Potential-based shaping is only policy-invariant for the
        # trainer's own discount.
        "gamma": args.gamma,
        "pad_to_size": pad_to_size,
        "fog_of_war": args.fog_of_war,
        "engine_overrides": args.engine_overrides,
        "gold_scale": args.gold_scale,
        "turn_scale": args.turn_scale,
        "unit_count_scale": args.unit_count_scale,
        # The bot workers' and the eval env's seat (self-play workers follow
        # --swap-players).
        "agent_seat": args.agent_seat,
        "flat_action_version": resolve_flat_action_version(
            args.flat_action_version,
            action_space_type=args.action_space_type,
            checkpoint_path=getattr(args, "resume_from", None),
        ),
    }


def run_record(args, env_kwargs: dict[str, Any]) -> dict[str, Any]:
    """What ``config.json`` records: the arguments, with the env settings as resolved.

    ``env_kwargs`` is what :func:`build_env_kwargs` resolved, which is what
    every env was built with. ``vars(args)`` alone recorded
    ``flat_action_version: null`` whether a resume kept a version-1 table or
    a cold start used version 2.
    """
    record = dict(vars(args))
    record["flat_action_version"] = env_kwargs["flat_action_version"]
    record["env_kwargs"] = dict(env_kwargs)
    return record


def _per_call(freq: int, n_envs: int) -> int:
    """Convert a frequency in timesteps to SB3 callback calls (one per vec step)."""
    return max(1, int(freq) // max(1, int(n_envs)))


def train_self_play(args) -> Path:
    """
    Self-play (or mixed) training.

    Args:
        args: Parsed command-line arguments

    Returns:
        Path to the log directory
    """
    mixed = args.mode == "mixed"
    bot_ratio = args.bot_ratio if mixed else 0.0

    logger.info("\n" + "=" * 60)
    logger.info("Mixed Training (Self-Play + Bots)" if mixed else "Self-Play Training")
    logger.info("=" * 60 + "\n")

    # Resolved before any output exists: it reads a --resume-from
    # checkpoint's flat table, which can fail.
    env_kwargs = build_env_kwargs(args)

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"{'mixed_training' if mixed else 'self_play'}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    pool_dir = log_dir / "opponent_pool" if args.use_opponent_pool else None

    # Set random seed
    set_random_seed(args.seed)

    # Create opponent pool if enabled
    opponent_pool = None
    if args.use_opponent_pool:
        opponent_pool = OpponentPool(max_size=args.pool_size, selection_strategy=args.pool_strategy, save_dir=str(pool_dir))
        logger.info("Created opponent pool (max size: %d, strategy: %s)", args.pool_size, args.pool_strategy)

    logger.info("Env: %s", env_kwargs)

    # One VecEnv for everything. In mixed mode ``round(n_envs * bot_ratio)``
    # workers play ``--bot-opponent`` and the rest self-play: SB3 cannot swap
    # the training env during learn(), so the mix has to live inside it.
    logger.info("Creating %d environments (bot ratio %.2f)...", args.n_envs, bot_ratio)
    vec_env = make_self_play_vec_env(
        n_envs=args.n_envs,
        seed=args.seed,
        use_subprocess=(args.n_envs > 1 and args.subprocess),
        opponent_pool=opponent_pool,
        swap_players=args.swap_players,
        bot_ratio=bot_ratio,
        bot_opponent=args.bot_opponent,
        bot_opponent_kwargs=args.bot_opponent_kwargs,
        latest_opponent_prob=args.latest_opponent_prob,
        **env_kwargs,
    )

    # Wrap with monitor for logging
    vec_env = VecMonitor(vec_env)

    # Evaluate against a fixed scripted bot: a self-play eval opponent moves
    # with the learner, so its win rate says nothing about progress.
    # Same VecMonitor layering as the training env: evaluate_policy then
    # reports whole-episode reward/length, and SB3's eval callback does not
    # warn about mismatched env types.
    eval_env = VecMonitor(
        DummyVecEnv([lambda: make_maskable_env(opponent=args.eval_opponent, seed=args.seed + 10_000, **env_kwargs)])
    )

    # Import MaskablePPO
    try:
        from sb3_contrib import MaskablePPO

        logger.info("Using MaskablePPO from sb3-contrib")
    except ImportError:
        raise ImportError("sb3-contrib is required for self-play training. Install with: pip install sb3-contrib")

    # Create or load model
    if args.resume_from:
        logger.info("Resuming from checkpoint: %s", args.resume_from)
        model = MaskablePPO.load(args.resume_from, env=vec_env, device=args.device)
        # The loaded model keeps the checkpoint's action space. If the envs
        # now use another flat_discrete table (an explicit
        # --flat-action-version), record that on the model so the
        # checkpoints saved from here on decode (ModelBot) with the table
        # the policy actually plays.
        env_version = env_kwargs["flat_action_version"]
        if args.action_space_type == "flat_discrete" and flat_action_version_of(model) != env_version:
            logger.info(
                "Checkpoint was trained on flat_action_version %d; continuing on version %d",
                flat_action_version_of(model),
                env_version,
            )
            stamp_flat_action_version(model, env_version)
    else:
        from reinforcetactics.rl.bootstrap import _resolve_policy_kwargs

        model = MaskablePPO(
            "MultiInputPolicy",
            vec_env,
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
            # ppo.policy_kwargs (net_arch, features extractor); dotted class
            # names resolved the way the curriculum runner resolves them.
            policy_kwargs=_resolve_policy_kwargs(args.policy_kwargs),
        )

    # Create callbacks
    callbacks: list[BaseCallback] = []

    # Self-play callback. It gets the VecEnv itself (not a list of env
    # objects): it pushes opponent snapshots through env_method, which
    # reaches SubprocVecEnv workers, and raises if no worker is a
    # SelfPlayEnv. It also initializes the opponents at training start.
    self_play_callback = SelfPlayCallback(
        vec_env,
        opponent_pool=opponent_pool,
        update_freq=_per_call(args.opponent_update_freq, args.n_envs),
        add_to_pool_freq=_per_call(args.add_to_pool_freq, args.n_envs),
        min_win_rate_for_pool=args.min_win_rate_for_pool,
        verbose=1,
    )
    callbacks.append(self_play_callback)

    # Checkpoint callback. Atomic: vertex_train.py syncs logs/ while this runs
    # (and on SIGTERM), and an in-place save could publish a truncated zip.
    checkpoint_callback = AtomicCheckpointCallback(
        save_freq=_per_call(args.checkpoint_freq, args.n_envs),
        save_path=str(checkpoint_dir),
        name_prefix="mixed" if mixed else "self_play",
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback. MaskableEvalCallback forwards action masks during
    # evaluation so MaskablePPO selects only valid actions — using SB3's
    # plain EvalCallback would ignore masks.
    eval_callback = MaskableEvalCallback(
        eval_env,
        # Saved by the new-best hook, atomically (see the checkpoint callback),
        # to the same best_model/best_model.zip path SB3 would use.
        best_model_save_path=None,
        callback_on_new_best=SaveModelAtomicallyCallback(log_dir / "best_model" / "best_model.zip"),
        log_path=str(log_dir / "eval"),
        eval_freq=_per_call(args.eval_freq, args.n_envs),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        use_masking=True,
    )
    callbacks.append(eval_callback)

    # Save training config
    config = run_record(args, env_kwargs)
    config_path = log_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved config to %s", config_path)

    # Train
    logger.info("Starting training for %s timesteps...", f"{args.total_timesteps:,}")
    logger.info("Opponent update frequency: %d timesteps", args.opponent_update_freq)
    if mixed:
        logger.info("Bot ratio: %.2f%% (opponent: %s)", bot_ratio * 100, args.bot_opponent)
    if opponent_pool:
        logger.info("Add to pool frequency: %d timesteps", args.add_to_pool_freq)
        logger.info("Min win rate for pool: %.2f%%", args.min_win_rate_for_pool * 100)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=args.progress_bar,
        # Continue the step counter (and schedules) of a resumed run.
        reset_num_timesteps=not args.resume_from,
    )

    # Save final model
    final_path = log_dir / "final_model.zip"
    save_model_atomically(model, final_path)
    logger.info("Training complete! Model saved to %s", final_path)

    # Save final statistics
    stats = {
        "total_timesteps": args.total_timesteps,
        "final_win_rate": self_play_callback._get_average_win_rate(),
        "win_rate_history": self_play_callback.win_rate_history,
        "pool_additions": self_play_callback.pool_additions,
        "pool_size": opponent_pool.size if opponent_pool else 0,
    }
    stats_path = log_dir / "final_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    vec_env.close()
    eval_env.close()
    return log_dir


_ARG_TO_CONFIG_PATH = {
    "map_file": "env.map_file",
    "action_space_type": "env.action_space_type",
    "max_flat_actions": "env.max_flat_actions",
    "flat_action_version": "env.flat_action_version",
    "max_turns": "env.max_turns",
    "max_actions_per_turn": "env.max_actions_per_turn",
    "reward_config": "env.reward_config",
    "pad_to_size": "env.pad_to_size",
    "enabled_units": "env.enabled_units",
    "fog_of_war": "env.fog_of_war",
    "agent_seat": "env.agent_seat",
    "engine_overrides": "env.engine_overrides",
    "gold_scale": "env.gold_scale",
    "turn_scale": "env.turn_scale",
    "unit_count_scale": "env.unit_count_scale",
    # The scripted opponent of the bot workers in mixed mode.
    "bot_opponent": "env.opponent",
    "bot_opponent_kwargs": "env.opponent_kwargs",
    "subprocess": "env.use_subprocess",
    "max_steps": "env.max_steps",
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
    "policy_kwargs": "ppo.policy_kwargs",
    "swap_players": "self_play.swap_players",
    "opponent_update_freq": "self_play.opponent_update_freq",
    "use_opponent_pool": "self_play.use_opponent_pool",
    "pool_size": "self_play.pool_size",
    "pool_strategy": "self_play.pool_strategy",
    "add_to_pool_freq": "self_play.add_to_pool_freq",
    "min_win_rate_for_pool": "self_play.min_win_rate_for_pool",
    "latest_opponent_prob": "self_play.latest_opponent_prob",
    "bot_ratio": "self_play.bot_ratio",
    "eval_freq": "eval.eval_freq",
    "n_eval_episodes": "eval.n_eval_episodes",
    # The fixed scripted opponent of the evaluation env. Used to be unmapped,
    # so the config's value was ignored and the flag's own default ("bot")
    # played instead.
    "eval_opponent": "self_play.eval_opponent",
    "checkpoint_freq": "eval.checkpoint_freq",
    "log_dir": "logging.log_dir",
    "wandb": "logging.wandb",
    "wandb_project": "logging.wandb_project",
    "wandb_entity": "logging.wandb_entity",
}

# Config fields only some modes read (review follow-up: the report used to
# treat every mode alike, so a pure self-play run with a bot opponent set, or
# a pool knob with the pool off, passed --strict and did nothing).
# The bot workers exist only in mixed mode.
_BOT_WORKER_FIELDS = frozenset({"env.opponent", "env.opponent_kwargs", "self_play.bot_ratio"})
# The pool settings, read only with the opponent pool on.
_POOL_FIELDS = frozenset(
    {
        "self_play.pool_size",
        "self_play.pool_strategy",
        "self_play.add_to_pool_freq",
        "self_play.min_win_rate_for_pool",
        "self_play.latest_opponent_prob",
    }
)

# Every config field some mode of this script reads: the argparse-mapped
# ones, plus ``self_play.mixed_training`` (selects --mode mixed). A run reads
# the subset :func:`consumed_config_fields` returns for its mode; anything
# else a --config sets away from its default is reported (an error with
# --strict).
CONSUMED_CONFIG_FIELDS: frozenset[str] = frozenset(_ARG_TO_CONFIG_PATH.values()) | {"self_play.mixed_training"}
CONSUMED_ALGORITHMS: tuple[str, ...] = ("self_play", "mixed", "maskable_ppo")


def consumed_config_fields(mode: str, use_opponent_pool: bool) -> frozenset[str]:
    """The config fields a run in ``mode`` (``'self-play'`` / ``'mixed'``) reads."""
    consumed = CONSUMED_CONFIG_FIELDS
    if mode != "mixed":
        consumed -= _BOT_WORKER_FIELDS
    if not use_opponent_pool:
        consumed -= _POOL_FIELDS
    return consumed


def consumed_algorithms(mode: str) -> tuple[str, ...]:
    """The ``algorithm`` labels that describe a run in ``mode``: ``mixed`` only when it is one."""
    return CONSUMED_ALGORITHMS if mode == "mixed" else ("self_play", "maskable_ppo")


IGNORED_FIELD_HINTS: dict[str, str] = {
    "algorithm": "this script trains MaskablePPO by self-play; 'mixed' needs --mode mixed (or self_play.mixed_training: true)",
    "warm_start_path": "use --resume-from to continue a checkpoint",
    "curriculum.*": "read only by train_bootstrap.py",
    "ppo.use_action_masking": "self-play always trains MaskablePPO with masks",
    "ppo.lr_schedule": "not implemented for the SB3 trainers; the learning rate is constant",
    "ppo.purchase_explore_eps": "only the curriculum runner installs the purchase-exploration hook",
    "self_play.snapshot_freq": "read only by train_feudal_rl.py",
    "env.opponent": "the bot workers' opponent, read only in mixed mode (--mode mixed)",
    "env.opponent_kwargs": "the bot workers' opponent kwargs, read only in mixed mode (--mode mixed)",
    "self_play.bot_ratio": "read only in mixed mode (--mode mixed)",
    "self_play.pool_size": "read only with self_play.use_opponent_pool: true",
    "self_play.pool_strategy": "read only with self_play.use_opponent_pool: true",
    "self_play.add_to_pool_freq": "read only with self_play.use_opponent_pool: true",
    "self_play.min_win_rate_for_pool": "read only with self_play.use_opponent_pool: true",
    "self_play.latest_opponent_prob": "read only with self_play.use_opponent_pool: true",
    "eval.seed_offset": "read only by train_bootstrap.py",
    "eval.resample_eval_seeds": "read only by train_bootstrap.py",
    "eval.best_eligible_after": "read only by train_bootstrap.py",
    "logging.tensorboard": "TensorBoard logs always go to <log-dir>/tensorboard",
    "feudal.*": "read only by train_feudal_rl.py",
    "alphazero.*": "read only by train_alphazero.py",
}


def build_parser(config_path: str | None = None, *, cfg=None) -> argparse.ArgumentParser:
    """Build the argument parser, with defaults taken from the config if given.

    ``cfg`` (an already-loaded :class:`TrainingConfig`) takes precedence
    over ``config_path``.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON training config")
    pre_parser.add_argument(
        "--strict",
        action="store_true",
        help="Make config fields this script does not read an error (default: a warning)",
    )

    parser = argparse.ArgumentParser(
        description="Train RL agents with self-play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre_parser],
    )

    # Training mode
    parser.add_argument(
        "--mode",
        type=str,
        default="self-play",
        choices=["self-play", "mixed"],
        help="Training mode. 'mixed' trains on one VecEnv where round(n_envs * bot_ratio) workers play "
        "--bot-opponent and the rest self-play",
    )

    # Self-play settings
    parser.add_argument(
        "--swap-players",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the agent's seat (player 1 or 2) per episode",
    )
    parser.add_argument(
        "--opponent-update-freq", type=int, default=10000, help="How often to update opponent model (timesteps)"
    )

    # Opponent pool settings
    parser.add_argument("--use-opponent-pool", action="store_true", help="Use pool of historical opponents")
    parser.add_argument("--pool-size", type=int, default=10, help="Maximum size of opponent pool")
    parser.add_argument(
        "--pool-strategy",
        type=str,
        default="uniform",
        choices=["uniform", "recent", "prioritized"],
        help="Opponent selection strategy",
    )
    parser.add_argument("--add-to-pool-freq", type=int, default=50000, help="How often to add model to pool (timesteps)")
    parser.add_argument(
        "--min-win-rate-for-pool",
        type=float,
        default=0.55,
        help="Minimum win rate (over the games since the previous pool check) to add the model to the pool",
    )
    parser.add_argument(
        "--latest-opponent-prob",
        type=float,
        default=0.0,
        help="Once the pool has entries, the per-episode probability of playing the latest snapshot instead of "
        "a pool sample. 0.0 (default) plays a pool sample every episode, as before",
    )

    # Mixed training settings
    parser.add_argument("--bot-ratio", type=float, default=0.3, help="Fraction of workers playing a bot (mixed mode)")
    parser.add_argument(
        "--bot-opponent",
        type=str,
        default="bot",
        choices=accepted_names(),
        help="Scripted opponent of the bot workers (mixed mode)",
    )
    parser.add_argument(
        "--bot-opponent-kwargs",
        type=json.loads,
        default=None,
        help="Constructor kwargs for --bot-opponent as a JSON object (e.g. '{\"max_actions\": 10}')",
    )

    # Environment settings
    parser.add_argument("--map-file", type=str, default=None, help="Map CSV (default: random 20x20 map)")
    parser.add_argument(
        "--action-space",
        dest="action_space_type",
        type=str,
        default="multi_discrete",
        choices=["multi_discrete", "flat_discrete"],
        help="Action space type",
    )
    parser.add_argument("--max-flat-actions", type=int, default=512, help="Action-space size for flat_discrete")
    parser.add_argument(
        "--flat-action-version",
        type=int,
        default=None,
        choices=FLAT_ACTION_VERSIONS,
        help="flat_discrete decode-table layout (default: a --resume-from checkpoint's own, else the latest)",
    )
    parser.add_argument("--max-turns", type=int, default=None, help="Game-turn limit before a draw (default: none)")
    parser.add_argument(
        "--max-actions-per-turn", type=int, default=None, help="Per-turn action budget for both seats (default: none)"
    )
    parser.add_argument(
        "--reward-config", type=json.loads, default=None, help="Reward weights as a JSON object (overrides env defaults)"
    )
    parser.add_argument(
        "--pad-to-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("PAD_H", "PAD_W"),
        help="Zero-pad observations to this size (flat_discrete only)",
    )
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument(
        "--subprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run envs in worker processes (SubprocVecEnv); --no-subprocess uses DummyVecEnv",
    )
    parser.add_argument("--enabled-units", type=str, default=None, help="Comma-separated list of enabled unit types")
    parser.add_argument(
        "--fog-of-war", action=argparse.BooleanOptionalAction, default=False, help="Partial observability for both seats"
    )
    parser.add_argument(
        "--agent-seat",
        type=parse_agent_seat,
        default=1,
        help="Seat of the bot workers (mixed mode) and the eval env: 1, 2 or random (drawn per episode). "
        "Self-play workers follow --swap-players.",
    )
    parser.add_argument(
        "--engine-overrides",
        type=json.loads,
        default=None,
        help="Sparse overlay over the engine constants as a JSON object (see EnvConfig.engine_overrides)",
    )
    parser.add_argument("--gold-scale", type=float, default=None, help="Observation tanh divisor for gold (default: env's)")
    parser.add_argument(
        "--turn-scale", type=float, default=None, help="Observation tanh divisor for the turn (default: env's)"
    )
    parser.add_argument(
        "--unit-count-scale", type=float, default=None, help="Observation tanh divisor for unit counts (default: env's)"
    )

    # Training settings
    parser.add_argument("--total-timesteps", type=int, default=5000000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, or auto")
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True, help="Show SB3's progress bar")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (also used for reward shaping)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.05, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument(
        "--policy-kwargs",
        type=json.loads,
        default=None,
        help="MaskablePPO policy_kwargs as a JSON object (ignored with --resume-from, which keeps the checkpoint's)",
    )

    # Evaluation settings
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency (timesteps)")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument(
        "--eval-opponent",
        type=str,
        default="bot",
        choices=accepted_names(),
        help="Fixed scripted opponent the model is evaluated against (with --config: self_play.eval_opponent)",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Checkpoint save frequency (timesteps)")

    # Logging settings
    parser.add_argument("--log-dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="reinforcetactics-selfplay", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity name")

    if cfg is None and config_path:
        from reinforcetactics.rl.config import load_config

        cfg = load_config(config_path)
    if cfg is not None:
        from reinforcetactics.rl.config import config_to_argparse_defaults

        defaults = config_to_argparse_defaults(cfg, _ARG_TO_CONFIG_PATH)
        # ``self_play.mixed_training: true`` selects mixed mode unless
        # --mode is given on the command line.
        if cfg.self_play.mixed_training:
            defaults["mode"] = "mixed"
        parser.set_defaults(**defaults)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line and validate the settings the run will use.

    With ``--config``, the fields this run's mode does not read are
    reported: a warning, or an ``IgnoredConfigFieldError`` with ``--strict``.
    Every mapped flag is then written back into the config and validated
    (:func:`reinforcetactics.rl.config.effective_config`), so a flag cannot
    carry a value the config file would have been rejected for; a bad one
    is a usage error. The run then uses the validated values: the returned
    arguments carry them as coerced (``--reward-config '{"win": "1e3"}'``
    becomes 1000.0, bot kwargs ``{"max_actions": 10.0}`` become 10), not as
    typed. Validation used to coerce a copy that was then thrown away, so
    such values passed here and the envs rejected them raw.

    The vec-env builder's own worker checks run here too
    (:func:`reinforcetactics.rl.self_play.check_worker_split`): in mixed
    mode ``env.opponent: self`` passes config validation (and argparse does
    not check config-seeded defaults against ``--bot-opponent``'s choices),
    but leaves the bot workers without a scripted opponent. So does a
    ``--resume-from`` checkpoint that does not exist. Each used to fail only
    once the run had created its log directory.
    """
    from reinforcetactics.rl.config import check_ignored_config_fields, effective_config, load_config

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    cfg = load_config(pre_args.config) if pre_args.config else None
    parser = build_parser(cfg=cfg)
    args = parser.parse_args(argv)
    if cfg is not None:
        check_ignored_config_fields(
            cfg,
            consumed_config_fields(args.mode, args.use_opponent_pool),
            entry_point=f"train_self_play.py --mode {args.mode}",
            strict=args.strict,
            algorithms=consumed_algorithms(args.mode),
            hints=IGNORED_FIELD_HINTS,
        )
    try:
        effective_config(cfg, args, _ARG_TO_CONFIG_PATH, convert={"enabled_units": _enabled_units_list}, write_back=True)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        check_worker_split(
            args.n_envs,
            args.bot_ratio if args.mode == "mixed" else 0.0,
            bot_opponent=args.bot_opponent,
            bot_opponent_kwargs=args.bot_opponent_kwargs,
            latest_opponent_prob=args.latest_opponent_prob,
        )
    except (TypeError, ValueError) as exc:
        parser.error(
            f"--mode {args.mode}: {exc}. The workers are set by --n-envs, --bot-ratio, --bot-opponent and "
            "--bot-opponent-kwargs (with --config: env.n_envs, self_play.bot_ratio, env.opponent, env.opponent_kwargs)"
        )
    if args.resume_from and not any(Path(p).is_file() for p in (args.resume_from, f"{args.resume_from}.zip")):
        # MaskablePPO.load, like the flat-table version read, also tries the
        # path with ".zip" appended.
        parser.error(f"--resume-from: checkpoint '{args.resume_from}' does not exist")
    return args


def main(argv: list[str] | None = None):
    """Main entry point."""
    args = parse_args(argv)

    # Set device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Print settings
    logger.info("Starting training on %s", args.device)
    logger.info("Mode: %s", args.mode)
    logger.info("Total timesteps: %s", f"{args.total_timesteps:,}")
    logger.info("Parallel envs: %d", args.n_envs)
    logger.info("Opponent pool: %s", "enabled" if args.use_opponent_pool else "disabled")

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
            logger.info("Weights & Biases initialized")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")

    # Train
    log_dir = train_self_play(args)

    logger.info("Training complete! Logs saved to: %s", log_dir)

    if args.wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
