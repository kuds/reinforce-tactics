"""
Shared helpers for the training entry-point scripts.

Centralizes patterns that were copy-pasted across ``train_feudal_rl.py``,
``train_self_play.py``, and ``train_alphazero.py``: device resolution,
W&B initialization, and the ``argparse`` -> dataclass field mappings used
to bridge ``TrainingConfig`` into the script CLIs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` if available else ``"cpu"``.

    Other values (``"cpu"``, ``"cuda"``, ``"mps"``) are passed through
    unchanged so callers can also pass through user-specified devices.
    """
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def init_wandb(
    *,
    enabled: bool,
    project: str,
    entity: Optional[str],
    config: Mapping[str, Any],
    run_name_prefix: str,
) -> bool:
    """Initialize a Weights & Biases run if ``enabled``.

    Returns True if a run was started, False otherwise (W&B disabled,
    not installed, or init failed). Callers can use the return value
    to gate ``wandb.finish()`` at the end of training.
    """
    if not enabled:
        return False
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping W&B logging")
        return False

    run_name = f"{run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=project, entity=entity, config=dict(config), name=run_name)
    logger.info("Weights & Biases initialized: %s", run_name)
    return True


def finish_wandb(active: bool) -> None:
    """End the current W&B run if one was started."""
    if not active:
        return
    try:
        import wandb

        wandb.finish()
    except Exception:  # pylint: disable=broad-except
        pass


# Argparse-dest -> dotted TrainingConfig path. Scripts compose these with
# their own algorithm-specific mappings.
PPO_ARG_MAPPING: Dict[str, str] = {
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
}

COMMON_ARG_MAPPING: Dict[str, str] = {
    "total_timesteps": "total_timesteps",
    "seed": "seed",
    "device": "ppo.device",
    "n_envs": "env.n_envs",
    "max_steps": "env.max_steps",
    "opponent": "env.opponent",
}

EVAL_ARG_MAPPING: Dict[str, str] = {
    "eval_freq": "eval.eval_freq",
    "n_eval_episodes": "eval.n_eval_episodes",
    "checkpoint_freq": "eval.checkpoint_freq",
}

LOGGING_ARG_MAPPING: Dict[str, str] = {
    "log_dir": "logging.log_dir",
    "wandb": "logging.wandb",
    "wandb_project": "logging.wandb_project",
    "wandb_entity": "logging.wandb_entity",
}
