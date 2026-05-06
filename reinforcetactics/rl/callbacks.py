"""
Reusable SB3 callbacks for RL training.

- ``TrainingMetricsCallback`` captures per-rollout PPO metrics
  (rollout/* and train/*) into an in-memory list, working around SB3's
  Logger.dump() clearing values between rollouts.
- ``PeriodicEvalCallback`` runs ``evaluate_model`` every ``eval_freq``
  env steps, mirroring SB3's ``EvalCallback`` contract while capturing
  the project's full win/loss/draw breakdown plus optional per-step
  action-type counts and reward-component sums.

Both callbacks are designed to work with ``MaskablePPO`` from sb3-contrib
as well as plain ``PPO`` from stable-baselines3 — they don't import
sb3-contrib, so the module remains usable in non-masked training too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecNormalize

from reinforcetactics.rl.evaluation import evaluate_model


class TrainingMetricsCallback(BaseCallback):
    """Capture per-rollout PPO training metrics during ``model.learn()``.

    SB3's ``Logger.dump()`` clears ``name_to_value`` between rollouts, so
    rollout/* values are gone by the time the next callback hook fires. To
    work around this we capture rollout/* directly from
    ``self.model.ep_info_buffer`` at ``_on_rollout_end`` (the value source
    the logger itself uses) and read train/* from the logger at the *next*
    ``_on_rollout_start`` — by which point ``train()`` has run for the
    previous iteration and populated those keys. ``_on_training_end``
    picks up the final iteration that has no follow-up rollout.

    The accumulated records persist across multiple ``model.learn()``
    invocations, so multi-stage training produces one continuous timeline.
    """

    TRAIN_KEYS = (
        "train/approx_kl",
        "train/clip_fraction",
        "train/entropy_loss",
        "train/explained_variance",
        "train/learning_rate",
        "train/loss",
        "train/policy_gradient_loss",
        "train/value_loss",
    )

    def __init__(self) -> None:
        super().__init__()
        self.records: List[dict] = []
        self._pending: Optional[dict] = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        snapshot: dict = {"timesteps": self.num_timesteps}
        ep_buffer = getattr(self.model, "ep_info_buffer", None)
        if ep_buffer:
            rewards = [ep["r"] for ep in ep_buffer if "r" in ep]
            lengths = [ep["l"] for ep in ep_buffer if "l" in ep]
            if rewards:
                snapshot["rollout/ep_rew_mean"] = float(safe_mean(rewards))
            if lengths:
                snapshot["rollout/ep_len_mean"] = float(safe_mean(lengths))
        self._pending = snapshot

    def _commit_pending(self) -> None:
        if self._pending is None:
            return
        for key in self.TRAIN_KEYS:
            val = self.model.logger.name_to_value.get(key)
            if val is not None:
                self._pending[key] = float(val)
        # Only emit records that contain at least one metric beyond timesteps.
        if len(self._pending) > 1:
            self.records.append(self._pending)
        self._pending = None

    def _on_rollout_start(self) -> None:
        # train() of the previous iteration has run by this hook, so
        # train/* values are now populated in the logger.
        self._commit_pending()

    def _on_training_end(self) -> None:
        self._commit_pending()


class PeriodicEvalCallback(BaseCallback):
    """Run ``evaluate_model`` every ``eval_freq`` env steps.

    Mirrors SB3's ``EvalCallback`` contract — a single ``model.learn()``
    call drives evaluation at a fixed cadence — but captures the full
    win / loss / draw breakdown via the project's ``evaluate_model``
    helper (SB3's built-in callback only logs mean reward and episode
    length). When ``track_breakdown=True`` the callback also passes
    through to ``evaluate_model``'s breakdown mode, so each entry in
    ``self.results`` carries ``action_counts`` and ``reward_components``
    suitable for stacked-area "what is the agent doing over time" plots.

    The callback gates on ``num_timesteps`` (total env steps across all
    sub-envs) so the cadence is independent of ``n_envs``: ``eval_freq=
    100_000`` fires every 100,000 env steps regardless of how many
    parallel envs are rolling. Best model (by win rate, with avg reward
    as tiebreaker) is saved to ``save_dir/best_model.zip`` when
    ``save_dir`` is provided.
    """

    def __init__(
        self,
        eval_env: Any,
        eval_freq: int,
        n_eval_episodes: int = 30,
        eval_seed_base: int = 0,
        save_dir: Any = None,
        track_breakdown: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_seed_base = int(eval_seed_base)
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.track_breakdown = bool(track_breakdown)

        self.results: List[dict] = []
        self.best_win_rate: float = -1.0
        self._best_reward: float = float("-inf")
        self._last_eval_block: int = -1

    def _on_step(self) -> bool:
        # Trigger when num_timesteps crosses an eval_freq boundary. Using
        # block index (not modulo) avoids missing/double-firing when
        # num_timesteps jumps by n_envs > 1 each step.
        block = self.num_timesteps // self.eval_freq
        if block > self._last_eval_block:
            self._last_eval_block = block
            self._do_eval()
        return True

    def _do_eval(self) -> None:
        eval_seed = self.eval_seed_base + 1000 * self._last_eval_block
        m = evaluate_model(
            self.model,
            self.eval_env,
            n_episodes=self.n_eval_episodes,
            seed=eval_seed,
            track_breakdown=self.track_breakdown,
        )
        m["timesteps"] = int(self.num_timesteps)
        m["eval_seed"] = eval_seed
        self.results.append(m)

        # Tensorboard: log the most useful scalars so they show up alongside
        # the SB3-internal train/* and rollout/* curves.
        self.logger.record("eval/win_rate", m["win_rate"])
        self.logger.record("eval/mean_reward", m["avg_reward"])
        self.logger.record("eval/mean_ep_length", m["avg_length"])

        if self.verbose:
            print(
                f"  [eval @ {m['timesteps']:>9,}]  "
                f"WR={m['win_rate'] * 100:5.1f}%  "
                f"reward={m['avg_reward']:+8.1f} (+/-{m['std_reward']:5.1f})  "
                f"len={m['avg_length']:5.1f}  "
                f"W/L/D={m['wins']}/{m['losses']}/{m['draws']}"
            )

        # Save best by win rate, with avg_reward as a tiebreaker so we don't
        # latch onto the first 0%-WR snapshot.
        if self.save_dir is not None:
            score = (m["win_rate"], m["avg_reward"])
            best = (self.best_win_rate, self._best_reward)
            if score > best:
                self.best_win_rate = m["win_rate"]
                self._best_reward = m["avg_reward"]
                self.model.save(str(self.save_dir / "best_model.zip"))
                # If the training env is VecNormalize-wrapped, the model's
                # value head was trained against normalized rewards — without
                # the matching running-mean/std stats the saved policy isn't
                # reloadable later. Persist them alongside best_model.zip so
                # the pair stays consistent.
                if isinstance(self.training_env, VecNormalize):
                    self.training_env.save(str(self.save_dir / "best_vecnormalize.pkl"))
