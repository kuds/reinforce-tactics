"""
Purchase-exploration: ε-random override of the ``unit_type`` sub-action.

When a MaskablePPO rollout samples a ``create_unit`` action, with
probability ε this module replaces the policy-chosen ``unit_type`` with
a uniform draw over the env's currently-legal (enabled-in-config *and*
affordable-given-current-gold) unit types. The substitution is taken
from the env's own per-dimension action mask, so the substituted unit
is guaranteed to satisfy both criteria — there is no bypass of the
``enabled_units`` config or the gold check in
``GameState.create_unit``.

The hook lives inside ``policy.forward`` so the substituted action is
the one passed to ``env.step`` *and* the one stored in the rollout
buffer. ``log_prob`` is recomputed under the masked policy at the
substituted action, keeping PPO's importance ratio internally
consistent (numerator and denominator both score the same action).

Pieces:
- :func:`substitute_purchase_unit_types` — pure-numpy substitution,
  testable without sb3-contrib or torch.
- :func:`install_purchase_explore_hook` — wraps ``model.policy.forward``
  exactly once; idempotent across warm-loads.
- :class:`PurchaseExploreScheduleCallback` — anneals
  ``model.purchase_explore_eps`` over a stage budget, mirroring
  :class:`reinforcetactics.rl.callbacks.EntropyScheduleCallback`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Layout of the MultiDiscrete action vector used by ``StrategyGameEnv``:
# [action_type, unit_type, from_x, from_y, to_x, to_y]. ``create_unit``
# is action_type 0 and ``unit_type`` lives in slot 1.
CREATE_UNIT_ACTION_TYPE = 0
UNIT_TYPE_DIM_INDEX = 1


def substitute_purchase_unit_types(
    actions: np.ndarray,
    unit_type_masks: np.ndarray,
    eps: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly resample ``unit_type`` for ``create_unit`` actions.

    Args:
        actions: Integer array of shape ``(n_envs, 6)`` — MultiDiscrete
            actions in the layout
            ``[action_type, unit_type, from_x, from_y, to_x, to_y]``.
        unit_type_masks: Boolean array of shape ``(n_envs, 8)`` — the
            per-env legality mask for the ``unit_type`` sub-action,
            i.e. ``env.action_masks()[1]`` stacked across the VecEnv.
            ``True`` entries are the enabled+affordable unit types for
            that env's current state.
        eps: Substitution probability in ``[0, 1]``. Each create-unit
            row independently triggers a Bernoulli(eps); on success its
            ``unit_type`` is replaced by ``rng.choice`` over the True
            entries of that row's mask.
        rng: Numpy ``Generator`` used for both the Bernoulli draw and
            the uniform-over-legal pick. Pass a seeded generator for
            reproducibility.

    Returns:
        A new integer array of the same shape and dtype as ``actions``.
        Non-create-unit rows are returned unchanged. Create-unit rows
        with an empty mask are also returned unchanged (defensive: the
        env's masking logic should always leave at least one legal unit
        when create_unit is on the action_type mask, but if it doesn't
        we'd rather pass through than crash).

    Notes:
        Input is never mutated. ``eps == 0`` short-circuits to a copy.
    """
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"eps must be in [0, 1], got {eps}")
    out = np.array(actions, copy=True)
    if eps == 0.0:
        return out

    if unit_type_masks.ndim != 2:
        raise ValueError(f"unit_type_masks must be 2-D (n_envs, n_units), got shape {unit_type_masks.shape}")
    if unit_type_masks.shape[0] != out.shape[0]:
        raise ValueError(f"unit_type_masks n_envs={unit_type_masks.shape[0]} does not match actions n_envs={out.shape[0]}")

    n_envs = out.shape[0]
    for i in range(n_envs):
        if int(out[i, 0]) != CREATE_UNIT_ACTION_TYPE:
            continue
        legal = np.flatnonzero(unit_type_masks[i])
        if legal.size == 0:
            continue
        if rng.random() >= eps:
            continue
        out[i, UNIT_TYPE_DIM_INDEX] = int(rng.choice(legal))
    return out


def _slice_unit_type_mask(
    action_masks: Any,
    action_dims: Sequence[int],
) -> np.ndarray | None:
    """Extract the ``unit_type`` per-dim mask from MaskablePPO's input.

    sb3-contrib's :class:`MaskableMultiCategoricalDistribution.apply_masking`
    accepts a single concatenated array of shape
    ``(n_envs, sum(action_dims))`` (or 1-D for a single env). We mirror
    its splitting logic to pull out columns corresponding to the
    ``unit_type`` sub-action.

    Returns ``None`` if ``action_masks`` is missing, empty, or has an
    unexpected shape — the caller should treat that as "no mask, skip
    substitution" rather than silently sampling illegal units.
    """
    if action_masks is None:
        return None
    if hasattr(action_masks, "cpu"):  # torch tensor
        masks_np = action_masks.cpu().numpy()
    else:
        masks_np = np.asarray(action_masks)
    masks_np = masks_np.astype(bool)
    total = int(sum(action_dims))
    if masks_np.ndim == 1:
        if masks_np.shape[0] != total:
            return None
        masks_np = masks_np[None, :]
    elif masks_np.ndim == 2:
        if masks_np.shape[1] != total:
            return None
    else:
        return None
    start = int(sum(action_dims[:UNIT_TYPE_DIM_INDEX]))
    end = start + int(action_dims[UNIT_TYPE_DIM_INDEX])
    return masks_np[:, start:end]


def install_purchase_explore_hook(model: Any, eps: float, seed: int | None = None) -> None:
    """Install the ε-random unit-type substitution on ``model.policy.forward``.

    Idempotent: a model's policy is wrapped at most once, so calling
    twice (e.g. after ``model = MaskablePPO.load(...)``) does not nest
    wrappers. The current ε is read live from ``model.purchase_explore_eps``
    on every call, so :class:`PurchaseExploreScheduleCallback` can
    mutate that attribute without rebuilding the model.

    Args:
        model: A ``MaskablePPO`` instance (or anything exposing a
            ``policy`` with ``forward(obs, deterministic, action_masks)``
            and an ``evaluate_actions(obs, actions, action_masks)``
            method matching sb3-contrib's contract).
        eps: Initial value for ``model.purchase_explore_eps``. Use 0.0
            to install the hook in a no-op state (the wrap is still
            cheap; the substitution short-circuits when eps <= 0).
        seed: Optional seed for the substitution RNG. ``None`` falls
            back to ``model.seed`` if set, else system entropy.
    """
    import torch as th

    model.purchase_explore_eps = float(eps)
    if not 0.0 <= model.purchase_explore_eps <= 1.0:
        raise ValueError(f"eps must be in [0, 1], got {eps}")

    if seed is None:
        seed = getattr(model, "seed", None)
    model._purchase_explore_rng = np.random.default_rng(seed)

    # Models without a ``policy`` attribute (e.g. test stubs that
    # exercise the bootstrap loop without instantiating a real PPO)
    # only need the eps attribute set above; there's no forward()
    # to wrap, so skip the rest. Raise loudly when eps > 0 and no
    # policy exists, since that would silently disable the feature.
    policy = getattr(model, "policy", None)
    if policy is None:
        if model.purchase_explore_eps > 0.0:
            raise AttributeError(
                "install_purchase_explore_hook: model has no .policy attribute, "
                "cannot install the unit_type substitution hook. Either pass a "
                "real (Maskable)PPO instance, or set eps=0."
            )
        return
    if getattr(policy, "_purchase_explore_installed", False):
        return

    # Cache action_dims off the masked distribution so we don't have to
    # introspect the action space on every forward call. action_dims is
    # the ordered list of MultiDiscrete sub-action sizes — e.g.
    # ``[10, 8, W, H, W, H]`` for ``StrategyGameEnv``.
    action_dist = getattr(policy, "action_dist", None)
    action_dims = getattr(action_dist, "action_dims", None)
    if action_dims is None:
        # Fall back to action_space.nvec if the dist hasn't been built
        # yet (shouldn't happen post-construction, but be defensive).
        nvec = getattr(model.action_space, "nvec", None)
        if nvec is None:
            # Non-MultiDiscrete space (e.g. ``flat_discrete`` env mode
            # gives a plain ``Discrete``). The substitution can't apply
            # — sub-actions don't exist as a separate dim. If the user
            # asked for purchase exploration on this space, raise; if
            # eps == 0 the call is a no-op-by-default install from
            # bootstrap, so skip silently.
            if model.purchase_explore_eps > 0.0:
                raise TypeError(
                    f"purchase exploration only supports MultiDiscrete action spaces; got {type(model.action_space).__name__}"
                )
            return
        action_dims = list(int(x) for x in nvec)
    action_dims = list(int(x) for x in action_dims)
    if len(action_dims) <= UNIT_TYPE_DIM_INDEX:
        raise ValueError(f"action_dims {action_dims} has no unit_type slot at index {UNIT_TYPE_DIM_INDEX}")

    original_forward = policy.forward

    def wrapped_forward(obs, deterministic: bool = False, action_masks=None):
        actions, values, log_prob = original_forward(obs, deterministic=deterministic, action_masks=action_masks)
        eps_now = float(getattr(model, "purchase_explore_eps", 0.0))
        if deterministic or eps_now <= 0.0 or action_masks is None:
            return actions, values, log_prob

        ut_mask = _slice_unit_type_mask(action_masks, action_dims)
        if ut_mask is None:
            return actions, values, log_prob

        actions_np = actions.detach().cpu().numpy()
        # Reshape (n_envs, sum(action_dims)) -> (n_envs, len(action_dims))
        # The original forward already reshapes to action_space.shape
        # (i.e. (n_envs, 6) for our MultiDiscrete), but be defensive in
        # case a future SB3 change drops the reshape.
        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(1, -1)
        substituted = substitute_purchase_unit_types(actions_np, ut_mask, eps_now, model._purchase_explore_rng)
        if np.array_equal(substituted, actions_np):
            return actions, values, log_prob

        new_actions = th.as_tensor(substituted, device=actions.device, dtype=actions.dtype)
        # Recompute log_prob *and* values at the substituted action so
        # the rollout buffer's old_log_prob matches the executed action.
        # evaluate_actions re-runs the network; that's the price of
        # avoiding internal copies of forward(). It only fires when at
        # least one env got substituted, which is bounded above by eps
        # times the rate of create_unit sampling — typically a small
        # fraction of rollout steps.
        with th.no_grad():
            _v, new_log_prob, _ent = policy.evaluate_actions(obs, new_actions, action_masks=action_masks)
        return new_actions, values, new_log_prob

    policy.forward = wrapped_forward
    policy._purchase_explore_installed = True


class PurchaseExploreScheduleCallback(BaseCallback):
    """Anneal ``model.purchase_explore_eps`` over a stage budget.

    Mirrors :class:`reinforcetactics.rl.callbacks.EntropyScheduleCallback`:
    progress is computed against ``total_timesteps`` from the start of
    the current stage's ``learn()`` call (so it survives
    ``reset_num_timesteps=False`` curriculum runs). Linear or cosine
    interpolation between ``start`` and ``end``.

    The ε hook is read fresh on every rollout step (see
    :func:`install_purchase_explore_hook`), so writing
    ``model.purchase_explore_eps`` here is enough to drive the schedule.
    Callers must install the hook (with any non-zero initial ε) before
    attaching this callback; otherwise ε mutation has no effect.
    """

    _SCHEDULES = ("linear", "cosine")

    def __init__(
        self,
        start: float,
        end: float,
        total_timesteps: int,
        schedule: str = "linear",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
            raise ValueError(f"start/end must be in [0, 1], got start={start}, end={end}")
        if total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be > 0, got {total_timesteps}")
        if schedule not in self._SCHEDULES:
            raise ValueError(f"schedule must be one of {self._SCHEDULES}, got '{schedule}'")
        self.start = float(start)
        self.end = float(end)
        self.total_timesteps = int(total_timesteps)
        self.schedule = schedule
        self._stage_start_step: int | None = None

    def _on_training_start(self) -> None:
        self._stage_start_step = int(self.num_timesteps)

    def _value_at(self, progress: float) -> float:
        progress = max(0.0, min(1.0, progress))
        if self.schedule == "linear":
            return self.start + (self.end - self.start) * progress
        return self.end + 0.5 * (self.start - self.end) * (1.0 + math.cos(math.pi * progress))

    def _on_step(self) -> bool:
        if self._stage_start_step is None:
            self._stage_start_step = int(self.num_timesteps)
        elapsed = int(self.num_timesteps) - self._stage_start_step
        progress = elapsed / self.total_timesteps if self.total_timesteps > 0 else 1.0
        new_value = float(self._value_at(progress))
        setattr(self.model, "purchase_explore_eps", new_value)
        self.logger.record("train/purchase_explore_eps", new_value)
        return True
