"""
Imitation learning warm-start for MaskablePPO.

PPO has to discover good behaviour purely from environment reward — for
ability-heavy units (Sorcerer buffs, Mage paralyze, Cleric heals) the credit
assignment is sparse and the random-exploration phase rarely stumbles onto
the relevant combos. AlphaStar's solution was behaviour cloning from human
replays before RL. This module is the same idea using the project's scripted
``MediumBot`` / ``AdvancedBot`` as the expert source.

Pipeline
========

1. ``collect_demonstrations`` plays bot-vs-bot games while wrapping
   ``GameState`` mutator methods. Each time the demonstrator player takes a
   game-state action the wrapper records ``(observation, action_vec,
   per_dim_mask)`` — exactly the shapes that ``MaskableMultiInputPolicy``
   consumes.
2. ``behavior_clone`` runs masked cross-entropy over those demonstrations on
   a freshly constructed ``MaskablePPO`` policy. The value head is left
   untouched (PPO will learn it during fine-tuning).
3. ``make_warm_started_model`` is the convenience entry point that does both
   and returns a model ready to call ``model.learn(...)``.

Only the action space ``"multi_discrete"`` is supported — that is the layout
that MaskablePPO uses by default in this codebase.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from reinforcetactics.constants import ALL_UNIT_TYPES, UNIT_TYPE_TO_IDX
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import (
    AdvancedBot,
    BalancedRandomBot,
    MediumBot,
    NoopBot,
    RandomBot,
    SimpleBot,
)
from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.rl.observation import build_observation
from reinforcetactics.utils.file_io import FileIO

logger = logging.getLogger(__name__)


NUM_ACTION_TYPES = 10
NUM_UNIT_TYPES = 8


# Ordered list of per-dimension sizes for the MaskablePPO MultiDiscrete head.
def _per_dim_sizes(width: int, height: int) -> Tuple[int, ...]:
    return (NUM_ACTION_TYPES, NUM_UNIT_TYPES, width, height, width, height)


# Bot factory registry. Each entry returns a constructed bot bound to the
# given (game_state, player). Demonstrators (recorded) and opponents
# (not recorded) both flow through this registry.
BotFactory = Callable[[GameState, int], Any]


def _make_bot(name: str, rng: Optional[random.Random] = None) -> BotFactory:
    """Return a factory that builds the named bot for a (gs, player) pair."""
    name = name.lower()

    def factory(game_state: GameState, player: int) -> Any:
        if name in ("simple", "bot"):
            return SimpleBot(game_state, player=player)
        if name == "medium":
            return MediumBot(game_state, player=player)
        if name == "advanced":
            return AdvancedBot(game_state, player=player)
        if name == "noop":
            return NoopBot(game_state, player=player)
        if name == "random":
            return RandomBot(game_state, player=player, rng=rng)
        if name == "balanced_random":
            return BalancedRandomBot(game_state, player=player, rng=rng)
        raise ValueError(f"Unknown bot type for imitation: {name!r}")

    return factory


@dataclass
class Demonstration:
    """A single (obs, action, mask) triple captured from a demonstrator turn."""

    obs: Dict[str, np.ndarray]
    action: np.ndarray  # shape (6,), int64
    # Per-dimension boolean masks, same layout MaskablePPO consumes.
    at_mask: np.ndarray  # (10,)
    ut_mask: np.ndarray  # (8,)
    fx_mask: np.ndarray  # (W,)
    fy_mask: np.ndarray  # (H,)
    tx_mask: np.ndarray  # (W,)
    ty_mask: np.ndarray  # (H,)


@dataclass
class DemonstrationDataset:
    """Stacked demonstrations as numpy arrays ready for batched BC training."""

    obs: Dict[str, np.ndarray]
    actions: np.ndarray  # (N, 6), int64
    masks_concat: np.ndarray  # (N, 10+8+W+H+W+H), bool
    # Dimension sizes used to split ``masks_concat`` back per-dim if needed.
    dim_sizes: Tuple[int, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    @classmethod
    def from_list(cls, demos: List[Demonstration]) -> "DemonstrationDataset":
        if not demos:
            raise ValueError("Cannot build dataset from empty demonstration list")

        # Stack observation dict (assumes consistent key set).
        keys = list(demos[0].obs.keys())
        obs_stacked: Dict[str, np.ndarray] = {k: np.stack([d.obs[k] for d in demos], axis=0) for k in keys}

        actions = np.stack([d.action for d in demos], axis=0).astype(np.int64)

        # Concatenate per-dim masks in the order MaskablePPO splits them.
        mask_blocks = []
        for d in demos:
            mask_blocks.append(
                np.concatenate(
                    [d.at_mask, d.ut_mask, d.fx_mask, d.fy_mask, d.tx_mask, d.ty_mask],
                    axis=0,
                ).astype(np.bool_)
            )
        masks_concat = np.stack(mask_blocks, axis=0)

        dim_sizes = (
            demos[0].at_mask.shape[0],
            demos[0].ut_mask.shape[0],
            demos[0].fx_mask.shape[0],
            demos[0].fy_mask.shape[0],
            demos[0].tx_mask.shape[0],
            demos[0].ty_mask.shape[0],
        )

        return cls(obs=obs_stacked, actions=actions, masks_concat=masks_concat, dim_sizes=dim_sizes)


# ---------------------------------------------------------------------------
# Mask helper — mirrors StrategyGameEnv._build_masks but is callable without
# instantiating the env. Keeps imitation independent of env wiring while
# preserving the exact MaskablePPO contract (per-dimension union mask).
# ---------------------------------------------------------------------------


def _compute_masks(
    game_state: GameState,
    width: int,
    height: int,
    enabled_units: List[str],
) -> Tuple[
    np.ndarray,  # flat (10*W*H,) for obs.action_mask
    np.ndarray,  # at_mask (10,)
    np.ndarray,  # ut_mask (8,)
    np.ndarray,  # fx_mask (W,)
    np.ndarray,  # fy_mask (H,)
    np.ndarray,  # tx_mask (W,)
    np.ndarray,  # ty_mask (H,)
]:
    """Compute the env's mask layout for the current player.

    This is a thin replication of ``StrategyGameEnv._build_masks`` factored so
    that the demonstration recorder does not have to construct a full env.
    """
    legal_actions = game_state.get_legal_actions(player=game_state.current_player)
    area = width * height

    flat = np.zeros(NUM_ACTION_TYPES * area, dtype=np.float32)
    at = np.zeros(NUM_ACTION_TYPES, dtype=bool)
    ut = np.zeros(NUM_UNIT_TYPES, dtype=bool)
    fx = np.zeros(width, dtype=bool)
    fy = np.zeros(height, dtype=bool)
    tx = np.zeros(width, dtype=bool)
    ty = np.zeros(height, dtype=bool)

    # (action_key, action_type_idx, src_field, tgt_field) — same map as gym_env.
    action_map = StrategyGameEnv._ACTION_KEY_MAP

    def _pos(action: Dict[str, Any], fields: Any) -> Tuple[int, int]:
        if isinstance(fields, str):
            o = action[fields]
            return o.x, o.y
        return action[fields[0]], action[fields[1]]

    for key, (at_idx, src_fields, tgt_fields) in action_map.items():
        for action in legal_actions.get(key, []):
            at[at_idx] = True

            tx_, ty_ = _pos(action, tgt_fields)
            tx[tx_] = True
            ty[ty_] = True
            flat_idx = at_idx * area + ty_ * width + tx_
            if 0 <= flat_idx < flat.size:
                flat[flat_idx] = 1.0

            if src_fields is not None:
                sx, sy = _pos(action, src_fields)
                fx[sx] = True
                fy[sy] = True
            else:
                fx[tx_] = True
                fy[ty_] = True

            if key == "create_unit":
                ut[UNIT_TYPE_TO_IDX.get(action["unit_type"], 0)] = True

    # End turn always legal at canonical (0, 0).
    at[5] = True
    flat[5 * area] = 1.0
    fx[0] = True
    fy[0] = True
    tx[0] = True
    ty[0] = True

    if not ut.any():
        if enabled_units:
            ut[UNIT_TYPE_TO_IDX.get(enabled_units[0], 0)] = True
        else:
            ut[0] = True

    return flat, at, ut, fx, fy, tx, ty


# ---------------------------------------------------------------------------
# GameState method interception
# ---------------------------------------------------------------------------


class _ActionRecorder:
    """Captures (obs, action_vec, mask) triples from intercepted GameState calls.

    Wraps the action-mutating methods of a single ``GameState`` instance.
    Each wrapped method snapshots obs/mask BEFORE delegating to the real
    method, so the recorded observation reflects the pre-action state — which
    is what the policy needs to learn from.

    Only actions whose ``current_player`` matches ``demonstrator_player`` are
    recorded; the opposing bot's actions on the same game state are passed
    through untouched.
    """

    # Methods we intercept. Each entry maps to (action_type, extractor) where
    # extractor returns ``(unit_type_letter_or_None, (from_x, from_y), (to_x, to_y))``
    # given the call args.
    _INTERCEPT_METHODS = (
        "create_unit",
        "move_unit",
        "attack",
        "seize",
        "heal",
        "cure",
        "paralyze",
        "haste",
        "defence_buff",
        "attack_buff",
        "end_turn",
    )

    def __init__(
        self,
        game_state: GameState,
        demonstrator_player: int,
        width: int,
        height: int,
        enabled_units: List[str],
        fog_of_war: bool,
    ) -> None:
        self.game_state = game_state
        self.demonstrator_player = demonstrator_player
        self.width = width
        self.height = height
        self.enabled_units = enabled_units
        self.fog_of_war = fog_of_war
        self.demos: List[Demonstration] = []
        self._originals: Dict[str, Callable[..., Any]] = {}
        self._installed = False

    # -- snapshot helpers --------------------------------------------------

    def _snapshot(self, action: np.ndarray) -> None:
        flat, at, ut, fx, fy, tx, ty = _compute_masks(self.game_state, self.width, self.height, self.enabled_units)

        obs = build_observation(
            self.game_state,
            perspective_player=self.demonstrator_player,
            action_mask=flat,
            fog_of_war=self.fog_of_war,
        )

        # Sanity: ensure each component of the recorded action is permitted
        # by its per-dim mask; otherwise the masked-categorical log_prob
        # would be -inf during BC. Drop the demonstration rather than train
        # on a corrupt label. This is rare in practice (scripted bots only
        # play legal moves), but skipping is cheaper than diagnosing later.
        a_type = int(action[0])
        a_unit = int(action[1])
        a_fx, a_fy, a_tx, a_ty = (int(action[2]), int(action[3]), int(action[4]), int(action[5]))
        if not (at[a_type] and ut[a_unit] and fx[a_fx] and fy[a_fy] and tx[a_tx] and ty[a_ty]):
            logger.debug(
                "Skipping demonstration with mask-illegal action %s (likely sanitization edge)",
                action.tolist(),
            )
            return

        self.demos.append(
            Demonstration(
                obs=obs,
                action=action.astype(np.int64),
                at_mask=at,
                ut_mask=ut,
                fx_mask=fx,
                fy_mask=fy,
                tx_mask=tx,
                ty_mask=ty,
            )
        )

    def _is_demonstrator_turn(self) -> bool:
        return self.game_state.current_player == self.demonstrator_player

    # ut_mask is guaranteed to have at least one bit set; pick the first set
    # bit for non-create actions where unit_type is a "don't care" slot.
    def _default_unit_idx(self) -> int:
        for letter in self.enabled_units:
            return UNIT_TYPE_TO_IDX.get(letter, 0)
        return 0

    # -- wrappers (one per intercepted method) -----------------------------

    def _wrap_create_unit(self) -> Callable[..., Any]:
        original = self._originals["create_unit"]

        def wrapped(unit_type, x, y, player=None):
            target_player = player if player is not None else self.game_state.current_player
            if target_player == self.demonstrator_player:
                ut_idx = UNIT_TYPE_TO_IDX.get(unit_type, 0)
                self._snapshot(np.array([0, ut_idx, x, y, x, y], dtype=np.int64))
            return original(unit_type, x, y, player=player)

        return wrapped

    def _wrap_move_unit(self) -> Callable[..., Any]:
        original = self._originals["move_unit"]

        def wrapped(unit, to_x, to_y):
            if self._is_demonstrator_turn() and unit.player == self.demonstrator_player:
                self._snapshot(
                    np.array(
                        [1, self._default_unit_idx(), unit.x, unit.y, to_x, to_y],
                        dtype=np.int64,
                    )
                )
            return original(unit, to_x, to_y)

        return wrapped

    def _wrap_attack(self) -> Callable[..., Any]:
        original = self._originals["attack"]

        def wrapped(attacker, target):
            if self._is_demonstrator_turn() and attacker.player == self.demonstrator_player:
                self._snapshot(
                    np.array(
                        [2, self._default_unit_idx(), attacker.x, attacker.y, target.x, target.y],
                        dtype=np.int64,
                    )
                )
            return original(attacker, target)

        return wrapped

    def _wrap_seize(self) -> Callable[..., Any]:
        original = self._originals["seize"]

        def wrapped(unit):
            if self._is_demonstrator_turn() and unit.player == self.demonstrator_player:
                self._snapshot(
                    np.array(
                        [3, self._default_unit_idx(), unit.x, unit.y, unit.x, unit.y],
                        dtype=np.int64,
                    )
                )
            return original(unit)

        return wrapped

    def _wrap_heal_like(self, name: str, src_attr: str, action_type: int) -> Callable[..., Any]:
        original = self._originals[name]

        def wrapped(src, target):
            if self._is_demonstrator_turn() and getattr(src, "player", None) == self.demonstrator_player:
                self._snapshot(
                    np.array(
                        [action_type, self._default_unit_idx(), src.x, src.y, target.x, target.y],
                        dtype=np.int64,
                    )
                )
            return original(src, target)

        # ``src_attr`` accepted for API symmetry / documentation; not used
        # because both heal-like signatures are positional ``(src, target)``.
        del src_attr
        return wrapped

    def _wrap_end_turn(self) -> Callable[..., Any]:
        original = self._originals["end_turn"]

        def wrapped():
            if self._is_demonstrator_turn():
                self._snapshot(np.array([5, self._default_unit_idx(), 0, 0, 0, 0], dtype=np.int64))
            return original()

        return wrapped

    # -- install / uninstall ------------------------------------------------

    def install(self) -> None:
        if self._installed:
            return
        gs = self.game_state
        for name in self._INTERCEPT_METHODS:
            self._originals[name] = getattr(gs, name)

        gs.create_unit = self._wrap_create_unit()
        gs.move_unit = self._wrap_move_unit()
        gs.attack = self._wrap_attack()
        gs.seize = self._wrap_seize()
        gs.heal = self._wrap_heal_like("heal", "healer", action_type=4)
        gs.cure = self._wrap_heal_like("cure", "curer", action_type=4)
        gs.paralyze = self._wrap_heal_like("paralyze", "paralyzer", action_type=6)
        gs.haste = self._wrap_heal_like("haste", "sorcerer", action_type=7)
        gs.defence_buff = self._wrap_heal_like("defence_buff", "sorcerer", action_type=8)
        gs.attack_buff = self._wrap_heal_like("attack_buff", "sorcerer", action_type=9)
        gs.end_turn = self._wrap_end_turn()
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for name, fn in self._originals.items():
            setattr(self.game_state, name, fn)
        self._originals.clear()
        self._installed = False


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------


def record_episode(
    demonstrator: str = "medium",
    opponent: str = "medium",
    map_file: Optional[str] = None,
    enabled_units: Optional[List[str]] = None,
    max_turns: int = 200,
    fog_of_war: bool = False,
    seed: Optional[int] = None,
    demonstrator_player: int = 1,
) -> List[Demonstration]:
    """Play one bot-vs-bot game and return demos from ``demonstrator_player``.

    The demonstrator and opponent share the same ``GameState``. The
    demonstrator's mutator calls are intercepted and recorded; the
    opponent's calls flow through untouched.
    """
    if demonstrator_player not in (1, 2):
        raise ValueError(f"demonstrator_player must be 1 or 2, got {demonstrator_player}")

    rng = random.Random(seed) if seed is not None else None
    if map_file:
        map_data = FileIO.load_map(map_file)
    else:
        # Match StrategyGameEnv default (20x20, 2 players).
        map_data = FileIO.generate_random_map(20, 20, num_players=2)

    units = enabled_units if enabled_units is not None else ALL_UNIT_TYPES.copy()
    game_state = GameState(
        map_data,
        num_players=2,
        max_turns=max_turns,
        enabled_units=units,
        fog_of_war=fog_of_war,
    )
    if fog_of_war:
        game_state.update_visibility()

    width = game_state.grid.width
    height = game_state.grid.height

    recorder = _ActionRecorder(
        game_state=game_state,
        demonstrator_player=demonstrator_player,
        width=width,
        height=height,
        enabled_units=units,
        fog_of_war=fog_of_war,
    )
    recorder.install()

    try:
        demo_factory = _make_bot(demonstrator, rng=rng)
        opp_factory = _make_bot(opponent, rng=rng)

        opponent_player = 3 - demonstrator_player
        bots = {
            demonstrator_player: demo_factory(game_state, demonstrator_player),
            opponent_player: opp_factory(game_state, opponent_player),
        }

        # Hard cap on turns to prevent pathological infinite games when both
        # bots end every turn with no productive moves. ``max_turns`` already
        # ends the game inside GameState; the +50 buffer absorbs intra-turn
        # action loops.
        step_budget = max_turns * 4 + 50
        steps = 0
        while not game_state.game_over and steps < step_budget:
            current_bot = bots[game_state.current_player]
            current_bot.take_turn()
            steps += 1
    finally:
        recorder.uninstall()

    return recorder.demos


def collect_demonstrations(
    n_episodes: int = 50,
    demonstrator: str = "medium",
    opponent: str = "medium",
    map_file: Optional[str] = None,
    enabled_units: Optional[List[str]] = None,
    max_turns: int = 200,
    fog_of_war: bool = False,
    seed: Optional[int] = None,
    demonstrator_player: int = 1,
    progress: bool = False,
) -> DemonstrationDataset:
    """Collect demonstrations from ``n_episodes`` bot-vs-bot games."""
    all_demos: List[Demonstration] = []

    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        ep_demos = record_episode(
            demonstrator=demonstrator,
            opponent=opponent,
            map_file=map_file,
            enabled_units=enabled_units,
            max_turns=max_turns,
            fog_of_war=fog_of_war,
            seed=ep_seed,
            demonstrator_player=demonstrator_player,
        )
        all_demos.extend(ep_demos)
        if progress:
            logger.info(
                "imitation episode %d/%d collected %d demos (total %d)",
                ep + 1,
                n_episodes,
                len(ep_demos),
                len(all_demos),
            )

    if not all_demos:
        raise RuntimeError(
            "No demonstrations collected. The demonstrator bot did not produce "
            "any recordable actions — check bot/opponent compatibility with the map."
        )

    return DemonstrationDataset.from_list(all_demos)


# ---------------------------------------------------------------------------
# Behavior cloning
# ---------------------------------------------------------------------------


@dataclass
class BCStats:
    """Per-epoch BC training metrics."""

    epoch: int
    loss: float
    accuracy_action_type: float
    accuracy_full: float


def _iter_minibatches(
    dataset: DemonstrationDataset,
    batch_size: int,
    rng: np.random.Generator,
) -> Any:
    n = len(dataset)
    indices = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield indices[start : start + batch_size]


def behavior_clone(
    model: Any,
    dataset: DemonstrationDataset,
    n_epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    seed: int = 0,
    log_every: int = 1,
) -> List[BCStats]:
    """Behavior-clone ``model.policy`` on ``dataset`` via masked cross-entropy.

    Only the policy (action) head is updated. The value head is left alone —
    PPO's first updates will fit it; pre-training a value function from
    demonstrations is a separate (and noisier) problem.

    Args:
        model: A constructed ``MaskablePPO`` instance. Its policy must be a
            ``MaskableActorCriticPolicy`` over a ``MultiDiscrete`` action
            space (the default for ``make_maskable_env``).
        dataset: Output of :func:`collect_demonstrations`.
        n_epochs: Number of full passes over the dataset.
        batch_size: Mini-batch size for each gradient step.
        learning_rate: Adam learning rate for the BC phase.
        seed: RNG seed for batch shuffling.
        log_every: Emit a log message every N epochs.

    Returns:
        Per-epoch ``BCStats`` list. The final entry is also useful as a
        cheap regression sanity check in tests.
    """
    import torch as th  # local import: torch is heavy and only needed here

    policy = model.policy
    device = policy.device

    # Filter parameters belonging to the policy / feature extractor only.
    # ``value_net`` is preserved in MaskableActorCriticPolicy but excluded
    # from the BC optimizer to avoid biasing it toward demo states.
    bc_params = [p for n, p in policy.named_parameters() if not n.startswith("value_net")]
    optimizer = th.optim.Adam(bc_params, lr=learning_rate)

    rng = np.random.default_rng(seed)
    n = len(dataset)
    if n == 0:
        raise ValueError("Empty demonstration dataset")

    # Pre-convert obs to torch tensors per epoch's batches lazily.
    def _to_tensor(arr: np.ndarray) -> th.Tensor:
        return th.as_tensor(arr, device=device)

    stats: List[BCStats] = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_correct_atype = 0
        epoch_correct_full = 0
        epoch_count = 0

        for batch_idx in _iter_minibatches(dataset, batch_size, rng):
            obs_batch = {k: _to_tensor(v[batch_idx]) for k, v in dataset.obs.items()}
            actions = _to_tensor(dataset.actions[batch_idx]).long()
            masks = _to_tensor(dataset.masks_concat[batch_idx])

            _values, log_prob, _entropy = policy.evaluate_actions(obs_batch, actions, action_masks=masks)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(bc_params, max_norm=0.5)
            optimizer.step()

            with th.no_grad():
                # Greedy decode under masking for accuracy diagnostics.
                policy.set_training_mode(False)
                pred_actions, _ = policy.predict(
                    {k: v.cpu().numpy() for k, v in obs_batch.items()},
                    deterministic=True,
                    action_masks=masks.cpu().numpy(),
                )
                policy.set_training_mode(True)

                pred_t = th.as_tensor(pred_actions, device=device).long()
                # action_type matches column 0
                epoch_correct_atype += int((pred_t[:, 0] == actions[:, 0]).sum().item())
                epoch_correct_full += int((pred_t == actions).all(dim=1).sum().item())
                epoch_count += int(actions.shape[0])
                epoch_loss += float(loss.item()) * int(actions.shape[0])

        stat = BCStats(
            epoch=epoch + 1,
            loss=epoch_loss / max(epoch_count, 1),
            accuracy_action_type=epoch_correct_atype / max(epoch_count, 1),
            accuracy_full=epoch_correct_full / max(epoch_count, 1),
        )
        stats.append(stat)
        if (epoch + 1) % log_every == 0:
            logger.info(
                "BC epoch %d/%d  loss=%.4f  action_type_acc=%.3f  full_acc=%.3f  (n=%d)",
                stat.epoch,
                n_epochs,
                stat.loss,
                stat.accuracy_action_type,
                stat.accuracy_full,
                epoch_count,
            )

    policy.set_training_mode(False)
    return stats


# ---------------------------------------------------------------------------
# One-shot convenience entry
# ---------------------------------------------------------------------------


def make_warm_started_model(
    env: Any,
    n_episodes: int = 50,
    n_epochs: int = 5,
    demonstrator: str = "medium",
    opponent: str = "medium",
    map_file: Optional[str] = None,
    enabled_units: Optional[List[str]] = None,
    max_turns: int = 200,
    fog_of_war: bool = False,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    seed: int = 0,
    ppo_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, DemonstrationDataset, List[BCStats]]:
    """Build a MaskablePPO model and warm-start it via behavior cloning.

    The ``env`` argument is the environment MaskablePPO will be trained on
    after this call (single env or vec env). The demonstration generation is
    independent and uses fresh GameState instances configured to mirror the
    env (same map / unit roster / fog-of-war flag), so the BC observations
    have the same shape and semantics as what the policy will see during
    PPO fine-tuning.

    Returns:
        Tuple of (model, dataset, bc_stats). Call ``model.learn(...)`` next.
    """
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise ImportError("sb3-contrib is required for imitation warm-start. Install with: pip install sb3-contrib") from exc

    dataset = collect_demonstrations(
        n_episodes=n_episodes,
        demonstrator=demonstrator,
        opponent=opponent,
        map_file=map_file,
        enabled_units=enabled_units,
        max_turns=max_turns,
        fog_of_war=fog_of_war,
        seed=seed,
        progress=True,
    )
    logger.info("Collected %d demonstrations across %d episodes", len(dataset), n_episodes)

    default_ppo: Dict[str, Any] = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "verbose": 0,
    }
    if ppo_kwargs:
        default_ppo.update(ppo_kwargs)

    model = MaskablePPO("MultiInputPolicy", env, **default_ppo)

    bc_stats = behavior_clone(
        model=model,
        dataset=dataset,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )

    return model, dataset, bc_stats
