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
    MasterBot,
    MediumBot,
    MixedBot,
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

# Action-type index for end_turn (matches _wrap_end_turn's snapshot and
# StrategyGameEnv._ACTION_KEY_MAP). Scripted bots emit exactly one end_turn
# per game-turn vs. many move / build / attack actions per turn, so the raw
# demonstration mix is ~10:1 against end_turn. The cross-entropy loss then
# suppresses the end_turn logit in ~90% of gradient updates, producing the
# never-end-turn attractor observed when the BC checkpoint is loaded into
# the curriculum. The class-balancing path in ``behavior_clone`` upweights
# this single class to neutralise the imbalance.
END_TURN_ACTION_IDX = 5


# Ordered list of per-dimension sizes for the MaskablePPO MultiDiscrete head.
def _per_dim_sizes(width: int, height: int) -> Tuple[int, ...]:
    return (NUM_ACTION_TYPES, NUM_UNIT_TYPES, width, height, width, height)


# Bot factory registry. Each entry returns a constructed bot bound to the
# given (game_state, player). Demonstrators (recorded) and opponents
# (not recorded) both flow through this registry.
BotFactory = Callable[[GameState, int], Any]


def _make_bot(
    name: str,
    rng: Optional[random.Random] = None,
    stochastic_tiebreak: bool = False,
    tiebreak_rng: Optional[random.Random] = None,
) -> BotFactory:
    """Return a factory that builds the named bot for a (gs, player) pair.

    Args:
        name: Bot type (simple / medium / advanced / mixed / random / etc.).
        rng: Optional ``random.Random`` consumed by stochastic bots
            (random, balanced_random, mixed). Existing behaviour --
            stochastic bots draw actions from this stream.
        stochastic_tiebreak: If True, the deterministic bots
            (simple/medium/advanced/master) also receive a per-episode
            rng for shuffle-before-sort tiebreaking. Without this,
            repeated episodes against deterministic bots produce
            byte-identical games. Scoring logic is unchanged.
        tiebreak_rng: Optional dedicated rng for deterministic-bot
            tiebreaking. When provided AND ``stochastic_tiebreak`` is
            True, this is used instead of ``rng`` so tiebreak shuffles
            do not consume the same rng state that stochastic bots
            draw from. Decouples the two rng streams: toggling
            ``stochastic_tiebreak`` on the same seed leaves
            random-bot action traces byte-identical (only the
            tiebreak choices of the deterministic bot change).
            Falls back to ``rng`` for backwards compatibility when
            unset.
    """
    name = name.lower()

    # Pick the rng deterministic bots will use for tiebreaks. Prefer
    # the dedicated tiebreak_rng (decoupled streams) and fall back to
    # the shared rng (legacy path) when not provided. ``None`` when
    # stochastic_tiebreak is False -- bots stay fully deterministic.
    det_rng = (tiebreak_rng if tiebreak_rng is not None else rng) if stochastic_tiebreak else None

    def factory(game_state: GameState, player: int) -> Any:
        if name in ("simple", "bot"):
            return SimpleBot(game_state, player=player, rng=det_rng)
        if name == "medium":
            return MediumBot(game_state, player=player, rng=det_rng)
        if name == "mixed":
            return MixedBot(game_state, player=player, rng=rng)
        if name == "advanced":
            return AdvancedBot(game_state, player=player, rng=det_rng)
        if name == "master":
            return MasterBot(game_state, player=player, rng=det_rng)
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
    """A single (obs, action, mask) triple captured from a demonstrator turn.

    Two per-dim mask views are stored:

      * ``*_mask`` (load-bearing for the BC loss) -- narrowed to one-hot on
        placeholder dims (see ``_snapshot``). Avoids gradient leakage on dims
        whose recorded value is a default rather than a real demonstrator
        choice.
      * ``env_*_mask`` (load-bearing for diagnostics) -- the un-narrowed
        env-style union mask, identical to what ``StrategyGameEnv._build_masks``
        would produce at this state. Used by ``behavior_clone`` to compute
        per-dim loss / accuracy and an "honest" joint accuracy that doesn't
        get a free trivial match on placeholder dims.

    Storing both keeps the load shape stable and lets the loss / metric
    decisions diverge per call site without recomputing masks at train
    time.
    """

    obs: Dict[str, np.ndarray]
    action: np.ndarray  # shape (6,), int64
    # Narrowed per-dimension boolean masks, fed to MaskablePPO for the loss.
    at_mask: np.ndarray  # (10,)
    ut_mask: np.ndarray  # (8,)
    fx_mask: np.ndarray  # (W,)
    fy_mask: np.ndarray  # (H,)
    tx_mask: np.ndarray  # (W,)
    ty_mask: np.ndarray  # (H,)
    # Un-narrowed env-style per-dim masks (the union over all legal actions
    # at this state) -- used for honest accuracy diagnostics, not the loss.
    env_at_mask: Optional[np.ndarray] = None  # (10,)
    env_ut_mask: Optional[np.ndarray] = None  # (8,)
    env_fx_mask: Optional[np.ndarray] = None  # (W,)
    env_fy_mask: Optional[np.ndarray] = None  # (H,)
    env_tx_mask: Optional[np.ndarray] = None  # (W,)
    env_ty_mask: Optional[np.ndarray] = None  # (H,)


@dataclass
class EpisodeOutcome:
    """Outcome of one demonstration episode.

    Captured so callers can assess *demonstrator quality* per scenario --
    a scenario where the demonstrator loses most games yields BC labels
    biased toward losing trajectories, which downgrades the BC ceiling.
    """

    demonstrator_player: int
    winner: Optional[int]  # 1, 2, or None for draw / unresolved
    end_reason: Optional[str]
    n_turns: int
    n_demos: int

    @property
    def demonstrator_won(self) -> bool:
        return self.winner == self.demonstrator_player

    @property
    def demonstrator_lost(self) -> bool:
        return self.winner is not None and self.winner != self.demonstrator_player

    @property
    def is_draw(self) -> bool:
        return self.winner is None


@dataclass
class ScenarioStats:
    """Aggregated demonstrator outcomes for one scenario.

    Surfaces the BC ceiling per (demonstrator, opponent, map) triple: if
    AdvancedBot only beats RandomBot in 40% of games on this map, BC
    labels from that scenario teach the policy losing patterns at a 60%
    rate. Surfacing this *before* PPO fine-tuning lets the user adjust
    scenario weights / demonstrator choice rather than discover the
    ceiling after a long training run.
    """

    name: str
    demonstrator: str
    opponent: str
    map_file: Optional[str] = None
    n_episodes: int = 0
    demo_wins: int = 0
    demo_losses: int = 0
    draws: int = 0
    # Action-loop timeouts: the episode driver exited via step_budget
    # without GameState.end_game ever running. Tracked separately from
    # ``draws`` so users can distinguish 'map_turns_draw' (the game
    # legitimately ended with no winner) from 'bot stuck in an
    # intra-turn loop' (a failure mode that would otherwise inflate the
    # draws column and look like the scenario is producing draws).
    step_budget_exhausted: int = 0
    total_demos: int = 0
    total_turns: int = 0
    end_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def total_games(self) -> int:
        return self.demo_wins + self.demo_losses + self.draws + self.step_budget_exhausted

    @property
    def demo_win_rate(self) -> float:
        return self.demo_wins / self.total_games if self.total_games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.total_games if self.total_games else 0.0

    @property
    def avg_turns(self) -> float:
        return self.total_turns / self.total_games if self.total_games else 0.0

    @property
    def avg_demos_per_game(self) -> float:
        return self.total_demos / self.total_games if self.total_games else 0.0

    def record(self, outcome: EpisodeOutcome) -> None:
        if outcome.demonstrator_won:
            self.demo_wins += 1
        elif outcome.demonstrator_lost:
            self.demo_losses += 1
        elif outcome.end_reason == "step_budget_exhausted":
            # Distinguishable from a genuine draw: this is a bot-stall
            # signature, not an outcome of play.
            self.step_budget_exhausted += 1
        else:
            self.draws += 1
        self.total_demos += outcome.n_demos
        self.total_turns += outcome.n_turns
        reason = outcome.end_reason or "unknown"
        self.end_reasons[reason] = self.end_reasons.get(reason, 0) + 1


@dataclass
class DemonstrationDataset:
    """Stacked demonstrations as numpy arrays ready for batched BC training.

    ``masks_concat`` holds the *narrowed* per-dim masks (placeholder dims
    collapsed to one-hot at the recorded value -- consumed by the BC loss).
    ``env_masks_concat`` holds the *un-narrowed* env union masks, consumed
    by per-dim / honest-joint diagnostics. The env view is optional for
    backward compatibility with datasets built before the field existed
    (those fall back to ``masks_concat`` at diagnostic time, which gives
    the old over-optimistic numbers but doesn't break anything).
    """

    obs: Dict[str, np.ndarray]
    actions: np.ndarray  # (N, 6), int64
    masks_concat: np.ndarray  # (N, 10+8+W+H+W+H), bool -- narrowed (loss)
    # Dimension sizes used to split ``masks_concat`` back per-dim if needed.
    dim_sizes: Tuple[int, ...] = field(default_factory=tuple)
    # Un-narrowed env-style per-dim masks for honest diagnostics. None for
    # older datasets serialized before this field existed.
    env_masks_concat: Optional[np.ndarray] = None  # (N, 10+8+W+H+W+H), bool
    # Per-scenario outcomes. Empty when the dataset is built by
    # ``from_list`` alone; populated by ``collect_demonstrations`` /
    # ``collect_demonstrations_multi`` so callers can plot demonstrator
    # win-rates alongside BC training curves.
    scenario_stats: List[ScenarioStats] = field(default_factory=list)

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

        # Concatenate the narrowed per-dim masks in the order MaskablePPO
        # splits them. These feed the BC loss.
        mask_blocks = []
        for d in demos:
            mask_blocks.append(
                np.concatenate(
                    [d.at_mask, d.ut_mask, d.fx_mask, d.fy_mask, d.tx_mask, d.ty_mask],
                    axis=0,
                ).astype(np.bool_)
            )
        masks_concat = np.stack(mask_blocks, axis=0)

        # Concatenate the env-style (un-narrowed) per-dim masks if every
        # demonstration carries them. All-or-nothing keeps the field
        # honest: a partially populated stack would silently mix narrowed
        # and un-narrowed rows, defeating the diagnostic purpose.
        env_masks_concat: Optional[np.ndarray] = None
        if all(d.env_at_mask is not None for d in demos):
            env_blocks = []
            for d in demos:
                env_blocks.append(
                    np.concatenate(
                        [
                            d.env_at_mask,
                            d.env_ut_mask,
                            d.env_fx_mask,
                            d.env_fy_mask,
                            d.env_tx_mask,
                            d.env_ty_mask,
                        ],
                        axis=0,
                    ).astype(np.bool_)
                )
            env_masks_concat = np.stack(env_blocks, axis=0)

        dim_sizes = (
            demos[0].at_mask.shape[0],
            demos[0].ut_mask.shape[0],
            demos[0].fx_mask.shape[0],
            demos[0].fy_mask.shape[0],
            demos[0].tx_mask.shape[0],
            demos[0].ty_mask.shape[0],
        )

        return cls(
            obs=obs_stacked,
            actions=actions,
            masks_concat=masks_concat,
            dim_sizes=dim_sizes,
            env_masks_concat=env_masks_concat,
        )


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

        # Action mask is recorded on the Demonstration separately (per-dim
        # masks for MaskablePPO); we deliberately do NOT include it in the
        # obs dict — the policy's observation space no longer carries it.
        obs = build_observation(
            self.game_state,
            perspective_player=self.demonstrator_player,
            action_mask=None,
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

        # Narrow "don't-care" per-dim masks to one-hot at the recorded
        # value. evaluate_actions returns a joint log_prob summed across
        # all 6 head dims; without this narrowing the BC loss bleeds
        # gradient into heads whose recorded value is a placeholder, not
        # a real demonstrator choice. Two failure modes the leak drives:
        #
        #   * end_turn (atype=5) records (unit=0, fx=0, fy=0, tx=0, ty=0).
        #     The full mask keeps every legal create/move/attack/seize
        #     bit set across the unit / position dims; log P(0 | union)
        #     is then >> log(1), so each end_turn demo pushes the unit
        #     head toward Warrior AND the position heads toward (0, 0).
        #     With ``end_turn_weight`` auto-balancing at ~10x, that
        #     pressure is amplified and bleeds into non-end_turn states
        #     because the position / unit heads are shared across atypes.
        #   * Non-create, non-end_turn (move / attack / seize / heal /
        #     ability) records the unit dim as ``_default_unit_idx()``
        #     (always 0 = enabled_units[0]); the position dims hold the
        #     real source / target. Only the unit dim is a placeholder,
        #     and it leaks the same way -- every such demo pushes the
        #     unit head toward 0 (Warrior), biasing create_unit toward
        #     Warrior across the board.
        #
        # Narrowing a placeholder dim's mask to ``{recorded_value}`` makes
        # log P(value | {value}) = log 1 = 0, so the loss contributes no
        # gradient to that head -- the only gradient that flows on a
        # placeholder dim is from real demos where that dim is a real
        # choice. The action_type mask (``at``) is left alone: every
        # demo's atype is a real choice and the BC head should learn it
        # against the full legal-atype distribution.
        #
        # Snapshot the un-narrowed env-style masks first -- diagnostics
        # downstream (per-dim accuracy, honest full_action_acc) need to
        # see the real decision distribution, not the artificially
        # collapsed loss view.
        env_at = at.copy()
        env_ut = ut.copy()
        env_fx = fx.copy()
        env_fy = fy.copy()
        env_tx = tx.copy()
        env_ty = ty.copy()

        if a_type == END_TURN_ACTION_IDX:
            ut = np.zeros(NUM_UNIT_TYPES, dtype=bool)
            ut[a_unit] = True
            fx = np.zeros(self.width, dtype=bool)
            fx[a_fx] = True
            fy = np.zeros(self.height, dtype=bool)
            fy[a_fy] = True
            tx = np.zeros(self.width, dtype=bool)
            tx[a_tx] = True
            ty = np.zeros(self.height, dtype=bool)
            ty[a_ty] = True
        elif a_type != 0:  # non-create_unit, non-end_turn: only unit dim is a placeholder
            ut = np.zeros(NUM_UNIT_TYPES, dtype=bool)
            ut[a_unit] = True

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
                env_at_mask=env_at,
                env_ut_mask=env_ut,
                env_fx_mask=env_fx,
                env_fy_mask=env_fy,
                env_tx_mask=env_tx,
                env_ty_mask=env_ty,
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

        # ``setattr`` instead of ``gs.foo = wrapped`` because the latter trips
        # mypy's ``method-assign`` rule. The runtime behaviour is identical:
        # both bind the wrapped closure on the instance, shadowing the class
        # method until ``uninstall`` restores the original.
        setattr(gs, "create_unit", self._wrap_create_unit())
        setattr(gs, "move_unit", self._wrap_move_unit())
        setattr(gs, "attack", self._wrap_attack())
        setattr(gs, "seize", self._wrap_seize())
        setattr(gs, "heal", self._wrap_heal_like("heal", "healer", action_type=4))
        setattr(gs, "cure", self._wrap_heal_like("cure", "curer", action_type=4))
        setattr(gs, "paralyze", self._wrap_heal_like("paralyze", "paralyzer", action_type=6))
        setattr(gs, "haste", self._wrap_heal_like("haste", "sorcerer", action_type=7))
        setattr(gs, "defence_buff", self._wrap_heal_like("defence_buff", "sorcerer", action_type=8))
        setattr(gs, "attack_buff", self._wrap_heal_like("attack_buff", "sorcerer", action_type=9))
        setattr(gs, "end_turn", self._wrap_end_turn())
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


def _play_episode(
    demonstrator: str,
    opponent: str,
    map_file: Optional[str],
    enabled_units: Optional[List[str]],
    max_turns: int,
    fog_of_war: bool,
    seed: Optional[int],
    demonstrator_player: int,
    stochastic_tiebreak: bool = False,
) -> Tuple[List[Demonstration], EpisodeOutcome]:
    """Internal driver. Plays one game, returns demos plus end-state outcome.

    ``record_episode`` is the public façade that returns only the demos
    (kept stable for callers); ``collect_demonstrations`` uses this
    function directly so it can aggregate scenario-level outcome stats.
    """
    if demonstrator_player not in (1, 2):
        raise ValueError(f"demonstrator_player must be 1 or 2, got {demonstrator_player}")

    rng = random.Random(seed) if seed is not None else None
    # Dedicated rng for deterministic-bot tiebreaks. Derived from the
    # same seed via a fixed XOR offset so the run is reproducible, but
    # the stream is INDEPENDENT of ``rng`` -- toggling
    # ``stochastic_tiebreak`` on a given seed leaves the stochastic
    # bots' (random / balanced_random / mixed) action traces
    # byte-identical. Without this split, a deterministic-bot shuffle
    # consumes the shared rng and silently shifts RandomBot's next
    # .choice() draw, making the two modes incomparable on the same
    # seed. The 0xC0FFEEBAD constant is arbitrary; just needs to be
    # stable across runs.
    tiebreak_rng = random.Random(seed ^ 0xC0FFEEBAD) if (seed is not None and stochastic_tiebreak) else None
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
        # Both factories share the per-episode rng for stochastic-bot
        # action draws (random / balanced_random / mixed). When
        # ``stochastic_tiebreak`` is True, _make_bot threads
        # ``tiebreak_rng`` into the deterministic bots so equal-scoring
        # decisions resolve randomly per episode -- the only way to get
        # distinct trajectories from a fully-deterministic matchup
        # (e.g. AdvancedBot vs AdvancedBot) on the same map. ``rng``
        # and ``tiebreak_rng`` are independent streams so toggling
        # ``stochastic_tiebreak`` on the same seed only affects the
        # deterministic-bot tiebreaks.
        demo_factory = _make_bot(
            demonstrator,
            rng=rng,
            stochastic_tiebreak=stochastic_tiebreak,
            tiebreak_rng=tiebreak_rng,
        )
        opp_factory = _make_bot(
            opponent,
            rng=rng,
            stochastic_tiebreak=stochastic_tiebreak,
            tiebreak_rng=tiebreak_rng,
        )

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

    # Distinguish action-loop timeouts (loop exited via step_budget,
    # game_state.game_over still False) from legitimate map-cap draws
    # (GameState.end_game ran with winner=None, end_reason='max_turns_draw'
    # or similar). Without this both look identical -- winner=None,
    # end_reason=None vs end_reason='max_turns_draw' -- and the scenario
    # stats' draws column lumps both together, hiding bot-stall
    # failures behind apparent map draws.
    end_reason = game_state.end_reason
    if not game_state.game_over and end_reason is None:
        end_reason = "step_budget_exhausted"

    outcome = EpisodeOutcome(
        demonstrator_player=demonstrator_player,
        winner=game_state.winner,
        end_reason=end_reason,
        n_turns=game_state.turn_number,
        n_demos=len(recorder.demos),
    )
    return recorder.demos, outcome


def record_episode(
    demonstrator: str = "medium",
    opponent: str = "medium",
    map_file: Optional[str] = None,
    enabled_units: Optional[List[str]] = None,
    max_turns: int = 200,
    fog_of_war: bool = False,
    seed: Optional[int] = None,
    demonstrator_player: int = 1,
    stochastic_tiebreak: bool = False,
) -> List[Demonstration]:
    """Play one bot-vs-bot game and return demos from ``demonstrator_player``.

    The demonstrator and opponent share the same ``GameState``. The
    demonstrator's mutator calls are intercepted and recorded; the
    opponent's calls flow through untouched.

    ``stochastic_tiebreak=True`` resolves equal-scoring bot decisions
    randomly per episode -- required to get distinct trajectories from
    N runs of a fully-deterministic matchup like
    ``demonstrator='advanced', opponent='advanced'`` (which otherwise
    produces N byte-identical games).
    """
    demos, _ = _play_episode(
        demonstrator=demonstrator,
        opponent=opponent,
        map_file=map_file,
        enabled_units=enabled_units,
        max_turns=max_turns,
        fog_of_war=fog_of_war,
        seed=seed,
        demonstrator_player=demonstrator_player,
        stochastic_tiebreak=stochastic_tiebreak,
    )
    return demos


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
    scenario_name: Optional[str] = None,
    stochastic_tiebreak: bool = False,
) -> DemonstrationDataset:
    """Collect demonstrations from ``n_episodes`` bot-vs-bot games.

    The returned dataset's ``scenario_stats`` is populated with a single
    :class:`ScenarioStats` summarising demonstrator W/L/D, average turn
    count, and end-reason histogram across the ``n_episodes`` games --
    surfaces demonstrator quality so callers can sanity-check BC label
    quality before training.

    When ``stochastic_tiebreak=True``, deterministic bots receive the
    per-episode rng so equal-scoring decisions resolve randomly. Each
    episode gets a different ``seed + ep`` rng, so N episodes produce
    N distinct trajectories instead of N copies of the same game.
    """
    all_demos: List[Demonstration] = []
    stats = ScenarioStats(
        name=scenario_name or f"{demonstrator}_vs_{opponent}",
        demonstrator=demonstrator,
        opponent=opponent,
        map_file=map_file,
        n_episodes=n_episodes,
    )

    for ep in range(n_episodes):
        ep_seed = None if seed is None else seed + ep
        ep_demos, outcome = _play_episode(
            demonstrator=demonstrator,
            opponent=opponent,
            map_file=map_file,
            enabled_units=enabled_units,
            max_turns=max_turns,
            fog_of_war=fog_of_war,
            seed=ep_seed,
            demonstrator_player=demonstrator_player,
            stochastic_tiebreak=stochastic_tiebreak,
        )
        all_demos.extend(ep_demos)
        stats.record(outcome)
        if progress:
            logger.info(
                "imitation episode %d/%d collected %d demos (total %d) winner=%s turns=%d",
                ep + 1,
                n_episodes,
                len(ep_demos),
                len(all_demos),
                outcome.winner if outcome.winner is not None else "draw",
                outcome.n_turns,
            )

    if not all_demos:
        raise RuntimeError(
            "No demonstrations collected. The demonstrator bot did not produce "
            "any recordable actions — check bot/opponent compatibility with the map."
        )

    dataset = DemonstrationDataset.from_list(all_demos)
    dataset.scenario_stats = [stats]
    return dataset


# ---------------------------------------------------------------------------
# Multi-scenario collection
# ---------------------------------------------------------------------------


@dataclass
class DemonstrationScenario:
    """One scenario in a curated mix of demonstration sources.

    A scenario bundles a (map, unit roster, demonstrator, opponent) so that
    each entry produces demonstrations targeted at a specific behaviour. For
    example, a scenario that enables only ``["W", "S"]`` (Warrior + Sorcerer)
    forces the demonstrator down decision branches that exercise the
    Sorcerer's buff abilities — yielding far more attack_buff / haste demos
    per episode than running ``MediumBot`` on a random map with all 8 units.

    Attributes:
        map_file: Path to map CSV (``None`` -> generated random map matching
            ``StrategyGameEnv``'s default).
        enabled_units: Subset of unit codes to enable for this scenario. The
            scripted bots only spawn / consider units in this list.
        demonstrator: Bot name to record (``simple|medium|advanced``).
        opponent: Bot name on the other side (``simple|medium|advanced|random|
            balanced_random|noop``).
        n_episodes: Episodes to play for this scenario.
        max_turns: Per-episode turn cap.
        fog_of_war: Whether to enable FOW for this scenario.
        demonstrator_player: 1 or 2 — which side the recorder follows.
        weight: Optional sampling weight applied at concat time. Values >1
            duplicate demos from this scenario; <1 subsamples. Use sparingly
            — duplication grows memory linearly.
        stochastic_tiebreak: When True, the deterministic bots
            (simple/medium/advanced/master) receive a per-episode rng
            that shuffles equal-scoring decisions before they sort.
            Required for any scenario whose demonstrator AND opponent
            are both deterministic -- without it, every episode plays
            the byte-identical same game and N episodes give 1 unique
            trajectory. Default ``False`` preserves backward
            compatibility (existing tests / tournaments unchanged).
    """

    map_file: Optional[str] = None
    enabled_units: Optional[List[str]] = None
    demonstrator: str = "medium"
    opponent: str = "medium"
    n_episodes: int = 10
    max_turns: int = 200
    fog_of_war: bool = False
    demonstrator_player: int = 1
    weight: float = 1.0
    name: Optional[str] = None  # Free-form label used in logs only
    stochastic_tiebreak: bool = False


def _grid_dims_from_map(map_file: Optional[str]) -> Tuple[int, int]:
    """Return (width, height) for a map source matching the env's logic."""
    if map_file:
        map_data = FileIO.load_map(map_file)
    else:
        map_data = FileIO.generate_random_map(20, 20, num_players=2)
    # GameState wraps map_data into a TileGrid; the env exposes
    # game_state.grid.{width,height} as the source of truth. Build a
    # disposable game_state to match exactly.
    gs = GameState(map_data, num_players=2)
    return gs.grid.width, gs.grid.height


def _resample_dataset(
    dataset: DemonstrationDataset,
    weight: float,
    rng: np.random.Generator,
) -> DemonstrationDataset:
    """Up- or down-sample a dataset to ``round(weight * len(dataset))`` rows."""
    n = len(dataset)
    target = max(1, int(round(weight * n)))
    if target == n:
        # Build a fresh DemonstrationDataset rather than aliasing the
        # input. The numpy arrays are still shared (no-op resampling
        # never copies data), but ``scenario_stats`` gets a new list
        # so a caller that later mutates ``returned.scenario_stats``
        # does not also mutate the original's stats. Matches the
        # contract of the non-trivial paths below (which always return
        # a fresh dataset object).
        return DemonstrationDataset(
            obs=dict(dataset.obs),
            actions=dataset.actions,
            masks_concat=dataset.masks_concat,
            dim_sizes=dataset.dim_sizes,
            env_masks_concat=dataset.env_masks_concat,
            scenario_stats=list(dataset.scenario_stats),
        )
    if target < n:
        idx = rng.choice(n, size=target, replace=False)
    else:
        # Keep all originals, then upsample the remainder with replacement.
        extra = rng.choice(n, size=target - n, replace=True)
        idx = np.concatenate([np.arange(n), extra])
    obs = {k: v[idx] for k, v in dataset.obs.items()}
    env_masks_concat = dataset.env_masks_concat[idx] if dataset.env_masks_concat is not None else None
    # ``scenario_stats`` describes the ORIGINAL demonstrator outcomes for
    # this scenario, independent of any down/up-sampling applied to the
    # demo rows -- carry it through unchanged so the user sees the real
    # game-level WR / turn counts even after resampling.
    return DemonstrationDataset(
        obs=obs,
        actions=dataset.actions[idx],
        masks_concat=dataset.masks_concat[idx],
        dim_sizes=dataset.dim_sizes,
        env_masks_concat=env_masks_concat,
        scenario_stats=list(dataset.scenario_stats),
    )


def _concat_datasets(parts: List[DemonstrationDataset]) -> DemonstrationDataset:
    """Concatenate datasets that share the same dim_sizes and obs key set."""
    if not parts:
        raise ValueError("No datasets to concatenate")
    if len(parts) == 1:
        return parts[0]

    ref = parts[0]
    for i, p in enumerate(parts[1:], start=1):
        if p.dim_sizes != ref.dim_sizes:
            raise ValueError(
                f"Scenario {i} has incompatible mask dim_sizes {p.dim_sizes} "
                f"vs {ref.dim_sizes}. All scenarios must share grid size and "
                f"action-space layout."
            )
        if set(p.obs.keys()) != set(ref.obs.keys()):
            raise ValueError(
                f"Scenario {i} obs keys {sorted(p.obs.keys())} disagree with "
                f"{sorted(ref.obs.keys())} (likely a fog_of_war mismatch)."
            )
        for k in ref.obs:
            if p.obs[k].shape[1:] != ref.obs[k].shape[1:]:
                raise ValueError(f"Scenario {i} obs[{k!r}] shape {p.obs[k].shape} incompatible with {ref.obs[k].shape}.")

    obs = {k: np.concatenate([p.obs[k] for p in parts], axis=0) for k in ref.obs}
    actions = np.concatenate([p.actions for p in parts], axis=0)
    masks_concat = np.concatenate([p.masks_concat for p in parts], axis=0)
    # Only carry env_masks_concat through if every part has it; otherwise the
    # diagnostic field becomes meaningless (mixed narrowed / un-narrowed
    # rows). The all-or-nothing contract matches ``from_list``.
    if all(p.env_masks_concat is not None for p in parts):
        env_masks_concat: Optional[np.ndarray] = np.concatenate([p.env_masks_concat for p in parts], axis=0)
    else:
        env_masks_concat = None
    merged_stats: List[ScenarioStats] = []
    for p in parts:
        merged_stats.extend(p.scenario_stats)
    return DemonstrationDataset(
        obs=obs,
        actions=actions,
        masks_concat=masks_concat,
        dim_sizes=ref.dim_sizes,
        env_masks_concat=env_masks_concat,
        scenario_stats=merged_stats,
    )


def collect_demonstrations_multi(
    scenarios: List[DemonstrationScenario],
    seed: int = 0,
    progress: bool = False,
    shuffle: bool = True,
) -> DemonstrationDataset:
    """Round-robin collect demonstrations across multiple scenarios.

    Each scenario contributes ``n_episodes`` games; recorded demos are
    concatenated into a single dataset. All scenarios must share the same
    grid size and fog-of-war setting (otherwise mask shapes / obs shapes
    don't stack cleanly).

    Args:
        scenarios: Ordered list of scenarios. Order does not affect the
            final dataset when ``shuffle=True``.
        seed: Global RNG seed; per-scenario seeds derive from this.
        progress: If True, log per-scenario yields.
        shuffle: Shuffle the concatenated dataset so mini-batches mix
            scenarios (matters for BC — without shuffling, the optimizer
            sees all of scenario 1, then all of scenario 2, etc.).

    Returns:
        A single ``DemonstrationDataset`` ready to pass to ``behavior_clone``.

    Raises:
        ValueError: If any two scenarios disagree on grid size or
            fog_of_war (the obs/mask shapes would not stack).
    """
    if not scenarios:
        raise ValueError("scenarios must contain at least one entry")

    # Pre-validate grid dimensions so users see the failure before any
    # episodes are simulated (collecting demos is the slow part).
    ref_dims = _grid_dims_from_map(scenarios[0].map_file)
    for i, sc in enumerate(scenarios[1:], start=1):
        dims = _grid_dims_from_map(sc.map_file)
        if dims != ref_dims:
            raise ValueError(
                f"Scenario {i} ({sc.name or sc.map_file or '<random>'}) has "
                f"grid {dims} but scenario 0 has {ref_dims}. All scenarios "
                f"must share the same map dimensions."
            )

    rng = np.random.default_rng(seed)
    parts: List[DemonstrationDataset] = []
    for i, sc in enumerate(scenarios):
        scenario_seed = seed + 10_000 * (i + 1)
        if progress:
            logger.info(
                "scenario %d/%d %s: %s vs %s, %d episodes, units=%s",
                i + 1,
                len(scenarios),
                sc.name or sc.map_file or "<random map>",
                sc.demonstrator,
                sc.opponent,
                sc.n_episodes,
                sc.enabled_units or "all",
            )
        ds = collect_demonstrations(
            n_episodes=sc.n_episodes,
            demonstrator=sc.demonstrator,
            opponent=sc.opponent,
            map_file=sc.map_file,
            enabled_units=sc.enabled_units,
            max_turns=sc.max_turns,
            fog_of_war=sc.fog_of_war,
            seed=scenario_seed,
            demonstrator_player=sc.demonstrator_player,
            progress=False,
            scenario_name=sc.name or sc.map_file or f"scenario_{i}",
            stochastic_tiebreak=sc.stochastic_tiebreak,
        )
        if sc.weight != 1.0:
            ds = _resample_dataset(ds, sc.weight, rng)
        if progress:
            s = ds.scenario_stats[0] if ds.scenario_stats else None
            if s is not None:
                logger.info(
                    "  -> %d demos | demonstrator W/L/D %d/%d/%d (WR=%.1f%%) | avg_turns=%.1f | action_type histogram: %s",
                    len(ds),
                    s.demo_wins,
                    s.demo_losses,
                    s.draws,
                    100.0 * s.demo_win_rate,
                    s.avg_turns,
                    np.bincount(ds.actions[:, 0], minlength=NUM_ACTION_TYPES).tolist(),
                )
            else:
                logger.info(
                    "  -> %d demos (action_type histogram: %s)",
                    len(ds),
                    np.bincount(ds.actions[:, 0], minlength=NUM_ACTION_TYPES).tolist(),
                )
        parts.append(ds)

    combined = _concat_datasets(parts)

    if shuffle:
        order = rng.permutation(len(combined))
        env_masks_concat = combined.env_masks_concat[order] if combined.env_masks_concat is not None else None
        combined = DemonstrationDataset(
            obs={k: v[order] for k, v in combined.obs.items()},
            actions=combined.actions[order],
            masks_concat=combined.masks_concat[order],
            dim_sizes=combined.dim_sizes,
            env_masks_concat=env_masks_concat,
            scenario_stats=combined.scenario_stats,
        )

    return combined


def format_scenario_stats_table(stats: List[ScenarioStats]) -> str:
    """Render a per-scenario W/L/D + avg_turns table as plain text.

    Designed for direct ``print()`` in notebooks. Columns: scenario name,
    demonstrator vs opponent, games, demonstrator W/L/D, action-loop
    timeouts (T), win-rate %, average game length in turns, average
    demos collected per game.

    The T (timeout) column distinguishes action-loop timeouts -- where
    the per-episode step budget was exhausted while both bots remained
    in-turn -- from genuine map-cap draws (counted in D). Without this
    split, bot-stall failures would silently inflate the draw count
    and look like a property of the scenario rather than a bug.
    """
    if not stats:
        return "(no scenario stats collected)"

    header = (
        f"{'scenario':<32s}  {'matchup':<28s}  {'games':>5s}  "
        f"{'W':>4s} {'L':>4s} {'D':>4s} {'T':>4s}  {'WR':>6s}  {'turns':>6s}  {'demos/g':>8s}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    tot_w = tot_l = tot_d = tot_t = tot_demos = 0
    tot_turns = 0
    for s in stats:
        matchup = f"{s.demonstrator} vs {s.opponent}"
        lines.append(
            f"{s.name:<32s}  {matchup:<28s}  {s.total_games:>5d}  "
            f"{s.demo_wins:>4d} {s.demo_losses:>4d} {s.draws:>4d} {s.step_budget_exhausted:>4d}  "
            f"{100.0 * s.demo_win_rate:>5.1f}%  {s.avg_turns:>6.1f}  "
            f"{s.avg_demos_per_game:>8.1f}"
        )
        tot_w += s.demo_wins
        tot_l += s.demo_losses
        tot_d += s.draws
        tot_t += s.step_budget_exhausted
        tot_demos += s.total_demos
        tot_turns += s.total_turns
    tot_games = tot_w + tot_l + tot_d + tot_t
    tot_wr = (tot_w / tot_games) if tot_games else 0.0
    tot_avg_turns = (tot_turns / tot_games) if tot_games else 0.0
    tot_avg_demos = (tot_demos / tot_games) if tot_games else 0.0
    lines.append(sep)
    lines.append(
        f"{'TOTAL':<32s}  {'':<28s}  {tot_games:>5d}  "
        f"{tot_w:>4d} {tot_l:>4d} {tot_d:>4d} {tot_t:>4d}  "
        f"{100.0 * tot_wr:>5.1f}%  {tot_avg_turns:>6.1f}  {tot_avg_demos:>8.1f}"
    )
    return "\n".join(lines)


def load_scenarios_from_yaml(path: str) -> List[DemonstrationScenario]:
    """Parse a YAML file into a list of ``DemonstrationScenario``.

    Schema:
        scenarios:
          - name: <str>
            map_file: <path|null>
            enabled_units: [<unit codes>]
            demonstrator: <bot name>
            opponent: <bot name>
            n_episodes: <int>
            max_turns: <int>
            fog_of_war: <bool>
            demonstrator_player: <1|2>
            weight: <float>

    Unknown keys raise ``ValueError`` so typos surface immediately.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML is required to load scenario configs") from exc

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw_list = cfg.get("scenarios", [])
    if not isinstance(raw_list, list) or not raw_list:
        raise ValueError(f"{path}: top-level 'scenarios' must be a non-empty list")

    valid_keys = {f for f in DemonstrationScenario.__dataclass_fields__}
    scenarios: List[DemonstrationScenario] = []
    for i, entry in enumerate(raw_list):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: scenarios[{i}] must be a mapping")
        unknown = set(entry) - valid_keys
        if unknown:
            raise ValueError(f"{path}: scenarios[{i}] has unknown keys: {sorted(unknown)}")
        scenarios.append(DemonstrationScenario(**entry))
    return scenarios


# ---------------------------------------------------------------------------
# Behavior cloning
# ---------------------------------------------------------------------------


# Order matches ``_per_dim_sizes`` and the MaskablePPO MultiDiscrete head
# split. Kept as a module-level constant so logging / serialization /
# plotting agree on the dim labels without each call site re-naming them.
PER_DIM_NAMES: Tuple[str, ...] = ("at", "ut", "fx", "fy", "tx", "ty")


@dataclass
class BCStats:
    """Per-epoch BC training metrics.

    The legacy fields (``loss``, ``accuracy_action_type``, ``accuracy_full``)
    remain so older notebooks / plotters keep working. The new fields surface
    per-dimension diagnostics under the *un-narrowed* env masks -- the loss
    view (narrowed masks) gives trivially-perfect log_prob on placeholder
    dims, which hides whether the per-dim heads are actually learning the
    decision distribution.

    Per-dim metrics are reported in :data:`PER_DIM_NAMES` order
    (action_type, unit_type, from_x, from_y, to_x, to_y). ``per_dim_loss``
    is mean masked NLL per dim; ``per_dim_accuracy`` is the greedy-argmax
    match-rate per dim. ``accuracy_full_env_mask`` is the joint greedy-
    argmax match-rate computed under env masks -- the honest analogue to
    ``accuracy_full`` (which is inflated by the loss-view narrowing).
    """

    epoch: int
    loss: float
    accuracy_action_type: float
    accuracy_full: float
    # Per-dim metrics computed under the un-narrowed env-style masks.
    # Empty tuple when ``DemonstrationDataset.env_masks_concat`` is None
    # (older datasets serialized before the env-mask field existed).
    per_dim_loss: Tuple[float, ...] = field(default_factory=tuple)
    per_dim_accuracy: Tuple[float, ...] = field(default_factory=tuple)
    # Honest joint accuracy: greedy argmax of every head under the env
    # mask matches the recorded action. ``accuracy_full`` above is the
    # legacy (narrowed-mask) view, which trivially scores 100% on
    # placeholder dims and so over-reports joint quality. 0.0 when the
    # dataset has no env_masks_concat.
    accuracy_full_env_mask: float = 0.0


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
    end_turn_weight: Optional[float] = None,
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
        end_turn_weight: Per-sample loss weight applied to demonstrations
            whose action_type is ``end_turn`` (index 5). ``None`` (default)
            auto-computes ``n_non_end / n_end`` so the aggregate gradient
            contribution from end_turn samples matches that of all other
            classes combined -- corrects the ~10:1 demonstration imbalance
            that otherwise drives the policy into the never-end-turn
            attractor. Pass ``1.0`` to disable weighting. ``0.0`` would
            drop end_turn entirely (don't).

    Returns:
        Per-epoch ``BCStats`` list. The final entry is also useful as a
        cheap regression sanity check in tests.
    """
    import torch as th  # local import: torch is heavy and only needed here
    import torch.nn.functional as F  # noqa: N812  (canonical alias)

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

    # Auto-balance the end_turn class when no explicit weight was passed.
    # The ratio is computed once over the whole dataset (not per-batch) so
    # mini-batches that happen to contain zero end_turn samples don't
    # destabilise the gradient scale.
    action_types = dataset.actions[:, 0]
    n_end = int((action_types == END_TURN_ACTION_IDX).sum())
    n_non_end = int(action_types.shape[0] - n_end)
    if end_turn_weight is None:
        resolved_end_turn_weight = float(n_non_end) / max(n_end, 1) if n_end > 0 else 1.0
    else:
        resolved_end_turn_weight = float(end_turn_weight)
    logger.info(
        "BC class balance: %d end_turn / %d non_end demos -> end_turn_weight=%.3f",
        n_end,
        n_non_end,
        resolved_end_turn_weight,
    )

    # Pre-convert obs to torch tensors per epoch's batches lazily.
    def _to_tensor(arr: np.ndarray) -> th.Tensor:
        return th.as_tensor(arr, device=device)

    n_dims = len(dataset.dim_sizes) if dataset.dim_sizes else 6
    # Cumulative offsets for in-place per-dim slicing of the concatenated
    # masks. Computed once to avoid repeating the cumsum on every batch.
    dim_splits = list(dataset.dim_sizes) if dataset.dim_sizes else [NUM_ACTION_TYPES, NUM_UNIT_TYPES, 0, 0, 0, 0]
    # Whether the dataset carries the un-narrowed env-style masks needed
    # for the diagnostic metrics. Older datasets do not; the diagnostic
    # fields stay zero in that case (legacy ``accuracy_full`` is still
    # populated for backwards compatibility).
    has_env_masks = dataset.env_masks_concat is not None
    if not has_env_masks:
        logger.warning(
            "BC dataset has no env_masks_concat; per-dim diagnostics will be "
            "empty and accuracy_full uses the narrowed mask (over-reports). "
            "Re-collect demos with the current imitation.py to populate the "
            "env-mask view."
        )

    # Probe whether the policy exposes the standard ActorCriticPolicy
    # forward path (extract_features -> mlp_extractor -> action_net). All
    # MaskablePPO policies in this codebase do; the guard exists so a
    # custom policy lacking these attributes cleanly skips per-dim metrics
    # rather than raising during training.
    can_compute_per_dim = all(hasattr(policy, attr) for attr in ("extract_features", "mlp_extractor", "action_net"))

    stats: List[BCStats] = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_correct_atype = 0
        epoch_correct_full = 0
        epoch_count = 0
        # Per-dim aggregators (NLL sum and greedy-correct count under the
        # un-narrowed env mask, per dim). Length matches ``n_dims``.
        per_dim_nll_sum = [0.0] * n_dims
        per_dim_correct = [0] * n_dims
        epoch_correct_full_env = 0
        epoch_env_mask_count = 0

        for batch_idx in _iter_minibatches(dataset, batch_size, rng):
            obs_batch = {k: _to_tensor(v[batch_idx]) for k, v in dataset.obs.items()}
            actions = _to_tensor(dataset.actions[batch_idx]).long()
            masks = _to_tensor(dataset.masks_concat[batch_idx])

            _values, log_prob, _entropy = policy.evaluate_actions(obs_batch, actions, action_masks=masks)
            # Weighted NLL: end_turn samples get ``resolved_end_turn_weight``,
            # everything else stays at 1.0. Normalised by batch size N (not
            # ``sum_w``) so each end_turn sample contributes exactly ``w``x
            # the gradient of a non-end_turn sample regardless of batch
            # composition. Dividing by ``sum_w`` (the prior formulation)
            # partially cancelled the upweighting: a batch that happened
            # to draw many end_turn samples had a larger denominator that
            # diluted the per-sample weight, and an all-end_turn batch
            # collapsed back to effective weight 1. ``.mean()`` preserves
            # the "end_turn samples weigh ``w``x" contract per batch.
            sample_w = th.ones_like(log_prob)
            sample_w = th.where(
                actions[:, 0] == END_TURN_ACTION_IDX,
                th.full_like(log_prob, resolved_end_turn_weight),
                sample_w,
            )
            loss = -(log_prob * sample_w).mean()

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(bc_params, max_norm=0.5)
            optimizer.step()

            with th.no_grad():
                batch_n = int(actions.shape[0])
                epoch_count += batch_n
                epoch_loss += float(loss.item()) * batch_n

                # Diagnostic forward pass: recover raw per-dim logits from
                # the policy and compute per-dim metrics under BOTH the
                # narrowed mask (matches the legacy accuracy fields) and
                # the un-narrowed env mask (the honest view).
                #
                # Replicating the forward avoids the GPU->CPU->numpy->GPU
                # round-trip ``policy.predict`` triggered (it serializes
                # the whole obs dict per batch). With ``no_grad`` and
                # eval mode this is ~free compared to the round-trip on
                # 25k-demo datasets.
                if not can_compute_per_dim:
                    continue
                policy.set_training_mode(False)
                try:
                    raw_logits = _policy_raw_action_logits(policy, obs_batch)
                finally:
                    policy.set_training_mode(True)

                per_dim_logits = th.split(raw_logits, dim_splits, dim=-1)

                # Greedy argmax under the narrowed (loss) mask -- matches
                # the legacy ``accuracy_full`` definition exactly. Kept for
                # backward compatibility with existing plots / JSON dumps.
                narrowed_per_dim = th.split(masks.float(), dim_splits, dim=-1)
                greedy_narrowed = []
                for logits_i, mask_i in zip(per_dim_logits, narrowed_per_dim):
                    masked_i = logits_i.masked_fill(~mask_i.bool(), -1e8)
                    greedy_narrowed.append(masked_i.argmax(dim=-1))
                greedy_narrowed_stack = th.stack(greedy_narrowed, dim=-1)
                epoch_correct_atype += int((greedy_narrowed_stack[:, 0] == actions[:, 0]).sum().item())
                epoch_correct_full += int((greedy_narrowed_stack == actions).all(dim=-1).sum().item())

                if has_env_masks:
                    env_masks_batch = _to_tensor(dataset.env_masks_concat[batch_idx])
                    env_per_dim = th.split(env_masks_batch.float(), dim_splits, dim=-1)
                    greedy_env = []
                    for i, (logits_i, mask_i) in enumerate(zip(per_dim_logits, env_per_dim)):
                        mask_bool = mask_i.bool()
                        masked_i = logits_i.masked_fill(~mask_bool, -1e8)
                        log_probs_i = F.log_softmax(masked_i, dim=-1)
                        true_i = actions[:, i]
                        # Per-dim NLL under the env mask -- this is what
                        # surfaces "the policy can't tell legal value
                        # apart in this dim" failure modes (e.g. fx/fy
                        # never converge for non-create actions because
                        # the loss view narrows them to one-hot).
                        nll_i = -log_probs_i.gather(-1, true_i.unsqueeze(-1)).squeeze(-1)
                        per_dim_nll_sum[i] += float(nll_i.sum().item())
                        greedy_i = masked_i.argmax(dim=-1)
                        per_dim_correct[i] += int((greedy_i == true_i).sum().item())
                        greedy_env.append(greedy_i)
                    greedy_env_stack = th.stack(greedy_env, dim=-1)
                    epoch_correct_full_env += int((greedy_env_stack == actions).all(dim=-1).sum().item())
                    epoch_env_mask_count += batch_n

        denom = max(epoch_count, 1)
        env_denom = max(epoch_env_mask_count, 1)
        per_dim_loss = (
            tuple(s / env_denom for s in per_dim_nll_sum) if has_env_masks else ()
        )
        per_dim_accuracy = (
            tuple(c / env_denom for c in per_dim_correct) if has_env_masks else ()
        )
        accuracy_full_env_mask = (epoch_correct_full_env / env_denom) if has_env_masks else 0.0
        stat = BCStats(
            epoch=epoch + 1,
            loss=epoch_loss / denom,
            accuracy_action_type=epoch_correct_atype / denom,
            accuracy_full=epoch_correct_full / denom,
            per_dim_loss=per_dim_loss,
            per_dim_accuracy=per_dim_accuracy,
            accuracy_full_env_mask=accuracy_full_env_mask,
        )
        stats.append(stat)
        if (epoch + 1) % log_every == 0:
            if has_env_masks:
                # Per-dim diagnostics in the order PER_DIM_NAMES advertises.
                # The headline ``full_acc`` is the legacy (narrowed) view;
                # ``full_env`` is the honest one. Both printed because the
                # gap is itself diagnostic (large gap = placeholder dims
                # carrying most of the legacy "match").
                per_dim_str = "  ".join(
                    f"{name}_loss={loss_val:.3f}/acc={acc_val:.2f}"
                    for name, loss_val, acc_val in zip(PER_DIM_NAMES, per_dim_loss, per_dim_accuracy)
                )
                logger.info(
                    "BC epoch %d/%d  loss=%.4f  atype_acc=%.3f  full_acc=%.3f  "
                    "full_env=%.3f  (n=%d)\n              per-dim: %s",
                    stat.epoch,
                    n_epochs,
                    stat.loss,
                    stat.accuracy_action_type,
                    stat.accuracy_full,
                    stat.accuracy_full_env_mask,
                    epoch_count,
                    per_dim_str,
                )
            else:
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


def _policy_raw_action_logits(policy: Any, obs_batch: Dict[str, Any]) -> Any:
    """Recover the raw (pre-mask) per-head logit tensor from an SB3 policy.

    Replicates ``ActorCriticPolicy.evaluate_actions``'s forward up to
    ``action_net`` so the caller can split per-dim logits without going
    through the (slow) ``policy.predict`` round-trip. Returns a tensor
    of shape ``(B, sum(dim_sizes))``.

    Handles both shared and split features extractors. Centralised so
    when SB3 reshuffles internals there's a single site to update.
    """
    features = policy.extract_features(obs_batch)
    if isinstance(features, tuple):
        # share_features_extractor=False: (pi_features, vf_features).
        pi_features = features[0]
    else:
        pi_features = features
    latent_pi = policy.mlp_extractor.forward_actor(pi_features)
    return policy.action_net(latent_pi)


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
    scenarios: Optional[List[DemonstrationScenario]] = None,
    end_turn_weight: Optional[float] = None,
    stochastic_tiebreak: bool = False,
) -> Tuple[Any, DemonstrationDataset, List[BCStats]]:
    """Build a MaskablePPO model and warm-start it via behavior cloning.

    The ``env`` argument is the environment MaskablePPO will be trained on
    after this call (single env or vec env). The demonstration generation is
    independent and uses fresh GameState instances configured to mirror the
    env (same map / unit roster / fog-of-war flag), so the BC observations
    have the same shape and semantics as what the policy will see during
    PPO fine-tuning.

    When ``scenarios`` is provided, the single-source ``demonstrator`` /
    ``opponent`` / ``map_file`` / ``enabled_units`` / ``max_turns`` /
    ``fog_of_war`` / ``stochastic_tiebreak`` arguments are ignored and
    demonstrations are collected via :func:`collect_demonstrations_multi`
    instead -- each scenario's own ``stochastic_tiebreak`` field then
    controls the behaviour per scenario.

    ``stochastic_tiebreak`` only applies to the single-source path. When
    True, the deterministic-bot demonstrator/opponent receive a
    per-episode rng so equal-scoring decisions resolve randomly --
    required to get distinct trajectories from an all-deterministic
    matchup (e.g. ``demonstrator='advanced', opponent='advanced'``)
    which otherwise produces N byte-identical copies of one game.

    Returns:
        Tuple of (model, dataset, bc_stats). Call ``model.learn(...)`` next.
    """
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise ImportError("sb3-contrib is required for imitation warm-start. Install with: pip install sb3-contrib") from exc

    if scenarios:
        dataset = collect_demonstrations_multi(scenarios, seed=seed, progress=True)
        logger.info("Collected %d demonstrations across %d scenarios", len(dataset), len(scenarios))
    else:
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
            stochastic_tiebreak=stochastic_tiebreak,
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
        end_turn_weight=end_turn_weight,
    )

    return model, dataset, bc_stats


# ---------------------------------------------------------------------------
# Post-training BC evaluation
#
# Quick standalone eval of a BC checkpoint against the bot ladder
# (simple / medium / advanced) using the curriculum's first-stage env
# template. Run this immediately after the BC build to decouple
# "BC quality" from "PPO can't recover from a bad BC start" -- without
# it, the only signal on BC quality is the supervised training metrics
# (loss / action_type_acc / full_action_acc) which proxy poorly for
# actual gameplay performance.
#
# Critically uses :func:`reinforcetactics.rl.bootstrap.make_stage_env`
# so the eval env matches the production curriculum env down to
# ``reward_config`` and ``max_actions_per_turn``. Lifted from the
# notebook (section 3e) precisely because the manual env construction
# there hit a debugging trap when those kwargs were silently omitted
# (run 20260524_225835: WR=0% / reward=-30,169 was a measurement
# artifact, not BC failure). Centralising the env construction kills
# the bug class permanently.
# ---------------------------------------------------------------------------


def evaluate_bc_against_bot_ladder(
    model: Any,
    cfg: Any,
    *,
    n_episodes: int = 30,
    opponents: Tuple[str, ...] = ("simple", "medium", "advanced"),
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate a BC checkpoint against scripted bots on the first stage's env.

    Designed to be called right after BC training to surface the BC
    policy's actual gameplay WR before committing to a multi-hour PPO
    run. Uses :func:`bootstrap.make_stage_env` so the eval env matches
    the production curriculum env exactly -- omitting kwargs like
    ``reward_config`` and ``max_actions_per_turn`` (the bug we hit in
    run 20260524_225835) is structurally impossible through this path.

    Args:
        model: Trained BC ``MaskablePPO`` instance.
        cfg: The :class:`TrainingConfig` whose ``curriculum.stages[0]``
            and ``env`` define the eval env template.
        n_episodes: Episodes per opponent. 30 is a reasonable default
            (~1-2 min per opponent on skirmish; short games against
            random/balanced_random, longer against medium/advanced).
        opponents: Bot ladder to evaluate against. Default
            ``(simple, medium, advanced)`` covers the canonical "is BC
            broken? where's the ceiling? where's the gap?" questions.
            Adding ``random`` / ``balanced_random`` is sometimes useful
            for diagnosing exploration vs strategy issues.
        seed: Seed for env construction. ``None`` defaults to
            ``cfg.seed + 7777`` (matches the notebook's convention --
            a fresh offset distinct from the training loop's eval
            seed so the BC sanity eval isn't a deterministic copy of
            an in-training eval episode).
        deterministic: Pass-through to :func:`evaluate_model`. Default
            ``True`` matches eval-time PPO behaviour.

    Returns:
        Dict mapping opponent name to the full metrics dict from
        :func:`evaluate_model` (``win_rate``, ``avg_reward``,
        ``avg_length``, ``avg_turns``, ``wins`` / ``losses`` /
        ``draws``, ``end_reasons``, etc.). Suitable for direct
        serialisation alongside other BC artifacts.

    Example:
        >>> results = evaluate_bc_against_bot_ladder(bc_model, cfg)
        >>> results["simple"]["win_rate"]  # > 0.4 means BC is at least
        ...                                #   competent vs trivial opponent
    """
    from reinforcetactics.rl.bootstrap import make_stage_env
    from reinforcetactics.rl.evaluation import evaluate_model

    if not cfg.curriculum.stages:
        raise ValueError("cfg.curriculum.stages is empty; no stage to template the eval env on")
    first_stage = cfg.curriculum.stages[0]

    eval_seed = seed if seed is not None else cfg.seed + 7777

    results: Dict[str, Dict[str, Any]] = {}
    for opp in opponents:
        env = make_stage_env(first_stage, cfg.env, opponent=opp, seed=eval_seed)
        try:
            metrics = evaluate_model(
                model,
                env,
                n_episodes=n_episodes,
                deterministic=deterministic,
                seed=eval_seed,
            )
        finally:
            env.close()
        results[opp] = metrics
    return results
