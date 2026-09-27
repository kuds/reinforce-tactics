"""What ``StrategyGameEnv`` accepts for its opponent and its reward weights.

The env's input contract, in one small module the config layer can check a
training config against without building an env: the reward keys the env
reads (:data:`KNOWN_REWARD_KEYS`) with their single set of defaults, and the
opponent names it plays (derived from the bot registry). It depends only on
the bot registry; ``gym_env`` re-exports every name here.
"""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from reinforcetactics.game.bot_registry import accepted_names as accepted_bot_names
from reinforcetactics.game.bot_registry import canonical_name as canonical_bot_name
from reinforcetactics.game.bot_registry import is_scripted_name, validate_scripted_kwargs

# ---------------------------------------------------------------------------
# Opponents
# ---------------------------------------------------------------------------

# ``StrategyGameEnv(opponent=...)`` (and ``opponent_type`` set
# before a reset) takes ``None`` (no opponent: the caller plays the other
# seat itself), ``"self"``, or any scripted-bot name the bot registry
# resolves (``bot_registry.accepted_names()``: every entry of SCRIPTED_BOTS,
# ``"master"`` and ``"noop"`` included, plus the historic ``"bot"`` alias for
# ``"simple"``). Anything else is a ValueError: an unknown string used to
# fall through to "no opponent" and train against nothing (review rlenv-10 /
# rulebots-7). ``"random"`` runs RandomBot with its default
# ``max_actions=20`` (more of a stress test than a weak baseline);
# ``"balanced_random"`` runs BalancedRandomBot, whose throughput scales with
# army size -- a lighter stepping stone between ``"noop"`` and ``"random"``.
# ``"self"``: the opponent (a snapshot of the agent under training) comes
# from ``set_self_play_opponent_factory`` -- registered by
# ``rl.self_play.SelfPlayEnv`` -- so reset() can rebind a fresh one to the
# new game_state every episode.
SELF_PLAY_OPPONENT = "self"


def accepted_opponents() -> tuple[str, ...]:
    """Every opponent string ``StrategyGameEnv`` accepts (``None`` is accepted too)."""
    return (SELF_PLAY_OPPONENT, *accepted_bot_names())


def resolve_opponent(opponent: str | None) -> str | None:
    """Validate an ``opponent`` value and return its canonical form.

    ``None`` -> ``None``, ``"self"`` -> ``"self"``, a scripted-bot name or
    alias -> its canonical registry name (``"bot"`` -> ``"simple"``).

    Raises:
        ValueError: Anything else, listing :func:`accepted_opponents`.
    """
    if opponent is None or opponent == SELF_PLAY_OPPONENT:
        return opponent
    if isinstance(opponent, str) and is_scripted_name(opponent):
        return canonical_bot_name(opponent)
    raise ValueError(
        f"Unknown opponent {opponent!r}. Expected None or one of: {', '.join(accepted_opponents())} "
        "(see reinforcetactics.game.bot_registry)."
    )


def validate_opponent_kwargs(opponent: str | None, opponent_kwargs: Mapping[str, Any] | None) -> None:
    """Check ``opponent_kwargs`` against the opponent they will be passed to.

    Scripted bots: the keys must be constructor parameters of that bot
    (``bot_registry.validate_scripted_kwargs``; MixedBot's values are
    checked too). ``None`` / ``"self"`` take no kwargs. Non-empty kwargs
    that no constructor would receive used to be dropped silently.

    Raises:
        ValueError: Unknown opponent, or kwargs the opponent does not take.
        TypeError: ``opponent_kwargs`` is not a mapping.
    """
    name = resolve_opponent(opponent)
    if name is None or name == SELF_PLAY_OPPONENT:
        if opponent_kwargs:
            raise ValueError(
                f"opponent_kwargs {sorted(opponent_kwargs)} given for opponent={opponent!r}, which takes none; "
                "they apply only to scripted opponents"
            )
        return
    validate_scripted_kwargs(name, opponent_kwargs)


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------

# Default reward weights. ``StrategyGameEnv(reward_config=...)`` overlays a
# subset of keys; every key in :data:`KNOWN_REWARD_KEYS` is accepted and any
# other key is a ValueError (review rlenv-16: a typo used to be merged in and
# never read). The env indexes these directly, so this dict is the only
# default for each weight.
#
# Weights are tuned so that capturing the enemy HQ dominates the alternative
# of farming kills against a respawning opponent. Old defaults made a single
# kill (+10) worth more than a turn of seize progress (+1), pushing the
# policy into a kill-farm local optimum that never finishes the game. Now:
# capture (+200) >> a full kill loop, and seize_progress (+5) > a typical
# attack hit.
DEFAULT_REWARD_CONFIG: dict[str, float] = {
    "win": 1000.0,
    "loss": -1000.0,
    "draw": -200.0,
    "income_diff": 0.05,
    "unit_diff": 0.3,
    "structure_control": 1.0,
    "invalid_action": -10.0,
    # ``turn_penalty`` defaults to 0.0. Earlier defaults charged a
    # per-end_turn cost to incentivize game progress, but that made
    # ``end_turn`` the only individually-negative-valued action and PPO
    # converged to a "never end the turn" attractor -- episodes truncated at
    # ``max_steps`` with very few game-turns elapsed and 0% win rate.
    # Per-turn pressure now lives in ``win_speed_bonus`` (terminal-only,
    # can't be dodged by stalling) and in gamma's natural discount.
    "turn_penalty": 0.0,
    # ``win_speed_bonus`` scales linearly with how many turns remained when
    # the agent won: at turn 1, a winning agent gets the full bonus; at
    # ``max_turns`` it gets 0. Stacks with the ``win`` / ``win_by_*``
    # terminals and only fires on the agent's own win. With ``max_turns``
    # unset the bonus is skipped (no horizon to normalize against). Default
    # 0.0 keeps old configs behaviour-identical; bootstrap.yaml sets the
    # active magnitude.
    "win_speed_bonus": 0.0,
    # Step-limit truncation charge (see ``step``). 0.0: SB3 bootstraps the
    # value of a truncated state, which already prices "unfinished".
    "truncation": 0.0,
    # Action rewards
    "create_unit": 0.5,
    "move": 0.0,
    # Reward per damage point the agent's units deal (nominal damage, as the
    # engine reports it): their own attacks, on the attack step, and their
    # counter-attacks when the opponent attacks them, on the end_turn step
    # (the opponent's turn plays out inside it).
    "damage_scale": 0.05,
    # Charge per HP the agent's units *lose* in combat (a negative
    # magnitude): to the opponent's attacks, on the end_turn step, and to
    # the counter-attack on the agent's own attack, on the attack step.
    # Measured as HP actually lost in both places (a unit with 1 HP left
    # loses 1, however hard the killing blow). Together with
    # ``damage_scale`` this makes combat shaping net-zero-sum whichever side
    # swings first: with ``damage_taken_scale = -damage_scale`` a mutual
    # trade nets ~0 and only decisive combat (dealing more than you take)
    # pays. Counters the kill/damage-farm draw attractor where two armies
    # trade blows to the max-turns clock while collecting only the
    # dealt-damage half of the exchange. Default 0.0 leaves legacy reward
    # shapes unchanged.
    "damage_taken_scale": 0.0,
    # Reward per enemy unit killed: by the agent's attack, or by its
    # counter-attack during the opponent's turn.
    "kill": 5.0,
    # Charge per agent unit lost (a negative magnitude; the mirror of
    # ``kill``): a counter-attack that kills the attacker, on the attack
    # step, and each unit the opponent kills, on the end_turn step. Default
    # 0.0 (off).
    "unit_lost": 0.0,
    "seize_progress": 5.0,
    "capture": 200.0,
    "cure": 5.0,
    "heal_scale": 0.5,  # reward per HP healed
    "paralyze": 8.0,
    "haste": 6.0,
    "defence_buff": 5.0,
    "attack_buff": 5.0,
    # Opponent-capture penalty. Fires once per capturable tile the opponent
    # seizes during their turn (tracked in the end_turn branch of
    # _execute_action). Tiered to encode two distinct behaviours: neutral
    # captures punish ignoring the capture race (the failure mode that
    # collapsed skirmish_simple), owned captures punish letting the opponent
    # take ground we already held. Defaults are 0.0 so old reward_configs
    # are unaffected; configs/ppo/bootstrap.yaml sets the active magnitudes.
    "enemy_neutral_capture": 0.0,
    "enemy_owned_capture": 0.0,
}

# Keys read only when present, so their absence means something: a per-type
# ``<type>_capture`` replaces ``capture`` for that structure type, and
# ``win_by_<end_reason>`` replaces ``win`` for a win that ended that way.
OPTIONAL_REWARD_KEYS: frozenset[str] = frozenset(
    {"tower_capture", "building_capture", "hq_capture", "win_by_hq_capture", "win_by_elimination"}
)

# Every reward_config key the env reads. The config layer validates YAML
# reward_config blocks against this set (``validate_reward_config``).
KNOWN_REWARD_KEYS: frozenset[str] = frozenset(DEFAULT_REWARD_CONFIG) | OPTIONAL_REWARD_KEYS


def validate_reward_config(reward_config: Mapping[str, Any] | None) -> None:
    """Check a reward_config overlay: known keys, finite real-number values.

    Raises:
        TypeError: Not a mapping, or a value that is not a real number
            (a YAML ``'3e-4'`` string or a bool, say).
        ValueError: A key outside :data:`KNOWN_REWARD_KEYS`, or a NaN / inf.
    """
    if reward_config is None:
        return
    if not isinstance(reward_config, Mapping):
        raise TypeError(f"reward_config must be a mapping, got {type(reward_config).__name__}")
    unknown = sorted(str(k) for k in reward_config if k not in KNOWN_REWARD_KEYS)
    if unknown:
        raise ValueError(f"Unknown reward_config keys {unknown}. Known keys: {', '.join(sorted(KNOWN_REWARD_KEYS))}")
    for key, value in reward_config.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"reward_config[{key!r}] must be a number, got {value!r} ({type(value).__name__})")
        if not math.isfinite(float(value)):
            raise ValueError(f"reward_config[{key!r}] must be finite, got {value!r}")
