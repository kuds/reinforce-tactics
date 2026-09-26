"""The resolved rule configuration of one game (review core-16).

``engine_overrides`` is a sparse overlay over the engine constants in
``rules.py`` (economy, unit stats, structure HP, the unit cap, the combat
model and the optional terrain and turn rules), so balance can be varied
and recorded as config instead of a code edit. ``EngineConfig`` is that
overlay validated and resolved into the tables the engine reads, built once
per game by ``EngineConfig.from_overrides``. It used to be spread over
GameState's ``_resolve_*`` helpers and ``__init__``; on its own it can be
built, compared and tested without a map or a GameState.

The config is fixed for the whole game, which is what lets
``GameState.clone_for_search`` share one instance between a game and all
its search clones instead of copying it. ``frozen`` stops its fields being
rebound; the tables themselves stay plain dicts (the engine hands
``unit_data[code]`` to every ``Unit`` as its stats, and callers have always
read them as dicts), which the engine never mutates.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from reinforcetactics.core.terrain_rules import TERRAIN_RULE_KEYS, TerrainRules
from reinforcetactics.rules import (
    BUILDING_INCOME,
    HEADQUARTERS_INCOME,
    MAX_UNITS_PER_PLAYER,
    STARTING_GOLD,
    TOWER_INCOME,
    UNIT_DATA,
)

if TYPE_CHECKING:
    from reinforcetactics.core.grid import TileGrid

# Every engine_overrides key some resolver reads. An unknown key is
# rejected: a misspelt rule (``forest_concealement: true``) would otherwise
# silently play the default game. New override keys must be added here.
ENGINE_OVERRIDE_KEYS = frozenset(
    {
        "starting_gold",
        "headquarters_income",
        "building_income",
        "tower_income",
        "tower_health",
        "building_health",
        "headquarters_health",
        "damage_model",
        "max_units_per_player",
        "unit_data",
        "begin_first_turn",
        "legacy_end_rules",
        *TERRAIN_RULE_KEYS,
    }
)

# Override key -> structure tile-type code. Lets a balance sweep tune
# capture difficulty (e.g. ``headquarters_health: 30`` halves a Warrior's
# HQ-capture time) from the config surface instead of editing rules.py.
STRUCTURE_HEALTH_KEYS = {
    "tower_health": "t",
    "building_health": "b",
    "headquarters_health": "h",
}

DAMAGE_MODELS = ("flat", "hp_scaled")


@dataclass(frozen=True)
class EngineConfig:
    """One game's rules: its ``engine_overrides`` overlay, resolved over ``rules.py``.

    Build it with :meth:`from_overrides`; :meth:`to_overrides` gives back
    the overlay saves and replays record. GameState exposes every field
    under its old attribute name (``gs.unit_data``, ``gs.starting_gold``,
    ...), read-only.
    """

    # The overlay as the game was given it (a shallow copy), which is what
    # saves and replays write: recording it verbatim, rather than something
    # rebuilt from the resolved fields below, keeps those files unchanged.
    overrides: dict[str, Any]
    # Full per-unit stat table: a deep copy of rules.UNIT_DATA taken when
    # the game is created, with the overlay's per-field deltas applied.
    unit_data: dict[str, dict[str, Any]]
    # Gold per turn by structure: {"headquarters", "building", "tower"}.
    income_rates: dict[str, int]
    starting_gold: int
    # "flat" (HP-independent damage) or "hp_scaled" (see _resolve_damage_model).
    damage_model: str
    # Structure max-HP overrides, {tile_code: hp}; only overridden codes appear.
    structure_health: dict[str, int]
    # Hard ceiling on units per player (action-space and economy guardrail),
    # enforced by both create_unit and the legal actions so the cap shows
    # up in the action mask, not just as a rejected action.
    max_units_per_player: int
    # Movement costs, path-based Knight Charge, forest concealment, HQ
    # always known: all off by default (see core/terrain_rules.py).
    terrain_rules: TerrainRules
    # Player 1's turn-0 start-of-turn processing (see _resolve_begin_first_turn).
    begin_first_turn: bool
    # The pre-2026-09 end rules, for old replays (see _resolve_legacy_end_rules).
    legacy_end_rules: bool

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, Any] | None = None) -> EngineConfig:
        """Validate an ``engine_overrides`` overlay and resolve every rule it sets.

        Every key is optional; an absent key keeps the ``rules.py`` value,
        so ``None`` / ``{}`` is the game as shipped. Bad input fails loud
        (``KeyError`` for an unknown key, unit code or tile code,
        ``ValueError`` for a bad value): a typo in a balance sweep must not
        silently train on the default rules. ``rules.UNIT_DATA`` is read
        here, at game creation, so an in-place edit of it (as the balance
        notebook makes) reaches every game created afterwards, and a game's
        own copy never reaches the module table.
        """
        overlay: dict[str, Any] = dict(overrides) if overrides else {}
        # The resolution order fixes which error an overlay with several
        # problems reports; it is the order GameState always resolved them in.
        unit_data, income_rates, starting_gold = _resolve_economy(overlay)
        damage_model = _resolve_damage_model(overlay)
        structure_health = _resolve_structure_health(overlay)
        max_units_per_player = _resolve_max_units_per_player(overlay)
        terrain_rules = TerrainRules.from_overrides(overlay)
        return cls(
            overrides=overlay,
            unit_data=unit_data,
            income_rates=income_rates,
            starting_gold=starting_gold,
            damage_model=damage_model,
            structure_health=structure_health,
            max_units_per_player=max_units_per_player,
            terrain_rules=terrain_rules,
            begin_first_turn=_resolve_begin_first_turn(overlay),
            legacy_end_rules=_resolve_legacy_end_rules(overlay),
        )

    def to_overrides(self) -> dict[str, Any]:
        """The sparse overlay this config was built from, as saves record it.

        A deep copy, so the caller may keep or edit it; ``from_overrides``
        of the result equals this config.
        """
        return copy.deepcopy(self.overrides)

    def apply_structure_health(self, grid: TileGrid) -> None:
        """Overlay the structure max-HP overrides onto a freshly-built grid.

        ``TileGrid`` constructs structure tiles at the ``rules.py`` HP, so
        this runs right after grid creation while every structure is at full
        health -- setting both ``max_health`` and ``health`` keeps the tile
        consistent (regen scales off ``max_health``; capture resets to it).
        """
        if not self.structure_health:
            return
        for row in grid.tiles:
            for tile in row:
                override_hp = self.structure_health.get(tile.type)
                if override_hp is not None and tile.is_capturable():
                    tile.max_health = override_hp
                    tile.health = override_hp


def _resolve_economy(overrides: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, int], int]:
    """Merge the overlay's economy and unit-stat keys over the module constants.

    Returns ``(unit_data, income_rates, starting_gold)`` fully resolved.
    ``unit_data`` is a deep copy of :data:`UNIT_DATA` with per-unit,
    per-field deltas applied (so the shared module dict is never mutated).
    Also rejects unknown top-level keys. Unknown unit codes / stat fields
    raise ``KeyError`` / ``ValueError`` early -- a typo in a balance sweep
    should fail loud, not silently train on the wrong stats.
    """
    unit_data = copy.deepcopy(UNIT_DATA)
    income_rates = {
        "headquarters": HEADQUARTERS_INCOME,
        "building": BUILDING_INCOME,
        "tower": TOWER_INCOME,
    }
    starting_gold = STARTING_GOLD
    if not overrides:
        return unit_data, income_rates, starting_gold
    unknown = set(overrides) - ENGINE_OVERRIDE_KEYS
    if unknown:
        raise KeyError(f"engine_overrides: unknown key(s) {sorted(unknown)} (valid: {sorted(ENGINE_OVERRIDE_KEYS)})")

    if "starting_gold" in overrides:
        starting_gold = int(overrides["starting_gold"])
    for ov_key, rate_key in (
        ("headquarters_income", "headquarters"),
        ("building_income", "building"),
        ("tower_income", "tower"),
    ):
        if ov_key in overrides:
            income_rates[rate_key] = int(overrides[ov_key])

    unit_overrides = overrides.get("unit_data") or {}
    for code, fields in unit_overrides.items():
        if code not in unit_data:
            raise KeyError(f"engine_overrides.unit_data: unknown unit code '{code}'")
        for field, value in fields.items():
            if field not in unit_data[code]:
                raise ValueError(
                    f"engine_overrides.unit_data['{code}']: unknown stat field '{field}' (valid: {sorted(unit_data[code])})"
                )
            unit_data[code][field] = value
    return unit_data, income_rates, starting_gold


def _resolve_max_units_per_player(overrides: dict[str, Any]) -> int:
    """Resolve the per-player unit cap from the overlay.

    Defaults to :data:`MAX_UNITS_PER_PLAYER`. A positive int is required
    -- a cap <= 0 would forbid all unit creation, which is never the
    intent and should fail loud rather than silently soft-lock a game.

    The cap is a *creation gate*, not a retroactive trim: it blocks new
    ``create_unit`` calls once a player is at the cap but never removes
    existing units, so a scenario that starts a side at or above the cap
    (or a sweep that sets the cap below the starting army) simply can't
    grow until attrition drops the count. It is therefore a soft ceiling
    on growth, not a hard guarantee of ``<= cap`` units at every instant.
    """
    if "max_units_per_player" not in overrides:
        return MAX_UNITS_PER_PLAYER
    val = int(overrides["max_units_per_player"])
    if val <= 0:
        raise ValueError(f"engine_overrides.max_units_per_player must be a positive int, got {val}")
    return val


def _resolve_damage_model(overrides: dict[str, Any]) -> str:
    """Resolve the combat damage model from the overlay.

    ``"flat"`` (default) reproduces legacy HP-independent damage.
    ``"hp_scaled"`` multiplies outgoing damage by the attacker's current
    HP fraction (decisive combat; consistent with seize, which is already
    HP-scaled). An unknown value fails loud rather than silently training
    on an unintended combat model.
    """
    model = overrides.get("damage_model", "flat")
    if model not in DAMAGE_MODELS:
        raise ValueError(f"engine_overrides.damage_model must be 'flat' or 'hp_scaled', got {model!r}")
    return model


def _resolve_structure_health(overrides: dict[str, Any]) -> dict[str, int]:
    """Resolve per-structure max-HP overrides into ``{tile_code: hp}``.

    Only keys present in the overlay appear in the result; absent
    structures keep their ``rules.py`` defaults. Non-positive values fail
    loud (a structure with <=0 HP would be captured on the first seize /
    be nonsensical for regen).
    """
    resolved: dict[str, int] = {}
    for ov_key, code in STRUCTURE_HEALTH_KEYS.items():
        if ov_key in overrides:
            val = int(overrides[ov_key])
            if val <= 0:
                raise ValueError(f"engine_overrides.{ov_key} must be a positive int, got {val}")
            resolved[code] = val
    return resolved


def _resolve_begin_first_turn(overrides: dict[str, Any]) -> bool:
    """Resolve ``begin_first_turn`` from the overlay.

    Start-of-turn processing (income, structure healing, status and
    cooldown ticks, the visibility update; see ``GameState._begin_turn``)
    runs in ``end_turn`` for the player whose turn is starting, so Player
    1's very first turn never got it: it plays turn 0 on its starting gold
    alone, while every later turn -- Player 2's first one included --
    collects income first. ``False`` (default) keeps that schedule.
    ``True`` runs ``_begin_turn(1)`` when the game is created, so Player 1
    also collects income before its first move. Recorded with the rest of
    ``engine_overrides`` (saves and replay ``game_info``) so a balance
    sweep can toggle it and replays reproduce it. Non-bool values fail
    loud: ``"false"`` would otherwise read as true.
    """
    return _resolve_bool_override(overrides, "begin_first_turn")


def _resolve_legacy_end_rules(overrides: dict[str, Any]) -> bool:
    """Resolve ``legacy_end_rules`` from the overlay.

    ``True`` plays by the end rules of games recorded before September
    2026 (review core-7), so their replays play back as they were played:
    any HQ capture wins the game for the capturer, whatever the seat
    count; nobody is eliminated, so a seat that lost its units or resigned
    keeps its turns, income and structures; and with three or more seats
    the game ends only when a single player has units left.
    ``replay_actions.replay_game_state_kwargs`` sets it for those replays;
    nothing else should.
    """
    return _resolve_bool_override(overrides, "legacy_end_rules")


def _resolve_bool_override(overrides: dict[str, Any], key: str) -> bool:
    """A bool engine override, default False. Non-bools fail loud: ``"false"`` would read as true."""
    value = overrides.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"engine_overrides.{key} must be a bool, got {value!r}")
    return value
