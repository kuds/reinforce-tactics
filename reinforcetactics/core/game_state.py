"""
Core game state management without rendering dependencies.
Fixed version: removed duplicate methods, added type hints, controlled logging.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import logging
import os
import random
import struct
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from reinforcetactics.core.grid import TileGrid
from reinforcetactics.core.mechanics import GameMechanics, same_side
from reinforcetactics.core.terrain_rules import TERRAIN_RULE_KEYS, TerrainRules
from reinforcetactics.core.unit import Unit
from reinforcetactics.core.visibility import (
    UNEXPLORED,
    VISIBLE,
    StructureSnapshot,
    VisibilityMap,
    get_visible_units,
)
from reinforcetactics.rules import (
    ALL_UNIT_TYPES,
    BUILDING_INCOME,
    HEADQUARTERS_INCOME,
    MAX_UNITS_PER_PLAYER,
    STARTING_GOLD,
    TOWER_INCOME,
    UNIT_DATA,
    TileType,
)

# Debug mode: with RT_CHECK_CACHE=1, every legal-action cache hit is
# recomputed and compared, so a mutator that forgets to invalidate fails
# loudly instead of handing bots and masks a stale action set.
_CHECK_LEGAL_ACTION_CACHE = os.environ.get("RT_CHECK_CACHE") == "1"

# Version of the save format written by ``GameState.to_dict``. Saves without
# the field are version 1 (everything before it existed) and still load.
# 2: adds the fields ``from_dict`` needs to resume a game exactly
# (winning_action_index, healing_totals, per-unit has_moved, fog-of-war
# attack snapshot and ambushed flag, the fog-of-war state). Early version 2
# saves also carry padding metadata (original_map_width/height,
# map_padding_offset_x/y, original_map_data); nothing ever set it, so it
# always described the saved grid itself, and ``from_dict`` ignores it
# (with a warning should the offsets not be zero). Later additions, optional
# on load: per-unit haste_refreshed and the fog-of-war pre-move view
# cancel_move restores.
SAVE_FORMAT_VERSION = 2

# Unit attribute types a search clone can share with the original unit.
_IMMUTABLE_UNIT_FIELD_TYPES = (type(None), bool, int, float, str, tuple, frozenset)

# Configure logging
logger = logging.getLogger(__name__)


def derive_seed(*parts: object) -> int:
    """A stable 63-bit seed derived from ``parts`` (e.g. a run seed and a game id).

    SHA-256 rather than ``hash()``, which is salted per process and would
    give a different seed on every run.
    """
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") >> 1


class GameState:
    """Manages the core game state without rendering."""

    ALL_UNIT_TYPES = ALL_UNIT_TYPES

    # Every engine_overrides key some resolver reads. An unknown key is
    # rejected: a misspelt rule (``forest_concealement: true``) would
    # otherwise silently play the default game. New override keys must be
    # added here.
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

    @staticmethod
    def _resolve_engine_overrides(
        overrides: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, int], int]:
        """Merge a sparse override overlay over the module engine constants.

        Returns ``(unit_data, income_rates, starting_gold)`` fully resolved.
        ``unit_data`` is a deep copy of :data:`UNIT_DATA` with per-unit,
        per-field deltas applied (so the shared module dict is never
        mutated). Unknown unit codes / stat fields raise ``KeyError`` /
        ``ValueError`` early -- a typo in a balance sweep should fail loud,
        not silently train on the wrong stats.
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
        unknown = set(overrides) - GameState.ENGINE_OVERRIDE_KEYS
        if unknown:
            raise KeyError(
                f"engine_overrides: unknown key(s) {sorted(unknown)} (valid: {sorted(GameState.ENGINE_OVERRIDE_KEYS)})"
            )

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
                        f"engine_overrides.unit_data['{code}']: unknown stat field "
                        f"'{field}' (valid: {sorted(unit_data[code])})"
                    )
                unit_data[code][field] = value
        return unit_data, income_rates, starting_gold

    @staticmethod
    def _resolve_max_units_per_player(overrides: dict[str, Any]) -> int:
        """Resolve the per-player unit cap from the engine-override overlay.

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
        if "max_units_per_player" not in (overrides or {}):
            return MAX_UNITS_PER_PLAYER
        val = int(overrides["max_units_per_player"])
        if val <= 0:
            raise ValueError(f"engine_overrides.max_units_per_player must be a positive int, got {val}")
        return val

    @staticmethod
    def _resolve_damage_model(overrides: dict[str, Any]) -> str:
        """Resolve the combat damage model from the engine-override overlay.

        ``"flat"`` (default) reproduces legacy HP-independent damage.
        ``"hp_scaled"`` multiplies outgoing damage by the attacker's current
        HP fraction. An unknown value fails loud rather than silently
        training on an unintended combat model.
        """
        model = (overrides or {}).get("damage_model", "flat")
        if model not in ("flat", "hp_scaled"):
            raise ValueError(f"engine_overrides.damage_model must be 'flat' or 'hp_scaled', got {model!r}")
        return model

    @staticmethod
    def _resolve_begin_first_turn(overrides: dict[str, Any]) -> bool:
        """Resolve ``begin_first_turn`` from the engine-override overlay.

        Start-of-turn processing (income, structure healing, status and
        cooldown ticks, the visibility update; see ``_begin_turn``) runs in
        ``end_turn`` for the player whose turn is starting, so Player 1's
        very first turn never got it: it plays turn 0 on its starting gold
        alone, while every later turn -- Player 2's first one included --
        collects income first. ``False`` (default) keeps that schedule.
        ``True`` runs ``_begin_turn(1)`` when the game is created, so Player
        1 also collects income before its first move. Recorded with the rest
        of ``engine_overrides`` (saves and replay ``game_info``) so a balance
        sweep can toggle it and replays reproduce it. Non-bool values fail
        loud: ``"false"`` would otherwise read as true.
        """
        return GameState._resolve_bool_override(overrides, "begin_first_turn")

    @staticmethod
    def _resolve_legacy_end_rules(overrides: dict[str, Any]) -> bool:
        """Resolve ``legacy_end_rules`` from the engine-override overlay.

        ``True`` plays by the end rules of games recorded before September
        2026 (review core-7), so their replays play back as they were
        played: any HQ capture wins the game for the capturer, whatever the
        seat count; nobody is eliminated, so a seat that lost its units or
        resigned keeps its turns, income and structures; and with three or
        more seats the game ends only when a single player has units left.
        ``replay_actions.replay_game_state_kwargs`` sets it for those
        replays; nothing else should.
        """
        return GameState._resolve_bool_override(overrides, "legacy_end_rules")

    @staticmethod
    def _resolve_bool_override(overrides: dict[str, Any], key: str) -> bool:
        """A bool engine override, default False. Non-bools fail loud: ``"false"`` would read as true."""
        value = (overrides or {}).get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"engine_overrides.{key} must be a bool, got {value!r}")
        return value

    @staticmethod
    def map_team_declarations(map_data: Any, num_players: int) -> dict[int, int]:
        """Teams a map declares, as ``{player: team}`` (empty if it declares none).

        The canonical team encoding is the structure tile code
        ``type_player_team`` (``Tile.team``), e.g. ``h_3_1``: player 3's HQ,
        team 1. Declare it on each player's HQ; other structures may repeat
        it but must agree. Owners outside ``1..num_players`` are ignored (a
        1v1v1 map played by two seats).

        Raises:
            ValueError: if one player is declared on two different teams, or
                a team id is not a positive int.
        """
        grid = map_data if isinstance(map_data, TileGrid) else TileGrid(map_data)
        declared: dict[int, int] = {}
        for row in grid.tiles:
            for tile in row:
                if tile.team is None or tile.player is None or not 1 <= tile.player <= num_players:
                    continue
                if tile.team <= 0:
                    raise ValueError(f"Tile ({tile.x}, {tile.y}) declares team {tile.team}; team ids must be positive")
                known = declared.setdefault(tile.player, tile.team)
                if known != tile.team:
                    raise ValueError(
                        f"The map puts player {tile.player} on team {known} and on team {tile.team} "
                        f"(tile ({tile.x}, {tile.y}))"
                    )
        return declared

    @classmethod
    def _resolve_teams(
        cls, grid: TileGrid, num_players: int, teams: dict[int, int] | None, map_teams: bool = True
    ) -> dict[int, int]:
        """Resolve every seat's team from the map's declarations and an explicit ``teams``.

        Both sources may declare a player; they must agree (``map_teams=False``
        ignores the map's). A player neither
        declares is a team of its own: its player number when nothing at
        all is declared (free-for-all, so 1v1 and 1v1v1 maps are unchanged),
        otherwise a fresh id after the declared ones so it cannot collide
        with a declared team.

        Raises:
            ValueError: on a conflicting or malformed declaration, or when
                every seat ends up on one team (nobody left to fight).
        """
        declared = cls.map_team_declarations(grid, num_players) if map_teams else {}
        for player, team in (teams or {}).items():
            if not isinstance(player, int) or not 1 <= player <= num_players:
                raise ValueError(f"teams: player {player!r} is not a seat of this {num_players}-player game")
            if not isinstance(team, int) or isinstance(team, bool) or team <= 0:
                raise ValueError(f"teams: team id for player {player} must be a positive int, got {team!r}")
            if declared.get(player, team) != team:
                raise ValueError(f"teams puts player {player} on team {team}, but the map declares team {declared[player]}")
            declared[player] = team

        resolved: dict[int, int] = {}
        next_free = max(declared.values(), default=0) + 1
        for player in range(1, num_players + 1):
            if player in declared:
                resolved[player] = declared[player]
            elif not declared:
                resolved[player] = player
            else:
                resolved[player] = next_free
                next_free += 1
        if num_players >= 2 and len(set(resolved.values())) < 2:
            raise ValueError(f"teams put all {num_players} players on one team; a game needs at least two teams")
        return resolved

    # YAML override key -> structure tile-type code. Lets a balance sweep tune
    # capture difficulty (e.g. ``headquarters_health: 30`` halves a Warrior's
    # HQ-capture time) from the config surface instead of editing rules.py.
    _STRUCTURE_HEALTH_KEYS = {
        "tower_health": "t",
        "building_health": "b",
        "headquarters_health": "h",
    }

    @classmethod
    def _resolve_structure_health(cls, overrides: dict[str, Any]) -> dict[str, int]:
        """Resolve per-structure max-HP overrides into ``{tile_code: hp}``.

        Only keys present in ``overrides`` appear in the result; absent
        structures keep their ``rules.py`` defaults. Non-positive values
        fail loud (a structure with <=0 HP would be captured on the first
        seize / be nonsensical for regen).
        """
        resolved: dict[str, int] = {}
        for ov_key, code in cls._STRUCTURE_HEALTH_KEYS.items():
            if ov_key in (overrides or {}):
                val = int(overrides[ov_key])
                if val <= 0:
                    raise ValueError(f"engine_overrides.{ov_key} must be a positive int, got {val}")
                resolved[code] = val
        return resolved

    @staticmethod
    def _resolve_rng(rng: Any | None, seed: int | None) -> tuple[int | None, Any]:
        """Return ``(seed, rng)``: the caller's source, or a game-owned ``random.Random``.

        A caller-supplied ``rng`` is used as is, and ``seed`` is recorded
        only if the caller names one (the engine cannot know how that
        source was seeded). Otherwise a missing seed is drawn from OS
        entropy -- not from ``random``, whose state a seeded caller may
        have fixed for its own purposes.
        """
        if seed is not None:
            seed = int(seed)  # numpy ints are not valid random.Random seeds
        if rng is not None:
            return seed, rng
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "big") >> 1
        return seed, random.Random(seed)

    def _rng_state_for_save(self) -> str | None:
        """The game RNG's exact position, compact enough for a JSON save.

        The seed alone only reproduces a game from turn 0; a mid-game save
        also needs the stream position so a reload rolls what the unsaved
        game would have rolled. None for a caller-supplied source whose
        state cannot be captured.
        """
        if not isinstance(self.rng, random.Random):
            return None
        try:
            version, internal, gauss_next = self.rng.getstate()
        except NotImplementedError:  # e.g. random.SystemRandom
            return None
        packed = base64.b64encode(struct.pack(f"<{len(internal)}I", *internal)).decode("ascii")
        return f"{version}:{gauss_next!r}:{packed}"

    @staticmethod
    def _rng_from_save(encoded: str) -> random.Random:
        """Rebuild the RNG ``_rng_state_for_save`` captured."""
        version, gauss_next, packed = encoded.split(":", 2)
        raw = base64.b64decode(packed)
        internal = struct.unpack(f"<{len(raw) // 4}I", raw)
        rng = random.Random()
        rng.setstate((int(version), internal, None if gauss_next == "None" else float(gauss_next)))
        return rng

    def _apply_structure_health_overrides(self) -> None:
        """Overlay resolved structure-HP overrides onto the freshly-built grid.

        ``TileGrid`` constructs structure tiles at the ``rules.py`` HP, so
        this runs right after grid creation while every structure is at full
        health -- setting both ``max_health`` and ``health`` keeps the tile
        consistent (regen scales off ``max_health``; capture resets to it).
        """
        if not self.structure_health:
            return
        for row in self.grid.tiles:
            for tile in row:
                override_hp = self.structure_health.get(tile.type)
                if override_hp is not None and tile.is_capturable():
                    tile.max_health = override_hp
                    tile.health = override_hp

    def __init__(
        self,
        map_data,
        num_players: int = 2,
        max_turns: int | None = None,
        enabled_units: list[str] | None = None,
        fog_of_war: bool = False,
        engine_overrides: dict[str, Any] | None = None,
        rng: Any | None = None,
        seed: int | None = None,
        teams: dict[int, int] | None = None,
        map_teams: bool = True,
    ) -> None:
        """
        Initialize the game state.

        Args:
            map_data: Tile codes by row: a pandas DataFrame, numpy array or
                list of lists
            num_players: Number of players (default 2)
            max_turns: Maximum turns for the game (None = unlimited)
            enabled_units: List of enabled unit types (default all units enabled)
            fog_of_war: Enable fog of war (default False for backward compatibility)
            rng: Optional random source exposing ``random()`` used for
                engine-side stochastic outcomes -- currently only the Rogue
                evade roll in ``mechanics.attack_unit``. ``None`` (default)
                gives the game its own ``random.Random(seed)``; the engine
                never reads the module-global ``random``. Replays are
                unaffected either way: they apply recorded outcomes
                directly instead of re-rolling (``utils/replay_actions.py``).
            seed: Seed for the game's own RNG (ignored for sampling when
                ``rng`` is given, but still recorded). ``None`` draws one
                from OS entropy. Kept in ``self.seed`` and written to saves
                and replays, so any game can be re-run with the same
                combat rolls. The RL env passes one derived from its
                episode seed, the tournament runner one derived from
                ``rng_seed`` and the game id.
            engine_overrides: Optional sparse overlay over the non-YAML
                engine constants (``rules.py``), so balance can be
                varied/recorded as config instead of a code edit. Shape::

                    {
                      "starting_gold": int,
                      "headquarters_income": int,
                      "building_income": int,
                      "tower_income": int,
                      "tower_health": int,         # structure max-HP overrides
                      "building_health": int,      #   (capture-difficulty lever)
                      "headquarters_health": int,
                      "damage_model": "flat" | "hp_scaled",  # combat model
                      "max_units_per_player": int,  # per-player unit cap
                      "begin_first_turn": bool,    # P1 turn-0 start-of-turn
                      "legacy_end_rules": bool,    # pre-2026-09 replays only
                      "unit_data": {CODE: {field: value}},  # sparse deltas
                      # optional terrain rules, see core/terrain_rules.py:
                      "terrain_move_cost": {TILE_CODE: cost},
                      "charge_distance": "displacement" | "path",
                      "forest_concealment": bool,
                      "hq_always_visible": bool,
                    }

                Every key is optional; absent keys fall back to the module
                constant, so ``None`` / ``{}`` is byte-identical to today.
                Unknown keys raise ``KeyError`` (see ``ENGINE_OVERRIDE_KEYS``).
                The resolved tables (``self.unit_data``, ``self.income_rates``,
                ``self.starting_gold``) are this game's single source of
                truth -- units and income read them, never the global
                constant -- so an override can't leak or be half-applied.
            teams: Optional ``{player: team}``. Teams can also be declared by
                the map (``type_player_team`` structure codes, see
                ``map_team_declarations``); the two must agree. Players
                nobody declares are each their own team, so by default every
                game is free-for-all. Teammates are allies for every rule
                (``are_allies``); the game ends when one team is left.
            map_teams: Whether the map's ``type_player_team`` codes declare
                teams (default). ``False`` ignores them: saves and replays
                written before teams existed were played free-for-all even
                on a map with those codes (the old 2v2 map put one player on
                two teams), and are loaded that way.
        """
        self.grid = TileGrid(map_data)
        self.units: list[Unit] = []
        # Monotonic per-game unit-id counter. Stamped on every newly
        # created unit; written into the replay log alongside each
        # action so the v3 replay player can look up units by id
        # rather than by brittle (x, y) position. Restored from saves
        # in ``from_dict`` so post-load creations don't collide.
        self._next_unit_id: int = 0
        self.current_player: int = 1
        self.num_players: int = num_players
        # Player -> team id for every seat (see _resolve_teams). Every
        # hostility rule goes through are_allies/are_enemies, which read it.
        self._teams_arg: dict[int, int] | None = dict(teams) if teams else None
        self._map_teams: bool = map_teams
        self.teams: dict[int, int] = self._resolve_teams(self.grid, num_players, teams, map_teams)
        # Seats knocked out of the game (review core-7): they hold no units
        # or structures and end_turn skips them. With more than two teams a
        # player is eliminated when it loses its last HQ, its last unit or
        # resigns, and the game goes on until one team is left; with two
        # teams the first HQ capture still ends the game outright.
        self.eliminated_players: set[int] = set()
        self.engine_overrides: dict[str, Any] = dict(engine_overrides) if engine_overrides else {}
        (
            self.unit_data,
            self.income_rates,
            self.starting_gold,
        ) = self._resolve_engine_overrides(self.engine_overrides)
        # Combat damage model (engine-side, config-surfaced via engine_overrides
        # so it's snapshotted into config.json like the economy). "flat"
        # (default, legacy) = HP-independent damage; "hp_scaled" = damage
        # multiplied by the attacker's current HP fraction (decisive combat;
        # consistent with seize, which is already HP-scaled).
        self.damage_model: str = self._resolve_damage_model(self.engine_overrides)
        # Per-structure max-HP overrides (capture-difficulty lever). Resolved
        # from engine_overrides and overlaid onto the grid built above; absent
        # keys keep rules.py defaults. Snapshotted into config.json via the
        # verbatim engine_overrides log, same as damage_model / economy.
        self.structure_health: dict[str, int] = self._resolve_structure_health(self.engine_overrides)
        self._apply_structure_health_overrides()
        # Hard ceiling on units-per-player (action-space + economy guardrail).
        # Enforced in both create_unit and get_legal_actions so the cap shows
        # up in the action mask, not just as a rejected action.
        self.max_units_per_player: int = self._resolve_max_units_per_player(self.engine_overrides)
        # Optional terrain rules (movement costs, path-based Knight Charge,
        # forest concealment, HQ always known). All off by default, which is
        # the game as shipped; see core/terrain_rules.py.
        self.terrain_rules: TerrainRules = TerrainRules.from_overrides(self.engine_overrides)
        self.begin_first_turn: bool = self._resolve_begin_first_turn(self.engine_overrides)
        self.legacy_end_rules: bool = self._resolve_legacy_end_rules(self.engine_overrides)
        self.player_gold: dict[int, int] = {i: self.starting_gold for i in range(1, num_players + 1)}
        # Cumulative structure auto-heal totals per player (HP restored and
        # gold spent by ``heal_units_on_structures`` over the whole game).
        # The per-turn stats returned via ``end_turn()`` are routinely
        # discarded by callers (bots call ``end_turn`` internally, the gym
        # env drops the return value), so this game-lifetime accumulator is
        # the only reliable way for diagnostics to answer "how much gold did
        # each side silently spend on auto-heal" after the fact.
        self.healing_totals: dict[int, dict[str, int]] = {i: {"hp": 0, "gold": 0} for i in range(1, num_players + 1)}
        self.game_over: bool = False
        self.winner: int | None = None
        # Why the game ended. Populated alongside ``game_over`` by
        # ``_set_game_over``. Values: ``hq_capture``, ``elimination``,
        # ``max_turns_draw``, ``resign``. Replays surface this so videos
        # and the load-game UI can explain *how* a game finished without
        # re-deriving it from actions[].
        self.end_reason: str | None = None
        # Index into ``action_history`` of the action that flipped
        # ``game_over``. None until the game ends. Lets replay viewers
        # jump to the decisive moment and lets analysis count any
        # post-victory actions a bot may have queued.
        self.game_over_action_index: int | None = None
        self.turn_number: int = 0
        self.mechanics = GameMechanics()
        # Engine-side RNG for stochastic combat outcomes (Rogue evade). Every
        # game owns one (review core-10): the old default, the module-global
        # ``random``, made seeded tournaments, AlphaZero evals and BC datasets
        # irreproducible and was shared by every thread. The seed is recorded
        # even when drawn from entropy, so any game can be re-run.
        self.seed: int | None
        self.rng: Any
        self.seed, self.rng = self._resolve_rng(rng, seed)

        # Fog of war settings
        self.fog_of_war: bool = fog_of_war
        # FOW method for future compatibility when different algorithms are added
        # Current options: 'simple_radius' (Option A from proposal)
        # Future options: 'line_of_sight', 'hybrid'
        self.fog_of_war_method: str = "simple_radius" if fog_of_war else "none"
        # Built (and first computed) by _init_visibility at the end of __init__
        self.visibility_maps: dict[int, VisibilityMap] = {}

        # Enabled unit types (defaults to all if not specified)
        self.enabled_units: list[str] = enabled_units if enabled_units is not None else self.ALL_UNIT_TYPES.copy()

        # Optional map file reference for saving
        self.map_file_used: str | None = None

        # The tile codes the grid was built from (a 2D list), for saves and
        # replays. Every coordinate the engine uses or records -- units,
        # structures, action_history -- is on this grid: a GUI game's map is
        # the UI-padded one (FileIO.load_map(for_ui=True)), so its saves and
        # replays carry that padding too, and stay self-consistent.
        self.initial_map_data: list[list[str]] = np.asarray(map_data, dtype=object).tolist()

        # Player configurations (human vs bot)
        self.player_configs: list[dict[str, Any]] = []

        # Maximum turns for the game (None = unlimited)
        self.max_turns: int | None = max_turns

        # Action history for replay
        self.action_history: list[dict[str, Any]] = []
        self.game_start_time: datetime = datetime.now()

        # Cached legal actions per player (see get_legal_actions)
        self._legal_actions_cache: dict[int, dict[str, list[Any]]] = {}
        self._legal_actions_cache_valid: bool = False

        # Every player starts with a computed fog-of-war view. Callers used to
        # have to call update_visibility() after construction, and a game
        # built without it showed nothing at all until the first move.
        self._init_visibility()
        # Turn 0 for Player 1 (see _resolve_begin_first_turn). Last, so the
        # whole state exists; the default leaves turn 0 as it always was.
        if self.begin_first_turn:
            self._begin_turn(self.current_player)

    def reset(self, map_data) -> None:
        """Reset the game state."""
        self.__init__(
            map_data,
            self.num_players,
            self.max_turns,
            self.enabled_units,
            self.fog_of_war,
            engine_overrides=self.engine_overrides,
            rng=self.rng,
            seed=self.seed,
            teams=self._teams_arg,
            map_teams=self._map_teams,
        )

    def _invalidate_cache(self) -> None:
        """Invalidate cached values."""
        self._legal_actions_cache_valid = False
        self._legal_actions_cache.clear()

    def _set_game_over(self, winner: int | None, end_reason: str) -> None:
        """Single chokepoint for flipping ``game_over``.

        Records the winner, end reason, and the index of the action that
        caused the game to end (or -1 if no action was recorded yet,
        e.g. a max-turns draw that triggers before any new action is
        appended). Idempotent: first call wins; later attempts to set a
        different reason are ignored so a post-game action can't
        overwrite the real cause.
        """
        if self.game_over:
            return
        self.game_over = True
        self.winner = winner
        self.end_reason = end_reason
        self.game_over_action_index = len(self.action_history) - 1 if self.action_history else -1

    # ------------------------------------------------------------------
    # Teams and elimination
    # ------------------------------------------------------------------

    def team_of(self, player: int) -> int | None:
        """The team ``player`` plays on (None for a player outside this game)."""
        return self.teams.get(player)

    def are_allies(self, player_a: int | None, player_b: int | None) -> bool:
        """Same player or teammates. ``None`` (a neutral owner) is nobody's ally."""
        return same_side(player_a, player_b, self.teams)

    def are_enemies(self, player_a: int | None, player_b: int | None) -> bool:
        """Two players on different teams. ``None`` (neutral) is nobody's enemy either."""
        return player_a is not None and player_b is not None and not self.are_allies(player_a, player_b)

    def is_eliminated(self, player: int) -> bool:
        """Whether ``player`` has been knocked out of the game (see ``eliminated_players``)."""
        return player in self.eliminated_players

    def _active_players(self) -> list[int]:
        """Seats still in the game, in turn order."""
        return [p for p in range(1, self.num_players + 1) if p not in self.eliminated_players]

    def _starting_team_count(self) -> int:
        return len(set(self.teams.values()))

    def _player_owns_hq(self, player: int) -> bool:
        return any(
            tile.type == TileType.HEADQUARTERS.value and tile.player == player for row in self.grid.tiles for tile in row
        )

    def _eliminate_player(self, player: int, reason: str, by_player: int | None = None) -> None:
        """Knock ``player`` out of the game (review core-7).

        While other teams play on, its units are removed and every structure
        it still owns becomes neutral (health kept, so an enemy mid-seize
        keeps its progress); end_turn skips it from now on, so it gets no
        turns, income or new units. When one team is left the game ends with
        ``reason`` as its end reason and the board is left as it stands, as
        a decided game's always was; the winner is ``by_player`` (whose
        action decided it) when it is on the winning team, else that team's
        lowest-numbered remaining player. In games with more than two seats
        the elimination is also recorded as an ``eliminate`` action so
        replays and analysis see it; a two-seat game ends at its first
        elimination, which game_info already describes, so its action log is
        unchanged.
        """
        if self.game_over or player in self.eliminated_players:
            return
        self.eliminated_players.add(player)
        if self.num_players > 2:
            self.record_action("eliminate", eliminated_player=player, reason=reason)
        logger.debug("Player %d eliminated (%s)", player, reason)

        remaining_teams = {self.teams[p] for p in self._active_players()}
        if len(remaining_teams) > 1:
            # In place: bots and the renderer may hold a reference to the list.
            self.units[:] = [u for u in self.units if u.player != player]
            for row in self.grid.tiles:
                for tile in row:
                    if tile.is_capturable() and tile.player == player:
                        tile.player = None
            self._invalidate_cache()
            return
        if not remaining_teams:
            self._set_game_over(winner=None, end_reason=reason)
            return
        winning_team = remaining_teams.pop()
        if by_player is not None and self.teams.get(by_player) == winning_team and by_player not in self.eliminated_players:
            winner = by_player
        else:
            winner = min(p for p in self._active_players() if self.teams[p] == winning_team)
        self._set_game_over(winner=winner, end_reason=reason)

    def _check_player_eliminated(self, defeated_player: int) -> None:
        """Eliminate ``defeated_player`` if it has just lost its last unit.

        Called when one of its units dies. In a 1v1 this ends the game with
        the opponent as the winner, as it always did; with more seats the
        player is knocked out and the game ends only when one team is left
        (see ``_eliminate_player``).
        """
        if any(u.player == defeated_player for u in self.units):
            return
        if self.legacy_end_rules:
            self._legacy_last_player_standing(defeated_player, "elimination")
            return
        self._eliminate_player(defeated_player, "elimination", by_player=self.current_player)

    def _legacy_last_player_standing(self, defeated_player: int, reason: str) -> None:
        """The pre-September-2026 end check after ``defeated_player`` lost its units (see ``legacy_end_rules``).

        Two seats: the other player wins. More: the game ends when only one
        player has units left (after a resign, also when none has). Nobody
        is eliminated otherwise.
        """
        if self.num_players == 2:
            self._set_game_over(winner=2 if defeated_player == 1 else 1, end_reason=reason)
            return
        players_with_units = {u.player for u in self.units}
        if len(players_with_units) == 1:
            self._set_game_over(winner=players_with_units.pop(), end_reason=reason)
        elif not players_with_units and reason == "resign":
            self._set_game_over(winner=None, end_reason=reason)

    def _on_hq_captured(self, capturer: int, previous_owner: int | None) -> None:
        """Apply the end rule for an HQ that ``capturer`` just took from ``previous_owner``.

        Two teams (1v1, 2v2): capturing an enemy HQ wins the game for the
        capturer's team, as it always did. More than two teams
        (free-for-all): the previous owner is eliminated once it holds no HQ
        any more, and play goes on for everyone else. A neutral HQ (left by
        an eliminated player) is just a structure: taking it ends nothing.
        """
        if previous_owner is None or self.game_over:
            return
        if self.legacy_end_rules or self._starting_team_count() <= 2:
            self._set_game_over(winner=capturer, end_reason="hq_capture")
        elif not self._player_owns_hq(previous_owner):
            self._eliminate_player(previous_owner, "hq_capture", by_player=capturer)

    def _init_visibility(self) -> None:
        """Build every player's fog-of-war map from scratch and compute it.

        Each player starts knowing where every HQ is and who owns it: the
        HQs are recorded in the last-seen memory at the current turn (and
        their tiles count as explored), so observations, the renderer and
        LLM prompts all show them, as the rules say ("enemy HQ is always
        known"). Their later HP and owner are only learnt by seeing them.
        """
        self.visibility_maps = {}
        if not self.fog_of_war:
            return
        hq_tiles = [
            self.grid.tiles[y][x]
            for x, y in self.grid.structure_positions
            if self.grid.tiles[y][x].type == TileType.HEADQUARTERS.value
        ]
        for player in range(1, self.num_players + 1):
            vis_map = VisibilityMap(self.grid.width, self.grid.height, player)
            for tile in hq_tiles:
                vis_map.remember_structure(tile, self.turn_number)
            self.visibility_maps[player] = vis_map
        self.update_visibility()

    def update_visibility(self, player: int | None = None) -> None:
        """
        Update visibility maps for fog of war.

        The engine calls this itself whenever a player's vision can change
        (construction and load, moves, unit creation and placement, captures,
        deaths, turn changes), so callers never need to.

        Args:
            player: Specific player to update, or None to update all players
        """
        if not self.fog_of_war:
            return

        # Legality under fog of war reads visibility (attackable targets, and
        # the units a player's pathfinding may treat as obstacles), so a
        # visibility change is a legality change.
        self._invalidate_cache()

        if player is not None:
            if player in self.visibility_maps:
                self.visibility_maps[player].update(self)
                self.visibility_maps[player].clear_stale_unit_memory(max_turns=10, current_turn=self.turn_number)
        else:
            for vis_map in self.visibility_maps.values():
                vis_map.update(self)
                vis_map.clear_stale_unit_memory(max_turns=10, current_turn=self.turn_number)

    def get_visible_units_for_player(self, player: int, include_own: bool = True) -> list[Unit]:
        """
        Get units visible to a specific player.

        Args:
            player: Player to get visible units for
            include_own: Whether to include the player's own units

        Returns:
            List of visible units
        """
        return get_visible_units(self, player, include_own)

    def is_position_visible(self, x: int, y: int, player: int) -> bool:
        """
        Check if a position is visible to a player.

        Args:
            x: X coordinate
            y: Y coordinate
            player: Player to check visibility for

        Returns:
            True if position is visible (or if fog of war is disabled)
        """
        if not self.fog_of_war:
            return True

        vis_map = self.visibility_maps.get(player)
        if vis_map is None:
            return True

        return vis_map.is_visible(x, y)

    def is_position_explored(self, x: int, y: int, player: int) -> bool:
        """
        Check if a position has been explored by a player.

        Args:
            x: X coordinate
            y: Y coordinate
            player: Player to check exploration for

        Returns:
            True if position is explored (or if fog of war is disabled)
        """
        if not self.fog_of_war:
            return True

        vis_map = self.visibility_maps.get(player)
        if vis_map is None:
            return True

        return vis_map.is_explored(x, y)

    def known_structure(self, player: int, x: int, y: int) -> StructureSnapshot | None:
        """What ``player`` knows about the structure at ``(x, y)``.

        The one view of structures under fog of war: ``to_numpy(for_player)``,
        the renderer and the LLM prompt all read it, so none of them can show
        a player more than it knows (review core-5, critic-integration-3,
        pygame-12).

        Returns:
            None when there is no structure at ``(x, y)`` or ``player`` has
            never seen it. Otherwise a snapshot with ``owner``, ``health``
            and ``turn_seen``: the live state (``turn_seen`` = this turn)
            without fog of war or while ``player`` can see the tile, else
            the state when ``player`` last saw it. Every HQ is known from
            the start of the game (see ``_init_visibility``).
        """
        tile = self.grid.get_tile(x, y)
        if tile is None or not tile.is_capturable():
            return None
        vis_map = self.visibility_maps.get(player) if self.fog_of_war else None
        if vis_map is None or vis_map.is_visible(x, y):
            return StructureSnapshot(
                tile_type=tile.type, owner=tile.player, health=tile.health, position=(x, y), turn_seen=self.turn_number
            )
        return vis_map.get_last_seen_structure(x, y)

    def pathing_units(self, player: int) -> list[Unit]:
        """The units ``player``'s pathfinding treats as present: its blocking view.

        Without fog of war that is every unit. Under fog of war it is the
        player's own units plus the units on tiles it can see. Letting a
        hidden enemy block paths and destinations would reveal it through
        the move mask (review core-5), so pathfinding plans around the units
        the player knows of and ``move_unit`` resolves a collision with a
        hidden unit when the move is carried out (the ambush rule, see
        ``_resolve_ambush``). Public so the GUI's movement overlay plans with
        the same view and shows exactly the tiles the engine allows.
        """
        if not self.fog_of_war:
            return self.units
        # Teammates' units count as known wherever they stand: teams don't
        # share vision, but a hidden teammate treated as absent would let a
        # unit end its move on the teammate's tile.
        return [u for u in self.units if self.are_allies(u.player, player) or self.is_position_visible(u.x, u.y, player)]

    def capture_visible_enemies_for_unit(self, unit: Unit) -> None:
        """
        Capture which enemy units are currently visible to a unit's owner.

        This is used for fog of war to prevent "move to discover, then attack"
        exploitation. Call this when a unit starts its action (is selected).

        Args:
            unit: The unit starting its action
        """
        if not self.fog_of_war:
            unit.visible_enemies_at_action_start = None
            return

        # The snapshot is taken once, when the action begins. A unit that has
        # already moved this action keeps it: re-selecting an ambushed unit in
        # the GUI (its move can't be cancelled) must not add the ambusher, or
        # any enemy the move revealed, to its attack targets.
        if unit.has_moved and unit.visible_enemies_at_action_start is not None:
            return

        visible_positions = set()
        for enemy in self.units:
            if self.are_enemies(enemy.player, unit.player):
                if self.is_position_visible(enemy.x, enemy.y, unit.player):
                    visible_positions.add((enemy.x, enemy.y))

        unit.visible_enemies_at_action_start = visible_positions

    def is_enemy_attackable_by_unit(self, unit: Unit, enemy: Unit) -> bool:
        """
        Check if an enemy is attackable by a unit considering FOW pre-move snapshot.

        In fog of war mode, a unit can only attack enemies that were visible
        when the unit started its action, not enemies discovered by moving.

        Args:
            unit: The attacking unit
            enemy: The potential target

        Returns:
            True if the enemy can be attacked
        """
        if not self.fog_of_war:
            return True  # No FOW, all visible enemies are attackable

        # If no snapshot was captured, fall back to current visibility
        if unit.visible_enemies_at_action_start is None:
            return self.is_position_visible(enemy.x, enemy.y, unit.player)

        # Check if enemy's position was in the pre-move snapshot
        return (enemy.x, enemy.y) in unit.visible_enemies_at_action_start

    def is_unit_type_enabled(self, unit_type: str) -> bool:
        """Check if a unit type is enabled for this game."""
        return unit_type in self.enabled_units

    def set_enabled_units(self, enabled_units: list[str]) -> None:
        """Set the list of enabled unit types."""
        self.enabled_units = enabled_units
        self._invalidate_cache()

    def get_unit_at_position(self, x: int, y: int) -> Unit | None:
        """Get the unit at a grid position.

        A linear scan, deliberately: a persistent position index would go
        stale whenever a unit is moved or removed outside GameState's own
        methods, and the replay applier, the rule bots' look-ahead and many
        tests do exactly that. The hot path (pathfinding) builds its own
        occupancy set per call instead (review core-20).
        """
        for unit in self.units:
            if unit.x == x and unit.y == y:
                return unit
        return None

    def record_action(self, action_type: str, **kwargs) -> None:
        """
        Record an action for replay purposes.

        Coordinates are recorded as given, on this game's grid: a replay
        stores ``initial_map_data`` with them, and its playback translates
        both onto its own display padding the same way.

        Args:
            action_type: Type of action (move, attack, create_unit, etc.)
            **kwargs: Action-specific parameters
        """
        # Don't log anything once the game has been decided. Without this,
        # bots that don't break their per-unit loop on game_over append
        # cosmetic moves (and end_turn) after the winning action, which
        # makes len(actions) > winning_action_index + 1 and inflates
        # turn_number past the real end of the game. The single-chokepoint
        # guard here covers all 12 record_action sites in one place.
        if self.game_over:
            return

        action_record = {
            "turn": self.turn_number,
            "player": self.current_player,
            "type": action_type,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.action_history.append(action_record)

    # ------------------------------------------------------------------
    # Legality predicates
    # ------------------------------------------------------------------
    # Each rule is written once and used twice: ``_compute_legal_actions``
    # enumerates what a player may do with it, and the action methods below
    # reject anything else with it. The action methods used to trust their
    # callers, so any caller that did not pre-filter against the legal list
    # (multi_discrete policies, LLM bots, the rule bots' knight charge, the
    # GUI) could spawn units anywhere, attack across the map, act out of
    # turn or seize an HQ several times in one turn (review core-2). One
    # definition per rule is what keeps the mask and the engine agreeing: an
    # offered action the engine rejects traps a deterministic policy, and an
    # accepted action the mask never offers is an exploit.
    #
    # Two gates apply only on execution, not in enumeration: the game must
    # not be over, and it must be the acting player's turn.
    # ``get_legal_actions(player)`` still answers for any player at any time
    # (masks and prompts are built for a player's own turn in practice).

    @staticmethod
    def _is_ready_unit(unit: Unit, player: int) -> bool:
        """``unit`` belongs to ``player``, is alive and is not paralyzed."""
        return unit.player == player and unit.health > 0 and not unit.is_paralyzed()

    def _under_unit_cap(self, player: int) -> bool:
        return sum(1 for u in self.units if u.player == player) < self.max_units_per_player

    def _is_free_spawn_tile(self, player: int, x: int, y: int) -> bool:
        """An in-bounds, empty Building owned by ``player`` (HQs and towers never spawn)."""
        tile = self.grid.get_tile(x, y)
        return (
            tile is not None
            and tile.type == TileType.BUILDING.value
            and tile.player == player
            and self.get_unit_at_position(x, y) is None
        )

    def _can_afford(self, player: int, unit_type: str) -> bool:
        return self.player_gold[player] >= self.unit_data[unit_type]["cost"]

    def _find_paths(
        self,
        unit: Unit,
        blocked: set[tuple[int, int]] | None = None,
        came_from: dict[tuple[int, int], tuple[int, int]] | None = None,
    ) -> dict[tuple[int, int], int]:
        """Every tile ``unit`` can reach this turn -> tiles stepped, in search order.

        Walkable tiles within its movement (under the game's terrain move
        costs), passing through friendly units but never enemies; tiles
        holding a friendly unit are included, as a path may cross them. The
        blockers are collected once per search (or once per
        ``get_legal_actions`` call, for all of a player's units: pass
        ``blocked``), so each tile the search examines costs a set lookup
        rather than a scan of every unit (review core-20). They come from the
        units the player knows of (``pathing_units``: all of them without fog
        of war), so a hidden enemy neither blocks a path nor reveals itself
        through the move mask (review core-5). ``came_from`` receives the
        search's path tree (the ambush rule walks it).
        """
        if blocked is None:
            blocked = self.mechanics.movement_blockers(self.pathing_units(unit.player), unit, self.teams)
        return unit.find_paths(
            self.grid.width,
            self.grid.height,
            self.mechanics.passability(self.grid, blocked),
            self.terrain_rules.move_cost_fn(self.grid),
            came_from,
        )

    def _move_paths(
        self,
        unit: Unit,
        occupied: set[tuple[int, int]] | None = None,
        blocked: set[tuple[int, int]] | None = None,
        came_from: dict[tuple[int, int], tuple[int, int]] | None = None,
    ) -> dict[tuple[int, int], int]:
        """``_find_paths`` restricted to tiles ``unit`` may end on (no known unit there)."""
        if occupied is None:
            occupied = {(u.x, u.y) for u in self.pathing_units(unit.player)}
        return {pos: steps for pos, steps in self._find_paths(unit, blocked, came_from).items() if pos not in occupied}

    def get_reachable_positions(self, unit: Unit) -> list[tuple[int, int]]:
        """Tiles ``unit`` can move through this turn, including ones friends stand on.

        Same result as ``unit.get_reachable_positions`` with
        ``can_move_to_position`` as its predicate over ``pathing_units``,
        but under the game's terrain move costs and without scanning every
        unit per tile: for bots, overlays and anything else that plans paths.
        """
        return list(self._find_paths(unit))

    def get_move_destinations(self, unit: Unit) -> list[tuple[int, int]]:
        """Tiles ``unit`` may legally end a move on (reachable and empty), in search order.

        Ignores whose turn it is and whether the unit may still move; see
        ``get_legal_actions`` for that.
        """
        return list(self._move_paths(unit))

    def _resolve_ambush(self, unit: Unit, path: list[tuple[int, int]]) -> tuple[tuple[int, int], Unit | None]:
        """Walk ``unit`` along its planned ``path`` and return where it really stops.

        The ambush rule (fog of war only). The path (start tile first) was
        planned around the units the player can see, so a hidden unit may
        stand on it. The first tile the unit cannot pass (a hidden enemy; a
        teammate's unit, like its own, is passed through) or end on (a
        hidden unit on the destination) stops it on the last tile before
        that one where no unit stands, at worst its start tile.

        Returns:
            ``(stop_tile, blocker)``; ``blocker`` is None when the path is clear.
        """
        last = len(path) - 1
        for i in range(1, last + 1):
            x, y = path[i]
            blocker = self.get_unit_at_position(x, y)
            if blocker is None:
                continue
            if i < last and self.mechanics.can_move_to_position(
                x, y, self.grid, self.units, moving_unit=unit, teams=self.teams
            ):
                continue  # a unit it may pass through
            for j in range(i - 1, 0, -1):
                if self.get_unit_at_position(*path[j]) is None:
                    return path[j], blocker
            return path[0], blocker
        return path[last], None

    def _can_attack_target(self, unit: Unit, target: Unit) -> bool:
        """A living enemy within ``unit``'s reach that fog of war lets it attack.

        Under fog of war the target must have been visible when the unit
        started its action (``is_enemy_attackable_by_unit``), so moving to
        discover an enemy does not also let the unit hit it.
        """
        return (
            self.are_enemies(target.player, unit.player)
            and target.health > 0
            and self.mechanics.can_reach(unit, target.x, target.y, self.grid)
            and (not self.fog_of_war or self.is_enemy_attackable_by_unit(unit, target))
        )

    def _can_paralyze_target(self, unit: Unit, target: Unit) -> bool:
        """Mage off cooldown, an attackable enemy in paralyze range that is not already paralyzed.

        Re-casting on a paralyzed target would only refresh the status (a
        near no-op) and inflate the action space, the same reason heal,
        cure and the buffs skip an ally that already has the effect.
        """
        return (
            unit.can_use_paralyze()
            and not target.is_paralyzed()
            and self.mechanics.in_ability_range("paralyze", unit, target)
            and self._can_attack_target(unit, target)
        )

    def _can_heal_target(self, unit: Unit, target: Unit) -> bool:
        return unit.type == "C" and self.mechanics.is_healable_ally(unit, target, self.teams)

    def _can_cure_target(self, unit: Unit, target: Unit) -> bool:
        return unit.type == "C" and self.mechanics.is_curable_ally(unit, target, self.teams)

    def _can_haste_target(self, unit: Unit, target: Unit) -> bool:
        return unit.can_use_haste() and self.mechanics.is_hasteable_ally(unit, target)

    def _can_defence_buff_target(self, unit: Unit, target: Unit) -> bool:
        return unit.can_use_defence_buff() and self.mechanics.is_defence_buffable_ally(unit, target, self.teams)

    def _can_attack_buff_target(self, unit: Unit, target: Unit) -> bool:
        return unit.can_use_attack_buff() and self.mechanics.is_attack_buffable_ally(unit, target, self.teams)

    def _can_seize(self, unit: Unit) -> bool:
        """``unit`` stands on a structure neither its player nor a teammate owns."""
        tile = self.grid.get_tile(unit.x, unit.y)
        return tile is not None and tile.is_capturable() and not self.are_allies(tile.player, unit.player)

    def _may_act(self, action: str, unit: Unit, target: Unit | None = None, rule: Callable[[], bool] | None = None) -> bool:
        """Validate one unit action before it is applied; log why when it is not.

        Rejects when the game is over; when ``unit`` (or ``target``) is no
        longer in play -- a stale reference, e.g. a bot still holding a unit
        that died to a counter earlier in its loop; when it is not the
        unit's player's turn; when the unit is dead or paralyzed; when the
        action slot it needs is spent (``can_move`` for a move,
        ``can_attack`` for attacks, abilities and seizing); or when
        ``rule`` (the action's target/range predicate) fails.
        """
        if self.game_over:
            reason = "the game is over"
        elif unit not in self.units or (target is not None and target not in self.units):
            reason = "a unit involved is no longer in play"
        elif unit.player != self.current_player:
            reason = f"it is player {self.current_player}'s turn"
        elif not self._is_ready_unit(unit, unit.player):
            reason = "the unit is dead or paralyzed"
        elif not (unit.can_move if action == "move" else unit.can_attack):
            reason = "the unit has already spent that action this turn"
        elif rule is not None and not rule():
            reason = "the target is out of range, on the wrong side, or a precondition fails"
        else:
            return True
        logger.debug("Rejected %s by player %d %s at (%d, %d): %s", action, unit.player, unit.type, unit.x, unit.y, reason)
        return False

    def _consume_action(self, unit: Unit) -> None:
        """Spend ``unit``'s action for this turn -- or its haste, if it has one.

        Every action method that uses up a unit's action (attack, seize and
        the abilities; not a move, which only spends ``can_move``) ends with
        this, so haste works the same whoever drives the engine (review
        core-8). A unit without haste is done for the turn
        (``can_move``/``can_attack`` False, exactly as before). A hasted unit
        instead uses up its haste and gets one more full action this turn: it
        may move again (a fresh move, so a Knight's charge distance restarts
        here) and act again. Before this lived here, only callers that ran
        ``end_unit_turn`` after an action (the GUI, the rule bots) granted
        the extra action; the RL env, MCTS and LLM bots never did.

        ``haste_refreshed`` marks the refreshed unit until it moves or acts
        again, so ``end_unit_turn`` called right after the action (as the
        GUI and bots still do) leaves it its extra action instead of ending
        it.
        """
        if unit.is_hasted:
            unit.is_hasted = False
            unit.can_move = True
            unit.can_attack = True
            unit.has_moved = False
            unit.original_x = unit.x
            unit.original_y = unit.y
            unit.distance_moved = 0
            # FOW: the extra action starts from a fresh snapshot of what its
            # owner sees (captured lazily by move_unit).
            unit.visible_enemies_at_action_start = None
            unit.haste_refreshed = True
        else:
            unit.can_move = False
            unit.can_attack = False
            unit.haste_refreshed = False
        # The move before this action can no longer be cancelled.
        unit.pre_move_visibility = None

    @staticmethod
    def _noop_attack_result() -> dict[str, Any]:
        """What ``attack`` returns when it applies nothing (``damage`` 0)."""
        return {
            "attacker_alive": True,
            "target_alive": True,
            "damage": 0,
            "counter_damage": 0,
            "charge_bonus": False,
            "flank_bonus": False,
            "evade": False,
            "attack_buff": False,
            "defence_buff": False,
        }

    def place_unit(self, unit_type: str, x: int, y: int, player: int) -> Unit:
        """Put a unit on the board for a test, scenario or other setup.

        Not a game action, so none of ``create_unit``'s rules apply: no gold
        is charged, nothing is recorded in the action history, and any tile
        (walkable or not), any player and any unit type (enabled or not) is
        accepted whoever's turn it is. The unit gets the next ``unit_id``
        and starts ready to act (``can_move``/``can_attack`` True), as if it
        had begun the turn on that tile; a unit made with ``create_unit``
        instead waits for its player's next turn. Because nothing is
        recorded, a replay cannot rebuild placed units from its action log.

        Raises:
            ValueError: for an unknown unit type, an off-board position or
                an occupied tile -- states the engine cannot represent.
        """
        if unit_type not in self.unit_data:
            raise ValueError(f"Unknown unit type: {unit_type!r}")
        if self.grid.get_tile(x, y) is None:
            raise ValueError(f"({x}, {y}) is off the {self.grid.width}x{self.grid.height} board")
        if self.get_unit_at_position(x, y) is not None:
            raise ValueError(f"({x}, {y}) is already occupied")

        unit = Unit(unit_type, x, y, player, stats=self.unit_data[unit_type])
        unit.unit_id = self._next_unit_id
        self._next_unit_id += 1
        unit.can_move = True
        unit.can_attack = True
        self.units.append(unit)
        self._invalidate_cache()
        # A placed unit can reveal (or stand in) fog for every player.
        self.update_visibility()
        return unit

    def create_unit(self, unit_type: str, x: int, y: int, player: int | None = None) -> Unit | None:
        """
        Create a unit at the specified position.

        Rejected (returns None, changes and records nothing) unless the game
        is running, ``player`` is the current player, ``unit_type`` is
        enabled, the player is under the unit cap and can afford it, and
        ``(x, y)`` is an empty Building the player owns -- the same rules
        ``get_legal_actions`` offers creates by. For test or scenario setup
        use :meth:`place_unit`.

        Args:
            unit_type: 'W', 'M', 'C', 'B', or 'A'
            x: Grid x coordinate
            y: Grid y coordinate
            player: Player number (defaults to current player)

        Returns:
            Unit if created, None if failed
        """
        if player is None:
            player = self.current_player

        if self.game_over:
            logger.debug("Cannot create unit: the game is over")
            return None

        if player != self.current_player:
            logger.debug(f"Cannot create unit for player {player}: it is player {self.current_player}'s turn")
            return None

        if player in self.eliminated_players:
            logger.debug(f"Cannot create unit for player {player}: eliminated")
            return None

        if unit_type not in self.unit_data:
            logger.warning(f"Unknown unit type: {unit_type}")
            return None

        if unit_type not in self.enabled_units:
            logger.debug(f"Cannot create unit: {unit_type} is not enabled in this game")
            return None

        # Enforce the per-player unit cap. Mirrored in get_legal_actions so
        # the RL action mask hides create_unit at the cap rather than the
        # agent issuing a rejected action and eating the invalid_action
        # penalty.
        if not self._under_unit_cap(player):
            logger.debug(f"Cannot create unit: player {player} at unit cap ({self.max_units_per_player})")
            return None

        if not self._is_free_spawn_tile(player, x, y):
            logger.debug(f"Cannot create unit at ({x}, {y}): not an empty building owned by player {player}")
            return None

        cost = self.unit_data[unit_type]["cost"]
        if not self._can_afford(player, unit_type):
            logger.debug(f"Cannot create unit: insufficient gold ({self.player_gold[player]} < {cost})")
            return None

        # Create the unit
        self.player_gold[player] -= cost
        unit = Unit(unit_type, x, y, player, stats=self.unit_data[unit_type])
        unit.unit_id = self._next_unit_id
        self._next_unit_id += 1
        self.units.append(unit)
        self._invalidate_cache()
        # The new unit sees from its first moment (review core-12).
        self.update_visibility(player)

        # Record action. unit_id lets the replay player rebuild its
        # id -> Unit map on the fly (v3 schema), so subsequent
        # actions can find this unit even after it moves.
        self.record_action("create_unit", unit_type=unit_type, x=x, y=y, player=player, unit_id=unit.unit_id)

        logger.debug(f"Player {player} created {unit_type} at ({x}, {y})")
        return unit

    def move_unit(self, unit: Unit, to_x: int, to_y: int) -> bool:
        """
        Move a unit to a new position.

        Under fog of war the destination only has to be legal by what the
        player can see (see ``pathing_units``), and the unit takes the
        shortest such path the breadth-first search finds first (it tries
        up, down, left, right from each tile). If a hidden unit stands on
        that path or on the destination the unit is ambushed: it stops on
        the last free tile before it (possibly where it started), the move
        is spent (``unit.ambushed`` is set and ``cancel_move`` refuses to
        undo it) and is recorded to where the unit really stopped (with
        ``ambushed: True``), and the ambusher comes into view. Read
        ``unit.x``/``unit.y`` for where it ended up.

        Args:
            unit: Unit to move
            to_x: Target x coordinate
            to_y: Target y coordinate

        Returns:
            bool: True if the move happened (ambushed or not)
        """
        from_x, from_y = unit.x, unit.y

        # Actor gate (see _may_act). Among other things it rejects duplicate
        # moves -- bot/RL/LLM call sites don't all gate on ``unit.can_move``
        # before calling, and ``get_reachable_positions`` ignores it, so a
        # unit could otherwise move more than once per turn -- and stale
        # references: a unit can die mid-loop from a counter-attack while the
        # bot still holds it, and moving it would log an event the replay
        # player (which only sees self.units) can't reproduce (PR #360 audit).
        if not self._may_act("move", unit):
            return False

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        steps = self._move_paths(unit, came_from=came_from).get((to_x, to_y))
        if steps is None:
            logger.debug(f"Cannot move to ({to_x}, {to_y}): not reachable or occupied")
            return False

        # FOW: Snapshot pre-move enemy visibility so the unit cannot attack
        # enemies it discovers by moving. The UI's input_handler captures this
        # at unit-selection time; for RL/LLM/bot code paths that drive
        # move_unit directly, capture lazily here just before the move.
        if self.fog_of_war and unit.visible_enemies_at_action_start is None:
            self.capture_visible_enemies_for_unit(unit)
        # FOW: remember what the mover's side saw before the move, so a
        # cancel_move can take back what the move revealed (review core-9).
        if self.fog_of_war and unit.player in self.visibility_maps:
            unit.pre_move_visibility = self.visibility_maps[unit.player].copy()

        # FOW ambush rule: without fog of war the path was planned around
        # every unit, so it is always clear.
        ambusher = None
        if self.fog_of_war:
            path = [(to_x, to_y)]
            while path[-1] != (from_x, from_y):
                path.append(came_from[path[-1]])
            path.reverse()
            (to_x, to_y), ambusher = self._resolve_ambush(unit, path)
            if ambusher is not None:
                # Tiles actually stepped (for the path-length Knight's Charge).
                steps = path.index((to_x, to_y))
                logger.debug(
                    f"{unit.type} ambushed by {ambusher.type} at ({ambusher.x}, {ambusher.y}); stopped at ({to_x}, {to_y})"
                )

        # Execute move
        unit.move_to(to_x, to_y)
        if self.terrain_rules.charge_distance == "path":
            # Optional rule: the Knight's Charge counts the tiles along the
            # path, not the straight-line displacement move_to recorded.
            unit.distance_moved = steps
        unit.can_move = False  # Consume move action
        # An ambushed move is spent: cancel_move refuses to undo it, or a
        # human could scout with it for free and re-plan around the ambusher.
        unit.ambushed = ambusher is not None
        unit.haste_refreshed = False  # the haste-granted action has begun

        # Record action (where the unit really went, so replays need no
        # knowledge of the ambush rule)
        self.record_action(
            "move",
            unit_type=unit.type,
            from_x=from_x,
            from_y=from_y,
            to_x=to_x,
            to_y=to_y,
            player=unit.player,
            actor_unit_id=unit.unit_id,
            **({"ambushed": True} if ambusher is not None else {}),
        )

        logger.debug(f"Moved {unit.type} from ({from_x}, {from_y}) to ({to_x}, {to_y})")
        self._invalidate_cache()

        # Update visibility for the moving player
        self.update_visibility(unit.player)

        return True

    def attack(self, attacker: Unit, target: Unit) -> dict[str, Any]:
        """
        Execute an attack.

        Args:
            attacker: Attacking unit
            target: Target unit

        Returns:
            dict: Attack results. An illegal attack (see ``_may_act``; the
            target must also be a living enemy within reach that fog of war
            lets the attacker see) changes nothing and returns the no-op
            result with ``damage`` 0. An executed attack always deals at
            least 1, so ``result["damage"] > 0`` tells callers whether the
            attack happened.
        """
        # Rejections return the same shape as a clean attack so callers
        # that index into the result dict don't KeyError.
        if not self._may_act("attack", attacker, target, lambda: self._can_attack_target(attacker, target)):
            return self._noop_attack_result()

        result = self.mechanics.attack_unit(
            attacker, target, self.grid, self.units, damage_model=self.damage_model, rng=self.rng, teams=self.teams
        )

        # Record action. The extra fields (attacker_killed, counter_damage,
        # *_hp_after, evade, *_bonus) are what makes the replay
        # self-describing -- without them the replay player would have
        # to re-call mechanics.attack_unit, which re-rolls Rogue evade
        # RNG (mechanics.py: ``random.random() < evade_chance``) and
        # recomputes damage against the replay's potentially-diverged
        # unit HP. Recording the outcome lets replay apply it directly.
        self.record_action(
            "attack",
            attacker_type=attacker.type,
            attacker_pos=(attacker.x, attacker.y),
            target_type=target.type,
            target_pos=(target.x, target.y),
            damage=result["damage"],
            target_killed=not result["target_alive"],
            attacker_killed=not result["attacker_alive"],
            counter_damage=result["counter_damage"],
            attacker_hp_after=attacker.health if result["attacker_alive"] else 0,
            target_hp_after=target.health if result["target_alive"] else 0,
            evade=result["evade"],
            charge_bonus=result["charge_bonus"],
            flank_bonus=result["flank_bonus"],
            attack_buff=result["attack_buff"],
            defence_buff=result["defence_buff"],
            player=attacker.player,
            attacker_unit_id=attacker.unit_id,
            target_unit_id=target.unit_id,
        )

        # Handle unit deaths
        if not result["target_alive"]:
            target_tile = self.grid.get_tile(target.x, target.y)
            if target_tile.is_capturable() and target_tile.health < target_tile.max_health:
                target_tile.regenerating = True
            defeated_player = target.player
            self.units.remove(target)
            self._invalidate_cache()
            # A dead unit stops giving its owner vision (review core-12).
            self.update_visibility(defeated_player)
            self._check_player_eliminated(defeated_player)

        if not result["attacker_alive"]:
            attacker_tile = self.grid.get_tile(attacker.x, attacker.y)
            if attacker_tile.is_capturable() and attacker_tile.health < attacker_tile.max_health:
                attacker_tile.regenerating = True
            defeated_player = attacker.player
            if attacker in self.units:
                self.units.remove(attacker)
            self._invalidate_cache()
            self.update_visibility(defeated_player)
            self._check_player_eliminated(defeated_player)

        # Spend the attacker's action (only if still alive; a hasted attacker
        # gets its extra action instead, see _consume_action)
        if result["attacker_alive"]:
            self._consume_action(attacker)
        self._invalidate_cache()

        return result

    def _use_ability(
        self,
        action: str,
        actor: Unit,
        target: Unit,
        can_target: Callable[[Unit, Unit], bool],
        apply: Callable[[], Any],
        rejected: Any,
        actor_pos_field: str,
        record_fields: Callable[[Any], dict[str, Any]] | None = None,
        after_apply: Callable[[], None] | None = None,
    ) -> Any:
        """The shared body of the targeted abilities (paralyze, heal, cure, haste, the buffs).

        Validates with ``_may_act`` and ``can_target`` (the ability's
        ``_can_*_target`` predicate, the one its legal actions are listed
        with), returning ``rejected`` and changing nothing when that fails.
        Otherwise applies the mechanics call ``apply``; if it took effect,
        spends ``actor``'s action (``_consume_action``), runs
        ``after_apply``, records ``action`` with the fields ``actor_pos_field``
        (the actor's position), ``target_pos``, ``record_fields(result)``,
        ``player``, ``actor_unit_id`` and ``target_unit_id`` -- in that order,
        the record layout replays and saves have always had -- and
        invalidates the legal-action cache.

        Returns:
            ``apply``'s result, or ``rejected``.
        """
        if not self._may_act(action, actor, target, lambda: can_target(actor, target)):
            return rejected
        result = apply()
        # heal_unit returns the HP it restored (-1 if refused), the others a
        # bool; either way the ability took effect iff result > 0.
        if result > 0:
            self._consume_action(actor)
            if after_apply is not None:
                after_apply()
            self.record_action(
                action,
                **{actor_pos_field: (actor.x, actor.y)},
                target_pos=(target.x, target.y),
                **(record_fields(result) if record_fields is not None else {}),
                player=actor.player,
                actor_unit_id=actor.unit_id,
                target_unit_id=target.unit_id,
            )
            self._invalidate_cache()
        return result

    def paralyze(self, paralyzer: Unit, target: Unit) -> bool:
        """Paralyze a target unit. Returns False, changing nothing, if illegal."""
        return self._use_ability(
            "paralyze",
            paralyzer,
            target,
            self._can_paralyze_target,
            lambda: self.mechanics.paralyze_unit(paralyzer, target, self.teams),
            rejected=False,
            actor_pos_field="paralyzer_pos",
        )

    def heal(self, healer: Unit, target: Unit) -> int:
        """Heal a target unit. Returns the HP healed; 0, changing nothing, if illegal."""
        return self._use_ability(
            "heal",
            healer,
            target,
            self._can_heal_target,
            lambda: self.mechanics.heal_unit(healer, target, self.teams),
            rejected=0,
            actor_pos_field="healer_pos",
            # target_hp_after lets the replay player set HP directly
            # instead of re-calling mechanics.heal_unit (the only path
            # today that could observe HEAL_AMOUNT drift between save
            # and replay).
            record_fields=lambda amount: {"amount": amount, "target_hp_after": target.health},
        )

    def cure(self, curer: Unit, target: Unit) -> bool:
        """Cure a target unit's paralysis. Returns False, changing nothing, if illegal."""
        return self._use_ability(
            "cure",
            curer,
            target,
            self._can_cure_target,
            lambda: self.mechanics.cure_unit(curer, target, self.teams),
            rejected=False,
            actor_pos_field="curer_pos",
        )

    def haste(self, sorcerer: Unit, target: Unit) -> bool:
        """
        Sorcerer grants Haste to a target unit.

        Haste gives the target one extra full action this turn (a move and an
        attack, ability or seize). The target must be one of the Sorcerer's
        own units, alive, unparalyzed and not already hasted. If it has
        already spent its action this turn, the extra action is granted at
        once; otherwise it is granted when the target spends its current one
        (see ``_consume_action``) -- by acting, or when its controller ends
        its action (``end_unit_turn``: the GUI's Wait, a bot done with it).
        The RL action space has no Wait, so there a hasted unit's first
        action ends only by acting.

        Args:
            sorcerer: The Sorcerer unit using Haste
            target: The target friendly unit

        Returns:
            bool: True if Haste was successfully applied (False, changing
            nothing, if illegal)
        """

        def grant_if_already_acted() -> None:
            if not (target.can_move or target.can_attack):
                # Already done for the turn: the extra action starts now.
                self._consume_action(target)
                # haste_refreshed shields the unit that just acted from the
                # end_unit_turn its controller calls next. The target didn't
                # act, so its controller's next end_unit_turn (the GUI's Wait)
                # must end the extra action, not be swallowed.
                target.haste_refreshed = False

        return self._use_ability(
            "haste",
            sorcerer,
            target,
            self._can_haste_target,
            lambda: self.mechanics.haste_unit(sorcerer, target),
            rejected=False,
            actor_pos_field="sorcerer_pos",
            record_fields=lambda _: {"target_type": target.type},
            after_apply=grant_if_already_acted,
        )

    def defence_buff(self, sorcerer: Unit, target: Unit) -> bool:
        """
        Sorcerer grants Defence Buff to a target unit.

        Args:
            sorcerer: The Sorcerer unit using Defence Buff
            target: The target friendly unit

        Returns:
            bool: True if Defence Buff was successfully applied (False,
            changing nothing, if illegal)
        """
        return self._use_ability(
            "defence_buff",
            sorcerer,
            target,
            self._can_defence_buff_target,
            lambda: self.mechanics.defence_buff_unit(sorcerer, target, self.teams),
            rejected=False,
            actor_pos_field="sorcerer_pos",
            record_fields=lambda _: {"target_type": target.type},
        )

    def attack_buff(self, sorcerer: Unit, target: Unit) -> bool:
        """
        Sorcerer grants Attack Buff to a target unit.

        Args:
            sorcerer: The Sorcerer unit using Attack Buff
            target: The target friendly unit

        Returns:
            bool: True if Attack Buff was successfully applied (False,
            changing nothing, if illegal)
        """
        return self._use_ability(
            "attack_buff",
            sorcerer,
            target,
            self._can_attack_buff_target,
            lambda: self.mechanics.attack_buff_unit(sorcerer, target, self.teams),
            rejected=False,
            actor_pos_field="sorcerer_pos",
            record_fields=lambda _: {"target_type": target.type},
        )

    def seize(self, unit: Unit) -> dict[str, Any]:
        """Seize the structure the unit is on.

        An illegal seize (see ``_may_act``; the unit must also stand on a
        structure its player does not own) changes and records nothing and
        returns a result without ``damage``. Checking ``can_attack`` here is
        what stops one unit from seizing several times a turn -- repeated
        SEIZEs from an LLM took a 50-HP HQ in one turn (review aibots-1).
        """
        tile = self.grid.get_tile(unit.x, unit.y)
        if not self._may_act("seize", unit, rule=lambda: self._can_seize(unit)):
            return {"captured": False, "game_over": False, "structure_type": tile.type if tile else None}
        previous_owner = tile.player
        result = self.mechanics.seize_structure(unit, tile, self.teams)

        # Record action. tile_hp_after / tile_owner_after let the v2
        # replay player set tile state directly instead of re-calling
        # mechanics.seize_structure -- which would decrement by
        # unit.health and only match the original if every prior
        # action's HP-mutation reproduced exactly. Without this,
        # any replay that starts from a partially-damaged tile (e.g.
        # a unit-test that pokes tile.health) desyncs immediately.
        # mechanics.seize_structure also clears tile.regenerating on
        # any successful call, so we don't need to record it separately.
        self.record_action(
            "seize",
            unit_type=unit.type,
            position=(unit.x, unit.y),
            structure_type=tile.type,
            captured=result["captured"],
            tile_hp_after=tile.health,
            tile_owner_after=tile.player,
            player=unit.player,
            actor_unit_id=unit.unit_id,
        )

        if result["captured"] and tile.type == TileType.HEADQUARTERS.value:
            self._on_hq_captured(unit.player, previous_owner)
            # mechanics flags every HQ capture; whether it ended the game is
            # the engine's call (not in a free-for-all, nor for a neutral HQ).
            result["game_over"] = self.game_over

        self._consume_action(unit)
        self._invalidate_cache()

        # A captured structure gives its vision to the capturer and takes it
        # from the previous owner (review core-12).
        if result["captured"]:
            self.update_visibility()

        return result

    def heal_units_on_structures(self, player: int) -> dict[str, Any]:
        """
        Heal units on owned structures at the start of their turn.

        Healing amounts:
        - Tower: 1 HP
        - HQ/Building: 2 HP

        Cost formula: (heal_amount / unit_max_hp) * unit_cost (rounded)

        Args:
            player: Player number whose units to heal

        Returns:
            Dict with healing statistics
        """
        stats = {"total_healed": 0, "total_cost": 0, "units_healed": []}

        # Find enemy HQ for distance calculations
        enemy_hq_pos = None
        for row in self.grid.tiles:
            for tile in row:
                if tile.type == TileType.HEADQUARTERS.value and self.are_enemies(tile.player, player):
                    enemy_hq_pos = (tile.x, tile.y)
                    break
            if enemy_hq_pos:
                break

        # Collect units that need healing on owned structures
        units_to_heal = []

        for unit in self.units:
            if unit.player != player:
                continue
            if unit.health >= unit.max_health:
                continue

            tile = self.grid.get_tile(unit.x, unit.y)
            if not tile or tile.player != player:
                continue

            # Determine heal amount based on structure type
            heal_amount = 0
            structure_name = ""

            if tile.type == TileType.TOWER.value:
                heal_amount = 1
                structure_name = "Tower"
            elif tile.type == TileType.HEADQUARTERS.value:
                heal_amount = 2
                structure_name = "Headquarters"
            elif tile.type == TileType.BUILDING.value:
                heal_amount = 2
                structure_name = "Building"

            if heal_amount > 0:
                distance = float("inf")
                if enemy_hq_pos:
                    distance = abs(unit.x - enemy_hq_pos[0]) + abs(unit.y - enemy_hq_pos[1])

                units_to_heal.append(
                    {"unit": unit, "heal_amount": heal_amount, "structure_name": structure_name, "distance": distance}
                )

        # Sort by distance to enemy HQ (closest first - priority)
        units_to_heal.sort(key=lambda x: x["distance"])

        # Process healing for each unit
        for heal_data in units_to_heal:
            unit = heal_data["unit"]
            requested_heal = heal_data["heal_amount"]
            structure_name = heal_data["structure_name"]

            max_possible_heal = unit.max_health - unit.health
            desired_heal = min(requested_heal, max_possible_heal)

            unit_cost = self.unit_data[unit.type]["cost"]
            cost_per_hp = unit_cost / unit.max_health

            actual_heal = 0
            actual_cost = 0

            if structure_name == "Tower":
                # Towers: All or nothing (1 HP)
                total_cost = round(cost_per_hp * desired_heal)
                if self.player_gold[player] >= total_cost:
                    actual_heal = desired_heal
                    actual_cost = total_cost
            else:  # HQ or Building - allow partial healing
                for hp in range(desired_heal, 0, -1):
                    cost = round(cost_per_hp * hp)
                    if self.player_gold[player] >= cost:
                        actual_heal = hp
                        actual_cost = cost
                        break

            if actual_heal > 0:
                old_health = unit.health
                unit.health = min(unit.health + actual_heal, unit.max_health)
                self.player_gold[player] -= actual_cost

                stats["total_healed"] += actual_heal
                stats["total_cost"] += actual_cost
                stats["units_healed"].append(
                    {
                        "unit_type": unit.type,
                        "position": (unit.x, unit.y),
                        "structure": structure_name,
                        "healed": actual_heal,
                        "cost": actual_cost,
                        "old_health": old_health,
                        "new_health": unit.health,
                    }
                )

                logger.debug(
                    f"Healed {unit.type} on {structure_name} at ({unit.x}, {unit.y}): "
                    f"{actual_heal} HP ({old_health} → {unit.health}) for ${actual_cost}"
                )

        # Game-lifetime accumulator (see __init__): the per-call stats
        # returned below are usually discarded by callers, so this is
        # what diagnostics read after the game.
        totals = self.healing_totals.setdefault(player, {"hp": 0, "gold": 0})
        totals["hp"] += stats["total_healed"]
        totals["gold"] += stats["total_cost"]

        return stats

    def end_turn(self) -> dict[str, Any]:
        """End the current player's turn and pass to the next player."""
        # No-op once the game has ended. Prevents turn_number from being
        # bumped past the winning turn (which would otherwise make
        # game_info.turns disagree with max(action.turn) in the replay)
        # and avoids running structure regen / paralysis decrement /
        # max_turns checks on a finished game.
        if self.game_over:
            return {"total": 0, "healing": {"total_healed": 0, "total_cost": 0, "units_healed": []}}

        # Everything below changes what is legal (can_move/can_attack resets,
        # paralysis and cooldown ticks, income, healing, current_player), and
        # nothing below reads the legal-action cache, so one invalidation up
        # front covers every exit path. Without it, a turn in which nothing
        # else mutates state hands the next player the actions cached at the
        # end of its previous turn (e.g. RandomBot's empty list).
        self._invalidate_cache()

        # Record action
        self.record_action("end_turn", player=self.current_player)

        # Reset structures that were vacated this turn
        for unit in self.units:
            if unit.player == self.current_player and unit.has_moved:
                old_tile = self.grid.get_tile(unit.original_x, unit.original_y)
                if (unit.x, unit.y) != (unit.original_x, unit.original_y):
                    self.mechanics.reset_structure_if_vacated(old_tile, self.units)

        # Regenerate structures
        self.mechanics.regenerate_structures(self.grid, self.units)

        # Vision a move revealed can no longer be taken back (cancel_move).
        for unit in self.units:
            unit.pre_move_visibility = None

        # Move to the next seat still in the game (review core-7: eliminated
        # players get no turns, so no income and no new units either).
        # Checked once per full round, after all players have gone: the
        # max_turns limit, as it always was.
        for _ in range(self.num_players):
            self.current_player += 1
            if self.current_player > self.num_players:
                self.current_player = 1
                self.turn_number += 1
                if self.max_turns is not None and self.turn_number >= self.max_turns:
                    self._set_game_over(winner=None, end_reason="max_turns_draw")
                    return {"total": 0, "healing": {"total_healed": 0, "total_cost": 0, "units_healed": []}}
            if self.current_player not in self.eliminated_players:
                break

        return self._begin_turn(self.current_player)

    def _begin_turn(self, player: int) -> dict[str, Any]:
        """Start-of-turn processing for ``player``, whose turn is starting (review core-13).

        In order: paralysis, cooldown and buff-duration ticks for the
        player's units; re-arming them (a unit still paralyzed after the tick
        stays disabled) and resetting their per-turn move/haste bookkeeping;
        income; auto-healing on owned structures; the player's fog-of-war
        update. ``end_turn`` runs it for every turn but Player 1's first,
        which by default starts without it (engine override
        ``begin_first_turn``; see ``_resolve_begin_first_turn``).

        Returns:
            The income breakdown (``calculate_income``) with the healing
            stats under ``"healing"`` -- what ``end_turn`` returns.
        """
        # Tick paralysis, ability cooldowns and buff durations of the
        # player's units (only theirs: durations count the unit's own turns)
        self.mechanics.tick_statuses(self.units, player)

        for unit in self.units:
            if unit.player == player:
                if not unit.is_paralyzed():
                    unit.can_move = True
                    unit.can_attack = True
                else:
                    unit.can_move = False
                    unit.can_attack = False

                unit.original_x = unit.x
                unit.original_y = unit.y
                unit.has_moved = False
                unit.ambushed = False
                unit.distance_moved = 0
                unit.is_hasted = False
                unit.haste_refreshed = False
                # FOW: Clear stale snapshot so it gets recaptured before
                # this unit's next move (see move_unit lazy capture).
                unit.visible_enemies_at_action_start = None
            unit.selected = False

        # Calculate and apply income
        income_data = self.mechanics.calculate_income(player, self.grid, self.income_rates)
        self.player_gold[player] += income_data["total"]

        # Heal units on structures after income collection
        healing_stats = self.heal_units_on_structures(player)
        income_data["healing"] = healing_stats

        # Update visibility for the new current player
        self.update_visibility(player)

        self._invalidate_cache()
        return income_data

    def resign(self, player: int | None = None) -> None:
        """``player`` (default: the current player) resigns.

        Its units are removed and it is eliminated (see ``_eliminate_player``):
        in a 1v1 the opponent wins at once, as before; with more seats the
        others play on until one team is left. Resigning on your own turn
        leaves you the current player until ``end_turn`` hands the turn on
        (the GUI does that for you); an eliminated player has nothing left
        to do but end its turn. No-op once the game is over or for a player
        already out.
        """
        if player is None:
            player = self.current_player
        if self.game_over or player in self.eliminated_players:
            return

        self.record_action("resign", player=player)

        # Remove resigning player's units (in place: bots and the renderer
        # may hold a reference to the list)
        self.units[:] = [u for u in self.units if u.player != player]
        self._invalidate_cache()

        if self.legacy_end_rules:
            self._legacy_last_player_standing(player, "resign")
            return
        self._eliminate_player(player, "resign")

    def end_unit_turn(self, unit: Unit, force_end: bool = False) -> bool:
        """End ``unit``'s current action through the engine (the GUI's Wait).

        Haste is applied by the engine when a unit spends its action
        (``_consume_action``), so nobody needs to call this after an action
        to get the extra one. Called right after an action that haste
        refreshed (``unit.haste_refreshed``), it keeps the extra action and
        returns True, so the GUI and bots that still call it there are
        unaffected. Otherwise a hasted unit that ends its action without
        acting spends its haste on it and is refreshed (True), and any other
        unit is done for the turn (False). ``force_end`` always ends the
        turn. Goes through ``Unit.end_unit_turn`` and invalidates the
        legal-action cache, which the unit-level call cannot.

        Returns:
            True if the unit can still act (haste was consumed).
        """
        if unit.haste_refreshed and not force_end:
            unit.haste_refreshed = False
            return True
        unit.haste_refreshed = False
        can_still_act = unit.end_unit_turn(force_end=force_end)
        unit.ambushed = False  # the action is over (see move_unit)
        self._invalidate_cache()
        return can_still_act

    def can_cancel_move(self, unit: Unit) -> bool:
        """Whether ``cancel_move`` would undo ``unit``'s move (the GUI offers it only then).

        On the current player's turn, the unit must have moved and not acted
        since (the GUI's post-move menu), its starting tile must be free, and
        the move must not have run into a fog-of-war ambush: an ambushed move
        is spent (see ``move_unit``). Under fog of war only the latest action
        can be cancelled: once another action followed the move, it may have
        used what the move revealed (another unit's attack on an enemy the
        move uncovered), which no restore can take back. It also needs the
        side's view from before the move (``pre_move_visibility``) to put
        back; a save written before that was saved has none.
        """
        if (
            self.game_over
            or unit not in self.units
            or unit.player != self.current_player
            or not unit.has_moved
            or not unit.can_attack
            or unit.ambushed
        ):
            return False
        occupant = self.get_unit_at_position(unit.original_x, unit.original_y)
        if occupant is not None and occupant is not unit:
            return False
        return not (self.fog_of_war and (unit.pre_move_visibility is None or not self._move_is_latest_action(unit)))

    def _move_is_latest_action(self, unit: Unit) -> bool:
        """Whether the last recorded action is ``unit``'s move to where it stands."""
        last = self.action_history[-1] if self.action_history else None
        return (
            last is not None
            and last.get("type") == "move"
            and last.get("actor_unit_id") == unit.unit_id
            and (last.get("to_x"), last.get("to_y")) == (unit.x, unit.y)
        )

    def cancel_move(self, unit: Unit) -> bool:
        """Take back ``unit``'s move, returning it to where its action started (review core-9).

        Refused (returns False, changing nothing) when ``can_cancel_move`` is
        False. Takes the move out of the record too: when the move is the
        latest action its record is removed, so the game reads as if it never
        happened; otherwise a ``cancel_move`` action is recorded for replays
        to apply. Under fog of war the mover's side also loses what the move
        revealed -- its visibility map is restored to its pre-move state and
        recomputed from where units now stand -- so moving and cancelling
        can't be used to scout.

        Returns:
            True if the move was cancelled.
        """
        if not self.can_cancel_move(unit):
            return False
        origin = (unit.original_x, unit.original_y)
        moved_to = (unit.x, unit.y)
        is_latest = self._move_is_latest_action(unit)
        if not unit.cancel_move():
            return False

        if is_latest:
            self.action_history.pop()
        else:
            self.record_action(
                "cancel_move",
                unit_type=unit.type,
                from_x=moved_to[0],
                from_y=moved_to[1],
                to_x=origin[0],
                to_y=origin[1],
                player=unit.player,
                actor_unit_id=unit.unit_id,
            )

        snapshot, unit.pre_move_visibility = unit.pre_move_visibility, None
        if self.fog_of_war:
            if snapshot is not None:
                self.visibility_maps[unit.player] = snapshot
            # Any attack snapshot taken since the move (by this unit or
            # another one yet to move, e.g. one that moved and was cancelled
            # in turn) may hold enemies only the move revealed; drop them so
            # they are taken again from the restored view. Units that moved
            # took theirs before this move, the latest action.
            for other in self.units:
                if other.player == unit.player and not other.has_moved:
                    other.visible_enemies_at_action_start = None
            # Re-derive what the side sees from where its units stand now
            # (the unit back on its origin, any later mover where it went).
            # Without this the tiles it saw from where it moved to stayed
            # VISIBLE, so known_structure served their live state.
            self.update_visibility(unit.player)

        self._invalidate_cache()
        return True

    def get_legal_actions(self, player: int | None = None) -> dict[str, list[Any]]:
        """
        Get all legal actions for the current player.

        Returns:
            dict: Legal actions organized by type
        """
        if player is None:
            player = self.current_player

        # Return cached actions if available and cache is valid
        if self._legal_actions_cache_valid and player in self._legal_actions_cache:
            cached = self._legal_actions_cache[player]
            if _CHECK_LEGAL_ACTION_CACHE:
                fresh = self._compute_legal_actions(player)
                if fresh != cached:
                    stale = {k: (len(cached.get(k, [])), len(v)) for k, v in fresh.items() if cached.get(k) != v}
                    raise AssertionError(
                        f"Stale legal-action cache for player {player} "
                        f"(turn {self.turn_number}): (cached, fresh) counts by type {stale}"
                    )
            return cached

        legal_actions = self._compute_legal_actions(player)

        # Cache the result
        self._legal_actions_cache[player] = legal_actions
        self._legal_actions_cache_valid = True

        return legal_actions

    def _compute_legal_actions(self, player: int) -> dict[str, list[Any]]:
        """Enumerate ``player``'s legal actions from the current state (uncached)."""
        legal_actions = {
            "create_unit": [],
            "move": [],
            "attack": [],
            "paralyze": [],
            "heal": [],
            "cure": [],
            "haste": [],
            "defence_buff": [],
            "attack_buff": [],
            "seize": [],
            "end_turn": True,
        }

        # Every test below is one of the predicates the action methods
        # validate with (see "Legality predicates"), so everything offered
        # here is accepted on the player's turn and nothing else is. The
        # iteration order (units, reachable tiles, targets) is part of the
        # flat_discrete action encoding and must not change.

        # Building units (only at Buildings, not HQ)
        # Only include enabled unit types. Suppressed entirely once the player
        # is at the unit cap so the action mask matches create_unit's own
        # enforcement (no offered-then-rejected create actions).
        if self._under_unit_cap(player):
            for tile in self.grid.get_capturable_tiles(player):
                if self._is_free_spawn_tile(player, tile.x, tile.y):
                    for unit_type in self.enabled_units:
                        if self._can_afford(player, unit_type):
                            legal_actions["create_unit"].append({"unit_type": unit_type, "x": tile.x, "y": tile.y})

        # Unit actions. Every move search shares one occupancy set, and one
        # blocker set: who blocks a unit depends only on its player. Both come
        # from the units this player knows of (see ``pathing_units``).
        known_units = self.pathing_units(player)
        occupied = {(u.x, u.y) for u in known_units}
        blocked: set[tuple[int, int]] | None = None
        for unit in self.units:
            # Guard on health: dead units are normally removed synchronously
            # by ``attack`` (see self.units.remove), but the helpers below all
            # filter on ``health > 0`` defensively -- mirror that here so a
            # corpse left in ``self.units`` by any future deferred-removal path
            # (AoE, end-of-turn DoT, status damage) can't emit phantom actions.
            if not self._is_ready_unit(unit, player):
                continue

            # Movement: reachable tiles that are also free to end on
            if unit.can_move:
                if blocked is None:
                    blocked = self.mechanics.movement_blockers(known_units, unit, self.teams)
                for pos in self._move_paths(unit, occupied, blocked):
                    legal_actions["move"].append(
                        {"unit": unit, "from_x": unit.x, "from_y": unit.y, "to_x": pos[0], "to_y": pos[1]}
                    )

            if not unit.can_attack:
                continue

            # Combat: every enemy in reach (adjacent for melee, 1-2 for
            # Mages/Sorcerers, 2-3 or 2-4 on a mountain for Archers); Mages
            # can also paralyze one of them when off cooldown.
            for enemy in self.units:
                if self._can_attack_target(unit, enemy):
                    legal_actions["attack"].append({"attacker": unit, "target": enemy})
                    if self._can_paralyze_target(unit, enemy):
                        legal_actions["paralyze"].append({"paralyzer": unit, "target": enemy})

            # Healing / curing (Cleric only) - range 1..CLERIC_HEAL_RANGE
            if unit.type == "C":
                for ally in self.mechanics.get_healable_allies(unit, self.units, self.teams):
                    legal_actions["heal"].append({"healer": unit, "target": ally})
                for ally in self.mechanics.get_curable_allies(unit, self.units, self.teams):
                    legal_actions["cure"].append({"curer": unit, "target": ally})

            # Sorcerer abilities, each gated on its own cooldown
            if unit.can_use_haste():
                for ally in self.mechanics.get_hasteable_allies(unit, self.units):
                    legal_actions["haste"].append({"sorcerer": unit, "target": ally})
            if unit.can_use_defence_buff():
                for ally in self.mechanics.get_defence_buffable_allies(unit, self.units, self.teams):
                    legal_actions["defence_buff"].append({"sorcerer": unit, "target": ally})
            if unit.can_use_attack_buff():
                for ally in self.mechanics.get_attack_buffable_allies(unit, self.units, self.teams):
                    legal_actions["attack_buff"].append({"sorcerer": unit, "target": ally})

            # Seizing
            if self._can_seize(unit):
                legal_actions["seize"].append({"unit": unit, "tile": self.grid.get_tile(unit.x, unit.y)})

        return legal_actions

    # How clone_for_search treats each attribute (review core-18). Shared:
    # fixed for the whole game (configuration, terrain source, stateless
    # helpers), so the clone references the original's object. Dropped:
    # history and caches search never reads, replaced by empty values.
    # Anything else is deep-copied, so state added later is safe by default.
    _SEARCH_SHARED_ATTRS = frozenset(
        {
            "engine_overrides",
            "unit_data",
            "income_rates",
            "structure_health",
            "terrain_rules",
            "mechanics",
            "enabled_units",
            "initial_map_data",
            "player_configs",
            "game_start_time",
        }
    )
    _SEARCH_DROPPED_ATTRS: dict[str, Callable[[], Any]] = {
        "action_history": list,
        "_legal_actions_cache": dict,
        "_legal_actions_cache_valid": lambda: False,
    }

    def clone_for_search(self) -> GameState:
        """An independent copy of the game for tree search (MCTS), made cheaply.

        Plays exactly like ``copy.deepcopy(self)`` (same legal actions, same
        outcomes, including the combat RNG's position), and nothing done to
        the clone touches the original. It drops what search never reads --
        the action history (which grows all game and dominated deepcopy's
        cost) and the legal-action cache -- and shares, instead of copying,
        the terrain tiles, configuration and replay metadata, none of which
        change during a game. The clone's own ``action_history`` starts
        empty, so it cannot be saved as a replay of the whole game.
        """
        clone = GameState.__new__(GameState)
        for name, value in vars(self).items():
            if name in self._SEARCH_SHARED_ATTRS:
                setattr(clone, name, value)
            elif name in self._SEARCH_DROPPED_ATTRS:
                setattr(clone, name, self._SEARCH_DROPPED_ATTRS[name]())
            elif name == "grid":
                clone.grid = self._clone_grid_for_search(value)
            elif name == "units":
                clone.units = [self._clone_unit_for_search(unit) for unit in value]
            else:
                setattr(clone, name, copy.deepcopy(value))
        return clone

    @staticmethod
    def _clone_grid_for_search(grid: TileGrid) -> TileGrid:
        """New row lists and structure tiles; terrain tiles are shared.

        Only structures change during a game (owner, HP, regeneration), so
        plain terrain tiles are shared between the original and its clones.
        """
        clone = copy.copy(grid)
        clone.tiles = [[copy.copy(tile) if tile.is_capturable() else tile for tile in row] for row in grid.tiles]
        return clone

    @staticmethod
    def _clone_unit_for_search(unit: Unit) -> Unit:
        """A copy of ``unit`` whose containers are its own.

        ``attack_data`` is the unit type's stat entry (never mutated), so it
        stays shared; any other mutable value (the fog-of-war snapshot, the
        pre-move visibility map that ``cancel_move`` restores) is copied so
        mutating it in the clone cannot reach the original.
        """
        clone = copy.copy(unit)
        for name, value in vars(unit).items():
            if name != "attack_data" and not isinstance(value, _IMMUTABLE_UNIT_FIELD_TYPES):
                setattr(clone, name, copy.deepcopy(value))
        return clone

    @staticmethod
    def _unit_to_save_dict(unit: Unit) -> dict[str, Any]:
        """``unit.to_dict()`` plus, under fog of war, the view ``cancel_move`` restores.

        A unit whose move can still be cancelled carries its side's
        visibility map from before the move, so a game saved at that point
        can cancel it after loading, taking back what the move revealed.
        """
        data = unit.to_dict()
        if unit.pre_move_visibility is not None:
            data["pre_move_visibility"] = unit.pre_move_visibility.to_dict()
        return data

    def to_dict(self) -> dict[str, Any]:
        """Convert game state to dictionary for serialization.

        Records everything ``from_dict`` needs to resume the game exactly:
        ``from_dict(json(to_dict(g))).to_dict()`` equals ``json(to_dict(g))``,
        and both games offer every player the same legal actions (see
        tests/test_save_roundtrip_core.py). Not recorded: the engine RNG
        (``rng``; a loaded game rolls Rogue evades from the module-global
        ``random`` unless given a new one), the legal-action caches, and the
        UI-only ``Unit.selected``. Mutable containers are copied, so the
        result shares nothing with the live game.
        """
        return {
            "save_format_version": SAVE_FORMAT_VERSION,
            "timestamp": self.game_start_time.strftime("%Y-%m-%d %H-%M-%S"),
            "current_player": self.current_player,
            "num_players": self.num_players,
            "player_gold": dict(self.player_gold),
            "turn_number": self.turn_number,
            "game_over": self.game_over,
            "winner": self.winner,
            # from_dict has always read these two back; without them a
            # turn-limited game reloaded as unlimited and a finished game
            # lost how it ended.
            "end_reason": self.end_reason,
            "max_turns": self.max_turns,
            # The index of the action that ended the game, and the
            # game-lifetime auto-heal totals: both feed the replay's
            # integrity fields, which were wrong for continued games.
            "winning_action_index": self.game_over_action_index,
            "healing_totals": {p: dict(t) for p, t in self.healing_totals.items()},
            "map_file": self.map_file_used,
            # The exact tile codes the grid was built from (after any UI
            # padding), written for every save, not only map-file-less ones.
            # "tiles" below holds only capturable tiles, so without this a
            # random-map save could never be reloaded, and a save that
            # points at a map file breaks silently if that file is later
            # edited, renamed or padded differently: unit and structure
            # coordinates would land on different terrain. The cost is one
            # short string per tile: a fixed ~7 KB for the usual 24x24 padded
            # map at the save writer's indent=2, about a quarter of an
            # early-game save and a shrinking share as action_history grows.
            "map_data": [list(row) for row in self.initial_map_data],
            "player_configs": copy.deepcopy(self.player_configs),
            "enabled_units": list(self.enabled_units),
            "fog_of_war": self.fog_of_war,
            "fog_of_war_method": self.fog_of_war_method,
            # What each player has explored and remembers. Without it a
            # reloaded fog-of-war game re-fogged every tile out of current
            # sight and forgot every structure (critic-integration-11).
            # Keyed by str(player) so the dict is the same before and after
            # a JSON round trip.
            "fog_of_war_state": {str(p): vis_map.to_dict() for p, vis_map in self.visibility_maps.items()},
            # Persist the engine-constant overlay so a reloaded game runs under
            # the same balance (damage_model, structure HP, economy, unit cap)
            # it was saved under. Absent in pre-0.3.3 saves -> from_dict falls
            # back to {} (== module defaults), preserving backward-compat.
            "engine_overrides": copy.deepcopy(self.engine_overrides),
            # The combat RNG (review core-10): its seed, and its position so a
            # reloaded game rolls exactly what the unsaved game would have.
            "seed": self.seed,
            "rng_state": self._rng_state_for_save(),
            # Every seat's team and who is out (review core-4/core-7). Teams a
            # map declares are re-derived from map_data on load and must
            # agree; this also carries ones passed as GameState(teams=...).
            "teams": self.teams,
            "eliminated_players": sorted(self.eliminated_players),
            "units": [self._unit_to_save_dict(unit) for unit in self.units],
            "tiles": self.grid.to_dict()["tiles"],
            # Records are never edited once written, so a new list suffices.
            "action_history": list(self.action_history),
            # Restore the per-game unit-id counter on reload so newly
            # created units after load don't reuse retired ids
            # (which would let the replay v3 dispatch route an action
            # to the wrong unit -- exactly the brittleness this whole
            # schema bump is meant to eliminate).
            "next_unit_id": self._next_unit_id,
        }

    def to_numpy(self, for_player: int | None = None) -> dict[str, np.ndarray]:
        """
        Convert game state to numpy arrays for RL.

        Args:
            for_player: If specified and fog_of_war is enabled, filter observation
                        to only show what this player can see. If None, shows full state.

        Returns:
            dict with numpy arrays
        """
        # Grid representation
        grid_state = self.grid.to_numpy()

        # Unit representation (height x width x 8)
        #   [..., 0] = unit_type int (0 = empty, 1..8 = ALL_UNIT_TYPES)
        #   [..., 1] = absolute owner (0 = empty cell, else player number)
        #   [..., 2] = unit HP percentage in [0, 100]
        #   [..., 3] = exhausted flag (1.0 if the unit has no actions left
        #             this turn, 0.0 otherwise). Defined as
        #             ``not (can_move or can_attack)`` so it captures every
        #             way a unit spends its turn -- moving, attacking,
        #             seizing, healing, or casting in place -- not just
        #             movement. (A unit that moved but can still attack reads
        #             0.0; a unit that attacked without moving reads 1.0.)
        #             Consumed by build_observation as a per-unit "exhausted"
        #             signal for the policy.
        #   [..., 4] = paralyzed_turns (0..PARALYZE_DURATION + 1). Surfaces the
        #             Mage paralyze debuff so the policy can value attacking /
        #             defending paralyzed targets correctly.
        #   [..., 5] = is_hasted (0.0 / 1.0). Surfaces the Sorcerer haste
        #             buff (extra-action-this-turn) so the policy can see
        #             which units still have an action left.
        #   [..., 6] = defence_buff_turns (0..SORCERER_BUFF_DURATION).
        #             Surfaces the Sorcerer defence buff so the policy can
        #             account for the +50% damage reduction on the unit.
        #   [..., 7] = attack_buff_turns (0..SORCERER_BUFF_DURATION).
        #             Surfaces the Sorcerer attack buff so the policy can
        #             account for the +50% damage bonus on the unit.
        unit_state = np.zeros((self.grid.height, self.grid.width, 8), dtype=np.float32)

        # Encoding for all 8 unit types: W, M, C, A, K, R, S, B
        unit_type_encoding = {"W": 1, "M": 2, "C": 3, "A": 4, "K": 5, "R": 6, "S": 7, "B": 8}

        # Visibility mask for FOW
        visibility_state = np.full((self.grid.height, self.grid.width), VISIBLE, dtype=np.uint8)

        # The player whose knowledge filters the arrays: set only under fog
        # of war when it has a visibility map (otherwise everything counts as
        # visible, as is_position_visible has it).
        fog_player: int | None = None
        if self.fog_of_war and for_player is not None:
            vis_map = self.visibility_maps.get(for_player)
            if vis_map is not None:
                visibility_state = vis_map.to_numpy()
                fog_player = for_player

        for unit in self.units:
            # FOW: Only show units that are visible or owned by the player
            if fog_player is not None:
                if unit.player != fog_player and visibility_state[unit.y, unit.x] != VISIBLE:
                    continue

            unit_state[unit.y, unit.x, 0] = unit_type_encoding.get(unit.type, 0)
            unit_state[unit.y, unit.x, 1] = unit.player
            unit_state[unit.y, unit.x, 2] = (unit.health / unit.max_health) * 100
            unit_state[unit.y, unit.x, 3] = 0.0 if (unit.can_move or unit.can_attack) else 1.0
            # Status effects (raw turn counts; observation.py normalises
            # by their respective max durations to land in [0, 1]).
            unit_state[unit.y, unit.x, 4] = float(getattr(unit, "paralyzed_turns", 0))
            unit_state[unit.y, unit.x, 5] = 1.0 if getattr(unit, "is_hasted", False) else 0.0
            unit_state[unit.y, unit.x, 6] = float(getattr(unit, "defence_buff_turns", 0))
            unit_state[unit.y, unit.x, 7] = float(getattr(unit, "attack_buff_turns", 0))

        # FOW: show only what for_player knows of the board (review core-5)
        if fog_player is not None:
            # Never-explored tiles: terrain, owner and HP all unknown.
            grid_state[visibility_state == UNEXPLORED] = 0
            # Structures out of sight show their owner and HP as the player
            # last saw them (known_structure), not live: a capture or seize
            # made out of sight must not reach the observation. Only
            # structures have owner/HP that change, so plain terrain needs
            # nothing beyond the mask above.
            for x, y in self.grid.structure_positions:
                if visibility_state[y, x] == VISIBLE:
                    continue
                known = self.known_structure(fog_player, x, y)
                tile = self.grid.tiles[y][x]
                if known is None:
                    grid_state[y, x, 1:] = 0
                else:
                    grid_state[y, x, 1] = known.owner or 0
                    grid_state[y, x, 2] = (known.health / tile.max_health) * 100 if tile.max_health else 0

        result = {
            "grid": grid_state,
            "units": unit_state,
            "gold": np.array([self.player_gold[i] for i in range(1, self.num_players + 1)], dtype=np.float32),
            "current_player": self.current_player,
            "turn_number": self.turn_number,
        }

        # Add visibility layer when FOW is enabled
        if self.fog_of_war:
            result["visibility"] = visibility_state

        return result

    def save_to_file(self, filepath: str | None = None) -> str | None:
        """
        Save game state to file.

        Args:
            filepath: Path to save file (auto-generated if None)

        Returns:
            Path to saved file
        """
        from reinforcetactics.utils.file_io import FileIO

        return FileIO.save_game(self, filepath)

    def _get_player_type(self, config: dict[str, Any]) -> str:
        """
        Get the standardized player type for replay logs.

        Args:
            config: Player configuration dictionary

        Returns:
            Player type string: 'human', 'bot', 'llm', or 'rl'
        """
        if config.get("type") == "human":
            return "human"

        # Prefer the type already resolved by the app / tournament layers
        # (create_bots_from_config and the tournament runner both stamp it).
        resolved = config.get("player_type")
        if resolved:
            return resolved

        # Fallback for configs that never went through those layers.
        # Deferred import: the engine must not import the game layer at
        # module load (core stays self-contained); this only runs on the
        # save-replay path.
        from reinforcetactics.game.bot_registry import player_type

        return player_type(config.get("bot_type", ""))

    @staticmethod
    def build_player_config(
        player_no: int, name: str, player_type: str, temperature: float | None = None, max_tokens: int | None = None
    ) -> dict[str, Any]:
        """
        Build a standardized player config for replay logs.

        Args:
            player_no: Player number (1, 2, etc.)
            name: Display name for the player/bot
            player_type: One of 'human', 'bot', 'llm', 'rl'
            temperature: LLM temperature (only for llm type)
            max_tokens: LLM max tokens (only for llm type)

        Returns:
            Standardized player config dictionary
        """
        config: dict[str, Any] = {"player_no": player_no, "type": player_type, "name": name}

        # Add LLM-specific fields
        if player_type == "llm":
            config["temperature"] = temperature
            config["max_tokens"] = max_tokens

        return config

    def save_replay_to_file(self, filepath: str | None = None) -> str | None:
        """
        Save replay to file.

        Args:
            filepath: Path to replay file (auto-generated if None)

        Returns:
            Path to saved replay
        """
        from reinforcetactics.utils.file_io import FileIO

        # Build player_configs for replay
        # If already in standardized format (has 'player_no'), use directly
        # Otherwise, transform from old format for backward compatibility
        enhanced_player_configs = []

        for i, config in enumerate(self.player_configs):
            player_num = i + 1

            # Check if already in standardized format
            if "player_no" in config:
                enhanced_player_configs.append(config)
            else:
                # Transform from old format (player_name, player_type, bot_type, etc.)
                player_name = config.get("player_name", config.get("name", "Unknown"))

                # Always use _get_player_type to map old format types (e.g., 'computer' -> 'bot')
                player_type = self._get_player_type(config)

                enhanced_config = {"player_no": player_num, "type": player_type, "name": player_name}

                # Add LLM-specific fields if applicable
                if player_type == "llm":
                    enhanced_config["temperature"] = config.get("temperature", None)
                    enhanced_config["max_tokens"] = config.get("max_tokens", None)

                enhanced_player_configs.append(enhanced_config)

        from reinforcetactics import __version__ as _rt_version

        # Final-state snapshot doubles as a replay-integrity checksum;
        # see runner._save_replay for the same fields.
        final_units_by_player: dict[int, list] = {}
        for u in self.units:
            final_units_by_player.setdefault(u.player, []).append(u)
        final_counts = {p: len(us) for p, us in final_units_by_player.items()}
        final_hp = {p: sum(u.health for u in us) for p, us in final_units_by_player.items()}

        game_info = {
            "num_players": self.num_players,
            "max_turns": self.max_turns,
            "total_turns": self.turn_number,
            "winner": self.winner,
            "game_over": self.game_over,
            "end_reason": self.end_reason,
            "winning_action_index": self.game_over_action_index,
            "start_time": self.game_start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "map_file": self.map_file_used,
            # The grid the actions' coordinates refer to (see record_action)
            "initial_map": self.initial_map_data,
            "player_configs": enhanced_player_configs,
            "enabled_units": self.enabled_units,
            "fog_of_war": self.fog_of_war,
            "fog_of_war_method": self.fog_of_war_method,
            # Seed of the combat RNG, so the game can be re-run (core-10).
            "seed": self.seed,
            "library_version": _rt_version,
            "replay_schema_version": 3,
            "final_unit_counts": final_counts,
            "final_hp_totals": final_hp,
            # Structure auto-heal economics (HP restored / gold spent per
            # player over the whole game). Queryable without re-simulating
            # the action log, and doubles as a replay-integrity checksum:
            # playback re-executes end_turn, so a faithful replay's
            # re-accumulated healing_totals must match these values.
            "healing_totals": {p: dict(t) for p, t in self.healing_totals.items()},
            # What the replay's GameState must be built with to play the log
            # back faithfully (see replay_actions.replay_game_state_kwargs):
            # the balance overlay (e.g. begin_first_turn gives Player 1 turn-0
            # income its first creates may spend) and the teams (explicit
            # ones are not in the map). eliminated_players is informational.
            "engine_overrides": self.engine_overrides,
            "begin_first_turn": self.begin_first_turn,
            "teams": self.teams,
            "eliminated_players": sorted(self.eliminated_players),
        }

        return FileIO.save_replay(self.action_history, game_info, filepath)

    @staticmethod
    def saved_map_data(save_data: dict[str, Any]) -> pd.DataFrame | None:
        """Return the terrain a save recorded via ``to_dict``, or None.

        Saves written before the terrain was recorded return None; their
        callers fall back to reloading ``save_data["map_file"]``.
        """
        terrain = save_data.get("map_data")
        if not terrain:
            return None
        return pd.DataFrame(terrain)

    @classmethod
    def from_dict(cls, save_data: dict[str, Any], map_data=None) -> GameState:
        """
        Restore game state from dictionary.

        Loads every save format version up to ``SAVE_FORMAT_VERSION``; a
        field an older save lacks takes the value a new game would have (a
        version 1 fog-of-war save, for one, gets fog rebuilt from the
        current board).

        Args:
            save_data: Dictionary with saved game data
            map_data: Map data (2D array). ``None`` rebuilds the grid from the
                terrain ``to_dict`` records under ``"map_data"``.

        Returns:
            Restored GameState instance

        Raises:
            ValueError: If ``map_data`` is None and the save predates recorded
                terrain (it only names a map file, or none for random maps).
        """
        if map_data is None:
            map_data = cls.saved_map_data(save_data)
            if map_data is None:
                raise ValueError("Save has no recorded terrain ('map_data'); pass the map explicitly")

        # Nothing below depends on the version yet (every field falls back
        # to a new game's value), so an unknown one only earns a warning. A
        # hand-edited, non-numeric version must not make the load crash.
        raw_version = save_data.get("save_format_version", 1)
        try:
            version = int(raw_version)
        except (TypeError, ValueError):
            logger.warning("Save format version %r is not a number; loading what it recognises", raw_version)
        else:
            if version > SAVE_FORMAT_VERSION:
                logger.warning(
                    "Save format version %s is newer than this version of the game (%s); loading what it recognises",
                    version,
                    SAVE_FORMAT_VERSION,
                )

        # Every container read from save_data is copied: the caller keeps its
        # dict, and a game must not share lists with it (or with the
        # class-level ALL_UNIT_TYPES) that either side could mutate.
        # Extract enabled_units from save data (default to all if not present for backward compatibility)
        saved_units = save_data.get("enabled_units")
        enabled_units = list(saved_units) if saved_units is not None else list(cls.ALL_UNIT_TYPES)

        # Extract fog_of_war from save data (default to False for backward compatibility)
        fog_of_war = save_data.get("fog_of_war", False)

        # Extract fog_of_war_method (default to 'simple_radius' if FOW enabled, 'none' otherwise)
        fog_of_war_method = save_data.get("fog_of_war_method", "simple_radius" if fog_of_war else "none")

        max_turns = save_data.get("max_turns")
        # Restore the engine-constant overlay (damage_model / structure HP /
        # economy / unit cap). Absent in pre-0.3.3 saves -> {} == module
        # defaults, byte-identical to the old load behaviour.
        engine_overrides = copy.deepcopy(save_data.get("engine_overrides") or {})
        # JSON turns the int player keys into strings. A save from before
        # teams existed has none and was played free-for-all, whatever team
        # codes its map carries: load it that way (map_teams=False).
        teams = {int(p): int(t) for p, t in (save_data.get("teams") or {}).items()} or None
        game = cls(
            map_data,
            save_data.get("num_players", 2),
            max_turns=max_turns,
            enabled_units=enabled_units,
            fog_of_war=fog_of_war,
            engine_overrides=engine_overrides,
            # Saves from before the seed was recorded get a fresh one.
            seed=save_data.get("seed"),
            teams=teams,
            map_teams="teams" in save_data,
        )
        if save_data.get("rng_state"):
            # Continue the saved stream exactly. The seed stays what the save
            # recorded (None for a caller-supplied rng), not a fresh draw.
            game.rng = cls._rng_from_save(save_data["rng_state"])
            game.seed = save_data.get("seed")
        game.eliminated_players = {int(p) for p in save_data.get("eliminated_players", [])}

        # Restore the fog of war method
        game.fog_of_war_method = fog_of_war_method

        try:
            game.game_start_time = datetime.strptime(save_data["timestamp"], "%Y-%m-%d %H-%M-%S")
        except (KeyError, TypeError, ValueError):
            pass  # keep "now" for saves without a usable timestamp
        game.current_player = save_data.get("current_player", 1)
        game.turn_number = save_data.get("turn_number", 0)
        game.game_over = save_data.get("game_over", False)
        game.winner = save_data.get("winner")
        game.end_reason = save_data.get("end_reason")
        game.game_over_action_index = save_data.get("winning_action_index")
        # Restore unit-id counter. Old saves predate this field; ``from_dict``
        # for the units themselves leaves ``unit.unit_id = None`` in that
        # case and ``find_unit_by_id`` falls back to position-based lookup.
        game._next_unit_id = save_data.get("next_unit_id", 0)

        # Fix player_gold dictionary key type (JSON serializes as strings)
        saved_gold = save_data.get("player_gold", {})
        game.player_gold = {int(k): v for k, v in saved_gold.items()}

        # Game-lifetime auto-heal totals (version 2+; older saves restart at 0)
        for p, totals in (save_data.get("healing_totals") or {}).items():
            game.healing_totals[int(p)] = {"hp": int(totals.get("hp", 0)), "gold": int(totals.get("gold", 0))}

        game.map_file_used = save_data.get("map_file")

        # Early version 2 saves carry padding metadata, always zero offsets
        # (see SAVE_FORMAT_VERSION). Only a script calling the since-removed
        # GameState.set_map_metadata could have saved others, and then the
        # saved action_history is not on the saved grid.
        offsets = (save_data.get("map_padding_offset_x") or 0, save_data.get("map_padding_offset_y") or 0)
        if offsets != (0, 0):
            logger.warning(
                "Save records map padding offsets %s; its action history is not on its grid, so a replay "
                "saved from this game will misplace the actions before the save",
                offsets,
            )

        # Restore player_configs (backward compatible with old saves)
        game.player_configs = copy.deepcopy(save_data.get("player_configs", []))

        # Restore units, with this game's stats (engine_overrides may change
        # them) rather than the module defaults, so max_health and attack
        # match the game the save was made in.
        game.units = []
        for unit_data in save_data.get("units", []):
            unit = Unit.from_dict(unit_data, stats=game.unit_data[unit_data["type"]])
            if game.fog_of_war and unit_data.get("pre_move_visibility") is not None:
                unit.pre_move_visibility = VisibilityMap.from_dict(
                    unit_data["pre_move_visibility"], game.grid.width, game.grid.height, unit.player
                )
            game.units.append(unit)

        # Restore tile states
        for tile_data in save_data.get("tiles", []):
            x, y = tile_data["x"], tile_data["y"]
            if 0 <= x < game.grid.width and 0 <= y < game.grid.height:
                tile = game.grid.tiles[y][x]
                # Restore a recorded neutral owner too: an eliminated
                # player's structures turn neutral (review core-7).
                if "player" in tile_data:
                    tile.player = tile_data["player"] or None
                if tile_data.get("health") is not None:
                    tile.health = tile_data["health"]
                if tile_data.get("regenerating") is not None:
                    tile.regenerating = tile_data["regenerating"]

        # Restore action history (for continuing replay recording from a loaded save)
        game.action_history = copy.deepcopy(save_data.get("action_history", []))

        # Fog of war: restore what each player had explored and remembers.
        # The maps __init__ computed describe the fresh map, not this game.
        # A version 1 save has no such state, so its fog is rebuilt from the
        # current board (anything explored before the save is lost).
        if game.fog_of_war:
            saved_fog = save_data.get("fog_of_war_state") or {}
            if all(str(p) in saved_fog for p in range(1, game.num_players + 1)):
                game.visibility_maps = {
                    p: VisibilityMap.from_dict(saved_fog[str(p)], game.grid.width, game.grid.height, p)
                    for p in range(1, game.num_players + 1)
                }
            else:
                game._init_visibility()

        game._invalidate_cache()
        return game
