"""
Core game state management without rendering dependencies.
Fixed version: removed duplicate methods, added type hints, controlled logging.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import random
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

import numpy as np

from reinforcetactics.core import legal_actions, serialization
from reinforcetactics.core.actions import ACTION_KINDS, ACTOR_KEYS, ActionResult
from reinforcetactics.core.engine_config import ENGINE_OVERRIDE_KEYS, EngineConfig
from reinforcetactics.core.fog import FogOfWar
from reinforcetactics.core.grid import TileGrid
from reinforcetactics.core.legal_actions import TARGET_RULES
from reinforcetactics.core.mechanics import GameMechanics, same_side
from reinforcetactics.core.serialization import SAVE_FORMAT_VERSION as SAVE_FORMAT_VERSION  # re-exported
from reinforcetactics.core.terrain_rules import TerrainRules
from reinforcetactics.core.unit import Unit
from reinforcetactics.core.visibility import UNEXPLORED, VISIBLE, StructureSnapshot, VisibilityMap
from reinforcetactics.rules import ALL_UNIT_TYPES, TileType

# Debug mode: with RT_CHECK_CACHE=1, every legal-action cache hit is
# recomputed and compared, so a mutator that forgets to invalidate fails
# loudly instead of handing bots and masks a stale action set.
_CHECK_LEGAL_ACTION_CACHE = os.environ.get("RT_CHECK_CACHE") == "1"

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

    # Every engine_overrides key some resolver reads (see core/engine_config.py).
    ENGINE_OVERRIDE_KEYS = ENGINE_OVERRIDE_KEYS

    # ------------------------------------------------------------------
    # Rule configuration: read-only views of ``self.engine_config``
    # ------------------------------------------------------------------
    # The game's rules live in one frozen EngineConfig (review core-16);
    # these keep the names every reader has always used.

    @property
    def engine_overrides(self) -> dict[str, Any]:
        """The sparse override overlay the game was created with, as saves record it."""
        return self.engine_config.overrides

    @property
    def unit_data(self) -> dict[str, dict[str, Any]]:
        """Per-unit stats with the overrides applied; units and costs read these, never rules.UNIT_DATA."""
        return self.engine_config.unit_data

    @property
    def income_rates(self) -> dict[str, int]:
        """Gold per turn by structure type (``headquarters``, ``building``, ``tower``)."""
        return self.engine_config.income_rates

    @property
    def starting_gold(self) -> int:
        """Each player's gold at the start of the game."""
        return self.engine_config.starting_gold

    @property
    def damage_model(self) -> str:
        """``"flat"`` or ``"hp_scaled"`` combat damage (see ``mechanics.attack_unit``)."""
        return self.engine_config.damage_model

    @property
    def structure_health(self) -> dict[str, int]:
        """Structure max-HP overrides, ``{tile_code: hp}``."""
        return self.engine_config.structure_health

    @property
    def max_units_per_player(self) -> int:
        """The unit cap ``create_unit`` and the legal actions enforce."""
        return self.engine_config.max_units_per_player

    @property
    def terrain_rules(self) -> TerrainRules:
        """The optional terrain rules (movement costs, charge distance, vision), all off by default."""
        return self.engine_config.terrain_rules

    @property
    def begin_first_turn(self) -> bool:
        """Whether Player 1 got start-of-turn processing on turn 0."""
        return self.engine_config.begin_first_turn

    @property
    def legacy_end_rules(self) -> bool:
        """Whether the game plays by the pre-2026-09 end rules (old replays only)."""
        return self.engine_config.legacy_end_rules

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
                Resolved into ``self.engine_config`` (an ``EngineConfig``),
                whose tables (``self.unit_data``, ``self.income_rates``,
                ``self.starting_gold``, ...) are this game's single source of
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
        # The game's rules: engine_overrides validated and resolved over the
        # rules.py constants (economy, unit stats, structure HP, unit cap,
        # damage model, terrain and turn rules; see core/engine_config.py).
        # Fixed for the whole game and read through the properties above
        # (self.unit_data, self.starting_gold, ...), so an override can't
        # leak between games or be half-applied.
        self.engine_config: EngineConfig = EngineConfig.from_overrides(engine_overrides)
        # Structure max-HP overrides (capture-difficulty lever) go onto the
        # grid built above while every structure is still at full health.
        self.engine_config.apply_structure_health(self.grid)
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

        # Fog of war: each player's visibility map and the rules that read
        # them (core/fog.py). Its maps are built and first computed at the
        # end of __init__, once the whole state exists.
        self.fog: FogOfWar = FogOfWar(self, fog_of_war)

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
        self.fog.reset()
        # Turn 0 for Player 1 (see begin_first_turn). Last, so the
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

    # ------------------------------------------------------------------
    # Fog of war (the code is in core/fog.py)
    # ------------------------------------------------------------------
    # ``self.fog`` owns the visibility maps and the fog-of-war rules; these
    # keep the names the renderer, the observation builder, the LLM bot,
    # the gym env, the tests and the notebooks have always called.

    @property
    def fog_of_war(self) -> bool:
        """Whether the game is played under fog of war (``fog.enabled``)."""
        return self.fog.enabled

    @property
    def fog_of_war_method(self) -> str:
        """The visibility algorithm: ``"simple_radius"`` under fog of war, else ``"none"`` (``fog.method``)."""
        return self.fog.method

    @property
    def visibility_maps(self) -> dict[int, VisibilityMap]:
        """Each player's ``VisibilityMap`` (``fog.maps``; empty without fog of war)."""
        return self.fog.maps

    def update_visibility(self, player: int | None = None) -> None:
        """Recompute ``player``'s fog-of-war view, every player's when None (``FogOfWar.update``).

        The engine calls this itself whenever a player's vision can change
        (construction and load, moves, unit creation and placement, captures,
        deaths, turn changes), so callers never need to.
        """
        self.fog.update(player)

    def get_visible_units_for_player(self, player: int, include_own: bool = True) -> list[Unit]:
        """The units ``player`` can see, its own too unless ``include_own`` is False (``FogOfWar.visible_units``)."""
        return self.fog.visible_units(player, include_own)

    def is_position_visible(self, x: int, y: int, player: int) -> bool:
        """Whether ``(x, y)`` is in ``player``'s sight (always without fog of war; ``FogOfWar.is_visible``)."""
        return self.fog.is_visible(x, y, player)

    def is_position_explored(self, x: int, y: int, player: int) -> bool:
        """Whether ``player`` has explored ``(x, y)`` (always without fog of war; ``FogOfWar.is_explored``)."""
        return self.fog.is_explored(x, y, player)

    def known_structure(self, player: int, x: int, y: int) -> StructureSnapshot | None:
        """What ``player`` knows about the structure at ``(x, y)``: live in sight, else as last seen.

        The one view of structures under fog of war (``FogOfWar.known_structure``);
        None when there is none there or ``player`` has never seen it.
        """
        return self.fog.known_structure(player, x, y)

    def pathing_units(self, player: int) -> list[Unit]:
        """The units ``player``'s pathfinding treats as present (``FogOfWar.pathing_units``).

        Every unit without fog of war; under it, the player's own and its
        teammates' units and the units it can see. Public so the GUI's
        movement overlay plans with the same view and shows exactly the
        tiles the engine allows.
        """
        return self.fog.pathing_units(player)

    def capture_visible_enemies_for_unit(self, unit: Unit) -> None:
        """Snapshot the enemies ``unit`` may attack this action: those its owner sees now.

        Prevents "move to discover, then attack" under fog of war. The GUI
        calls this when a unit is selected; ``move_unit`` takes it lazily
        otherwise (``FogOfWar.capture_visible_enemies``).
        """
        self.fog.capture_visible_enemies(unit)

    def is_enemy_attackable_by_unit(self, unit: Unit, enemy: Unit) -> bool:
        """Whether fog of war lets ``unit`` attack ``enemy``: seen when its action began (``FogOfWar.is_enemy_attackable``)."""
        return self.fog.is_enemy_attackable(unit, enemy)

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
    # Validating actions (the rules are in core/legal_actions.py)
    # ------------------------------------------------------------------
    # Each rule is written once, in core/legal_actions.py, and used twice:
    # ``enumerate_legal_actions`` offers what a player may do with it, and
    # the validators below, which the action methods and ``is_legal`` call,
    # reject anything else with it (review core-2). They add the two gates
    # that apply only on execution, not in enumeration: the game must not be
    # over, and it must be the acting player's turn.

    def get_reachable_positions(self, unit: Unit) -> list[tuple[int, int]]:
        """Tiles ``unit`` can move through this turn, including ones friends stand on.

        Same result as ``unit.get_reachable_positions`` with
        ``can_move_to_position`` as its predicate over ``pathing_units``,
        but under the game's terrain move costs and without scanning every
        unit per tile: for bots, overlays and anything else that plans paths.
        """
        return list(legal_actions.find_paths(self, unit))

    def get_move_destinations(self, unit: Unit) -> list[tuple[int, int]]:
        """Tiles ``unit`` may legally end a move on (reachable and empty), in search order.

        Ignores whose turn it is and whether the unit may still move; see
        ``get_legal_actions`` for that.
        """
        return list(legal_actions.move_paths(self, unit))

    def _may_act(
        self,
        action: str,
        unit: Unit,
        target: Unit | None = None,
        rule: Callable[[], bool] | None = None,
        log: bool = True,
    ) -> bool:
        """Validate one unit action before it is applied; log why when it is not.

        Rejects when the game is over; when ``unit`` (or ``target``) is no
        longer in play -- a stale reference, e.g. a bot still holding a unit
        that died to a counter earlier in its loop; when it is not the
        unit's player's turn; when the unit is dead or paralyzed; when the
        action slot it needs is spent (``can_move`` for a move,
        ``can_attack`` for attacks, abilities and seizing); or when
        ``rule`` (the action's target/range predicate) fails. ``log=False``
        is for ``is_legal``, which asks without attempting anything.
        """
        if self.game_over:
            reason = "the game is over"
        elif unit not in self.units or (target is not None and target not in self.units):
            reason = "a unit involved is no longer in play"
        elif unit.player != self.current_player:
            reason = f"it is player {self.current_player}'s turn"
        elif not legal_actions.is_ready_unit(unit, unit.player):
            reason = "the unit is dead or paralyzed"
        elif not (unit.can_move if action == "move" else unit.can_attack):
            reason = "the unit has already spent that action this turn"
        elif rule is not None and not rule():
            reason = "the target is out of range, on the wrong side, or a precondition fails"
        else:
            return True
        if log:
            logger.debug("Rejected %s by player %d %s at (%d, %d): %s", action, unit.player, unit.type, unit.x, unit.y, reason)
        return False

    def _may_target(self, action: str, actor: Unit, target: Unit, log: bool = True) -> bool:
        """``_may_act`` for the targeted action ``action`` (a ``TARGET_RULES`` key) from ``actor`` on ``target``."""
        rule = TARGET_RULES[action]
        return self._may_act(action, actor, target, lambda: rule(self, actor, target), log=log)

    def _may_seize(self, unit: Unit, log: bool = True) -> bool:
        """``_may_act`` for ``unit`` seizing the structure it stands on."""
        return self._may_act("seize", unit, rule=lambda: legal_actions.can_seize(self, unit), log=log)

    def _move_steps(
        self,
        unit: Unit,
        to_x: int,
        to_y: int,
        came_from: dict[tuple[int, int], tuple[int, int]] | None = None,
        log: bool = True,
    ) -> int | None:
        """Tiles ``unit`` steps to end a legal move on ``(to_x, to_y)``; None if it may not move there.

        The move rule: ``_may_act``, and a destination among the tiles
        ``get_move_destinations`` (and so ``get_legal_actions``) offers,
        planned around the units the player knows of. ``came_from``
        receives the path search tree (``move_unit`` walks it for the
        ambush rule).
        """
        if not self._may_act("move", unit, log=log):
            return None
        steps = legal_actions.move_paths(self, unit, came_from=came_from).get((to_x, to_y))
        if steps is None and log:
            logger.debug(f"Cannot move to ({to_x}, {to_y}): not reachable or occupied")
        return steps

    def _may_create(self, unit_type: str, x: int, y: int, player: int, log: bool = True) -> bool:
        """Validate a ``create_unit`` before it is applied; log why when it is not.

        The game must be running, ``player`` must be the current player and
        still in the game, ``unit_type`` enabled, the player under the unit
        cap and able to afford it, and ``(x, y)`` an empty Building the
        player owns -- the same rules ``get_legal_actions`` offers creates
        by. ``log=False`` is for ``is_legal``.
        """
        level = logging.DEBUG
        if self.game_over:
            reason = "the game is over"
        elif player != self.current_player:
            reason = f"it is player {self.current_player}'s turn"
        elif player in self.eliminated_players:
            reason = "the player is eliminated"
        elif unit_type not in self.unit_data:
            reason, level = "unknown unit type", logging.WARNING
        elif unit_type not in self.enabled_units:
            reason = "the unit type is not enabled in this game"
        # The per-player unit cap. Mirrored in get_legal_actions so the RL
        # action mask hides create_unit at the cap rather than the agent
        # issuing a rejected action and eating the invalid_action penalty.
        elif not legal_actions.under_unit_cap(self, player):
            reason = f"the player is at the unit cap ({self.max_units_per_player})"
        elif not legal_actions.is_free_spawn_tile(self, player, x, y):
            reason = "not an empty building the player owns"
        elif not legal_actions.can_afford(self, player, unit_type):
            reason = f"insufficient gold ({self.player_gold[player]} < {self.unit_data[unit_type]['cost']})"
        else:
            return True
        if log:
            logger.log(level, "Cannot create %s at (%s, %s) for player %s: %s", unit_type, x, y, player, reason)
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
        self.fog.update()
        return unit

    def create_unit(self, unit_type: str, x: int, y: int, player: int | None = None) -> Unit | None:
        """
        Create a unit at the specified position.

        Rejected (returns None, changes and records nothing) unless the game
        is running, ``player`` is the current player, ``unit_type`` is
        enabled, the player is under the unit cap and can afford it, and
        ``(x, y)`` is an empty Building the player owns (``_may_create``) --
        the same rules ``get_legal_actions`` offers creates by. For test or
        scenario setup use :meth:`place_unit`.

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

        if not self._may_create(unit_type, x, y, player):
            return None

        # Create the unit
        self.player_gold[player] -= self.unit_data[unit_type]["cost"]
        unit = Unit(unit_type, x, y, player, stats=self.unit_data[unit_type])
        unit.unit_id = self._next_unit_id
        self._next_unit_id += 1
        self.units.append(unit)
        self._invalidate_cache()
        # The new unit sees from its first moment (review core-12).
        self.fog.update(player)

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

        # The move rule (see _move_steps). Its actor gate (_may_act) rejects,
        # among other things, duplicate moves -- bot/RL/LLM call sites don't
        # all gate on ``unit.can_move`` before calling, and
        # ``get_reachable_positions`` ignores it, so a unit could otherwise
        # move more than once per turn -- and stale references: a unit can
        # die mid-loop from a counter-attack while the bot still holds it,
        # and moving it would log an event the replay player (which only
        # sees self.units) can't reproduce (PR #360 audit).
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        steps = self._move_steps(unit, to_x, to_y, came_from)
        if steps is None:
            return False

        # FOW: Snapshot pre-move enemy visibility so the unit cannot attack
        # enemies it discovers by moving (the UI's input_handler captures this
        # at unit-selection time; for RL/LLM/bot code paths that drive
        # move_unit directly it is captured lazily here), and remember what
        # the mover's side saw, so a cancel_move can take back what the move
        # revealed (review core-9).
        self.fog.before_move(unit)

        # FOW ambush rule: without fog of war the path was planned around
        # every unit, so it is always clear.
        ambusher = None
        if self.fog_of_war:
            path = [(to_x, to_y)]
            while path[-1] != (from_x, from_y):
                path.append(came_from[path[-1]])
            path.reverse()
            (to_x, to_y), ambusher = self.fog.resolve_ambush(unit, path)
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
        self.fog.update(unit.player)

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
        if not self._may_target("attack", attacker, target):
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
            self.fog.update(defeated_player)
            self._check_player_eliminated(defeated_player)

        if not result["attacker_alive"]:
            attacker_tile = self.grid.get_tile(attacker.x, attacker.y)
            if attacker_tile.is_capturable() and attacker_tile.health < attacker_tile.max_health:
                attacker_tile.regenerating = True
            defeated_player = attacker.player
            if attacker in self.units:
                self.units.remove(attacker)
            # The counter-attack killed it: out of play, it has no action
            # left, so a caller still holding it reads it as done. Its flags
            # were left set (the action below is only spent for a survivor),
            # and the rule bots re-ran their whole decision on the corpse
            # until their recursion cap, every action refused (review
            # rulebots-8). A pending haste goes too, so ``end_unit_turn``
            # can't re-arm it either.
            attacker.can_move = attacker.can_attack = False
            attacker.is_hasted = attacker.haste_refreshed = False
            self._invalidate_cache()
            self.fog.update(defeated_player)
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
        apply: Callable[[], Any],
        rejected: Any,
        actor_pos_field: str,
        record_fields: Callable[[Any], dict[str, Any]] | None = None,
        after_apply: Callable[[], None] | None = None,
    ) -> Any:
        """The shared body of the targeted abilities (paralyze, heal, cure, haste, the buffs).

        Validates with ``_may_target`` (``_may_act`` and the ability's
        ``legal_actions.TARGET_RULES`` rule, the one its legal actions are
        listed by), returning ``rejected`` and changing nothing when that fails.
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
        if not self._may_target(action, actor, target):
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
        if not self._may_seize(unit):
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
            self.fog.update()

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
        ``begin_first_turn``; see ``core/engine_config.py``).

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
        self.fog.update(player)

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

        # FOW: the side's view goes back to what it was before the move.
        snapshot, unit.pre_move_visibility = unit.pre_move_visibility, None
        self.fog.undo_move(unit, snapshot)

        self._invalidate_cache()
        return True

    # ------------------------------------------------------------------
    # Actions by name
    # ------------------------------------------------------------------
    # The entry point for code that picks actions the way get_legal_actions
    # lists them (bots, MCTS, the gym env, the LLM bots, the GUI): a kind
    # (a get_legal_actions key, see ``core.actions.ACTION_KINDS``) and a
    # payload (one of that key's entries). Each of them used to keep its own
    # table from that shape to an action method, and from the method's
    # return value to "did it happen" (review core-14).

    def apply_action(self, kind: str, action: Mapping[str, Any]) -> ActionResult:
        """Carry out the action ``kind`` that ``action`` describes, through its action method.

        ``action`` has the shape of a ``get_legal_actions()[kind]`` entry:
        ``{"unit_type", "x", "y"}`` for ``create_unit`` (with an optional
        ``"player"``, the current player by default, as ``create_unit``
        takes), ``{"unit", "to_x", "to_y"}`` for ``move``, ``{"unit"}`` for
        ``seize``, ``{}`` for ``end_turn``, and the acting unit (under
        ``ACTOR_KEYS[kind]``) and ``"target"`` for the others. Other keys (an
        entry's ``from_x``/``from_y`` or ``tile``) are ignored. The method
        (``create_unit``, ``move_unit``, ``seize``, ``end_turn``, or the one
        named ``kind``) is looked up on the instance, so the action is
        validated and recorded exactly as by a direct call, and a wrapper
        installed on the instance (the imitation recorder's) sees it too.

        Returns:
            The method's return value, and whether the engine carried the
            action out: always what ``is_legal(kind, action)`` answered just
            before. A refused action changes nothing.

        Raises:
            ValueError: ``kind`` is not one of ``ACTION_KINDS``.
        """
        if kind == "create_unit":
            unit = self.create_unit(action["unit_type"], action["x"], action["y"], player=action.get("player"))
            return ActionResult(kind, unit is not None, unit)
        if kind == "move":
            moved = self.move_unit(action["unit"], action["to_x"], action["to_y"])
            return ActionResult(kind, bool(moved), moved)
        if kind == "seize":
            result = self.seize(action["unit"])
            return ActionResult(kind, "damage" in result, result)
        if kind == "end_turn":
            # end_turn returns the same empty breakdown for a game it ends on
            # max_turns and for one already over, which it leaves alone; so
            # whether it did anything is decided before the call.
            running = not self.game_over
            return ActionResult(kind, running, self.end_turn())
        if kind in TARGET_RULES:
            result = getattr(self, kind)(action[ACTOR_KEYS[kind]], action["target"])
            if kind == "attack":
                # An executed attack always deals at least 1 damage.
                return ActionResult(kind, result["damage"] > 0, result)
            # heal returns the HP it restored, the others a bool (see _use_ability).
            return ActionResult(kind, result > 0, result)
        raise ValueError(f"Unknown action kind {kind!r}; expected one of {', '.join(ACTION_KINDS)}")

    def is_legal(self, kind: str, action: Mapping[str, Any]) -> bool:
        """Whether ``apply_action(kind, action)`` would carry the action out now; changes nothing.

        Asks the rule the action method validates with (``_may_create``,
        ``_move_steps``, ``_may_target``, ``_may_seize``; ``end_turn`` only
        needs the game to be running), so ``apply_action(kind,
        action).accepted`` always equals it, and it holds for every entry
        of ``get_legal_actions()[kind]`` (``{}`` for ``end_turn``) while the
        game runs. Unlike ``get_legal_actions`` it answers for the current
        player only, as the action methods do: another player's action, and
        any action once the game is over, is not legal. It enumerates
        nothing (a move costs one path search), logs no refusal, and
        neither reads nor fills the legal-action cache.

        Under fog of war an attack or paralyze needs a target the attacker
        could see when its action started (``is_enemy_attackable_by_unit``):
        its snapshot if it has one, else what its player sees now. This
        answers from the snapshot as it stands and never takes one.
        ``attack`` and the abilities don't take one either, and
        ``move_unit`` takes it only once the move is accepted, so this is
        exactly what the method would decide. Taking one here would freeze
        the unit's targets at the moment of asking: an enemy another unit
        then uncovers would be attackable by ``get_legal_actions`` and the
        methods, but not by this.

        Raises:
            ValueError: ``kind`` is not one of ``ACTION_KINDS``.
        """
        if kind == "create_unit":
            player = action.get("player")
            if player is None:
                player = self.current_player
            return self._may_create(action["unit_type"], action["x"], action["y"], player, log=False)
        if kind == "move":
            return self._move_steps(action["unit"], action["to_x"], action["to_y"], log=False) is not None
        if kind == "seize":
            return self._may_seize(action["unit"], log=False)
        if kind == "end_turn":
            return not self.game_over
        if kind in TARGET_RULES:
            return self._may_target(kind, action[ACTOR_KEYS[kind]], action["target"], log=False)
        raise ValueError(f"Unknown action kind {kind!r}; expected one of {', '.join(ACTION_KINDS)}")

    def get_legal_actions(self, player: int | None = None) -> dict[str, list[Any]]:
        """Every action ``player`` (default: the current player) may take now, by kind.

        The enumeration is ``legal_actions.enumerate_legal_actions``; this
        caches its result per player until the state next changes (every
        mutator invalidates the cache), because the RL env, the bots and
        the GUI ask for the same actions many times between changes.

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

        actions = self._compute_legal_actions(player)

        # Cache the result
        self._legal_actions_cache[player] = actions
        self._legal_actions_cache_valid = True

        return actions

    def _compute_legal_actions(self, player: int) -> dict[str, list[Any]]:
        """Enumerate ``player``'s legal actions from the current state (uncached; ``enumerate_legal_actions``)."""
        return legal_actions.enumerate_legal_actions(self, player)

    # How clone_for_search treats each attribute (review core-18). Shared:
    # fixed for the whole game (configuration, terrain source, stateless
    # helpers), so the clone references the original's object. Dropped:
    # history and caches search never reads, replaced by empty values.
    # Anything else is deep-copied, so state added later is safe by default.
    _SEARCH_SHARED_ATTRS = frozenset(
        {
            "engine_config",
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
            elif name == "fog":
                # Deep-copied like any other state, but bound to the clone.
                clone.fog = value.copy_for(clone)
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
            vis_map = self.fog.maps.get(for_player)
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
                known = self.fog.known_structure(fog_player, x, y)
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

    # ------------------------------------------------------------------
    # Saves and replays (the code is in core/serialization.py)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The game as a JSON-ready save dict ``from_dict`` resumes exactly (``serialization.game_to_dict``)."""
        return serialization.game_to_dict(self)

    def save_to_file(self, filepath: str | None = None) -> str | None:
        """Save the game to a JSON file (auto-named if ``filepath`` is None); returns its path, None on failure."""
        return serialization.save_to_file(self, filepath)

    def save_replay_to_file(self, filepath: str | None = None) -> str | None:
        """Save the game's replay (action log and ``game_info``); returns its path, None on failure."""
        return serialization.save_replay_to_file(self, filepath)

    # Standardized replay-log player config: {"player_no", "type", "name"}
    # plus the LLM sampling fields for type "llm".
    build_player_config = staticmethod(serialization.build_player_config)
    # The terrain a save recorded (a DataFrame), or None for older saves.
    saved_map_data = staticmethod(serialization.saved_map_data)

    @classmethod
    def from_dict(cls, save_data: dict[str, Any], map_data=None) -> GameState:
        """Restore a game ``to_dict`` saved (``serialization.game_from_dict``).

        ``map_data`` None rebuilds the grid from the terrain the save
        recorded; raises ValueError for a save too old to have it.
        """
        return serialization.game_from_dict(cls, save_data, map_data)
