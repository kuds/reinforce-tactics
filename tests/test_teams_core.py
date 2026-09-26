"""Teams: one encoding, and allies for every hostility rule (review core-4).

Team membership is declared by the ``type_player_team`` structure code
(``Tile.team``) on each player's HQ, or passed as ``GameState(teams=...)``;
undeclared players are each their own team (free-for-all). Teammates are
allies for every rule: they are never attacked, paralyzed or seized, and they
can be healed, cured and buffed, flank for each other and pass through each
other. Before this, ``Tile.team`` was parsed and ignored, the bundled 2v2 map
gave players 3 and 4 nothing, and teammates could attack each other.
"""

from pathlib import Path

import numpy as np
import pytest

from reinforcetactics.constants import SORCERER_BUFF_DURATION
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import AdvancedBot, MasterBot, MediumBot, SimpleBot
from reinforcetactics.game.llm_bot import LLMBot
from reinforcetactics.utils.file_io import FileIO

MAPS_DIR = Path(__file__).resolve().parents[1] / "maps"


def _grid_with(tiles):
    """A 10x10 grass map with ``tiles`` ({(x, y): code}) placed on it."""
    grid = np.full((10, 10), "p", dtype=object)
    for (x, y), code in tiles.items():
        grid[y, x] = code
    return grid


def _four_player_map(extra=None):
    """HQs in the corners, players 1 & 3 declared on team 1 and 2 & 4 on team 2."""
    tiles = {(0, 0): "h_1_1", (9, 0): "h_2_2", (9, 9): "h_3_1", (0, 9): "h_4_2"}
    tiles.update(extra or {})
    return _grid_with(tiles)


@pytest.fixture
def team_game():
    """Four seats, players 1 & 3 against 2 & 4, declared on the HQ codes."""
    return GameState(_four_player_map(), num_players=4)


class TestTeamResolution:
    def test_undeclared_maps_are_free_for_all(self):
        one_v_one = GameState(FileIO.load_map(str(MAPS_DIR / "1v1" / "beginner.csv")), num_players=2)
        ffa = GameState(FileIO.load_map(str(MAPS_DIR / "1v1v1" / "triangle_arena.csv")), num_players=3)

        assert one_v_one.teams == {1: 1, 2: 2}
        assert ffa.teams == {1: 1, 2: 2, 3: 3}
        assert ffa.are_enemies(1, 2) and ffa.are_enemies(2, 3) and not ffa.are_allies(1, 3)

    def test_hq_suffix_declares_teams(self, team_game):
        assert team_game.teams == {1: 1, 2: 2, 3: 1, 4: 2}
        assert team_game.are_allies(1, 3) and team_game.are_allies(2, 4)
        assert team_game.are_enemies(1, 2) and team_game.are_enemies(3, 4)
        # A player is its own ally; a neutral owner is nobody's ally or enemy.
        assert team_game.are_allies(1, 1)
        assert not team_game.are_allies(1, None) and not team_game.are_enemies(1, None)

    def test_explicit_teams_without_map_declaration(self):
        grid = _grid_with({(0, 0): "h_1", (9, 0): "h_2", (9, 9): "h_3", (0, 9): "h_4"})

        game = GameState(grid, num_players=4, teams={1: 1, 2: 2, 3: 1, 4: 2})

        assert game.are_allies(1, 3) and game.are_enemies(1, 4)

    def test_explicit_teams_that_agree_with_the_map_are_accepted(self):
        game = GameState(_four_player_map(), num_players=4, teams={1: 1, 3: 1})

        assert game.teams == {1: 1, 2: 2, 3: 1, 4: 2}

    def test_explicit_teams_conflicting_with_the_map_raise(self):
        with pytest.raises(ValueError, match="player 3"):
            GameState(_four_player_map(), num_players=4, teams={3: 2})

    def test_map_declaring_a_player_on_two_teams_raises(self):
        with pytest.raises(ValueError, match="player 1"):
            GameState(_four_player_map({(1, 0): "b_1_2"}), num_players=4)

    def test_everyone_on_one_team_raises(self):
        with pytest.raises(ValueError, match="one team"):
            GameState(_grid_with({(0, 0): "h_1", (9, 9): "h_2"}), num_players=2, teams={1: 1, 2: 1})

    def test_partial_declaration_leaves_the_rest_solo(self):
        grid = _grid_with({(0, 0): "h_1_1", (9, 0): "h_2", (9, 9): "h_3_1", (0, 9): "h_4"})

        game = GameState(grid, num_players=4)

        assert game.are_allies(1, 3)
        assert game.are_enemies(2, 4) and game.are_enemies(2, 1) and game.are_enemies(4, 3)


class TestBundled2v2Maps:
    @pytest.mark.parametrize("map_path", sorted((MAPS_DIR / "2v2").glob("*.csv")), ids=lambda p: p.name)
    def test_every_seat_owns_an_hq_and_a_building_and_can_build(self, map_path):
        game = GameState(FileIO.load_map(str(map_path)), num_players=4)

        for player in range(1, 5):
            owned = {t.type for row in game.grid.tiles for t in row if t.player == player}
            assert {"h", "b"} <= owned, f"player {player} owns {owned}"
            assert game.mechanics.calculate_income(player, game.grid, game.income_rates)["total"] > 0
            assert game.get_legal_actions(player)["create_unit"], f"player {player} cannot build"
        # Two teams of two, alternating in turn order.
        assert game.teams == {1: 1, 2: 2, 3: 1, 4: 2}


class TestAlliesInEveryRule:
    def test_teammates_are_never_attack_or_paralyze_targets(self, team_game):
        mage = team_game.place_unit("M", 4, 4, 1)
        mate = team_game.place_unit("W", 5, 4, 3)
        foe = team_game.place_unit("W", 4, 5, 2)

        legal = team_game.get_legal_actions(1)

        assert [a["target"] for a in legal["attack"]] == [foe]
        assert [a["target"] for a in legal["paralyze"]] == [foe]
        assert team_game.attack(mage, mate)["damage"] == 0
        assert not team_game.paralyze(mage, mate)
        assert mate.health == mate.max_health and not mate.is_paralyzed()

    def test_support_abilities_reach_teammates_units(self, team_game):
        cleric = team_game.place_unit("C", 4, 4, 1)
        hurt = team_game.place_unit("W", 5, 4, 3)
        hurt.health = 5
        frozen = team_game.place_unit("W", 4, 5, 3)
        frozen.paralyzed_turns = 2

        legal = team_game.get_legal_actions(1)
        assert [a["target"] for a in legal["heal"]] == [hurt]
        assert [a["target"] for a in legal["cure"]] == [frozen]
        assert team_game.heal(cleric, hurt) > 0

    def test_buffs_reach_teammates_but_haste_does_not(self, team_game):
        sorcerer = team_game.place_unit("S", 4, 4, 1)
        mate = team_game.place_unit("W", 5, 4, 3)

        legal = team_game.get_legal_actions(1)
        assert mate in [a["target"] for a in legal["attack_buff"]]
        assert mate in [a["target"] for a in legal["defence_buff"]]
        # Haste is an extra action this turn; a teammate's unit can't act on
        # player 1's turn, so it is never a haste target.
        assert mate not in [a["target"] for a in legal["haste"]]
        assert not team_game.haste(sorcerer, mate)
        assert team_game.attack_buff(sorcerer, mate)
        # Stored one higher: cast outside its owner's turn (see the durations tests).
        assert mate.attack_buff_turns == SORCERER_BUFF_DURATION + 1

    def test_a_teammate_flanks_for_a_rogue(self, team_game):
        rogue = team_game.place_unit("R", 4, 4, 1)
        target = team_game.place_unit("B", 5, 4, 2)
        team_game.place_unit("W", 6, 4, 3)  # player 1's teammate, next to the target

        result = team_game.attack(rogue, target)

        assert result["damage"] > 0 and result["flank_bonus"] is True

    def test_units_pass_through_teammates_but_not_enemies(self):
        # A one-tile corridor: row 1 is walkable, row 0 and 2 are water.
        grid = np.full((3, 8), "w", dtype=object)
        grid[1, :] = "p"
        grid[0, 0], grid[0, 7], grid[2, 0], grid[2, 7] = "h_1_1", "h_2_2", "h_3_1", "h_4_2"
        game = GameState(grid, num_players=4)
        runner = game.place_unit("B", 1, 1, 1)  # movement 5
        game.place_unit("W", 2, 1, 3)

        assert (4, 1) in game.get_move_destinations(runner)

        blocked = GameState(grid, num_players=4)
        runner = blocked.place_unit("B", 1, 1, 1)
        blocked.place_unit("W", 2, 1, 2)
        assert (4, 1) not in blocked.get_move_destinations(runner)

    def test_under_fog_a_teammate_on_the_path_is_passed_not_an_ambush(self):
        # A one-tile corridor through the teammate: the only way to (5, 5).
        grid = np.full((12, 12), "w", dtype=object)
        grid[:, 5] = "p"
        grid[0, 5], grid[11, 5], grid[1, 5], grid[10, 5] = "h_1", "h_2", "h_3", "h_4"
        game = GameState(grid, num_players=4, fog_of_war=True, teams={1: 1, 3: 1, 2: 2, 4: 2}, seed=1)
        mover = game.place_unit("W", 5, 3, 1)
        game.place_unit("W", 5, 4, 3)  # player 1's teammate, in the corridor
        game.place_unit("W", 5, 9, 2)
        game.place_unit("W", 5, 8, 4)

        assert game.move_unit(mover, 5, 5)

        assert (mover.x, mover.y) == (5, 5) and not mover.ambushed
        assert "ambushed" not in game.action_history[-1]
        assert game.can_cancel_move(mover)

    def test_a_teammates_structure_cannot_be_seized(self):
        game = GameState(_four_player_map({(5, 5): "b_3", (6, 6): "b_2"}), num_players=4)
        on_mate = game.place_unit("W", 5, 5, 1)
        on_foe = game.place_unit("W", 6, 6, 1)

        seizable = [a["unit"] for a in game.get_legal_actions(1)["seize"]]
        assert seizable == [on_foe]
        assert "damage" not in game.seize(on_mate)
        assert game.grid.get_tile(5, 5).health == game.grid.get_tile(5, 5).max_health

    def test_fog_snapshot_only_holds_enemies(self):
        game = GameState(_four_player_map(), num_players=4, fog_of_war=True)
        unit = game.place_unit("W", 4, 4, 1)
        game.place_unit("W", 5, 4, 3)
        game.place_unit("W", 4, 5, 2)

        game.capture_visible_enemies_for_unit(unit)

        assert unit.visible_enemies_at_action_start == {(4, 5)}


class TestTwoTeamWin:
    def test_capturing_an_enemy_hq_wins_for_the_team(self, team_game):
        hq = team_game.grid.get_tile(9, 0)  # player 2's
        seizer = team_game.place_unit("W", 9, 0, 1)
        team_game.place_unit("W", 5, 5, 4)
        hq.health = 1

        result = team_game.seize(seizer)

        assert result["captured"] and result["game_over"]
        assert team_game.game_over and team_game.end_reason == "hq_capture"
        assert team_game.winner == 1 and team_game.are_allies(team_game.winner, 3)


class TestScriptedBotsTreatTeammatesAsAllies:
    @pytest.mark.parametrize("bot_cls", [SimpleBot, MediumBot, AdvancedBot, MasterBot])
    def test_bots_see_teammates_as_friendly(self, team_game, bot_cls):
        bot = bot_cls(team_game, player=1)

        assert bot._is_friendly(3) and not bot._is_enemy(3)
        assert bot._is_enemy(2) and bot._is_enemy(4)
        assert not bot._is_enemy(None) and not bot._is_friendly(None)

    def test_simple_bot_targets_skip_teammates(self):
        game = GameState(_four_player_map({(6, 4): "b_3"}), num_players=4)
        unit = game.place_unit("W", 4, 4, 1)
        game.place_unit("W", 5, 4, 3)

        target = SimpleBot(game, player=1).find_best_target(unit)

        # Nearest things are the teammate's unit and building; the pick must
        # be an enemy's unit or structure instead.
        kind, obj, _ = target
        owner = obj.player
        assert game.are_enemies(owner, 1), (kind, owner)

    def test_a_2v2_bot_game_never_attacks_a_teammate(self) -> None:
        game = GameState(FileIO.load_map(str(MAPS_DIR / "2v2" / "beginner.csv")), num_players=4, max_turns=12)
        bots = {1: SimpleBot(game, 1), 2: MediumBot(game, 2), 3: AdvancedBot(game, 3), 4: MasterBot(game, 4)}
        owner_of: dict[int, int] = {}
        for _ in range(12 * 4):
            if game.game_over:
                break
            bots[game.current_player].take_turn()
            for action in game.action_history:
                if action["type"] == "create_unit":
                    owner_of[action["unit_id"]] = action["player"]

        attacks = [a for a in game.action_history if a["type"] == "attack"]
        for attack in attacks:
            assert game.are_enemies(attack["player"], owner_of[attack["target_unit_id"]]), attack
        # Every seat played: all four built something.
        assert {a["player"] for a in game.action_history if a["type"] == "create_unit"} == {1, 2, 3, 4}


class _SilentLLMBot(LLMBot):
    def _get_api_key_from_env(self):
        return "test-key"

    def _get_env_var_name(self):
        return "TEST_API_KEY"

    def _get_default_model(self):
        return "test-model"

    def _get_supported_models(self):
        return ["test-model"]

    def _call_llm(self, messages):
        return '{"actions": []}'

    def _get_llm_sdk_version(self):
        return "test-sdk-1.0.0"


class TestLLMBotsTreatTeammatesAsAllies:
    """The LLM prompt listed a teammate's units and structures as enemies and offered them as targets."""

    @pytest.fixture
    def state(self):
        game = GameState(_four_player_map({(6, 4): "b_3", (2, 6): "b_2"}), num_players=4)
        game.player_gold.update({1: 111, 2: 222, 3: 333, 4: 444})
        game.place_unit("K", 4, 4, 1)  # moves 4: reaches the teammate's building
        cleric = game.place_unit("C", 4, 6, 1)
        teammate = game.place_unit("W", 5, 4, 3)
        teammate.health = 5  # hurt: healable, and a tempting target if it were an enemy
        game.place_unit("W", 4, 2, 2)
        game.place_unit("W", 8, 8, 4)
        return game, cleric, _SilentLLMBot(game, player=1, api_key="test-key")._serialize_game_state()

    def test_teammates_are_listed_as_allies_not_enemies(self, state):
        _game, _cleric, s = state
        assert (5, 4) not in {tuple(u["position"]) for u in s["enemy_units"]}
        assert [(tuple(u["position"]), u["player"]) for u in s["teammate_units"]] == [((5, 4), 3)]
        assert (6, 4) not in {tuple(b["position"]) for b in s["enemy_buildings"]}
        assert {tuple(b["position"]) for b in s["teammate_buildings"]} == {(6, 4), (9, 9)}
        assert (2, 6) in {tuple(b["position"]) for b in s["enemy_buildings"]}

    def test_combos_target_enemies_and_heal_teammates(self, state):
        _game, cleric, s = state
        combos = s["legal_actions"]
        assert all(tuple(c["then_attack"]) != (5, 4) for c in combos["move_then_attack"])
        seizable = {tuple(c["move_to"]) for c in combos["move_then_seize"]}
        assert (6, 4) not in seizable and (2, 6) in seizable
        # The Cleric reaches its teammate's hurt Warrior from beyond an adjacent tile.
        heals = {tuple(c["move_to"]) for c in combos["move_then_heal"] if tuple(c["then_heal"]) == (5, 4)}
        assert heals and any(abs(x - 5) + abs(y - 4) > 1 for x, y in heals)

    def test_opponent_gold_lists_each_enemy_still_in_the_game(self, state):
        game, _cleric, s = state
        assert s["opponent_gold"] == {"2": 222, "4": 444}

        game.resign(4)
        s = _SilentLLMBot(game, player=1, api_key="test-key")._serialize_game_state()
        assert s["opponent_gold"] == 222  # one opponent left

    def test_a_1v1_state_has_no_teammate_keys_and_a_single_opponent_gold(self):
        grid = _grid_with({(0, 0): "h_1", (9, 9): "h_2"})
        game = GameState(grid, num_players=2)
        s = _SilentLLMBot(game, player=1, api_key="test-key")._serialize_game_state()
        assert "teammate_units" not in s and "teammate_buildings" not in s
        assert s["opponent_gold"] == game.player_gold[2]
        assert _buildings_income(s) == {(9, 9): game.income_rates["headquarters"]}


def _buildings_income(state):
    return {tuple(b["position"]): b["income"] for b in state["enemy_buildings"]}


class TestTeamsPersist:
    def test_save_round_trip_keeps_explicit_teams(self):
        grid = _grid_with({(0, 0): "h_1", (9, 0): "h_2", (9, 9): "h_3", (0, 9): "h_4"})
        game = GameState(grid, num_players=4, teams={1: 1, 2: 2, 3: 1, 4: 2})

        import json

        restored = GameState.from_dict(json.loads(json.dumps(game.to_dict())))

        assert restored.teams == {1: 1, 2: 2, 3: 1, 4: 2}

    def test_replay_game_info_carries_teams(self, tmp_path, monkeypatch):
        from reinforcetactics.utils import file_io

        captured = {}
        monkeypatch.setattr(
            file_io.FileIO, "save_replay", staticmethod(lambda actions, info, path=None: captured.update(info) or "x")
        )
        grid = _grid_with({(0, 0): "h_1", (9, 0): "h_2", (9, 9): "h_3", (0, 9): "h_4"})
        GameState(grid, num_players=4, teams={1: 1, 2: 2, 3: 1, 4: 2}).save_replay_to_file()

        from reinforcetactics.utils.replay_actions import replay_game_state_kwargs

        rebuilt = GameState(grid, **replay_game_state_kwargs(captured))
        assert rebuilt.teams == {1: 1, 2: 2, 3: 1, 4: 2}

    def test_saves_and_replays_from_before_teams_load_free_for_all(self):
        """The old 2v2 map put player 1 on two teams (h_1_1 and h_1_2); games
        on it were played free-for-all and must still load and replay."""
        old_map = np.array(
            [
                ["h_1_1", "b_1", "p", "p", "b_2_1", "h_2_1"],
                ["b_1", "p", "p", "p", "b", "b_2_1"],
                ["p", "p", "t", "t", "p", "p"],
                ["p", "p", "t", "t", "p", "p"],
                ["b_1_2", "p", "p", "p", "p", "b_2_2"],
                ["h_1_2", "b_1_2", "p", "p", "b_2_2", "h_2_2"],
            ],
            dtype=object,
        )
        with pytest.raises(ValueError):
            GameState(old_map, num_players=4)
        old_save = GameState(old_map, num_players=4, map_teams=False).to_dict()
        del old_save["teams"], old_save["eliminated_players"]

        from reinforcetactics.utils.replay_actions import replay_game_state_kwargs

        assert GameState.from_dict(old_save).teams == {1: 1, 2: 2, 3: 3, 4: 4}
        assert GameState(old_map, **replay_game_state_kwargs({"num_players": 4})).teams == {1: 1, 2: 2, 3: 3, 4: 4}


def test_gui_2v2_new_game_is_played_in_teams(monkeypatch, tmp_path):
    """The New Game flow derives teams from the map, and a map without any
    (a random one) gets the mode's default 1&3 vs 2&4."""
    import pygame

    from reinforcetactics.app import game_loop
    from reinforcetactics.utils import settings as settings_module

    pygame.init()
    monkeypatch.setattr(settings_module, "_settings_instance", settings_module.Settings(str(tmp_path / "s.json")))
    observed = []
    monkeypatch.setattr(game_loop.GameSession, "run", lambda session: observed.append(dict(session.game.teams)) or "x")
    configs = [{"type": "human", "bot_type": None}] + [{"type": "computer", "bot_type": "SimpleBot"}] * 3
    try:
        game_loop.start_new_game(mode="2v2", selected_map=str(MAPS_DIR / "2v2" / "beginner.csv"), player_configs=configs)
        game_loop.start_new_game(mode="2v2", selected_map="random", player_configs=configs)
    finally:
        pygame.quit()

    assert observed == [{1: 1, 2: 2, 3: 1, 4: 2}, {1: 1, 2: 2, 3: 1, 4: 2}]
