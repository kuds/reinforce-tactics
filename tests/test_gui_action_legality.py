"""The GUI's unit actions follow the engine's rules.

Since the engine refuses illegal actions (review §1.2), a GUI that offers an
action the engine then refuses does nothing -- and it used to print success
and end the unit's turn anyway. The unit action menu now builds its targets
from ``get_legal_actions``, and a refused action leaves the unit's action
unspent (review pygame-9, critic-integration-2).
"""

from unittest.mock import Mock

import numpy as np
import pygame
import pytest

from reinforcetactics.app.action_executor import apply_targeted_action, execute_unit_action
from reinforcetactics.app.input_handler import InputHandler
from reinforcetactics.constants import TILE_SIZE
from reinforcetactics.core.game_state import GameState
from reinforcetactics.ui.menus.in_game.unit_action_menu import UnitActionMenu


def _plains_map(size=10):
    map_data = np.array([["p" for _ in range(size)] for _ in range(size)], dtype=object)
    map_data[0][0] = "h_1"
    map_data[size - 1][size - 1] = "h_2"
    return map_data


@pytest.fixture
def screen():
    pygame.init()
    return pygame.display.set_mode((TILE_SIZE * 10, TILE_SIZE * 10))


@pytest.fixture
def game():
    return GameState(_plains_map(), num_players=2)


def _menu_targets(screen, game, unit):
    menu = UnitActionMenu(screen, game, unit)
    return {a["type"]: a["targets"] for a in menu.actions}


def _attack_count(game):
    return sum(1 for a in game.action_history if a.get("type") == "attack")


class TestMenuTargetsComeFromTheEngine:
    def test_paralyze_offers_range_two_targets(self, screen, game):
        mage = game.place_unit("M", 2, 2, player=1)
        enemy = game.place_unit("W", 2, 4, player=2)  # Manhattan distance 2
        assert game._can_paralyze_target(mage, enemy)

        targets = _menu_targets(screen, game, mage)
        # The old menu listed only adjacent enemies for Paralyze.
        assert enemy in targets.get("paralyze", [])

    def test_paralyze_hidden_while_on_cooldown(self, screen, game):
        mage = game.place_unit("M", 2, 2, player=1)
        game.place_unit("W", 2, 3, player=2)
        mage.paralyze_cooldown = 2
        game._invalidate_cache()

        assert "paralyze" not in _menu_targets(screen, game, mage)

    def test_menu_targets_equal_legal_actions(self, screen, game):
        mage = game.place_unit("M", 4, 4, player=1)
        cleric = game.place_unit("C", 4, 5, player=1)
        ally = game.place_unit("W", 5, 4, player=1)
        ally.health = 3
        game.place_unit("W", 4, 3, player=2)
        game.place_unit("A", 6, 4, player=2)
        game._invalidate_cache()

        legal = game.get_legal_actions(player=1)
        for unit, actor_keys in (
            (mage, {"attack": "attacker", "paralyze": "paralyzer"}),
            (cleric, {"attack": "attacker", "heal": "healer", "cure": "curer"}),
        ):
            targets = _menu_targets(screen, game, unit)
            for kind, actor_key in actor_keys.items():
                expected = [a["target"] for a in legal[kind] if a[actor_key] is unit]
                assert targets.get(kind, []) == expected, (unit.type, kind)

    def test_fog_hides_attack_on_enemy_discovered_after_moving(self, screen):
        game = GameState(_plains_map(), num_players=2, fog_of_war=True)
        attacker = game.place_unit("W", 1, 1, player=1)
        target = game.place_unit("W", 6, 6, player=2)
        game.update_visibility(player=1)
        assert not game.is_position_visible(6, 6, player=1)

        # Snapshot what the unit sees when its action starts, then move it
        # next to the (still unseen) enemy, as the GUI does on selection.
        game.capture_visible_enemies_for_unit(attacker)
        attacker.x = attacker.original_x = 4
        attacker.y = attacker.original_y = 6
        assert game.move_unit(attacker, 5, 6)
        assert game.is_position_visible(6, 6, player=1)

        # Range alone would offer the attack; the engine forbids it.
        assert not game.is_enemy_attackable_by_unit(attacker, target)
        assert "attack" not in _menu_targets(screen, game, attacker)


class TestRefusedActionsKeepTheUnitsAction:
    def test_refused_single_target_attack_does_not_end_the_turn(self, game):
        warrior = game.place_unit("W", 2, 2, player=1)
        far_enemy = game.place_unit("W", 2, 7, player=2)
        selected = [warrior]

        result = execute_unit_action(game, {"type": "attack", "targets": [far_enemy]}, warrior, selected)

        assert result == (False, None, warrior)
        assert warrior.can_attack and warrior.can_move
        assert far_enemy.health == far_enemy.max_health
        assert _attack_count(game) == 0
        assert selected[0] is warrior

    def test_accepted_attack_ends_the_turn(self, game):
        warrior = game.place_unit("W", 2, 2, player=1)
        enemy = game.place_unit("W", 2, 3, player=2)
        selected = [warrior]

        result = execute_unit_action(game, {"type": "attack", "targets": [enemy]}, warrior, selected)

        assert result == (False, None, None)
        assert not warrior.can_attack
        assert _attack_count(game) == 1

    def test_refused_capture_does_not_end_the_turn(self, game):
        # Standing on its own HQ: seizing is illegal.
        warrior = game.place_unit("W", 0, 0, player=1)
        selected = [warrior]

        result = execute_unit_action(game, {"type": "capture", "targets": None}, warrior, selected)

        assert result == (False, None, warrior)
        assert warrior.can_attack and warrior.can_move

    def test_apply_targeted_action_reports_the_engines_decision(self, game):
        cleric = game.place_unit("C", 2, 2, player=1)
        healthy_ally = game.place_unit("W", 2, 3, player=1)
        hurt_ally = game.place_unit("W", 3, 2, player=1)
        hurt_ally.health = 2
        game._invalidate_cache()

        assert apply_targeted_action(game, "heal", cleric, healthy_ally) is False
        assert apply_targeted_action(game, "heal", cleric, hurt_ally) is True
        assert hurt_ally.health > 2

    def test_refused_target_click_reopens_the_menu(self, screen, game):
        warrior = game.place_unit("W", 2, 2, player=1)
        far_enemy = game.place_unit("W", 2, 7, player=2)
        renderer = Mock()
        renderer.screen = screen
        handler = InputHandler(game, renderer, bots={}, num_players=2)
        handler.target_selection_mode = True
        handler.target_selection_unit = warrior
        handler.target_selection_action = {"type": "attack", "targets": [far_enemy]}

        click = (far_enemy.x * TILE_SIZE + 1, far_enemy.y * TILE_SIZE + 1)
        assert handler._handle_target_selection_click(click, current_time=0) == "continue"

        assert warrior.can_attack and warrior.can_move
        assert isinstance(handler.active_menu, UnitActionMenu)
        assert handler.target_selection_mode is False
        assert far_enemy.health == far_enemy.max_health
