"""``GameState.clone_for_search`` (review core-18).

MCTS used ``copy.deepcopy`` on the whole GameState for the root and for every
new node, copying the ever-growing action history and the legal-action cache
(5-11 ms per node on a 20x20 board). The search clone must behave exactly
like a deep copy -- same legal actions, same outcomes, fully independent of
the original -- while skipping what search never reads.
"""

import copy
import logging
import random

import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import AdvancedBot, MediumBot
from reinforcetactics.rl.mcts import _execute_action_on_state, _resolve_action_refs
from reinforcetactics.utils.file_io import FileIO

logging.getLogger("reinforcetactics").setLevel(logging.WARNING)


def _midgame(fog_of_war: bool = False, turns: int = 14) -> GameState:
    """A seeded bot game on crossroads, stopped mid-game with units in contact."""
    gs = GameState(FileIO.load_map("maps/1v1/crossroads.csv"), num_players=2, fog_of_war=fog_of_war, seed=5)
    if fog_of_war:
        gs.update_visibility()
    bots = {1: AdvancedBot(gs, player=1, rng=random.Random(1)), 2: MediumBot(gs, player=2, rng=random.Random(2))}
    while gs.turn_number < turns and not gs.game_over:
        bots[gs.current_player].take_turn()
    assert not gs.game_over and len(gs.units) >= 4 and gs.action_history
    return gs


def _fingerprint(gs: GameState):
    """Everything that decides how the game continues."""
    return (
        sorted((u.unit_id, tuple(sorted(u.to_dict().items(), key=lambda kv: kv[0]))) for u in gs.units),
        [(t.x, t.y, t.player, t.health, t.regenerating) for row in gs.grid.tiles for t in row],
        dict(gs.player_gold),
        {p: dict(t) for p, t in gs.healing_totals.items()},
        (gs.current_player, gs.turn_number, gs.game_over, gs.winner, gs.end_reason, gs._next_unit_id),
        gs.rng.getstate(),
        {p: (v.state.tobytes(), sorted(v.last_seen_units)) for p, v in gs.visibility_maps.items()},
    )


def _canonical_actions(gs: GameState, player: int | None = None):
    """Legal actions with unit/tile references replaced by positions (order kept)."""

    def ref(value):
        return (type(value).__name__, value.x, value.y) if hasattr(value, "x") else value

    out = {}
    for key, actions in gs.get_legal_actions(player).items():
        out[key] = actions if key == "end_turn" else [tuple(sorted((k, ref(v)) for k, v in a.items())) for a in actions]
    return out


def _play_every_action(gs: GameState, limit: int = 200) -> None:
    """Apply the first legal action repeatedly (end_turn when nothing else is left)."""
    for _ in range(limit):
        if gs.game_over:
            return
        legal = gs.get_legal_actions()
        key = next((k for k in legal if k != "end_turn" and legal[k]), "end_turn")
        action = {} if key == "end_turn" else legal[key][0]
        _execute_action_on_state(gs, key, _resolve_action_refs(gs, key, action))


@pytest.mark.parametrize("fog_of_war", [False, True])
def test_clone_has_identical_legal_actions(fog_of_war):
    gs = _midgame(fog_of_war)
    clone = gs.clone_for_search()
    for player in (1, 2):
        assert _canonical_actions(clone, player) == _canonical_actions(gs, player)
    assert _fingerprint(clone) == _fingerprint(gs)


@pytest.mark.parametrize("fog_of_war", [False, True])
def test_mutating_the_clone_leaves_the_original_untouched(fog_of_war):
    gs = _midgame(fog_of_war)
    before = _fingerprint(gs)
    history_len = len(gs.action_history)
    legal_before = _canonical_actions(gs)

    clone = gs.clone_for_search()
    _play_every_action(clone)
    # Poke every kind of mutable state directly too.
    clone.units[0].health = 1
    clone.units[0].visible_enemies_at_action_start = {(0, 0)}
    clone.player_gold[1] += 999
    clone.healing_totals[1]["hp"] += 5
    clone.rng.random()
    for row in clone.grid.tiles:
        for tile in row:
            if tile.is_capturable():
                tile.health = 1
                tile.player = 2

    assert _fingerprint(gs) == before
    assert len(gs.action_history) == history_len
    assert _canonical_actions(gs) == legal_before


def test_clone_plays_exactly_like_a_deepcopy():
    gs = _midgame()
    deep, clone = copy.deepcopy(gs), gs.clone_for_search()
    _play_every_action(deep)
    _play_every_action(clone)
    assert _fingerprint(clone) == _fingerprint(deep)


def test_clone_drops_history_and_caches_and_shares_terrain():
    gs = _midgame()
    gs.get_legal_actions()  # warm the cache
    clone = gs.clone_for_search()

    assert clone.action_history == [] and gs.action_history
    assert clone._legal_actions_cache == {} and not clone._legal_actions_cache_valid
    assert clone.unit_data is gs.unit_data and clone.engine_overrides is gs.engine_overrides
    assert clone.initial_map_data is gs.initial_map_data
    for row, clone_row in zip(gs.grid.tiles, clone.grid.tiles, strict=True):
        for tile, clone_tile in zip(row, clone_row, strict=True):
            # Terrain never changes, so it is shared; structures are copied.
            assert (clone_tile is tile) != tile.is_capturable()
    assert all(c is not u for c, u in zip(clone.units, gs.units, strict=True))
    assert clone.rng is not gs.rng and clone.rng.getstate() == gs.rng.getstate()


def test_clone_copies_state_it_does_not_know_about():
    gs = _midgame(turns=2)
    gs.some_future_state = {"counts": [1, 2]}
    clone = gs.clone_for_search()
    clone.some_future_state["counts"].append(3)
    assert gs.some_future_state == {"counts": [1, 2]}


def test_the_clone_has_its_own_pre_move_visibility_map():
    """The map cancel_move restores is copied, so restoring and updating it in a clone can't reach the original."""
    gs = _midgame(fog_of_war=True)
    unit = next(u for u in gs.units if u.player == gs.current_player and u.can_move)
    destination = next(pos for pos in gs.get_move_destinations(unit) if pos != (unit.x, unit.y))
    assert gs.move_unit(unit, *destination)
    snapshot = unit.pre_move_visibility.state.copy()

    clone = gs.clone_for_search()
    clone_unit = next(u for u in clone.units if u.unit_id == unit.unit_id)
    assert clone_unit.pre_move_visibility is not unit.pre_move_visibility
    clone_unit.pre_move_visibility.state[:] = 2
    clone_unit.pre_move_visibility.last_seen_units[(0, 0)] = None

    assert (unit.pre_move_visibility.state == snapshot).all()
    assert (0, 0) not in unit.pre_move_visibility.last_seen_units
    assert gs.cancel_move(unit)


def test_mcts_search_clones_instead_of_deepcopying(monkeypatch):
    """The MCTS root and child states come from clone_for_search."""
    torch = pytest.importorskip("torch")
    from reinforcetactics.rl.alphazero_net import AlphaZeroNet
    from reinforcetactics.rl.mcts import MCTS

    gs = _midgame(turns=4)
    before = _fingerprint(gs)
    net = AlphaZeroNet(grid_height=gs.grid.height, grid_width=gs.grid.width, num_res_blocks=1, channels=8)
    net.eval()
    real_deepcopy = copy.deepcopy

    def guarded_deepcopy(obj, *args, **kwargs):
        assert not isinstance(obj, GameState), "MCTS deep-copied a whole GameState"
        return real_deepcopy(obj, *args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", guarded_deepcopy)
    with torch.no_grad():
        MCTS(network=net, grid_width=gs.grid.width, grid_height=gs.grid.height, num_simulations=8).search(gs)
    assert _fingerprint(gs) == before
