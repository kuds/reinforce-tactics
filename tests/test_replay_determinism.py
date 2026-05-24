"""End-to-end replay-equivalence tests.

For every end-condition the engine can produce (``elimination``,
``elimination`` via counter-kill, ``hq_capture``, ``max_turns_draw``,
``resign``), and for at least one scenario that triggers the Rogue
evade RNG, this module:

  1. plays a scripted game on a fresh ``GameState``
  2. writes the replay to a tmp file via the normal save path
  3. loads it back and runs it through ``_execute_replay_action``
     (the v2 dispatch added in phase 2)
  4. asserts the post-replay game_state matches the original on
     every checksum the runner now records: ``game_over``,
     ``winner``, ``end_reason``, per-player unit count, per-player
     HP total.

The Rogue-evade case is deliberately replayed with a *different*
``random.seed`` than the original run -- the v2 path applies
recorded HP-after / killed flags directly, so RNG drift between
record and playback must not change the outcome.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ``video.py`` initialises pygame, but ``_execute_replay_action``
# itself never renders, so we just need a headless SDL driver.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from reinforcetactics.core.game_state import GameState  # noqa: E402
from reinforcetactics.utils.file_io import FileIO  # noqa: E402
from reinforcetactics.utils.replay_actions import get_schema_version  # noqa: E402
from reinforcetactics.utils.video import _execute_replay_action  # noqa: E402


def _translate(x: int, y: int) -> Tuple[int, int]:
    """Identity coord translator (tests use unpadded maps directly)."""
    return (x, y)


# ----------------------------------------------------------------- map
# Small symmetric 1v1 layout with HQ + barracks in opposite corners
# and a forest in the middle (lets us put a Rogue on forest to trigger
# the higher evade probability for the RNG scenario).
SMALL_MAP = np.array(
    [
        ["o", "o", "o", "o", "o", "o"],
        ["o", "h_1", "b_1", "p", "p", "o"],
        ["o", "b_1", "p", "f", "p", "o"],
        ["o", "p", "f", "p", "b_2", "o"],
        ["o", "p", "p", "b_2", "h_2", "o"],
        ["o", "o", "o", "o", "o", "o"],
    ],
    dtype=object,
)


def _make_game(max_turns: int = 50) -> GameState:
    """Fresh game with both players flush so create_unit always succeeds."""
    game = GameState(SMALL_MAP, num_players=2)
    game.max_turns = max_turns
    game.player_gold[1] = 9999
    game.player_gold[2] = 9999
    return game


def _save_replay(game: GameState, tmp_path: Path) -> str:
    """Serialise a game's action history with the full v2 game_info."""
    final_p1 = [u for u in game.units if u.player == 1]
    final_p2 = [u for u in game.units if u.player == 2]
    game_info: Dict[str, Any] = {
        "winner": game.winner,
        "winner_name": "P1" if game.winner == 1 else ("P2" if game.winner == 2 else "Draw"),
        "turns": game.turn_number,
        "max_turns": game.max_turns,
        "initial_map": SMALL_MAP.tolist(),
        "game_over": game.game_over,
        "end_reason": game.end_reason,
        "winning_action_index": game.game_over_action_index,
        "replay_schema_version": 2,
        "final_units_p1": len(final_p1),
        "final_units_p2": len(final_p2),
        "final_hp_total_p1": sum(u.health for u in final_p1),
        "final_hp_total_p2": sum(u.health for u in final_p2),
        "player_configs": [
            {"player_no": 1, "type": "bot", "name": "P1"},
            {"player_no": 2, "type": "bot", "name": "P2"},
        ],
    }
    path = str(tmp_path / "replay.json")
    FileIO.save_replay(game.action_history, game_info, path)
    return path


def _replay(path: str) -> Tuple[GameState, Dict[str, Any]]:
    """Run a saved replay through the v2 executor and return final state."""
    data = FileIO.load_replay(path)
    game_info = data["game_info"]
    actions: List[Dict[str, Any]] = data["actions"]
    schema = get_schema_version(game_info)

    replay_game = GameState(SMALL_MAP, num_players=2)
    replay_game.max_turns = game_info["max_turns"]
    replay_game.player_gold[1] = 9999
    replay_game.player_gold[2] = 9999

    # Reseed to a *different* value than the original. v2 must not
    # depend on RNG state being reproduced -- it applies recorded
    # outcomes directly.
    random.seed(12345)

    for action in actions:
        _execute_replay_action(replay_game, action, _translate, schema)
        if replay_game.game_over:
            break

    return replay_game, game_info


def _assert_replay_matches(original: GameState, replay_path: str) -> None:
    """Core equivalence check used by every scenario."""
    replay_game, game_info = _replay(replay_path)

    # Engine-level equivalence
    assert replay_game.game_over == original.game_over, (
        f"game_over mismatch: original={original.game_over}, replay={replay_game.game_over}"
    )
    assert replay_game.winner == original.winner, f"winner mismatch: original={original.winner}, replay={replay_game.winner}"
    assert replay_game.end_reason == original.end_reason, (
        f"end_reason mismatch: original={original.end_reason}, replay={replay_game.end_reason}"
    )

    # Per-player unit counts and HP totals (the snapshot fields the
    # runner now writes -- if these match, no "ghost" units survived
    # in the replay).
    for player in (1, 2):
        orig_units = [u for u in original.units if u.player == player]
        repl_units = [u for u in replay_game.units if u.player == player]
        assert len(orig_units) == len(repl_units), (
            f"P{player} unit count mismatch: original={len(orig_units)}, replay={len(repl_units)}"
        )
        orig_hp = sum(u.health for u in orig_units)
        repl_hp = sum(u.health for u in repl_units)
        assert orig_hp == repl_hp, f"P{player} HP total mismatch: original={orig_hp}, replay={repl_hp}"

    # v2 invariant: no actions logged past the winning action.
    if original.game_over:
        winning_idx = game_info["winning_action_index"]
        assert winning_idx == len(_load_actions(replay_path)) - 1, (
            f"v2 replay should have no trailing actions: "
            f"winning_action_index={winning_idx}, len(actions)={len(_load_actions(replay_path))}"
        )


def _load_actions(path: str) -> List[Dict[str, Any]]:
    return FileIO.load_replay(path)["actions"]


# ---- helpers for forcing specific scenarios --------------------------
def _enable(unit) -> None:
    unit.can_move = True
    unit.can_attack = True


# ===================================================================
# Scenarios
# ===================================================================


def test_elimination_direct_kill(tmp_path):
    """P1's last unit is killed by a direct P2 attack."""
    g = _make_game()
    p1 = g.create_unit("W", 2, 1, player=1)
    p2 = g.create_unit("W", 2, 2, player=2)
    _enable(p1)
    _enable(p2)
    # Damage P1 so the next P2 attack finishes them off.
    p1.health = 1

    # Make it P2's turn cleanly so the attack is legal.
    g.end_turn()  # P1 -> P2 (no-op for P1)
    _enable(p2)
    g.attack(p2, p1)

    assert g.game_over and g.end_reason == "elimination" and g.winner == 2
    path = _save_replay(g, tmp_path)
    _assert_replay_matches(g, path)


def test_elimination_via_counter_kill(tmp_path):
    """P1's only unit dies *attacking* a stronger P2 unit (the original
    bug -- attacker_killed was never recorded in v1 replays)."""
    g = _make_game()
    attacker = g.create_unit("W", 2, 1, player=1)
    defender = g.create_unit("W", 2, 2, player=2)
    _enable(attacker)
    _enable(defender)
    # Glass cannon attacker: enough damage to provoke a counter, low
    # enough HP that the counter kills them.
    attacker.health = 1

    g.attack(attacker, defender)

    assert g.game_over, "expected game_over from counter-kill elimination"
    assert g.end_reason == "elimination"
    assert g.winner == 2, f"expected P2 to win, got winner={g.winner}"
    # Sanity-check the new field on the recorded attack.
    attack_record = next(a for a in g.action_history if a["type"] == "attack")
    assert attack_record["attacker_killed"] is True, "phase 1 should have recorded attacker_killed=True for this counter-kill"

    path = _save_replay(g, tmp_path)
    _assert_replay_matches(g, path)


def test_hq_capture(tmp_path):
    """A unit captures the enemy HQ -- ``hq_capture`` end_reason.

    Manipulates ``tile.health`` directly so a single seize finishes
    it. Pre-phase-4 the seize record didn't carry tile state, so the
    replay's ``mechanics.seize_structure`` started from full HQ
    health and the structure never flipped -- replay diverged. The
    v2 seize record (``tile_hp_after`` + ``tile_owner_after``) makes
    this case self-contained.
    """
    g = _make_game()
    seizer = g.create_unit("W", 4, 4, player=1)
    _enable(seizer)
    hq_tile = g.grid.get_tile(4, 4)  # h_2 location
    hq_tile.health = 1

    g.seize(seizer)

    assert g.game_over, "expected game_over from HQ capture"
    assert g.end_reason == "hq_capture"
    assert g.winner == 1

    # Sanity-check the new v2 seize fields.
    seize_record = next(a for a in g.action_history if a["type"] == "seize")
    assert seize_record["tile_owner_after"] == 1
    assert seize_record["captured"] is True

    path = _save_replay(g, tmp_path)
    _assert_replay_matches(g, path)


def test_max_turns_draw(tmp_path):
    """Both players end_turn until ``max_turns`` triggers a draw."""
    g = _make_game(max_turns=2)
    while not g.game_over:
        g.end_turn()

    assert g.end_reason == "max_turns_draw"
    assert g.winner is None

    path = _save_replay(g, tmp_path)
    _assert_replay_matches(g, path)


def test_resign(tmp_path):
    """P1 resigns -- P2 wins by ``resign``."""
    g = _make_game()
    g.create_unit("W", 2, 1, player=1)
    g.create_unit("W", 2, 2, player=2)
    g.resign(1)

    assert g.game_over and g.end_reason == "resign" and g.winner == 2

    path = _save_replay(g, tmp_path)
    _assert_replay_matches(g, path)


def test_rogue_forest_evade_outcome_is_captured(tmp_path):
    """Force a Rogue to attack from a forest tile (where the counter-evade
    probability is 15% + 15% = 30%). The recorded ``evade`` flag must
    reflect what actually happened, and the v2 replay must reproduce
    the same outcome under a different RNG seed.

    The user's phase 4 ask: "capture that outcome when there are aspects
    of RNG (like if the rogue evaded the attack when on a forest) so
    that they are entirely self contained".
    """
    random.seed(0)  # pin the original roll so the recorded outcome is reproducible

    g = _make_game()
    # The SMALL_MAP has an 'f' forest tile at (3, 2). Spawn a Rogue
    # there and a Warrior next to it so the attack provokes a counter
    # which then exercises the evade roll.
    forest_tile = g.grid.get_tile(3, 2)
    assert forest_tile.type == "f", f"expected forest at (3,2), got {forest_tile.type}"

    rogue = g.create_unit("R", 3, 2, player=1)
    target = g.create_unit("W", 3, 3, player=2)
    _enable(rogue)
    _enable(target)
    # Keep both well above one-shot range so the counter actually
    # gets a chance to fire (or evade).
    rogue.health = rogue.max_health
    target.health = max(rogue.health + 5, target.max_health)

    g.attack(rogue, target)

    attack_record = next(a for a in g.action_history if a["type"] == "attack")
    # The evade field is the captured RNG outcome; it must be a bool.
    assert isinstance(attack_record["evade"], bool)
    assert attack_record["attacker_pos"] == (3, 2), "attacker should still be on forest"

    path = _save_replay(g, tmp_path)
    # _replay re-seeds RNG to 12345 -- different from 0 above. The v2
    # path never re-rolls evade, so the recorded outcome must hold.
    _assert_replay_matches(g, path)


def test_rogue_evade_rng_independence(tmp_path):
    """A Rogue attack provokes the counter-evade RNG roll. The replay is
    run with a *different* RNG seed; v2 must reproduce the original
    outcome exactly because it applies recorded HP-after directly
    instead of re-rolling the engine."""
    # Pin the original run's RNG so the recorded outcome is reproducible
    # if anyone re-runs this test from scratch.
    random.seed(7)

    g = _make_game()
    rogue = g.create_unit("R", 2, 1, player=1)
    target = g.create_unit("W", 2, 2, player=2)
    _enable(rogue)
    _enable(target)
    # Let both survive the exchange so the counter-evade outcome
    # actually shows up as recorded HP deltas (an instant-kill skips
    # the counter path entirely).
    target.health = max(2, target.max_health // 2)

    g.attack(rogue, target)

    attack_record = next(a for a in g.action_history if a["type"] == "attack")
    # Whatever the roll happened to be, the replay must reproduce it.
    expected_attacker_hp = attack_record["attacker_hp_after"]
    expected_target_hp = attack_record["target_hp_after"]

    path = _save_replay(g, tmp_path)
    # Internally _replay seeds RNG to 12345 -- different from 7 above.
    replay_game, _ = _replay(path)

    rogue_replay = next((u for u in replay_game.units if u.type == "R" and u.player == 1), None)
    target_replay = next((u for u in replay_game.units if u.type == "W" and u.player == 2), None)

    if attack_record["attacker_killed"]:
        assert rogue_replay is None, "killed Rogue should not survive in v2 replay"
    else:
        assert rogue_replay is not None
        assert rogue_replay.health == expected_attacker_hp

    if attack_record["target_killed"]:
        assert target_replay is None
    else:
        assert target_replay is not None
        assert target_replay.health == expected_target_hp


# ===================================================================
# Schema-version dispatch sanity
# ===================================================================


def test_schema_version_default_is_1():
    assert get_schema_version({}) == 1
    assert get_schema_version({"replay_schema_version": 1}) == 1


def test_schema_version_v2():
    assert get_schema_version({"replay_schema_version": 2}) == 2


def test_no_trailing_actions_after_game_over(tmp_path):
    """Phase 1 ``record_action`` early-return: once game_over is set,
    no further actions should land in action_history regardless of how
    many bot-style calls land afterwards."""
    g = _make_game()
    g.create_unit("W", 2, 1, player=1)
    g.create_unit("W", 2, 2, player=2)
    g.resign(1)

    actions_after_game_over = len(g.action_history)

    # All these would normally append to action_history. Verify the
    # guard makes them no-ops.
    g.end_turn()
    g.record_action("create_unit", player=1, x=0, y=0, unit_type="W")
    g.end_turn()

    assert len(g.action_history) == actions_after_game_over
    assert g.game_over_action_index == len(g.action_history) - 1


# ===================================================================
# Engine stale-reference guards (ghost-action prevention)
# ===================================================================


def test_dead_unit_cannot_move(tmp_path):
    """After a unit dies (counter-killed), passing its stale reference
    to ``move_unit`` must be a silent no-op -- no engine mutation, no
    log entry. This is the engine bug that produced the v2 replay
    divergence we found in the balance_analysis baseline (PR #360
    audit: 90% of replays affected).
    """
    g = _make_game()
    attacker = g.create_unit("W", 2, 1, player=1)
    defender = g.create_unit("W", 2, 2, player=2)
    _enable(attacker)
    _enable(defender)
    attacker.health = 1  # guarantees counter-kill

    g.attack(attacker, defender)
    assert attacker not in g.units, "expected attacker to be counter-killed"

    log_len_before = len(g.action_history)
    moved = g.move_unit(attacker, 3, 1)  # bot still holds stale ref

    assert moved is False, "dead unit must not move"
    assert len(g.action_history) == log_len_before, "dead unit must not log"
    assert attacker.x == 2 and attacker.y == 1, "dead unit position must not change"


def test_dead_unit_cannot_attack_seize_or_buff(tmp_path):
    """The same stale-reference guard must apply across the whole
    action surface, not just move_unit."""
    g = _make_game()
    a = g.create_unit("W", 2, 1, player=1)
    d = g.create_unit("W", 2, 2, player=2)
    other = g.create_unit("W", 3, 1, player=1)
    _enable(a)
    _enable(d)
    _enable(other)
    a.health = 1

    g.attack(a, d)
    assert a not in g.units

    log_len = len(g.action_history)

    # Each of these should silently no-op.
    g.attack(a, other)  # dead attacker
    g.attack(other, a)  # dead target
    g.seize(a)
    g.heal(a, other)
    g.cure(a, other)
    g.paralyze(a, other)
    g.haste(a, other)
    g.defence_buff(a, other)
    g.attack_buff(a, other)

    assert len(g.action_history) == log_len, f"dead unit produced {len(g.action_history) - log_len} ghost actions"


# ===================================================================
# v2 helper resiliency for older replays with ghost actions
# ===================================================================


def test_v2_attack_helper_terminates_even_with_missing_units(tmp_path):
    """A pre-fix replay can contain a recorded eliminating attack whose
    target unit is missing in the replay state (because earlier ghost
    actions moved it elsewhere or removed it). The v2 helper must
    still trigger ``_check_player_eliminated`` on the recorded
    outcome so the replay reaches game_over on the recorded winning
    action index.
    """
    # Build a state where P2 has zero units and P1 has one. Synthesise
    # an attack action that records target_killed=True against a
    # non-existent target. The replay must terminate.
    g = _make_game()
    p1_only = g.create_unit("W", 2, 1, player=1)
    _enable(p1_only)
    # No P2 units exist. Game isn't over yet because no kill ran.
    assert not g.game_over

    from reinforcetactics.utils.replay_actions import apply_recorded_attack

    fake_killing_attack = {
        "type": "attack",
        "player": 1,
        "attacker_pos": [2, 1],
        "target_pos": [9, 9],  # nothing here
        "attacker_type": "W",
        "target_type": "W",
        "damage": 7,
        "target_killed": True,
        "attacker_killed": False,
        "counter_damage": 0,
        "attacker_hp_after": 15,
        "target_hp_after": 0,
        "evade": False,
        "charge_bonus": False,
        "flank_bonus": False,
        "attack_buff": False,
        "defence_buff": False,
    }
    apply_recorded_attack(g, fake_killing_attack, _translate)

    assert g.game_over, "v2 helper must still fire game_over even with missing target"
    assert g.end_reason == "elimination"
    assert g.winner == 1


def test_v2_seize_helper_applies_tile_state_even_without_seizer(tmp_path):
    """Pre-fix replays can record a seize whose unit is missing in the
    replay state. Tile mutations are fully described by tile_hp_after
    / tile_owner_after, so the helper must apply them regardless of
    whether the seizer is present, and fire ``hq_capture`` end-game
    on a recorded captured HQ.
    """
    g = _make_game()
    p1_unit = g.create_unit("W", 2, 1, player=1)
    _enable(p1_unit)

    from reinforcetactics.utils.replay_actions import apply_recorded_seize

    # Synthesise an HQ-capture seize with no unit at the target tile.
    hq_pos = [4, 4]  # h_2 location
    fake_hq_capture = {
        "type": "seize",
        "player": 1,
        "unit_type": "W",
        "position": hq_pos,
        "structure_type": "h",
        "captured": True,
        "tile_hp_after": 30,
        "tile_owner_after": 1,
    }
    hq_tile = g.grid.get_tile(*hq_pos)
    original_owner = hq_tile.player

    apply_recorded_seize(g, fake_hq_capture, _translate)

    assert hq_tile.player == 1, "tile owner must reflect recorded outcome"
    assert hq_tile.player != original_owner, "tile must have actually flipped"
    assert hq_tile.health == 30
    assert g.game_over, "captured HQ must trigger hq_capture end-game"
    assert g.end_reason == "hq_capture"
    assert g.winner == 1
