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
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

# ``video.py`` initialises pygame, but ``_execute_replay_action``
# itself never renders, so we just need a headless SDL driver.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from reinforcetactics.core.game_state import GameState
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.utils.replay_actions import get_schema_version
from reinforcetactics.utils.video import _execute_replay_action


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

    translate = lambda x, y: (x, y)
    for action in actions:
        _execute_replay_action(replay_game, action, translate, schema)
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
    assert replay_game.winner == original.winner, (
        f"winner mismatch: original={original.winner}, replay={replay_game.winner}"
    )
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
        assert orig_hp == repl_hp, (
            f"P{player} HP total mismatch: original={orig_hp}, replay={repl_hp}"
        )

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
    assert attack_record["attacker_killed"] is True, (
        "phase 1 should have recorded attacker_killed=True for this counter-kill"
    )

    path = _save_replay(g, tmp_path)
    _assert_replay_matches(g, path)


def test_hq_capture(tmp_path):
    """A unit captures the enemy HQ -- ``hq_capture`` end_reason.

    Seizes naturally across many turns rather than manipulating
    ``tile.health`` directly: the seize action record currently
    carries only ``captured: bool`` (no ``tile_hp_after`` yet), so
    the replay must produce the same tile-HP decrement trajectory
    organically. Phase 4 should extend the seize record so manual
    tile-state setup doesn't desync replays.
    """
    g = _make_game(max_turns=200)
    seizer = g.create_unit("W", 4, 4, player=1)
    _enable(seizer)

    # Seize -> end P1 turn -> end P2 turn -> repeat. Keeps the same
    # unit pinned on the HQ tile until the structure flips. Bounded
    # by a generous turn limit so a balance change to seize damage
    # can't hang the test.
    max_iters = 100
    for _ in range(max_iters):
        if g.game_over:
            break
        seizer.can_move = True
        seizer.can_attack = True
        g.seize(seizer)
        if g.game_over:
            break
        g.end_turn()  # P1 -> P2
        g.end_turn()  # P2 -> P1 (next round)

    assert g.game_over, "expected game_over from HQ capture within iteration budget"
    assert g.end_reason == "hq_capture"
    assert g.winner == 1

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
