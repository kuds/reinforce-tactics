"""flat_discrete decode tables: versioned layouts and truncation (review rlenv-11 / prior-13).

A flat_discrete checkpoint's Discrete index means "the i-th entry of the
legal-action table", so the table layout is part of the checkpoint. Version
1 is the layout every checkpoint before versioning was trained on; it drops
heals, casts and then attacks first when the legal set overflows
``max_flat_actions``. Version 2 (the default for new envs) drops moves first,
round-robin across units, and keeps combat and support. These tests pin:

* version 1 reproduces the base commit's tables byte for byte (against a
  frozen copy of that implementation, on seeded game states);
* the two versions agree whenever nothing is truncated;
* version 2's truncation order and round-robin sharing;
* truncation is visible in ``info`` / ``episode_stats`` and its warning is
  rate-limited;
* the version rides on the action space into an SB3 checkpoint, and
  ModelBot decodes old (unstamped) checkpoints with version 1.
"""

import logging
import random

import numpy as np
import pytest

from reinforcetactics.game.model_bot import ModelBot
from reinforcetactics.rl import gym_env
from reinforcetactics.rl.gym_env import (
    FLAT_ACTION_VERSION_LATEST,
    FLAT_ACTION_VERSION_LEGACY,
    FLAT_ACTION_VERSIONS,
    StrategyGameEnv,
    build_flat_actions,
    flat_action_table,
    flat_action_version_of,
    stamp_flat_action_version,
)

BEGINNER_MAP = "maps/1v1/beginner.csv"
SEEDED_MAPS = ("maps/1v1/beginner.csv", "maps/1v1/skirmish.csv", "maps/1v1/corner_points.csv")
CAPS = (1, 2, 3, 5, 8, 16, 32, 64, 512)


# ---------------------------------------------------------------------------
# Frozen copy of build_flat_actions as of commit 713ce02 (the last layout
# before versioning), with the encoding tables it read frozen alongside it.
# Do not edit: this is what version 1 must keep reproducing.
# ---------------------------------------------------------------------------

_FROZEN_ACTION_KEY_MAP = {
    "create_unit": (0, None, ("x", "y")),
    "move": (1, ("from_x", "from_y"), ("to_x", "to_y")),
    "attack": (2, "attacker", "target"),
    "seize": (3, "unit", "tile"),
    "heal": (4, "healer", "target"),
    "cure": (4, "curer", "target"),
    "paralyze": (6, "paralyzer", "target"),
    "haste": (7, "sorcerer", "target"),
    "defence_buff": (8, "sorcerer", "target"),
    "attack_buff": (9, "sorcerer", "target"),
}
_FROZEN_UNIT_TYPE_TO_IDX = {"W": 0, "M": 1, "C": 2, "A": 3, "K": 4, "R": 5, "S": 6, "B": 7}


def _frozen_action_pos(obj_or_dict, fields):
    if isinstance(fields, str):
        o = obj_or_dict[fields]
        return o.x, o.y
    return obj_or_dict[fields[0]], obj_or_dict[fields[1]]


def _build_flat_actions_713ce02(game_state, player, max_flat_actions):
    legal_actions = game_state.get_legal_actions(player=player)

    actions = []
    seen = set()

    for key, (at_idx, src_fields, tgt_fields) in _FROZEN_ACTION_KEY_MAP.items():
        for action in legal_actions.get(key, []):
            tx, ty = _frozen_action_pos(action, tgt_fields)

            if src_fields is not None:
                fx, fy = _frozen_action_pos(action, src_fields)
            else:
                fx, fy = tx, ty  # create_unit: from = building position

            ut_idx = 0
            if key == "create_unit":
                ut_idx = _FROZEN_UNIT_TYPE_TO_IDX.get(action["unit_type"], 0)

            action_key = (at_idx, ut_idx, fx, fy, tx, ty)
            if action_key not in seen:
                seen.add(action_key)
                actions.append(np.array(action_key, dtype=np.int32))

    end_turn_key = (5, 0, 0, 0, 0, 0)
    if end_turn_key not in seen:
        seen.add(end_turn_key)
        actions.append(np.array(end_turn_key, dtype=np.int32))

    if len(actions) > max_flat_actions:
        protected = [a for a in actions if int(a[0]) in (3, 5)]
        others = [a for a in actions if int(a[0]) not in (3, 5)]
        budget = max(0, max_flat_actions - len(protected))
        actions = others[:budget] + protected
        if len(actions) > max_flat_actions:
            actions = actions[-max_flat_actions:]

    return actions


# ---------------------------------------------------------------------------
# Seeded decision points
# ---------------------------------------------------------------------------


def _seeded_decision_points(map_file: str, seed: int, n_actions: int = 160):
    """Yield ``(game_state, player)`` at seeded decision points of a random-play game.

    Both seats play uniformly random legal actions (from the uncapped
    table, so the trajectory does not depend on any truncation), with an
    occasional forced end_turn so armies and turns both advance.
    """
    env = StrategyGameEnv(map_file=map_file, opponent=None, action_space_type="flat_discrete", max_flat_actions=100_000)
    env.reset(seed=seed)
    gs = env.game_state
    rng = random.Random(seed)
    try:
        for _ in range(n_actions):
            if gs.game_over:
                return
            player = gs.current_player
            yield gs, player
            full = build_flat_actions(gs, player, 100_000)
            action = full[rng.randrange(len(full))]
            env.execute_game_action(env._encode_action(action), player)
            if int(action[0]) != 5 and rng.random() < 0.08 and not gs.game_over:
                gs.end_turn()
    finally:
        env.close()


def _as_bytes(actions):
    return [(a.dtype.str, a.shape, a.tobytes()) for a in actions]


def _truncating_state():
    """A skirmish position whose ~110 legal actions include every kind of action.

    Player 1 (to move): a wounded Warrior in contact, an Archer, a Mage, a
    Cleric next to the Warrior, a Sorcerer on a neutral tower and a Knight;
    player 2: two Warriors and an Archer in reach. Returns
    ``(game_state, player, full_table)``.
    """
    from reinforcetactics.core.game_state import GameState
    from reinforcetactics.utils.file_io import FileIO

    gs = GameState(FileIO.load_map("maps/1v1/skirmish.csv"), num_players=2)
    for unit_type, x, y in (("W", 3, 3), ("A", 2, 4), ("M", 4, 2), ("C", 2, 2), ("S", 3, 2), ("K", 5, 4)):
        gs.place_unit(unit_type, x, y, 1)
    for unit_type, x, y in (("W", 3, 4), ("W", 4, 3), ("A", 5, 5)):
        gs.place_unit(unit_type, x, y, 2)
    gs.get_unit_at_position(3, 3).health -= 3
    full = build_flat_actions(gs, 1, 100_000)
    kinds = {int(a[0]) for a in full}
    assert {0, 1, 2, 3, 4, 5, 6, 7} <= kinds, kinds
    return gs, 1, full


# ---------------------------------------------------------------------------
# Version 1 is the base commit's layout
# ---------------------------------------------------------------------------


class TestLegacyLayoutIsFrozen:
    @pytest.mark.parametrize("map_file", SEEDED_MAPS)
    def test_version_1_reproduces_base_commit_tables_byte_for_byte(self, map_file):
        checked = truncated = 0
        for gs, player in _seeded_decision_points(map_file, seed=7):
            n_full = len(_build_flat_actions_713ce02(gs, player, 100_000))
            for cap in CAPS:
                expected = _build_flat_actions_713ce02(gs, player, cap)
                got = build_flat_actions(gs, player, cap, version=FLAT_ACTION_VERSION_LEGACY)
                assert _as_bytes(got) == _as_bytes(expected), (map_file, gs.turn_number, cap)
                checked += 1
                truncated += n_full > cap
        # The seeded games must actually exercise truncation.
        assert checked > 100 and truncated > 50

    def test_default_version_of_the_free_function_is_legacy(self):
        gs, player, _ = _truncating_state()
        for cap in (4, 9, 17):
            assert _as_bytes(build_flat_actions(gs, player, cap)) == _as_bytes(_build_flat_actions_713ce02(gs, player, cap))

    def test_legacy_env_reproduces_base_commit_tables(self):
        env = StrategyGameEnv(
            map_file="maps/1v1/skirmish.csv",
            opponent="random",
            action_space_type="flat_discrete",
            max_flat_actions=12,
            flat_action_version=FLAT_ACTION_VERSION_LEGACY,
        )
        env.reset(seed=3)
        rng = random.Random(3)
        for _ in range(60):
            (mask,) = env.action_masks()
            expected = _build_flat_actions_713ce02(env.game_state, env.agent_player, 12)
            assert _as_bytes(env._current_actions) == _as_bytes(expected)
            assert mask.sum() == len(expected)
            _, _, term, trunc, _ = env.step(rng.randrange(int(mask.sum())))
            if term or trunc:
                env.reset(seed=rng.randrange(1000))
        env.close()


# ---------------------------------------------------------------------------
# Version 2
# ---------------------------------------------------------------------------


class TestVersion2Truncation:
    @pytest.mark.parametrize("map_file", SEEDED_MAPS)
    def test_versions_agree_whenever_nothing_is_truncated(self, map_file):
        for gs, player in _seeded_decision_points(map_file, seed=11, n_actions=80):
            full = build_flat_actions(gs, player, 100_000, version=FLAT_ACTION_VERSION_LEGACY)
            v2 = flat_action_table(gs, player, len(full), version=2)
            assert not v2.truncated and v2.n_legal == len(full)
            assert _as_bytes(v2.actions) == _as_bytes(full)

    def test_moves_are_dropped_before_combat_support_and_seize(self):
        gs, player, full = _truncating_state()
        n_moves = sum(int(a[0]) == 1 for a in full)
        non_moves = [tuple(a) for a in full if int(a[0]) != 1]
        cap = len(non_moves) + max(1, n_moves // 3)
        table = flat_action_table(gs, player, cap, version=2)
        assert table.truncated and table.n_legal == len(full) and len(table.actions) == cap
        kept = [tuple(a) for a in table.actions]
        # Every non-move action survives; only moves were dropped.
        assert [a for a in kept if a[0] != 1] == non_moves

    def test_kept_entries_stay_in_canonical_order_with_end_turn_last(self):
        gs, player, full = _truncating_state()
        order = {tuple(a): i for i, a in enumerate(full)}
        for cap in (1, 2, 5, 9, 20, len(full) - 1):
            kept = [tuple(a) for a in build_flat_actions(gs, player, cap, version=2)]
            assert len(kept) == min(cap, len(full))
            assert [order[a] for a in kept] == sorted(order[a] for a in kept)
            assert kept[-1] == (5, 0, 0, 0, 0, 0)

    def test_moves_are_shared_round_robin_across_units(self):
        gs, player, full = _truncating_state()
        movers = {(int(a[2]), int(a[3])) for a in full if int(a[0]) == 1}
        if len(movers) < 2:
            pytest.skip("needs two units with moves")
        n_other = sum(int(a[0]) != 1 for a in full)
        cap = n_other + len(movers)  # room for exactly one move per unit
        kept_moves = [a for a in build_flat_actions(gs, player, cap, version=2) if int(a[0]) == 1]
        assert {(int(a[2]), int(a[3])) for a in kept_moves} == movers
        # Version 1 at the same cap keeps the first units' moves and drops the rest.
        legacy = build_flat_actions(gs, player, cap, version=1)
        assert {(int(a[2]), int(a[3])) for a in legacy if int(a[0]) == 1} != movers or len(movers) == 1

    def test_combat_is_kept_where_version_1_dropped_it(self):
        gs, player, full = _truncating_state()
        combat = [tuple(a) for a in full if int(a[0]) in (2, 4, 6, 7, 8, 9)]
        if not combat:
            pytest.skip("seeded state has no combat or support action")
        n_protected_v1 = sum(int(a[0]) in (3, 5) for a in full)
        n_before_combat = sum(int(a[0]) in (0, 1) for a in full)
        cap = n_protected_v1 + n_before_combat  # v1 fills the budget with creates/moves only
        v1 = {tuple(a) for a in build_flat_actions(gs, player, cap, version=1)}
        v2 = {tuple(a) for a in build_flat_actions(gs, player, cap, version=2)}
        assert not set(combat) & v1
        assert set(combat) <= v2

    def test_spread_indices(self):
        assert gym_env._spread_indices(10, 0) == []
        assert gym_env._spread_indices(10, 1) == [9]
        assert gym_env._spread_indices(10, 2) == [0, 9]
        assert gym_env._spread_indices(10, 4) == [0, 3, 6, 9]
        assert gym_env._spread_indices(3, 5) == [0, 1, 2]
        for n in range(1, 30):
            for k in range(1, n + 1):
                idx = gym_env._spread_indices(n, k)
                assert len(set(idx)) == k and idx == sorted(idx) and idx[-1] == n - 1


class TestVersionValidation:
    def test_unknown_version_rejected(self):
        gs, player, _ = _truncating_state()
        with pytest.raises(ValueError, match="flat_action_version"):
            build_flat_actions(gs, player, 8, version=3)
        with pytest.raises(ValueError, match="flat_action_version"):
            StrategyGameEnv(map_file=BEGINNER_MAP, opponent=None, action_space_type="flat_discrete", flat_action_version=0)

    def test_max_flat_actions_must_be_positive(self):
        with pytest.raises(ValueError, match="max_flat_actions"):
            StrategyGameEnv(map_file=BEGINNER_MAP, opponent=None, action_space_type="flat_discrete", max_flat_actions=0)
        gs, player, _ = _truncating_state()
        with pytest.raises(ValueError, match="max_flat_actions"):
            build_flat_actions(gs, player, 0)

    def test_versions_tuple(self):
        assert FLAT_ACTION_VERSIONS == (1, 2)
        assert FLAT_ACTION_VERSION_LEGACY == 1 and FLAT_ACTION_VERSION_LATEST == 2


# ---------------------------------------------------------------------------
# The env: default version, stamp, diagnostics
# ---------------------------------------------------------------------------


class TestEnvVersionAndDiagnostics:
    def test_new_envs_default_to_latest_and_stamp_their_action_space(self):
        env = StrategyGameEnv(map_file=BEGINNER_MAP, opponent=None, action_space_type="flat_discrete", max_flat_actions=32)
        assert env.flat_action_version == FLAT_ACTION_VERSION_LATEST
        assert env.action_space.flat_action_version == FLAT_ACTION_VERSION_LATEST
        assert flat_action_version_of(env) == FLAT_ACTION_VERSION_LATEST
        legacy = StrategyGameEnv(
            map_file=BEGINNER_MAP, opponent=None, action_space_type="flat_discrete", flat_action_version=1
        )
        assert flat_action_version_of(legacy) == 1

    def test_unstamped_space_reads_as_legacy_and_stamp_round_trips(self):
        from gymnasium import spaces

        space = spaces.Discrete(16)
        assert flat_action_version_of(space) == FLAT_ACTION_VERSION_LEGACY
        stamp_flat_action_version(space, 2)
        assert flat_action_version_of(space) == 2
        with pytest.raises(TypeError):
            stamp_flat_action_version(spaces.MultiDiscrete([2, 2]), 2)

    def test_truncation_is_reported_in_info_and_episode_stats(self):
        env = StrategyGameEnv(
            map_file="maps/1v1/skirmish.csv",
            opponent="noop",
            action_space_type="flat_discrete",
            max_flat_actions=6,
            max_steps=40,
        )
        env.reset(seed=0)
        gs = env.game_state
        # Enough units that the legal set far exceeds 6 entries.
        for x, y in ((2, 1), (3, 1), (1, 2)):
            gs.place_unit("W", x, y, env.agent_player)
        (mask,) = env.action_masks()
        assert mask.sum() == 6
        _, _, _, _, info = env.step(0)
        assert info["flat_actions_truncated"] is True
        assert info["n_legal_actions"] == 6
        assert info["n_legal_actions_pre_truncation"] > 6
        info = {}
        term = trunc = False
        while not (term or trunc):
            _, _, term, trunc, info = env.step(0)
        stats = info["episode_stats"]
        assert stats["truncated_steps"] >= 1
        assert stats["max_legal_actions"] > 6  # pre-truncation peak, can exceed the cap
        env.close()

    def test_evaluation_reports_the_truncated_rate(self):
        from reinforcetactics.rl.evaluation import evaluate_model

        class _FirstLegal:
            def predict(self, obs, action_masks=None, **kwargs):
                return int(np.flatnonzero(action_masks)[0]), None

        # Two buildings x 8 unit types of purchases on the first turn: far
        # more than 4 entries, so the opening decision is truncated.
        env = StrategyGameEnv(
            map_file="maps/1v1/skirmish.csv",
            opponent="noop",
            action_space_type="flat_discrete",
            max_flat_actions=4,
            max_steps=12,
        )
        result = evaluate_model(_FirstLegal(), env, n_episodes=1, seed=0)
        assert 0.0 < result["flat_truncated_rate"] <= 1.0
        assert result["max_legal_actions"] > 4
        env.close()

    def test_untruncated_step_reports_no_truncation(self):
        env = StrategyGameEnv(map_file=BEGINNER_MAP, opponent="noop", action_space_type="flat_discrete", max_flat_actions=512)
        env.reset(seed=0)
        env.action_masks()
        _, _, _, _, info = env.step(0)
        assert info["flat_actions_truncated"] is False
        assert info["n_legal_actions"] == info["n_legal_actions_pre_truncation"]
        env.close()


class TestTruncationWarningRateLimit:
    def test_repeated_truncation_logs_once_per_interval(self, caplog, monkeypatch):
        monkeypatch.setattr(gym_env, "_truncation_warning", gym_env._RateLimitedWarning(interval_s=3600.0))
        gs, player, _ = _truncating_state()
        with caplog.at_level(logging.WARNING, logger="reinforcetactics.rl.gym_env"):
            for _ in range(50):
                build_flat_actions(gs, player, 4, version=2)
                build_flat_actions(gs, player, 4, version=1)
        warnings = [r for r in caplog.records if "exceed max_flat_actions" in r.getMessage()]
        assert len(warnings) == 1
        assert gym_env._truncation_warning.suppressed == 99

    def test_warning_resumes_after_the_interval_with_a_suppressed_count(self, caplog, monkeypatch):
        limiter = gym_env._RateLimitedWarning(interval_s=10.0)
        clock = iter([0.0, 1.0, 2.0, 20.0])
        monkeypatch.setattr(gym_env.time, "monotonic", lambda: next(clock))
        with caplog.at_level(logging.WARNING, logger="reinforcetactics.rl.gym_env"):
            assert limiter("truncated %d", 1) is True
            assert limiter("truncated %d", 2) is False
            assert limiter("truncated %d", 3) is False
            assert limiter("truncated %d", 4) is True
        messages = [r.getMessage() for r in caplog.records]
        assert messages[0] == "truncated 1"
        assert messages[1].startswith("truncated 4") and "2 similar warnings suppressed" in messages[1]


# ---------------------------------------------------------------------------
# Checkpoints carry the version; ModelBot decodes with it
# ---------------------------------------------------------------------------


def _save_flat_checkpoint(tmp_path, name, *, version, strip_stamp=False):
    sb3_contrib = pytest.importorskip("sb3_contrib")
    from reinforcetactics.rl.masking import make_maskable_env

    env = make_maskable_env(
        map_file=BEGINNER_MAP,
        opponent="noop",
        action_space_type="flat_discrete",
        max_flat_actions=8,
        enabled_units=["W"],
        seed=0,
    )
    env.unwrapped.flat_action_version = version
    stamp_flat_action_version(env.unwrapped.action_space, version)
    model = sb3_contrib.MaskablePPO(
        "MultiInputPolicy",
        env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={"net_arch": [8]},
        seed=0,
        device="cpu",
    )
    if strip_stamp:
        # What a checkpoint saved before versioning looks like.
        del model.action_space.flat_action_version
    path = tmp_path / name
    model.save(str(path))
    env.close()
    return path


class TestCheckpointVersion:
    def test_version_survives_sb3_save_and_load(self, tmp_path):
        sb3_contrib = pytest.importorskip("sb3_contrib")
        path = _save_flat_checkpoint(tmp_path, "v2.zip", version=2)
        model = sb3_contrib.MaskablePPO.load(str(path), device="cpu")
        assert flat_action_version_of(model) == 2

    @pytest.mark.parametrize(("version", "strip", "expected"), [(2, False, 2), (1, False, 1), (2, True, 1)])
    def test_model_bot_decodes_with_the_checkpoint_version(self, tmp_path, monkeypatch, version, strip, expected):
        from reinforcetactics.core.game_state import GameState
        from reinforcetactics.utils.file_io import FileIO

        path = _save_flat_checkpoint(tmp_path, f"ckpt_{version}_{strip}.zip", version=version, strip_stamp=strip)
        gs = GameState(FileIO.load_map(BEGINNER_MAP), num_players=2, enabled_units=["W"])
        bot = ModelBot(gs, player=2, model_path=str(path))
        assert bot._flat_action_version == expected

        seen_versions = []
        real = gym_env.build_flat_actions

        def spy(*args, **kwargs):
            seen_versions.append(kwargs.get("version"))
            return real(*args, **kwargs)

        monkeypatch.setattr(gym_env, "build_flat_actions", spy)
        gs.end_turn()
        bot.take_turn()
        assert seen_versions and set(seen_versions) == {expected}
