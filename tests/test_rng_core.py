"""Every game owns a seeded combat RNG (review core-10 / persist-6).

The Rogue evade roll -- the only stochastic outcome in combat -- used to read
the module-global ``random`` whenever a ``GameState`` was built without an
``rng``, which the tournament runner, the AlphaZero trainer and the imitation
driver all did. Seeded tournaments, evaluations and BC datasets were
therefore not reproducible, and concurrent games shared one stream.

These tests pin the fixed behaviour: a game owns ``random.Random(seed)``
(the seed drawn from entropy when none is given, and recorded), the same
seed replays the same rolls whatever the global ``random`` does, the seed
and the stream position survive a save/load, the replay records the seed,
and a seeded tournament with Rogues is reproducible.
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState, derive_seed
from reinforcetactics.core.mechanics import GameMechanics
from reinforcetactics.core.unit import Unit
from reinforcetactics.tournament import TournamentConfig, TournamentRunner
from reinforcetactics.tournament.bots import BotDescriptor, BotType
from reinforcetactics.tournament.schedule import MapConfig


def _plains(size: int = 6) -> np.ndarray:
    md = np.array([["p"] * size for _ in range(size)], dtype=object)
    md[0][0] = "h_1"
    md[size - 1][size - 1] = "h_2"
    return md


def _evade_sequence(gs: GameState, rolls: int = 60) -> list[bool]:
    """Let a Rogue attack a Warrior ``rolls`` times; return whether each evaded.

    HP and action flags are reset between attacks (test setup), so every
    attack draws exactly one evade roll from the game's RNG.
    """
    rogue = gs.place_unit("R", 2, 2, 1)
    target = gs.place_unit("W", 3, 2, 2)
    out = []
    for _ in range(rolls):
        rogue.health, target.health = rogue.max_health, target.max_health
        rogue.can_move = rogue.can_attack = True
        gs._invalidate_cache()
        result = gs.attack(rogue, target)
        assert result["damage"] > 0
        out.append(result["evade"])
    return out


def _poison_global_random(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("the engine read the module-global random")

    monkeypatch.setattr(random, "random", _boom)


class TestGameOwnsItsRng:
    def test_default_game_has_a_recorded_seed_and_its_own_generator(self):
        gs = GameState(_plains())
        assert isinstance(gs.rng, random.Random)
        assert isinstance(gs.seed, int)
        # The recorded seed rebuilds the same stream.
        assert random.Random(gs.seed).random() == gs.rng.random()

    def test_explicit_seed_is_used_and_recorded(self):
        gs = GameState(_plains(), seed=np.int64(5))  # numpy ints are accepted
        assert gs.seed == 5
        assert gs.rng.random() == random.Random(5).random()

    def test_caller_rng_is_used_as_is(self):
        rng = random.Random(3)
        gs = GameState(_plains(), rng=rng)
        assert gs.rng is rng
        assert gs.seed is None  # the engine cannot know how it was seeded

    def test_unseeded_games_get_distinct_seeds(self):
        assert len({GameState(_plains()).seed for _ in range(5)}) == 5

    def test_same_seed_gives_identical_evade_outcomes(self, monkeypatch):
        random.seed(1)
        first = _evade_sequence(GameState(_plains(), seed=42))
        random.seed(2)  # a different global state must not matter
        second = _evade_sequence(GameState(_plains(), seed=42))
        assert first == second
        assert any(first) and not all(first)  # both outcomes were rolled
        assert _evade_sequence(GameState(_plains(), seed=43)) != first

    def test_engine_never_reads_the_global_random(self, monkeypatch):
        _poison_global_random(monkeypatch)
        _evade_sequence(GameState(_plains(), seed=1), rolls=20)

    def test_mechanics_without_rng_do_not_fall_back_to_global_random(self, monkeypatch):
        _poison_global_random(monkeypatch)
        grid_state = GameState(_plains())
        rogue, target = Unit("R", 2, 2, 1), Unit("W", 3, 2, 2)
        result = GameMechanics.attack_unit(rogue, target, grid_state.grid, [rogue, target])
        assert result["evade"] in (True, False)

    def test_reset_keeps_the_seed(self):
        gs = GameState(_plains(), seed=11)
        gs.reset(_plains())
        assert gs.seed == 11


class TestSeedPersistence:
    def test_save_records_seed_and_reload_continues_the_stream(self):
        gs = GameState(_plains(), seed=9)
        for _ in range(7):
            gs.rng.random()
        saved = json.loads(json.dumps(gs.to_dict()))
        assert saved["seed"] == 9
        restored = GameState.from_dict(saved)
        assert restored.seed == 9
        assert [restored.rng.random() for _ in range(5)] == [gs.rng.random() for _ in range(5)]

    def test_caller_rng_state_is_saved_without_inventing_a_seed(self):
        gs = GameState(_plains(), rng=random.Random(3))
        gs.rng.random()
        restored = GameState.from_dict(json.loads(json.dumps(gs.to_dict())))
        assert restored.seed is None
        assert restored.rng.random() == gs.rng.random()

    def test_old_save_without_seed_still_loads(self):
        saved = json.loads(json.dumps(GameState(_plains(), seed=9).to_dict()))
        del saved["seed"], saved["rng_state"]
        restored = GameState.from_dict(saved)
        assert isinstance(restored.seed, int)
        assert isinstance(restored.rng, random.Random)

    def test_replay_records_the_seed(self, tmp_path):
        gs = GameState(_plains(), seed=77)
        path = gs.save_replay_to_file(str(tmp_path / "replay.json"))
        with open(path) as f:
            assert json.load(f)["game_info"]["seed"] == 77


def test_derive_seed_is_stable_and_distinct():
    assert derive_seed(7, 3, "map") == derive_seed(7, 3, "map")
    assert derive_seed(7, 3, "map") != derive_seed(7, 4, "map")
    assert 0 <= derive_seed("x") < 2**63


class TestSeededTournamentIsReproducible:
    """Two runs with the same ``rng_seed`` must play the same games (persist-6).

    The global ``random`` is forced to "always evade" in one run and "never
    evade" in the other: before the fix the engine rolled with it, so the
    Rogue-heavy games diverged.
    """

    @staticmethod
    def _run(tmp_path: Path, global_roll: float, monkeypatch) -> list[tuple[str, str, int]]:
        monkeypatch.setattr(random, "random", lambda: global_roll)
        config = TournamentConfig(
            name="rng_core",
            maps=[MapConfig(path="maps/1v1/beginner.csv", max_turns=30)],
            games_per_side=1,
            save_replays=True,
            replay_dir=str(tmp_path / "replays"),
            output_dir=str(tmp_path),
            max_turns=30,
            rng_seed=7,
            enabled_units=["W", "R"],
        )
        bots = [
            BotDescriptor(name="Advanced", bot_type=BotType.ADVANCED),
            BotDescriptor(name="Medium", bot_type=BotType.MEDIUM),
        ]
        TournamentRunner(config).run(bots)
        games = []
        for path in sorted((tmp_path / "replays").rglob("*.json")):
            with open(path) as f:
                replay = json.load(f)
            actions = [{k: v for k, v in a.items() if k != "timestamp"} for a in replay["actions"]]
            digest = hashlib.sha256(json.dumps(actions, sort_keys=True, default=str).encode()).hexdigest()
            games.append((path.name.split("_id")[1][:4], digest, replay["game_info"].get("seed")))
        return games

    def test_same_rng_seed_same_games_with_rogues(self, tmp_path, monkeypatch):
        first = self._run(tmp_path / "a", 0.0, monkeypatch)
        second = self._run(tmp_path / "b", 0.99, monkeypatch)
        assert len(first) == 2
        assert [game[:2] for game in first] == [game[:2] for game in second]
        # Each game records its own engine seed, the same in both runs.
        assert first == second
        assert len({seed for _, _, seed in first}) == 2 and None not in {seed for _, _, seed in first}


@pytest.mark.parametrize("seed", [None, 5])
def test_alphazero_game_seed_derivation(seed):
    from reinforcetactics.rl.alphazero_trainer import AlphaZeroTrainer

    trainer = AlphaZeroTrainer.__new__(AlphaZeroTrainer)
    trainer.seed = seed
    trainer.history = {"iteration": [1, 2]}
    got = trainer._game_seed("self_play", 3)
    assert got == (None if seed is None else derive_seed(5, "self_play", 2, 3))


def test_a_config_driven_alphazero_run_is_seeded_by_the_config(tmp_path, monkeypatch):
    """``--config`` fed every AlphaZero option but the seed, so config runs were never reproducible."""
    import importlib.util
    import sys
    from pathlib import Path

    config = tmp_path / "az.yaml"
    config.write_text("algorithm: alphazero\nseed: 123\nenv:\n  enabled_units: [W, A]\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "train" / "train_alphazero.py"
    spec = importlib.util.spec_from_file_location("train_alphazero_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(sys, "argv", ["train_alphazero.py", "--config", str(config)])
    args = module.parse_args()
    assert (args.seed, args.enabled_units) == (123, ["W", "A"])

    monkeypatch.setattr(sys, "argv", ["train_alphazero.py", "--config", str(config), "--seed", "7"])
    assert module.parse_args().seed == 7  # the command line still wins
