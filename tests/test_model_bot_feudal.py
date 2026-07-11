"""Tests for the feudal ``.pt`` checkpoint branch of ``ModelBot``.

The SB3 ``.zip`` branch is exercised via integration tests and the existing
tournament smoke tests. This file focuses on the new feudal loader path:
extension dispatch, hyperparam-driven construction, grid-mismatch guard,
mask wiring (per-dim and AR), and inter-op with tournament discovery.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.model_bot import ModelBot
from reinforcetactics.rl.feudal_rl import FeudalRLAgent
from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.tournament.bots import discover_model_bots
from reinforcetactics.utils.file_io import FileIO

MAP_FILE = "maps/1v1/beginner.csv"


def _train_and_save(tmpdir: Path, *, autoregressive: bool = False, manager_horizon: int = 5) -> Path:
    """Train a tiny feudal agent against the beginner map, save a checkpoint."""
    torch.manual_seed(0)
    np.random.seed(0)

    env = StrategyGameEnv(
        map_file=MAP_FILE,
        opponent="random",
        render_mode=None,
        max_steps=50,
        max_turns=10,
        enabled_units=["W", "M", "A"],
    )
    env.reset(seed=0)
    agent = FeudalRLAgent(
        observation_space=env.observation_space,
        grid_width=env.grid_width,
        grid_height=env.grid_height,
        agent_player=1,
        device="cpu",
        autoregressive_worker=autoregressive,
    )
    agent.manager_horizon = manager_horizon
    agent.setup_training(learning_rate=3e-4)
    agent.reset_goal()
    buf = agent.collect_rollout(env, n_steps=32, gamma=0.99, gae_lambda=0.95, reward_scale=0.001)
    agent.update(buf, n_epochs=1, batch_size=8, clip_range=0.2, ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5)

    ckpt = tmpdir / ("feudal_ar.pt" if autoregressive else "feudal.pt")
    agent.save_checkpoint(str(ckpt))
    return ckpt


def _make_game_state() -> GameState:
    return GameState(FileIO.load_map(MAP_FILE), num_players=2)


class TestFeudalLoaderDispatch:
    def test_pt_suffix_loads_feudal(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = _train_and_save(Path(td))
            bot = ModelBot(_make_game_state(), player=2, model_path=str(ckpt))
            assert bot.is_feudal
            assert bot.feudal_agent is not None
            # SB3 model is *not* loaded for .pt
            assert bot.model is None

    def test_zip_suffix_does_not_set_feudal(self):
        # We don't have a .zip handy in tests, but we can confirm the loader
        # branch on a missing .zip does not silently produce a feudal bot.
        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "missing.zip"
            with pytest.raises(FileNotFoundError):
                ModelBot(_make_game_state(), player=2, model_path=str(fake))

    def test_grid_mismatch_raises(self, tmp_path):
        ckpt = _train_and_save(tmp_path)
        # Live game on a *different* map ⇒ different grid dims ⇒ refuse load.
        # We fake this by shrinking the grid via a hand-built GameState. The
        # easiest path is to load the same map but assert the saved hyperparams
        # carry the right dims; then simulate a mismatch by patching grid.width.
        gs = _make_game_state()
        gs.grid.width += 2  # pretend the live game has a wider grid
        with pytest.raises(ValueError, match="grid"):
            ModelBot(gs, player=2, model_path=str(ckpt))


class TestFeudalTakeTurn:
    def test_take_turn_runs_without_error(self, tmp_path):
        ckpt = _train_and_save(tmp_path)
        gs = _make_game_state()
        bot = ModelBot(gs, player=2, model_path=str(ckpt))
        # The bot needs to be the current player to take its turn. The map
        # starts with player 1 to move; advance until it's player 2's turn.
        gs.end_turn()
        assert gs.current_player == 2
        bot.take_turn()
        # take_turn always ends with control returning to player 1.
        assert gs.current_player in (1,) or gs.game_over

    def test_take_turn_with_autoregressive_worker(self, tmp_path):
        ckpt = _train_and_save(tmp_path, autoregressive=True)
        gs = _make_game_state()
        bot = ModelBot(gs, player=2, model_path=str(ckpt))
        assert bot.is_feudal
        assert bot.feudal_agent.autoregressive_worker is True
        gs.end_turn()
        bot.take_turn()
        assert gs.current_player in (1,) or gs.game_over


class TestSelfPlayOpponentFactory:
    """The new ``set_self_play_opponent_factory`` hook lets the training
    script swap in a snapshot-backed ModelBot as the env's opponent. We
    verify the factory is invoked on every reset() with the new game_state."""

    def test_factory_called_on_reset(self):
        env = StrategyGameEnv(
            map_file=MAP_FILE,
            opponent="self",
            render_mode=None,
            max_steps=50,
            max_turns=10,
            enabled_units=["W", "M", "A"],
        )
        calls = []

        from reinforcetactics.game.bot import RandomBot

        def factory(game_state, opponent_player):
            # Keep a real reference to game_state (not just its id) so CPython
            # cannot recycle a freed object's address between resets.
            calls.append((game_state, opponent_player))
            return RandomBot(game_state, player=opponent_player)

        env.set_self_play_opponent_factory(factory)
        env.reset(seed=0)
        env.reset(seed=1)
        env.reset(seed=2)
        # One factory call per reset, all with the freshly built game_state.
        assert len(calls) == 3
        # opponent_player is the *other* player from the agent (default 1 → 2).
        assert all(p == 2 for _, p in calls)
        # The three game_state instances should all be distinct objects (each
        # reset rebuilds GameState from initial_map_data).
        assert len({id(gs) for gs, _ in calls}) == 3

    def test_self_opponent_type_no_factory_is_safe(self):
        """If no factory is set, self-play env reset should leave opponent=None
        and ``_opponent_turn`` no-ops — never crash."""
        env = StrategyGameEnv(
            map_file=MAP_FILE,
            opponent="self",
            render_mode=None,
            max_steps=50,
            max_turns=10,
            enabled_units=["W", "M", "A"],
        )
        env.reset(seed=0)
        assert env.opponent is None
        # Step a few times to force opponent turns; should not crash.
        for _ in range(3):
            action = env.action_space.sample()
            action[0] = 5  # end_turn
            env.step(action)


class TestTournamentDiscoversFeudal:
    def test_discover_model_bots_picks_up_pt(self, tmp_path):
        _train_and_save(tmp_path)
        bots = discover_model_bots(str(tmp_path), test_models=True)
        # Exactly one .pt checkpoint in the dir → exactly one descriptor.
        assert len(bots) == 1
        assert bots[0].model_path.endswith(".pt")

    def test_discover_model_bots_filters_unloadable(self, tmp_path):
        # A bogus .pt file should be rejected by _test_model_file rather
        # than producing a broken descriptor.
        (tmp_path / "broken.pt").write_bytes(b"not a real checkpoint")
        bots = discover_model_bots(str(tmp_path), test_models=True)
        assert len(bots) == 0
