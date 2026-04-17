"""Tests for reinforcetactics.rl.evaluation."""

import numpy as np

from reinforcetactics.rl.evaluation import evaluate_model


class _StubEnv:
    """Minimal env stub. Returns a scripted sequence of (reward, done, winner)."""

    def __init__(self, episodes, has_action_masks=False, agent_player=1):
        # episodes: list of lists of (reward, terminated, winner_or_None)
        self._episodes = list(episodes)
        self._idx = -1
        self._step = 0
        self.agent_player = agent_player
        if has_action_masks:
            self.action_masks = self._action_masks  # type: ignore[attr-defined]

    def reset(self):
        self._idx += 1
        self._step = 0
        return {"obs": np.zeros(1)}, {}

    def step(self, action):
        ep = self._episodes[self._idx]
        r, term, winner = ep[self._step]
        self._step += 1
        info = {}
        if term and winner is not None:
            info["episode_stats"] = {"winner": winner}
        elif term:
            info["episode_stats"] = {"winner": None}
        return {"obs": np.zeros(1)}, r, term, False, info

    def _action_masks(self):
        # Return a tuple of 2 dim masks (exercises the tuple->flat conversion)
        return (np.array([True, False]), np.array([True, True]))


class _StubModel:
    """Model stub that records prediction kwargs."""

    def __init__(self):
        self.calls = []

    def predict(self, obs, **kwargs):
        self.calls.append(kwargs)
        return np.array([0, 0, 0, 0, 0, 0]), None


class TestEvaluateModel:
    def test_counts_wins_losses_draws(self):
        episodes = [
            [(1.0, True, 1)],  # win
            [(0.0, True, 2)],  # loss
            [(0.0, True, None)],  # draw
            [(0.5, False, None), (0.5, True, 1)],  # win, 2 steps
        ]
        env = _StubEnv(episodes)
        model = _StubModel()

        result = evaluate_model(model, env, n_episodes=4)

        assert result["wins"] == 2
        assert result["losses"] == 1
        assert result["draws"] == 1
        assert result["episodes"] == 4
        assert result["win_rate"] == 0.5
        assert result["avg_length"] == 1.25
        # rewards: [1.0, 0.0, 0.0, 1.0]
        assert result["avg_reward"] == 0.5

    def test_action_masks_passed_to_predict(self):
        episodes = [[(0.0, True, 1)]]
        env = _StubEnv(episodes, has_action_masks=True)
        model = _StubModel()

        evaluate_model(model, env, n_episodes=1)

        assert "action_masks" in model.calls[0]
        masks = model.calls[0]["action_masks"]
        assert masks.dtype == np.bool_
        # Concatenated (2 + 2 = 4 booleans)
        assert masks.shape == (4,)
        assert masks.tolist() == [True, False, True, True]

    def test_no_action_masks_when_env_lacks_method(self):
        episodes = [[(0.0, True, 1)]]
        env = _StubEnv(episodes, has_action_masks=False)
        model = _StubModel()

        evaluate_model(model, env, n_episodes=1)

        assert "action_masks" not in model.calls[0]

    def test_deterministic_flag_forwarded(self):
        env = _StubEnv([[(0.0, True, 1)]])
        model = _StubModel()
        evaluate_model(model, env, n_episodes=1, deterministic=False)
        assert model.calls[0]["deterministic"] is False

        env = _StubEnv([[(0.0, True, 1)]])
        model = _StubModel()
        evaluate_model(model, env, n_episodes=1, deterministic=True)
        assert model.calls[0]["deterministic"] is True

    def test_agent_player_2(self):
        # Agent plays as player 2; winner=2 should count as win.
        episodes = [[(0.0, True, 2)], [(0.0, True, 1)]]
        env = _StubEnv(episodes, agent_player=2)
        model = _StubModel()

        result = evaluate_model(model, env, n_episodes=2)
        assert result["wins"] == 1
        assert result["losses"] == 1
