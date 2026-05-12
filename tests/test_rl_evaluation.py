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

    def test_track_breakdown_default_omits_extra_keys(self):
        env = _StubEnv([[(0.0, True, 1)]])
        result = evaluate_model(_StubModel(), env, n_episodes=1)
        assert "action_counts" not in result
        assert "reward_components" not in result

    def test_track_breakdown_accumulates_action_and_reward_breakdown(self):
        # Stub env that emits action_type + reward_breakdown in step info.
        class _BreakdownEnv(_StubEnv):
            def __init__(self, episodes, agent_player=1):
                super().__init__(episodes, agent_player=agent_player)

            def step(self, action):
                ep = self._episodes[self._idx]
                r, term, winner = ep[self._step]
                self._step += 1
                info = {
                    "action_type": 1,  # "move"
                    "reward_breakdown": {
                        "action": 0.5,
                        "shaping_delta": 0.1,
                        "invalid_penalty": 0.0,
                        "terminal": 100.0 if term else 0.0,
                    },
                }
                if term and winner is not None:
                    info["episode_stats"] = {"winner": winner}
                elif term:
                    info["episode_stats"] = {"winner": None}
                return {"obs": np.zeros(1)}, r, term, False, info

        episodes = [
            [(0.5, False, None), (0.5, True, 1)],  # 2 steps, win
            [(0.5, False, None), (0.5, True, 2)],  # 2 steps, loss
        ]
        env = _BreakdownEnv(episodes)

        result = evaluate_model(_StubModel(), env, n_episodes=2, track_breakdown=True)

        assert result["action_counts"]["move"] == 4  # 2 eps * 2 steps
        assert result["action_counts"]["create_unit"] == 0
        assert result["reward_components"]["action"] == 2.0  # 0.5 * 4
        assert result["reward_components"]["terminal"] == 200.0  # 100 per terminal step

    def test_track_breakdown_handles_missing_info_fields(self):
        # Stub env without action_type / reward_breakdown — counters stay 0.
        env = _StubEnv([[(0.0, True, 1)]])
        result = evaluate_model(_StubModel(), env, n_episodes=1, track_breakdown=True)
        assert sum(result["action_counts"].values()) == 0
        assert all(v == 0.0 for v in result["reward_components"].values())

    def test_end_reasons_recorded_when_env_emits_them(self):
        # Env emits info["end_reason"] on terminal steps; evaluate_model
        # should classify each episode under both end_reasons and the
        # outcome_reasons matrix.
        class _ReasonEnv(_StubEnv):
            def __init__(self, episodes_with_reasons, agent_player=1):
                # episodes: list of (reward, terminated, winner, end_reason) tuples.
                super().__init__([[(r, t, w) for r, t, w, _ in ep] for ep in episodes_with_reasons], agent_player=agent_player)
                self._reasons = [[reason for _, _, _, reason in ep] for ep in episodes_with_reasons]

            def step(self, action):
                ep_reasons = self._reasons[self._idx]
                step_idx = self._step
                obs, r, term, trunc, info = super().step(action)
                if term and ep_reasons[step_idx] is not None:
                    info["end_reason"] = ep_reasons[step_idx]
                return obs, r, term, trunc, info

        episodes = [
            [(0.0, True, 1, "hq_capture")],  # win by HQ capture
            [(0.0, True, 1, "elimination")],  # win by elimination
            [(0.0, True, 2, "hq_capture")],  # loss by HQ capture
            [(0.0, True, None, "max_turns_draw")],  # draw
        ]
        env = _ReasonEnv(episodes)
        result = evaluate_model(_StubModel(), env, n_episodes=4)

        assert result["end_reasons"]["hq_capture"] == 2
        assert result["end_reasons"]["elimination"] == 1
        assert result["end_reasons"]["max_turns_draw"] == 1
        assert result["end_reasons"]["max_steps_truncate"] == 0
        assert result["outcome_reasons"]["wins_by_hq_capture"] == 1
        assert result["outcome_reasons"]["wins_by_elimination"] == 1
        assert result["outcome_reasons"]["losses_by_hq_capture"] == 1
        assert result["outcome_reasons"]["draws_by_max_turns_draw"] == 1

    def test_end_reasons_zero_when_env_does_not_emit(self):
        # Existing envs without end_reason should not error; counters stay 0.
        env = _StubEnv([[(0.0, True, 1)]])
        result = evaluate_model(_StubModel(), env, n_episodes=1)
        assert sum(result["end_reasons"].values()) == 0
        assert sum(result["outcome_reasons"].values()) == 0

    def test_captures_by_type_aggregated_across_episodes(self):
        """Per-structure capture counts must be summed into eval results."""

        class _CaptureEnv(_StubEnv):
            def __init__(self, captures_per_episode):
                # Each episode is a single terminal step; captures_per_episode is
                # a list of (towers, buildings, hqs) tuples.
                episodes = [[(0.0, True, 1)] for _ in captures_per_episode]
                super().__init__(episodes)
                self._captures = list(captures_per_episode)

            def step(self, action):
                obs, r, term, trunc, info = super().step(action)
                if term:
                    towers, buildings, hqs = self._captures[self._idx]
                    info["episode_stats"] = {
                        "winner": 1,
                        "captures_by_type": {"tower": towers, "building": buildings, "hq": hqs},
                    }
                return obs, r, term, trunc, info

        env = _CaptureEnv([(2, 1, 0), (0, 3, 1)])
        result = evaluate_model(_StubModel(), env, n_episodes=2)

        assert result["captures_by_type"] == {"tower": 2, "building": 4, "hq": 1}

    def test_captures_by_type_zero_when_env_silent(self):
        """Older envs that don't emit the breakdown contribute zero."""
        env = _StubEnv([[(0.0, True, 1)]])
        result = evaluate_model(_StubModel(), env, n_episodes=1)
        assert result["captures_by_type"] == {"tower": 0, "building": 0, "hq": 0}

    def test_trace_dir_not_created_when_no_trigger_matches(self, tmp_path):
        """Empty eval blocks must not leave empty trace folders behind.

        Regression for eval blocks where every episode finishes cleanly:
        we used to ``mkdir`` the eval-block subdir unconditionally,
        which produced one empty folder per eval cadence on
        Drive-backed run dirs (visible noise + sync overhead).
        """

        class _CleanEnv(_StubEnv):
            def step(self, action):
                obs, r, term, trunc, info = super().step(action)
                if term:
                    info["end_reason"] = "hq_capture"
                return obs, r, term, trunc, info

        env = _CleanEnv([[(0.0, True, 1)], [(0.0, True, 1)]])
        trace_dir = tmp_path / "eval_000050000"
        result = evaluate_model(_StubModel(), env, n_episodes=2, trace_dir=trace_dir)

        assert result["traces"] == []
        assert not trace_dir.exists()

    def test_trace_dir_created_lazily_on_first_trigger(self, tmp_path):
        """The eval-block dir is created the moment a trigger fires."""

        class _ReasonEnv(_StubEnv):
            def __init__(self, reasons):
                super().__init__([[(0.0, True, None)] for _ in reasons])
                self._reasons = list(reasons)

            def step(self, action):
                obs, r, term, trunc, info = super().step(action)
                if term:
                    info["end_reason"] = self._reasons[self._idx]
                return obs, r, term, trunc, info

        env = _ReasonEnv(["hq_capture", "max_steps_truncate", "max_turns_draw"])
        trace_dir = tmp_path / "eval_000100000"
        result = evaluate_model(_StubModel(), env, n_episodes=3, trace_dir=trace_dir)

        assert len(result["traces"]) == 1
        assert trace_dir.exists()
        files = sorted(p.name for p in trace_dir.iterdir())
        assert files == ["episode_0001_max_steps_truncate.jsonl"]
