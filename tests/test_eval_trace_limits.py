"""Bound failure traces without choosing different seeds in vectorized evaluation."""

import json
from pathlib import Path

import numpy as np
import pytest

from reinforcetactics.rl.evaluation import EvalEnvPool, _EpisodeTracker, evaluate_model, evaluate_model_vec


class _TraceEnv:
    agent_player = 1

    def __init__(self, completions=None):
        self.completions = completions if completions is not None else []

    def reset(self, seed=None):
        self.seed = seed
        self.steps = 0
        # The first matching episode finishes after several later matches.
        self.length = 5 if seed == 100 else 1
        return {"obs": np.zeros(1)}, {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps >= self.length
        info = {"turn": self.steps, "action_type": 5, "valid_action": True}
        if terminated:
            self.completions.append(self.seed)
            info.update(
                {
                    "end_reason": "hq_capture" if self.seed == 101 else "max_turns_draw",
                    "episode_stats": {"winner": None},
                }
            )
        return {"obs": np.zeros(1)}, 0.0, terminated, False, info

    def close(self):
        pass


class _Policy:
    def predict(self, obs, **kwargs):
        values = obs["obs"]
        return (np.zeros(values.shape[0], dtype=int) if values.ndim > 1 else 0), None


def _run(vectorized, **kwargs):
    completions = []
    if not vectorized:
        return evaluate_model(_Policy(), _TraceEnv(completions), **kwargs), completions
    pool = EvalEnvPool.from_envs([_TraceEnv(completions) for _ in range(3)])
    try:
        return evaluate_model_vec(_Policy(), pool, **kwargs), completions
    finally:
        pool.close()


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("max_traces,expected", [(None, [0, 2, 3, 4]), (2, [0, 2])])
def test_earliest_matching_plan_episodes_are_traced(tmp_path, vectorized, max_traces, expected):
    result, completions = _run(
        vectorized,
        n_episodes=5,
        seed=100,
        trace_dir=tmp_path,
        trace_end_reasons=("max_turns_draw",),
        max_traces=max_traces,
        include_episode_records=True,
    )
    indices = []
    for path in result["traces"]:
        with Path(path).open() as fh:
            header = json.loads(next(fh))["_header"]
            assert len(list(fh)) == header["ep_length"]
        indices.append(header["episode_index"])
    assert indices == expected
    assert len(list(tmp_path.glob("*.jsonl"))) == len(expected)
    assert [r["index"] for r in result["episode_records"] if r["trace_path"]] == expected
    if vectorized:
        assert completions.index(102) < completions.index(100)
        assert completions.index(103) < completions.index(100)


@pytest.mark.parametrize("vectorized", [False, True])
def test_zero_disables_trace_collection_and_writes(tmp_path, monkeypatch, vectorized):
    original = _EpisodeTracker.on_step

    def check_no_trace_buffer(self, *args, **kwargs):
        assert self.trace is None
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_EpisodeTracker, "on_step", check_no_trace_buffer)
    trace_dir = tmp_path / "disabled"
    result, _ = _run(vectorized, n_episodes=5, seed=100, trace_dir=trace_dir, max_traces=0)
    assert result["traces"] == []
    assert not trace_dir.exists()


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("max_traces", [-1, 1.5])
def test_invalid_trace_cap_fails_before_any_episode(vectorized, max_traces):
    with pytest.raises(ValueError, match="max_traces"):
        _run(vectorized, n_episodes=1, seed=100, max_traces=max_traces)
