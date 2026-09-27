"""A stochastic eval samples from its own seeded stream (review of the eval-gate package).

``model.predict(deterministic=False)`` draws from torch's global generator,
the stream MaskablePPO also samples its training actions from. With the
stochastic gate as the curriculum default, that made:

* eval-only settings (``n_eval_episodes``, ``eval_both_modes``, ...) change
  the policy that was trained, and
* a stochastic eval irreproducible from its checkpoint and ``eval_seed``.

A seeded stochastic evaluation now forks the generator, seeds it per episode
(serial) or per call (vectorized), and restores it afterwards.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from reinforcetactics.rl.evaluation import EvalEnvPool, evaluate_model, evaluate_model_vec, policy_sampling_seed

MAP = "maps/1v1/beginner.csv"


def _env():
    from reinforcetactics.rl.masking import make_maskable_env

    return make_maskable_env(
        map_file=MAP,
        opponent="noop",
        action_space_type="flat_discrete",
        max_flat_actions=64,
        max_steps=40,
        max_turns=5,
        enabled_units=["W"],
    )


@pytest.fixture(scope="module")
def model():
    from sb3_contrib import MaskablePPO

    return MaskablePPO("MultiInputPolicy", _env(), policy_kwargs={"net_arch": [8]}, n_steps=16, batch_size=16, seed=0)


def _outcome(m: dict[str, Any]) -> tuple:
    return (m["wins"], m["losses"], m["draws"], tuple(m["rewards"]), tuple(m["lengths"]))


class TestSerial:
    def test_seeded_stochastic_eval_is_reproducible_and_leaves_the_global_stream_alone(self, model):
        env = _env()
        torch.manual_seed(1234)
        before = torch.get_rng_state().clone()
        runs = [evaluate_model(model, env, n_episodes=3, seed=77, deterministic=False) for _ in range(3)]
        # The training stream is exactly where it was...
        assert torch.equal(torch.get_rng_state(), before)
        # ...and the eval is a function of the weights and the seed.
        assert _outcome(runs[0]) == _outcome(runs[1]) == _outcome(runs[2])

    def test_episode_draws_do_not_depend_on_how_many_episodes_run(self, model):
        env = _env()
        two = evaluate_model(model, env, n_episodes=2, seed=5, deterministic=False)
        four = evaluate_model(model, env, n_episodes=4, seed=5, deterministic=False)
        assert four["rewards"][:2] == two["rewards"] and four["lengths"][:2] == two["lengths"]

    def test_greedy_and_unseeded_evals_are_unchanged(self, model):
        env = _env()
        # Greedy draws nothing: identical with or without the fork.
        assert _outcome(evaluate_model(model, env, n_episodes=2, seed=3)) == _outcome(
            evaluate_model(model, env, n_episodes=2, seed=3)
        )
        # Without a seed there is nothing to reproduce: the global stream is used, as before.
        before = torch.get_rng_state().clone()
        evaluate_model(model, env, n_episodes=1, deterministic=False)
        assert not torch.equal(torch.get_rng_state(), before)

    def test_policy_seed_is_distinct_per_episode_and_seat(self):
        seeds = {policy_sampling_seed(s, seat) for s in range(50) for seat in (None, 1, 2)}
        assert len(seeds) == 150
        assert policy_sampling_seed(7) == policy_sampling_seed(7, None)


class TestVectorized:
    def test_seeded_stochastic_vec_eval_is_reproducible_and_leaves_the_global_stream_alone(self, model):
        pool = EvalEnvPool.from_envs([_env(), _env()])
        try:
            before = torch.get_rng_state().clone()
            runs = [evaluate_model_vec(model, pool, n_episodes=3, seed=11, deterministic=False) for _ in range(2)]
            assert torch.equal(torch.get_rng_state(), before)
            assert _outcome(runs[0]) == _outcome(runs[1])
        finally:
            pool.close()


# ---------------------------------------------------------------------------
# End to end: eval settings no longer change what is trained, and a row can be
# re-derived from best_model.zip and its eval_seed.
# ---------------------------------------------------------------------------


def _one_stage_config(tmp_path: Path, name: str, **eval_overrides: Any) -> Path:
    data: dict[str, Any] = {
        "seed": 3,
        "env": {
            "n_envs": 1,
            "use_subprocess": False,
            "action_space_type": "flat_discrete",
            "max_flat_actions": 64,
            "max_steps": 40,
            "max_turns": 5,
            "enabled_units": ["W"],
        },
        "ppo": {"n_steps": 64, "batch_size": 32, "n_epochs": 2, "policy_kwargs": {"net_arch": [16]}, "device": "cpu"},
        # The curriculum defaults: the stochastic gate (eval_deterministic false).
        "eval": {"eval_freq": 64, "n_eval_episodes": 1, "checkpoint_freq": 64, "eval_both_modes": False, **eval_overrides},
        "curriculum": {
            "max_retries": 0,
            "stages": [
                {
                    "name": "s1",
                    "map_file": MAP,
                    "opponent": "noop",
                    "promotion_win_rate": 0.0,
                    "patience": 1,
                    "min_timesteps_before_promotion": 256,
                    "max_timesteps": 320,
                }
            ],
        },
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _train(tmp_path: Path, name: str, **eval_overrides: Any):
    from reinforcetactics.rl.bootstrap import run_curriculum
    from reinforcetactics.rl.config import load_config

    cfg = load_config(_one_stage_config(tmp_path, name, **eval_overrides))
    out = tmp_path / name
    result = run_curriculum(cfg, out)
    with (out / "train_metrics.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return cfg, out, result, rows


def test_eval_settings_do_not_change_the_training_run(tmp_path):
    _, _, _, base = _train(tmp_path, "base")
    _, _, _, more_episodes = _train(tmp_path, "more_episodes", n_eval_episodes=3)
    _, _, _, both_modes = _train(tmp_path, "both_modes", eval_both_modes=True)
    assert len(base) >= 2 and any(r["train/loss"] for r in base)
    # Every optimizer diagnostic, update by update: the same run.
    assert more_episodes == base
    assert both_modes == base


def test_a_stochastic_row_is_reproduced_from_best_model_and_its_eval_seed(tmp_path):
    from sb3_contrib import MaskablePPO

    from reinforcetactics.rl.bootstrap import make_stage_env

    cfg, out, result, _ = _train(tmp_path, "repro", n_eval_episodes=3)
    rows = result["history"][0]["results"]
    saved = [r for r in rows if r["saved_best"]]
    assert saved and all(r["deterministic"] is False for r in rows)
    row = saved[-1]
    stage = cfg.curriculum.stages[0]
    env = make_stage_env(stage, cfg.env, seed=cfg.seed + cfg.eval.seed_offset, gamma=cfg.ppo.gamma)
    best = MaskablePPO.load(str(out / "s1" / "best_model.zip"))
    again = evaluate_model(
        best, env, n_episodes=cfg.eval.n_eval_episodes, seed=row["eval_seed"], deterministic=False, seats=row["seats"][:1]
    )
    assert _outcome(again) == _outcome(row)
