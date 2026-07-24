"""Tests for reinforcetactics.rl.bootstrap and the PromotionCallback."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from reinforcetactics.rl.bootstrap import (
    CurriculumStalled,
    _resolve_dotted,
    _resolve_policy_kwargs,
    _write_results_csv,
    run_curriculum,
)
from reinforcetactics.rl.callbacks import (
    EntropyScheduleCallback,
    PeriodicEvalCallback,
    PromotionCallback,
    TrainingMetricsCallback,
)
from reinforcetactics.rl.config import (
    CurriculumConfig,
    CurriculumStage,
    EnvConfig,
    EvalConfig,
    PPOConfig,
    TrainingConfig,
    config_from_dict,
    load_config,
)

# ---------------------------------------------------------------------------
# PromotionCallback unit tests (no env / no real model needed)
# ---------------------------------------------------------------------------


class _StubLogger:
    """Minimal stand-in for sb3 Logger; only tracks .record calls."""

    def __init__(self) -> None:
        self.records: dict[str, Any] = {}
        # TrainingMetricsCallback reads ``self.model.logger.name_to_value``.
        self.name_to_value: dict[str, Any] = {}

    def record(self, key: str, value: Any) -> None:
        self.records[key] = value


def _make_eval_stub() -> PeriodicEvalCallback:
    """Build a PeriodicEvalCallback shell whose ``results`` we can append to.

    We bypass __init__ so we don't need a real env / save_dir; the
    PromotionCallback only ever reads ``eval_callback.results``.
    """
    stub = PeriodicEvalCallback.__new__(PeriodicEvalCallback)
    stub.results = []  # type: ignore[attr-defined]
    return stub


def _step_promote(cb: PromotionCallback, num_timesteps: int = 0) -> bool:
    """Drive the callback's ``_on_step`` without standing up a full SB3 model."""
    cb.num_timesteps = num_timesteps  # type: ignore[attr-defined]
    return cb._on_step()


def _append_eval(eval_cb: PeriodicEvalCallback, win_rate: float) -> None:
    eval_cb.results.append({"win_rate": win_rate, "avg_reward": 0.0})


class _BestStubModel:
    """Minimal model stub for exercising PeriodicEvalCallback's best-save."""

    def __init__(self) -> None:
        self.num_timesteps = 0
        self.logger = _StubLogger()
        self.saved: list[str] = []

    def save(self, path: str) -> None:
        self.saved.append(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake", encoding="utf-8")


class TestBestCheckpointTimestep:
    """``PeriodicEvalCallback.best_timestep`` records the cumulative timestep
    of the highest-WR eval -- the basis for the curriculum runner's
    'how far into the stage was the peak' (skip-ahead) diagnostic."""

    def test_best_timestep_tracks_peak_not_last(self, tmp_path, monkeypatch):
        # Rising-then-falling WR so the peak is the MIDDLE eval (@200), not
        # the last -- a drifted-after-peak stage, the case the field exists for.
        wr_program = iter([0.30, 0.90, 0.50])

        def fake_eval(model, env, **kwargs):
            wr = next(wr_program)
            return {
                "win_rate": wr,
                "avg_reward": wr,
                "avg_length": 1.0,
                "avg_turns": 1.0,
                "std_reward": 0.0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "seize_available_rate": 0.0,
                "max_legal_actions": 0,
            }

        monkeypatch.setattr("reinforcetactics.rl.callbacks.evaluate_model", fake_eval)

        cb = PeriodicEvalCallback(
            eval_env=object(),
            eval_freq=100,
            n_eval_episodes=1,
            save_dir=tmp_path,
            verbose=0,
        )
        cb.model = _BestStubModel()

        for ts in (100, 200, 300):
            # SB3's BaseCallback.on_step syncs ``self.num_timesteps`` from the
            # model before invoking the user callback; mirror that here since
            # we call ``_do_eval`` directly.
            cb.num_timesteps = ts
            cb.model.num_timesteps = ts
            cb._last_eval_block = ts // 100
            cb._do_eval()

        assert cb.best_win_rate == pytest.approx(0.90)
        assert cb.best_timestep == 200  # the peak eval's cumulative timestep
        assert (tmp_path / "best_model.zip").exists()


def _wr_program_eval(wr_program, seen_seeds=None):
    """Build a fake ``evaluate_model`` that returns a scripted WR sequence and
    optionally records the seed each call received."""

    def fake_eval(model, env, **kwargs):
        if seen_seeds is not None:
            seen_seeds.append(kwargs.get("seed"))
        wr = next(wr_program)
        return {
            "win_rate": wr,
            "avg_reward": wr,
            "avg_length": 1.0,
            "avg_turns": 1.0,
            "std_reward": 0.0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "seize_available_rate": 0.0,
            "max_legal_actions": 0,
        }

    return fake_eval


def _drive_evals(cb, timesteps):
    """Mirror SB3's num_timesteps sync and fire one eval per entry."""
    for ts in timesteps:
        cb.num_timesteps = ts
        cb.model.num_timesteps = ts
        cb._last_eval_block = ts // cb.eval_freq
        cb._do_eval()


class TestEvalSeedStability:
    """Consecutive evals must measure the *same* problem set.

    Resampling the 80-episode benchmark every eval block makes
    ``PromotionCallback``'s "patience consecutive crossings" compare two
    different benchmarks, and turns ``best_model.zip`` into an argmax over
    many noisy estimates on different problems (winner's curse).
    """

    def test_seed_is_fixed_across_evals_by_default(self, tmp_path, monkeypatch):
        seeds: list = []
        monkeypatch.setattr(
            "reinforcetactics.rl.callbacks.evaluate_model",
            _wr_program_eval(iter([0.1, 0.2, 0.3]), seeds),
        )
        cb = PeriodicEvalCallback(
            eval_env=object(), eval_freq=100, n_eval_episodes=1, eval_seed_base=7, save_dir=tmp_path, verbose=0
        )
        cb.model = _BestStubModel()
        _drive_evals(cb, (100, 200, 300))
        assert seeds == [7, 7, 7]

    def test_resample_eval_seeds_restores_rotation(self, tmp_path, monkeypatch):
        seeds: list = []
        monkeypatch.setattr(
            "reinforcetactics.rl.callbacks.evaluate_model",
            _wr_program_eval(iter([0.1, 0.2, 0.3]), seeds),
        )
        cb = PeriodicEvalCallback(
            eval_env=object(),
            eval_freq=100,
            n_eval_episodes=1,
            eval_seed_base=7,
            save_dir=tmp_path,
            resample_eval_seeds=True,
            verbose=0,
        )
        cb.model = _BestStubModel()
        _drive_evals(cb, (100, 200, 300))
        assert seeds == [7 + 1000 * 1, 7 + 1000 * 2, 7 + 1000 * 3]


class TestBestEligibleAfter:
    """The stage-entry eval measures the *carry-in* policy.

    ``PeriodicEvalCallback`` is built fresh per stage but gates on the
    cumulative counter (the runner passes ``reset_num_timesteps=False``), so
    the first ``_on_step`` of every stage always fires an eval. Letting that
    eval claim ``best_model.zip`` means ``restore_best_checkpoint_between_
    stages`` can rewind the stage's own training.
    """

    def test_carry_in_eval_cannot_claim_best(self, tmp_path, monkeypatch):
        # Carry-in policy is the strongest thing this stage ever sees.
        monkeypatch.setattr(
            "reinforcetactics.rl.callbacks.evaluate_model",
            _wr_program_eval(iter([1.0, 0.40, 0.60])),
        )
        cb = PeriodicEvalCallback(
            eval_env=object(),
            eval_freq=100,
            n_eval_episodes=1,
            save_dir=tmp_path,
            best_eligible_after=100,
            verbose=0,
        )
        cb.model = _BestStubModel()
        cb.num_timesteps = 5_000
        cb._on_training_start()  # stage entry at a large cumulative counter
        _drive_evals(cb, (5_000, 5_100, 5_200))

        # The 1.0 carry-in eval is recorded but not eligible; best is the
        # strongest eval the stage actually produced.
        assert [r["win_rate"] for r in cb.results] == [1.0, 0.40, 0.60]
        assert [r["best_eligible"] for r in cb.results] == [False, True, True]
        assert [r["stage_steps"] for r in cb.results] == [0, 100, 200]
        assert cb.best_win_rate == pytest.approx(0.60)
        assert cb.best_timestep == 5_200

    def test_default_zero_keeps_legacy_eligibility(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "reinforcetactics.rl.callbacks.evaluate_model",
            _wr_program_eval(iter([1.0, 0.40])),
        )
        cb = PeriodicEvalCallback(eval_env=object(), eval_freq=100, n_eval_episodes=1, save_dir=tmp_path, verbose=0)
        cb.model = _BestStubModel()
        cb.num_timesteps = 5_000
        cb._on_training_start()
        _drive_evals(cb, (5_000, 5_100))
        assert cb.best_win_rate == pytest.approx(1.0)
        assert cb.best_timestep == 5_000

    def test_pre_window_evals_still_reach_the_promotion_callback(self, tmp_path, monkeypatch):
        """Eligibility gates the *checkpoint*, not the promotion streak --
        that is ``min_timesteps_before_promotion``'s job. Keep them separate.
        """
        monkeypatch.setattr(
            "reinforcetactics.rl.callbacks.evaluate_model",
            _wr_program_eval(iter([0.9, 0.9])),
        )
        cb = PeriodicEvalCallback(
            eval_env=object(),
            eval_freq=100,
            n_eval_episodes=1,
            save_dir=tmp_path,
            best_eligible_after=1_000_000,
            verbose=0,
        )
        cb.model = _BestStubModel()
        cb.num_timesteps = 5_000
        cb._on_training_start()
        _drive_evals(cb, (5_000, 5_100))

        promote = PromotionCallback(eval_callback=cb, threshold=0.8, patience=2, verbose=0)
        assert _step_promote(promote, num_timesteps=5_100) is False
        assert promote.promoted is True
        # ...while nothing was ever saved as best.
        assert cb.best_win_rate == -1.0
        assert not (tmp_path / "best_model.zip").exists()


class TestPeriodicEvalJsonlPersistence:
    """``results_jsonl_path`` gets one JSON line per eval, appended the
    moment the eval finishes — so a run killed mid-stage (the routine
    Colab death) keeps the in-progress stage's eval timeline on disk
    instead of losing it with ``self.results``."""

    def test_do_eval_appends_one_json_line_per_eval(self, tmp_path, monkeypatch):
        def fake_eval(model, env, **kwargs):
            return {
                "win_rate": 0.5,
                "avg_reward": 1.0,
                "avg_length": 1.0,
                "avg_turns": 1.0,
                "std_reward": 0.0,
                "wins": 1,
                "losses": 1,
                "draws": 0,
            }

        monkeypatch.setattr("reinforcetactics.rl.callbacks.evaluate_model", fake_eval)

        jsonl_path = tmp_path / "eval_results.jsonl"
        cb = PeriodicEvalCallback(
            eval_env=object(),
            eval_freq=100,
            n_eval_episodes=1,
            results_jsonl_path=jsonl_path,
            verbose=0,
        )
        cb.model = _BestStubModel()

        for ts in (100, 200):
            cb.num_timesteps = ts
            cb.model.num_timesteps = ts
            cb._last_eval_block = ts // 100
            cb._do_eval()
            # Appended immediately, not at stage end: after the first eval
            # the file already holds that eval's row.
            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == ts // 100

        rows = [json.loads(line) for line in lines]
        assert [r["timesteps"] for r in rows] == [100, 200]
        assert len(cb.results) == 2  # in-memory contract unchanged


class _MetricsStubModel:
    """Minimal model stub for exercising TrainingMetricsCallback commits."""

    def __init__(self) -> None:
        self.num_timesteps = 0
        self.logger = _StubLogger()
        self.ep_info_buffer = [{"r": 2.0, "l": 10}, {"r": 4.0, "l": 20}]
        self.ent_coef = 0.05


class TestTrainingMetricsCsvPersistence:
    """``csv_path`` mirrors every committed record to disk immediately so
    optimizer diagnostics (approx_kl, value_loss, ...) survive a Colab
    disconnect instead of dying with the in-memory ``records`` list."""

    def test_records_appended_to_csv_incrementally(self, tmp_path):
        csv_path = tmp_path / "train_metrics.csv"
        cb = TrainingMetricsCallback(csv_path=csv_path)
        cb.model = _MetricsStubModel()
        cb.model.logger.name_to_value = {
            "train/approx_kl": 0.01,
            "train/value_loss": 3.5,
        }

        cb.context = "stage_a"
        cb.num_timesteps = 100
        cb._on_rollout_end()
        cb._on_rollout_start()  # commits the pending record

        # First record is on disk already (survives a mid-run kill).
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2  # header + 1 row

        cb.context = "stage_b"
        cb.num_timesteps = 200
        cb._on_rollout_end()
        cb._on_training_end()

        assert len(cb.records) == 2
        assert cb.records[0]["stage"] == "stage_a"
        assert cb.records[0]["rollout/ep_rew_mean"] == pytest.approx(3.0)
        assert cb.records[0]["train/approx_kl"] == pytest.approx(0.01)
        assert cb.records[0]["train/ent_coef"] == pytest.approx(0.05)

        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert [r["stage"] for r in rows] == ["stage_a", "stage_b"]
        assert rows[1]["timesteps"] == "200"
        assert rows[0]["train/value_loss"] == "3.5"

    def test_no_csv_written_when_path_unset(self, tmp_path):
        cb = TrainingMetricsCallback()
        cb.model = _MetricsStubModel()
        cb.model.logger.name_to_value = {"train/approx_kl": 0.01}
        cb.num_timesteps = 100
        cb._on_rollout_end()
        cb._on_training_end()
        assert len(cb.records) == 1
        assert list(tmp_path.iterdir()) == []


class TestPromotionCallback:
    def test_validates_threshold_and_patience(self):
        eval_cb = _make_eval_stub()
        with pytest.raises(ValueError):
            PromotionCallback(eval_cb, threshold=1.5, patience=1)
        with pytest.raises(ValueError):
            PromotionCallback(eval_cb, threshold=0.9, patience=0)

    def test_does_not_promote_before_any_eval(self):
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, verbose=0)
        for _ in range(5):
            assert _step_promote(cb) is True
        assert cb.promoted is False

    def test_single_passing_eval_is_not_enough(self):
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, verbose=0)

        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb) is True
        assert cb.promoted is False
        assert cb._streak == 1

    def test_promotes_after_patience_consecutive_passes(self):
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=3, verbose=0)

        _append_eval(eval_cb, 0.92)
        assert _step_promote(cb) is True
        _append_eval(eval_cb, 0.91)
        assert _step_promote(cb) is True
        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb) is False
        assert cb.promoted is True

    def test_failing_eval_resets_streak(self):
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, verbose=0)

        _append_eval(eval_cb, 0.95)
        _step_promote(cb)
        assert cb._streak == 1

        _append_eval(eval_cb, 0.50)  # regression
        _step_promote(cb)
        assert cb._streak == 0
        assert cb.promoted is False

        _append_eval(eval_cb, 0.95)
        _step_promote(cb)
        _append_eval(eval_cb, 0.95)
        result = _step_promote(cb)
        assert result is False
        assert cb.promoted is True

    def test_consumes_multiple_new_results_in_one_step(self):
        # Defensive: if two evals somehow land between callback ticks,
        # the streak accounting must still be correct.
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, verbose=0)
        _append_eval(eval_cb, 0.95)
        _append_eval(eval_cb, 0.95)
        result = _step_promote(cb)
        assert result is False
        assert cb.promoted is True

    def test_min_timesteps_validates_non_negative(self):
        eval_cb = _make_eval_stub()
        with pytest.raises(ValueError):
            PromotionCallback(eval_cb, threshold=0.9, patience=2, min_timesteps=-1)

    def test_min_timesteps_default_zero_preserves_legacy_behaviour(self):
        # No min set -> behaves exactly as before (promote on first
        # eligible streak regardless of num_timesteps).
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, verbose=0)
        _append_eval(eval_cb, 0.95)
        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb, num_timesteps=8) is False
        assert cb.promoted is True

    def test_min_timesteps_blocks_promotion_in_pre_window(self):
        # Two passing evals during the pre-window must NOT promote -- the
        # whole point is to guarantee stage-specific training before
        # handoff, even from a strong carry-in policy.
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, min_timesteps=500_000, verbose=0)
        _append_eval(eval_cb, 0.95)
        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb, num_timesteps=8) is True  # pre-window eval @8 ignored
        assert _step_promote(cb, num_timesteps=100_000) is True  # still pre-window
        assert cb.promoted is False
        assert cb._streak == 0

    def test_min_timesteps_does_not_count_pre_window_evals_toward_streak(self):
        # Pre-window evals are discarded entirely; the streak must build
        # from post-min evals only. Otherwise a strong carry-in would
        # promote immediately on crossing the min boundary, defeating
        # the purpose.
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, min_timesteps=500_000, verbose=0)
        # Two passing pre-window evals (would normally trigger promotion).
        _append_eval(eval_cb, 0.95)
        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb, num_timesteps=200_000) is True
        assert cb.promoted is False
        # Cross into the window; existing pre-window results were discarded
        # at the boundary tick, so a single new eval is NOT enough yet.
        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb, num_timesteps=550_000) is True
        assert cb.promoted is False
        assert cb._streak == 1
        # A second post-window passing eval completes the streak.
        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb, num_timesteps=600_000) is False
        assert cb.promoted is True

    def test_min_timesteps_promotes_normally_after_window_opens(self):
        # After crossing min_timesteps with no prior evals, behaviour is
        # identical to the no-min case.
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, min_timesteps=500_000, verbose=0)
        # No pre-window evals. First post-window eval is single -> not enough.
        assert _step_promote(cb, num_timesteps=600_000) is True
        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb, num_timesteps=650_000) is True
        assert cb.promoted is False
        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb, num_timesteps=700_000) is False
        assert cb.promoted is True

    def test_min_timesteps_is_stage_relative_under_cumulative_counter(self):
        # The bootstrap runner trains with ``reset_num_timesteps=False``, so
        # ``num_timesteps`` is cumulative across stages. A stage entered at
        # 2M cumulative steps with min_timesteps=500k must still gate its
        # first 500k *stage* steps -- under the old absolute comparison the
        # gate was a silent no-op on every stage after the first (the exact
        # stages the v31 sweep added it for).
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, min_timesteps=500_000, verbose=0)
        cb.num_timesteps = 2_000_000  # carried in from prior stages
        cb._on_training_start()  # SB3 fires this at the top of learn()
        _append_eval(eval_cb, 0.95)
        _append_eval(eval_cb, 0.95)
        # 100k into the stage: still pre-window despite 2.1M absolute steps.
        assert _step_promote(cb, num_timesteps=2_100_000) is True
        assert cb.promoted is False
        assert cb._streak == 0
        # Window opens at stage-relative 500k; the streak builds fresh.
        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb, num_timesteps=2_600_000) is True
        assert cb._streak == 1
        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb, num_timesteps=2_650_000) is False
        assert cb.promoted is True


# ---------------------------------------------------------------------------
# Curriculum loading and validation via TrainingConfig
# ---------------------------------------------------------------------------


VALID_DICT: dict[str, Any] = {
    "algorithm": "maskable_ppo",
    "seed": 7,
    "env": {
        "n_envs": 2,
        "max_steps": 100,
        "max_turns": 20,
        "enabled_units": ["W"],
        "action_space_type": "flat_discrete",
    },
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 256,
    },
    "eval": {
        "eval_freq": 1000,
    },
    "curriculum": {
        "stages": [
            {
                "name": "stage_a",
                "map_file": "maps/1v1/starter.csv",
                "opponent": "random",
                "promotion_win_rate": 0.9,
                "patience": 2,
                "max_timesteps": 5000,
                "n_eval_episodes": 5,
            },
            {
                "name": "stage_b",
                "map_file": "maps/1v1/starter.csv",
                "opponent": "simple",
                "promotion_win_rate": 0.8,
                "patience": 1,
                "max_timesteps": 5000,
                "n_eval_episodes": 5,
            },
        ],
    },
}


class TestCurriculumLoading:
    def test_round_trips_through_dict(self):
        cfg = config_from_dict(VALID_DICT)
        assert cfg.seed == 7
        assert cfg.env.n_envs == 2
        assert cfg.eval.eval_freq == 1000
        assert len(cfg.curriculum.stages) == 2
        assert cfg.curriculum.stages[0].name == "stage_a"
        assert cfg.curriculum.stages[1].opponent == "simple"
        assert isinstance(cfg.ppo, PPOConfig)
        assert isinstance(cfg.env, EnvConfig)
        assert isinstance(cfg.curriculum, CurriculumConfig)
        assert isinstance(cfg.curriculum.stages[0], CurriculumStage)

    def test_round_trips_through_yaml_file(self, tmp_path):
        path = tmp_path / "bootstrap.yaml"
        path.write_text(yaml.safe_dump(VALID_DICT), encoding="utf-8")
        cfg = load_config(path)
        assert len(cfg.curriculum.stages) == 2
        assert cfg.curriculum.stages[0].promotion_win_rate == pytest.approx(0.9)

    def test_rejects_unknown_top_level_key(self):
        bad = dict(VALID_DICT)
        bad["unknown"] = 1
        with pytest.raises(ValueError, match="Unknown top-level keys"):
            config_from_dict(bad)

    def test_rejects_unknown_stage_field(self):
        bad = {
            **VALID_DICT,
            "curriculum": {
                "stages": [{**VALID_DICT["curriculum"]["stages"][0], "bogus": 1}],
            },
        }
        with pytest.raises(ValueError, match="Unknown keys for CurriculumStage"):
            config_from_dict(bad)

    def test_empty_stages_loads_but_runner_rejects(self, tmp_path):
        # An empty curriculum is a valid TrainingConfig (other algorithms
        # don't use it); it's run_curriculum that requires a non-empty list.
        cfg = config_from_dict({**VALID_DICT, "curriculum": {"stages": []}})
        assert cfg.curriculum.stages == []
        with pytest.raises(ValueError, match="cfg.curriculum.stages is empty"):
            run_curriculum(cfg, output_dir=tmp_path)

    def test_rejects_duplicate_stage_names(self):
        stage = VALID_DICT["curriculum"]["stages"][0]
        bad = {**VALID_DICT, "curriculum": {"stages": [stage, stage]}}
        with pytest.raises(ValueError, match="duplicate stage name"):
            config_from_dict(bad)

    def test_rejects_unknown_opponent(self):
        stage = {**VALID_DICT["curriculum"]["stages"][0], "opponent": "godlike"}
        bad = {**VALID_DICT, "curriculum": {"stages": [stage]}}
        with pytest.raises(ValueError, match="unknown opponent"):
            config_from_dict(bad)

    def test_accepts_balanced_random_opponent(self):
        stage = {**VALID_DICT["curriculum"]["stages"][0], "opponent": "balanced_random"}
        cfg = config_from_dict({**VALID_DICT, "curriculum": {"stages": [stage]}})
        assert cfg.curriculum.stages[0].opponent == "balanced_random"

    def test_min_timesteps_before_promotion_default_and_round_trip(self):
        # Default is 0 (legacy behaviour).
        cfg = config_from_dict(VALID_DICT)
        assert cfg.curriculum.stages[0].min_timesteps_before_promotion == 0
        # Setting it round-trips through the loader. Use a value under
        # VALID_DICT's stage max_timesteps so validation doesn't reject.
        stage = {**VALID_DICT["curriculum"]["stages"][0], "min_timesteps_before_promotion": 2_000}
        cfg = config_from_dict({**VALID_DICT, "curriculum": {"stages": [stage]}})
        assert cfg.curriculum.stages[0].min_timesteps_before_promotion == 2_000

    def test_min_timesteps_before_promotion_rejects_negative(self):
        stage = {**VALID_DICT["curriculum"]["stages"][0], "min_timesteps_before_promotion": -1}
        with pytest.raises(ValueError, match="min_timesteps_before_promotion"):
            config_from_dict({**VALID_DICT, "curriculum": {"stages": [stage]}})

    def test_min_timesteps_before_promotion_rejects_exceeding_max(self):
        # min > max would mean the stage can never promote -> fail loud.
        stage = {
            **VALID_DICT["curriculum"]["stages"][0],
            "max_timesteps": 100_000,
            "min_timesteps_before_promotion": 200_000,
        }
        with pytest.raises(ValueError, match="must be <= max_timesteps"):
            config_from_dict({**VALID_DICT, "curriculum": {"stages": [stage]}})

    def test_shipped_config_resolves_features_extractor_class(self):
        """``policy_kwargs.features_extractor_class`` in bootstrap.yaml is
        a dotted string. The resolver must turn it into the actual class
        before MaskablePPO is constructed."""
        from reinforcetactics.rl.extractors import SpatialFeatureExtractor

        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_config(repo_root / "configs" / "ppo" / "bootstrap.yaml")
        kwargs = cfg.ppo.as_sb3_kwargs().get("policy_kwargs") or {}
        # As loaded, the value is still a string (YAML can't carry classes).
        assert isinstance(kwargs.get("features_extractor_class"), str)
        # After resolution, it's the actual extractor class.
        resolved = _resolve_policy_kwargs(kwargs)
        assert resolved is not None
        assert resolved["features_extractor_class"] is SpatialFeatureExtractor

    def test_resolve_dotted_rejects_non_dotted(self):
        with pytest.raises(ValueError, match="is not dotted"):
            _resolve_dotted("SpatialFeatureExtractor")

    def test_resolve_policy_kwargs_none_passes_through(self):
        assert _resolve_policy_kwargs(None) is None
        assert _resolve_policy_kwargs({}) == {}

    def test_resolve_policy_kwargs_preserves_non_class_keys(self):
        resolved = _resolve_policy_kwargs({"net_arch": [64, 64]})
        assert resolved == {"net_arch": [64, 64]}

    def test_shipped_config_loads(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_config(repo_root / "configs" / "ppo" / "bootstrap.yaml")
        names = [s.name for s in cfg.curriculum.stages]
        # Earlier iterations included `noop` stages on each map as a
        # stage-0 sanity check; they actively prevented PPO from
        # learning (no opponent variance -> constant returns ->
        # advantages collapse to ~0 -> policy never updates). Reverting
        # to the original 6-stage layout (now 7, with the
        # `beginner_balanced_random` bridge for the map shift) lets
        # opponent randomness drive exploration the way PPO needs.
        assert names == [
            "starter_random",
            "starter_simple",
            "starter_medium",
            "beginner_balanced_random",
            "beginner_random_10",
            "beginner_random_15",
            "beginner_random_20",
            "beginner_simple",
            "beginner_mixed_50",
            "beginner_medium",
            "beginner_mixed_med_adv_50",
            "beginner_advanced",
            "intermediate_balanced_random",
            "intermediate_random_20",
            "intermediate_mixed_random_simple",
            "intermediate_simple",
            "intermediate_medium",
            "skirmish_balanced_random",
            "skirmish_random_10",
            "skirmish_random_15",
            "skirmish_random_20",
            "skirmish_mixed_25",
            "skirmish_mixed_50",
            "skirmish_simple",
            "skirmish_medium",
            "corner_points_balanced_random",
            "corner_points_random_10",
            "corner_points_random_15",
            "corner_points_random_20",
            "corner_points_mixed_25",
            "corner_points_mixed_50",
            "corner_points_simple",
            "corner_points_medium",
        ]
        assert "starter_noop" not in names, "noop stages broke PPO learning in earlier runs -- removing them was deliberate"
        assert "beginner_noop" not in names
        # Regression: PyYAML 1.1 parses ``3e-4`` (no decimal) as a string,
        # which then fails deep inside SB3's lr-schedule check.
        assert isinstance(cfg.ppo.learning_rate, (int, float))
        assert isinstance(cfg.ppo.ent_coef, (int, float))
        assert isinstance(cfg.ppo.clip_range, (int, float))
        by_name = {s.name: s for s in cfg.curriculum.stages}
        # Starter stages inherit env max_turns (no per-stage override). They
        # do bump ent_coef above the global ppo.ent_coef floor so the early
        # random/simple curriculum sees enough exploration; assert the
        # direction of the bump rather than that no override exists.
        assert by_name["starter_random"].max_turns is None
        starter_random = by_name["starter_random"]
        if starter_random.ent_coef is not None:
            assert starter_random.resolve_ent_coef(cfg.ppo) >= cfg.ppo.ent_coef
        # Beginner stages bump max_turns / max_steps for the bigger map.
        first_beginner = by_name["beginner_balanced_random"]
        assert first_beginner.max_turns is not None
        assert first_beginner.max_turns >= 30
        # Entropy bump on the FIRST beginner stage (map-shift exploration
        # shock). Cooled on later stages.
        assert first_beginner.ent_coef is not None
        assert first_beginner.resolve_ent_coef(cfg.ppo) > cfg.ppo.ent_coef
        sched = first_beginner.resolve_ent_coef_schedule()
        assert sched is not None, "first beginner stage should drive an entropy schedule"
        assert sched["start"] > sched["end"]
        # Reward-shape invariant on beginner stages: HQ capture is much
        # harder than elimination on the bigger map, so the two terminal
        # rewards must be equalized (or capture <= elimination). The
        # shipped config used to enforce this via a per-stage override
        # on beginner_random_20; the global-config rewrite equalises
        # win_by_hq_capture / win_by_elimination at the env level
        # instead, which has the same effect for the resolved stage
        # config. Test against the resolved dict so either design path
        # satisfies the invariant.
        beginner_random = by_name["beginner_random_20"]
        resolved = beginner_random.resolve_reward_config(cfg.env)
        assert resolved is not None
        assert resolved["win_by_hq_capture"] <= resolved["win_by_elimination"]
        # Policy MLP capacity: SB3 defaults net_arch to [64, 64] which is
        # undersized for a Dict obs (~734 input dims) feeding a flat-
        # discrete head with up to 512 logits. The shipped config bumps
        # both pi and vf to at least [128, 128].
        pk = cfg.ppo.policy_kwargs or {}
        net_arch = pk.get("net_arch")
        assert net_arch is not None, "expected ppo.policy_kwargs.net_arch in shipped config"
        assert isinstance(net_arch, dict)
        assert min(net_arch["pi"]) >= 128
        assert min(net_arch["vf"]) >= 128

    def test_reward_config_override_merges_with_defaults(self):
        # Stage override should *merge* over env.reward_config, not
        # replace it. So a stage that only overrides one key still gets
        # the rest of the env defaults.
        env = EnvConfig(reward_config={"win": 5000.0, "loss": -5000.0, "draw": -5000.0})
        stage = CurriculumStage(
            name="s",
            map_file="m.csv",
            opponent="random",
            reward_config={"win": 3000.0, "win_by_elimination": 3000.0},
        )
        resolved = stage.resolve_reward_config(env)
        assert resolved == {
            "win": 3000.0,  # overridden
            "loss": -5000.0,  # inherited
            "draw": -5000.0,  # inherited
            "win_by_elimination": 3000.0,  # added by stage
        }

    def test_reward_config_resolves_to_defaults_when_unset(self):
        env = EnvConfig(reward_config={"win": 5000.0, "loss": -5000.0})
        stage = CurriculumStage(name="s", map_file="m.csv", opponent="random")
        assert stage.resolve_reward_config(env) == {"win": 5000.0, "loss": -5000.0}

    def test_reward_config_returns_none_when_nothing_specified(self):
        env = EnvConfig(reward_config=None)
        stage = CurriculumStage(name="s", map_file="m.csv", opponent="random")
        assert stage.resolve_reward_config(env) is None

    def test_rejects_non_mapping_reward_config(self):
        common = dict(name="s", map_file="m.csv", opponent="random")
        with pytest.raises(TypeError, match="reward_config"):
            CurriculumStage(**common, reward_config=[1, 2, 3]).validate()


class TestBootstrapStagesAreConstructible:
    """Every stage in ``configs/ppo/bootstrap.yaml`` must build a working env.

    Without this, an opponent_kwargs / opponent-type mismatch (e.g. a
    MixedBot inner name the bot module doesn't recognise) hides until
    Colab actually hits that stage at runtime and crashes mid-curriculum.
    The smoke test exercises every stage's ``make_stage_env`` path with
    the bootstrap.yaml's exact kwargs, sweeping seeds so MixedBot's
    per-episode coin flip lands on both branches across the run.

    Specifically asserts:
      * Construction works (catches the original MixedBot ``unknown bot
        type 'random'`` crash that motivated this test).
      * Reset works across multiple distinct seeds, so MixedBot rebuilds
        a different inner bot per episode.
      * One env step succeeds, exercising the chosen inner bot's
        ``take_turn`` path -- catches mid-episode bugs in the inner
        scripted/random bot that pure construction would miss.
      * ``pad_to_size`` is resolved the way production resolves it
        (bootstrap.py's ``_resolve_curriculum_pad_size`` picks the
        per-axis max when the curriculum spans multiple map sizes),
        so any regression in the padded-obs path is caught here rather
        than at the first production step.
    """

    @pytest.fixture(scope="class")
    def shipped_cfg(self):
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_config(repo_root / "configs" / "ppo" / "bootstrap.yaml")
        # Resolve pad_to_size the same way ``run_curriculum`` does. The
        # shipped bootstrap.yaml spans 6x6 / 7x7 / 8x8 / 10x12 maps with
        # ``action_space_type=flat_discrete``; production therefore picks
        # ``pad_to_size=(10, 12)``. Without this resolution every stage's
        # env would build at its native size and the padded-obs code path
        # would be untested.
        from reinforcetactics.rl.bootstrap import _resolve_curriculum_pad_size

        cfg.env.pad_to_size = _resolve_curriculum_pad_size(cfg)
        return cfg

    def test_every_stage_env_constructs_resets_and_steps(self, shipped_cfg):
        import numpy as np

        from reinforcetactics.rl.bootstrap import make_stage_env

        # Sweep multiple distinct seeds per stage so MixedBot's per-episode
        # coin flip lands on both inner-bot branches. For p_hard=0.25
        # stages (the worst case in bootstrap.yaml), 8 distinct seeds gives
        # P(miss one branch) = 0.75^8 ~= 10%, vs the earlier 4-seed sweep
        # which left ~32% miss probability -- enough to silently regress
        # CI runs.
        seeds = list(range(8))
        failures: list[str] = []
        for stage in shipped_cfg.curriculum.stages:
            for seed in seeds:
                env = None
                try:
                    # Each (stage, seed) builds a fresh env. ``make_stage_env``
                    # forwards reward_config / max_actions_per_turn /
                    # pad_to_size etc. from ``shipped_cfg.env`` so the
                    # smoke env matches what run_curriculum would build.
                    env = make_stage_env(stage, shipped_cfg.env, seed=seed)
                    # Reset with a DIFFERENT seed than the construction
                    # seed so the opponent-rebuild path actually draws a
                    # fresh ``np_random.integers(...)``. With the same
                    # seed gym re-seeds np_random identically and the
                    # rebuild is byte-identical -- pointless.
                    env.reset(seed=seed + 10_000)
                    # Step once so the inner bot's ``take_turn`` path
                    # (executed inside ``end_turn``) is exercised. Pick
                    # the first legal action via the mask. Failures here
                    # catch bugs in the bot's per-turn logic that pure
                    # construction would miss.
                    mask = env.action_masks()
                    if isinstance(mask, tuple):
                        mask = mask[0]
                    action = int(np.argmax(np.asarray(mask, dtype=bool)))
                    env.step(action)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{stage.name} (seed={seed}): {type(exc).__name__}: {exc}")
                finally:
                    # ``finally`` so a failure in reset / step doesn't
                    # leak env handles (subprocess pipes under
                    # use_subprocess=True). Without this, a chain of
                    # stage failures could accumulate >100 leaked envs
                    # before pytest tears down.
                    if env is not None:
                        try:
                            env.close()
                        except Exception:  # noqa: BLE001
                            pass
        assert not failures, "\n".join(failures)


class TestCurriculumStageResolution:
    def test_resolves_to_defaults_when_unset(self):
        env = EnvConfig(max_steps=400, max_turns=20)
        ppo = PPOConfig(ent_coef=0.05)
        stage = CurriculumStage(name="s", map_file="m.csv", opponent="random")
        assert stage.resolve_max_steps(env) == 400
        assert stage.resolve_max_turns(env) == 20
        assert stage.resolve_ent_coef(ppo) == pytest.approx(0.05)

    def test_resolves_to_override_when_set(self):
        env = EnvConfig(max_steps=400, max_turns=20)
        ppo = PPOConfig(ent_coef=0.05)
        stage = CurriculumStage(
            name="s",
            map_file="m.csv",
            opponent="random",
            max_steps=800,
            max_turns=40,
            ent_coef=0.10,
        )
        assert stage.resolve_max_steps(env) == 800
        assert stage.resolve_max_turns(env) == 40
        assert stage.resolve_ent_coef(ppo) == pytest.approx(0.10)

    def test_rejects_invalid_overrides(self):
        common = dict(name="s", map_file="m.csv", opponent="random")
        with pytest.raises(ValueError, match="max_steps"):
            CurriculumStage(**common, max_steps=0).validate()
        with pytest.raises(ValueError, match="max_turns"):
            CurriculumStage(**common, max_turns=-1).validate()
        with pytest.raises(ValueError, match="ent_coef"):
            CurriculumStage(**common, ent_coef=-0.01).validate()


class TestCurriculumStageEntropySchedule:
    """``ent_coef`` accepts either a constant or a ``{start, end, schedule}``
    mapping. The mapping form drives :class:`EntropyScheduleCallback` so
    exploration noise can be cooled across a stage."""

    common = dict(name="s", map_file="m.csv", opponent="random")

    def test_resolve_initial_value_uses_schedule_start(self):
        ppo = PPOConfig(ent_coef=0.05)
        stage = CurriculumStage(
            **self.common,
            ent_coef={"start": 0.10, "end": 0.03, "schedule": "linear"},
        )
        # Initial seed before the callback takes over: the schedule's
        # start, not the ppo default.
        assert stage.resolve_ent_coef(ppo) == pytest.approx(0.10)

    def test_resolve_schedule_returns_descriptor(self):
        stage = CurriculumStage(
            **self.common,
            ent_coef={"start": 0.10, "end": 0.03, "schedule": "linear"},
        )
        sched = stage.resolve_ent_coef_schedule()
        assert sched == {"start": 0.10, "end": 0.03, "schedule": "linear"}

    def test_resolve_schedule_defaults_to_linear(self):
        stage = CurriculumStage(**self.common, ent_coef={"start": 0.10, "end": 0.03})
        sched = stage.resolve_ent_coef_schedule()
        assert sched is not None
        assert sched["schedule"] == "linear"

    def test_resolve_schedule_is_none_for_constant(self):
        stage = CurriculumStage(**self.common, ent_coef=0.10)
        assert stage.resolve_ent_coef_schedule() is None

    def test_resolve_schedule_is_none_when_unset(self):
        stage = CurriculumStage(**self.common)
        assert stage.resolve_ent_coef_schedule() is None

    def test_validate_rejects_unknown_schedule_keys(self):
        with pytest.raises(ValueError, match="unknown keys"):
            CurriculumStage(
                **self.common,
                ent_coef={"start": 0.1, "end": 0.0, "extra": 1},
            ).validate()

    def test_validate_rejects_missing_required_key(self):
        with pytest.raises(ValueError, match="missing required key 'end'"):
            CurriculumStage(**self.common, ent_coef={"start": 0.1}).validate()
        with pytest.raises(ValueError, match="missing required key 'start'"):
            CurriculumStage(**self.common, ent_coef={"end": 0.03}).validate()

    def test_validate_rejects_negative_endpoint(self):
        with pytest.raises(ValueError, match="ent_coef.end"):
            CurriculumStage(
                **self.common,
                ent_coef={"start": 0.1, "end": -0.01},
            ).validate()

    def test_validate_rejects_unknown_schedule_kind(self):
        with pytest.raises(ValueError, match="schedule must be 'linear' or 'cosine'"):
            CurriculumStage(
                **self.common,
                ent_coef={"start": 0.1, "end": 0.03, "schedule": "exponential"},
            ).validate()


# ---------------------------------------------------------------------------
# run_curriculum integration with fakes (no SB3, no real env)
# ---------------------------------------------------------------------------


@dataclass
class _FakeEnv:
    """Just enough of a Gym env for the callbacks not to complain."""

    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass
class _FakeModel:
    """In-memory stand-in for MaskablePPO. Records the order of learn()
    calls and feeds preprogrammed eval win_rates into the callback list
    so PromotionCallback drives the loop the same way it would in real
    training."""

    win_rate_program: dict[str, list[float]] = field(default_factory=dict)
    learn_calls: list[dict[str, Any]] = field(default_factory=list)
    save_calls: list[str] = field(default_factory=list)
    set_env_calls: list[Any] = field(default_factory=list)
    num_timesteps: int = 0
    # SB3's BaseCallback exposes ``self.logger`` as a read-only property
    # that delegates to ``self.model.logger``; we satisfy that by exposing
    # the stub here and (below) setting ``cb.model = self``.
    logger: _StubLogger = field(default_factory=_StubLogger)
    ep_info_buffer: list = field(default_factory=list)
    # Tracks the value of ``ent_coef`` that the runner set immediately
    # before each ``learn()`` call, parallel to ``learn_calls``.
    ent_coef: float = 0.0
    ent_coef_at_learn: list[float] = field(default_factory=list)
    # The full callback list passed to each ``learn()`` invocation,
    # so tests can assert which callbacks the runner installed for a
    # given stage (notably whether ``EntropyScheduleCallback`` was
    # added when the stage's ``ent_coef`` is a schedule mapping).
    callbacks_at_learn: list[list[Any]] = field(default_factory=list)
    _current_stage_idx: int = 0
    _stage_names: list[str] = field(default_factory=list)

    def set_env(self, env: Any) -> None:
        self.set_env_calls.append(env)
        self._current_stage_idx += 1

    def learn(
        self,
        total_timesteps: int,
        callback: list[Any],
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> None:
        stage_name = self._stage_names[self._current_stage_idx]
        program = list(self.win_rate_program.get(stage_name, []))
        # Snapshot whatever ``ent_coef`` the runner mutated us to prior
        # to this learn() call.
        self.ent_coef_at_learn.append(float(self.ent_coef))
        self.callbacks_at_learn.append(list(callback))

        # SB3 wires up callbacks before _on_step; mimic the parts we use.
        # Don't set ``cb.logger`` directly: BaseCallback.logger is a
        # read-only property that delegates to ``self.model.logger``.
        # SB3 also fires _on_training_start at the top of every learn()
        # call; stage-relative callbacks (PromotionCallback's min_timesteps
        # gate, the schedule callbacks) snapshot their stage offset there.
        for cb in callback:
            cb.model = self
            cb.num_timesteps = self.num_timesteps
            cb._on_training_start()

        # Each programmed entry simulates one eval block. We append directly
        # to PeriodicEvalCallback.results (bypassing the real evaluate_model
        # call, which would need a working env) and tick PromotionCallback
        # so it observes the new result and decides whether to short-circuit.
        # We skip PeriodicEvalCallback._on_step deliberately: its real path
        # would re-run evaluate_model against the stub env. The unit tests
        # for PromotionCallback already cover the streak logic; here we
        # verify the runner's loop / promotion / stall behaviour.
        eval_cb = next(c for c in callback if isinstance(c, PeriodicEvalCallback))
        promote_cb = next(c for c in callback if isinstance(c, PromotionCallback))
        for win_rate in program:
            eval_cb.results.append({"win_rate": win_rate, "avg_reward": 0.0})
            self.num_timesteps += 1
            # BaseCallback.on_step syncs num_timesteps from the model
            # before calling _on_step; mirror that so the stage-relative
            # min_timesteps gate sees the live counter.
            promote_cb.num_timesteps = self.num_timesteps
            if not promote_cb._on_step():
                self.learn_calls.append(
                    {
                        "stage": stage_name,
                        "total_timesteps": total_timesteps,
                        "exited_early": True,
                    }
                )
                return
        # No early exit -> stage exhausted its budget without promoting.
        self.learn_calls.append({"stage": stage_name, "total_timesteps": total_timesteps, "exited_early": False})

    set_parameters_calls: list[str] = field(default_factory=list)

    def save(self, path: str) -> None:
        self.save_calls.append(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake-checkpoint", encoding="utf-8")

    def set_parameters(self, path: str, exact_match: bool = True) -> None:
        # Mirrors SB3: swap policy/optimizer tensors only. The runner
        # uses this for warm_start_path and the between-stage best-
        # checkpoint restore; record the path + order for assertions.
        self.set_parameters_calls.append(str(path))


class _FakeModelSavesBest(_FakeModel):
    """Like _FakeModel but also drops ``best_model.zip`` into the eval
    callback's ``save_dir`` during ``learn()`` -- mirroring the real
    ``PeriodicEvalCallback`` best-by-WR save that the base fake
    deliberately skips -- so the runner's between-stage best-checkpoint
    restore path is actually exercised."""

    def learn(
        self,
        total_timesteps: int,
        callback: list[Any],
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> None:
        for cb in callback:
            if isinstance(cb, PeriodicEvalCallback) and cb.save_dir is not None:
                Path(cb.save_dir).mkdir(parents=True, exist_ok=True)
                (Path(cb.save_dir) / "best_model.zip").write_text("best", encoding="utf-8")
        super().learn(total_timesteps, callback, reset_num_timesteps, progress_bar)


def _make_cfg(stages: list[CurriculumStage]) -> TrainingConfig:
    cfg = TrainingConfig(
        algorithm="maskable_ppo",
        seed=0,
        env=EnvConfig(n_envs=1, enabled_units=["W"]),
        ppo=PPOConfig(),
        eval=EvalConfig(eval_freq=1),
        curriculum=CurriculumConfig(stages=stages),
    )
    cfg.validate()
    return cfg


def _stage(name: str, opp: str = "random", patience: int = 2, threshold: float = 0.9) -> CurriculumStage:
    return CurriculumStage(
        name=name,
        map_file="maps/1v1/starter.csv",
        opponent=opp,
        promotion_win_rate=threshold,
        patience=patience,
        max_timesteps=10_000,
        n_eval_episodes=2,
    )


def _setup_run(stages, win_rate_program, tmp_path):
    cfg = _make_cfg(stages)

    fake_model: _FakeModel | None = None

    def model_factory(vec_env, cfg_arg, output_dir):
        nonlocal fake_model
        fake_model = _FakeModel(
            win_rate_program=win_rate_program,
            _stage_names=[s.name for s in cfg_arg.curriculum.stages],
        )
        return fake_model

    train_envs: list[_FakeEnv] = []
    eval_envs: list[_FakeEnv] = []

    def train_env_factory(stage, cfg_arg):
        env = _FakeEnv()
        train_envs.append(env)
        return env

    def eval_env_factory(stage, cfg_arg):
        env = _FakeEnv()
        eval_envs.append(env)
        return env

    return cfg, model_factory, train_env_factory, eval_env_factory, train_envs, eval_envs, lambda: fake_model


class TestRunCurriculum:
    def test_advances_through_all_stages_when_each_promotes(self, tmp_path):
        stages = [_stage("a", patience=2), _stage("b", "simple", patience=2)]
        program = {
            # Two consecutive >= 0.9 evals trigger promotion.
            "a": [0.95, 0.95],
            "b": [0.92, 0.93],
        }
        cfg, mf, tef, eef, train_envs, eval_envs, get_model = _setup_run(stages, program, tmp_path)

        result = run_curriculum(
            cfg,
            output_dir=tmp_path,
            train_env_factory=tef,
            eval_env_factory=eef,
            model_factory=mf,
        )

        model = get_model()
        assert model is not None
        # Every stage's learn() was called and exited early via promotion.
        assert [c["stage"] for c in model.learn_calls] == ["a", "b"]
        assert all(c["exited_early"] for c in model.learn_calls)
        # set_env called for every stage after the first.
        assert len(model.set_env_calls) == len(stages) - 1
        # History reports both stages promoted.
        assert [h["stage"] for h in result["history"]] == ["a", "b"]
        assert all(h["promoted"] for h in result["history"])
        # Final checkpoint was written.
        assert Path(result["final_model_path"]).is_file()
        # All envs were closed.
        assert all(e.closed for e in train_envs)
        assert all(e.closed for e in eval_envs)

    def test_min_timesteps_gate_is_stage_relative_in_curriculum(self, tmp_path):
        # Stage 'b' requires 3 stage-relative steps before promotion can
        # fire. The fake model advances num_timesteps by 1 per eval and
        # never resets it between stages (mirroring
        # ``reset_num_timesteps=False``). Stage 'a' promotes after 1 eval,
        # so the cumulative counter is 1 entering 'b'; 'b' then sees two
        # passing evals at stage-relative steps 1 and 2 -- both inside the
        # pre-window, so 'b' must stall. Under the old cumulative
        # comparison the gate would have opened at absolute step 3 and
        # wrongly promoted on the second eval.
        stage_a = _stage("a", patience=1)
        stage_b = _stage("b", "simple", patience=1)
        stage_b.min_timesteps_before_promotion = 3
        program = {"a": [0.95], "b": [0.95, 0.95]}
        cfg, mf, tef, eef, *_ = _setup_run([stage_a, stage_b], program, tmp_path)

        with pytest.raises(CurriculumStalled) as excinfo:
            run_curriculum(
                cfg,
                output_dir=tmp_path,
                train_env_factory=tef,
                eval_env_factory=eef,
                model_factory=mf,
            )
        assert excinfo.value.stage_name == "b"

    def _run_with_best_saving_model(self, cfg, tmp_path):
        program = {"a": [0.95, 0.95], "b": [0.92, 0.93]}
        holder: dict[str, Any] = {}

        def model_factory(vec_env, cfg_arg, output_dir):
            m = _FakeModelSavesBest(
                win_rate_program=program,
                _stage_names=[s.name for s in cfg_arg.curriculum.stages],
            )
            holder["m"] = m
            return m

        run_curriculum(
            cfg,
            output_dir=tmp_path,
            train_env_factory=lambda s, c: _FakeEnv(),
            eval_env_factory=lambda s, c: _FakeEnv(),
            model_factory=model_factory,
        )
        return holder["m"]

    def test_restores_best_checkpoint_between_stages_by_default(self, tmp_path):
        # Default behaviour: after each promoted stage the runner reloads
        # that stage's best_model.zip into the in-memory model (so the
        # next stage inherits the peak policy, not the drifted end-of-
        # stage one). Both stages promote, so there are two restores --
        # the second also makes final_model.zip the best of the last
        # stage.
        cfg = _make_cfg([_stage("a", patience=2), _stage("b", "simple", patience=2)])
        assert cfg.curriculum.restore_best_checkpoint_between_stages is True
        m = self._run_with_best_saving_model(cfg, tmp_path)
        assert m.set_parameters_calls == [
            str(tmp_path / "a" / "best_model.zip"),
            str(tmp_path / "b" / "best_model.zip"),
        ]

    def test_restore_between_stages_can_be_disabled(self, tmp_path):
        # Legacy behaviour: carry the end-of-stage policy forward, no
        # best-checkpoint restore.
        cfg = _make_cfg([_stage("a", patience=2), _stage("b", "simple", patience=2)])
        cfg.curriculum.restore_best_checkpoint_between_stages = False
        m = self._run_with_best_saving_model(cfg, tmp_path)
        assert m.set_parameters_calls == []

    def test_applies_per_stage_ent_coef_and_env_overrides(self, tmp_path):
        # First stage inherits everything from cfg defaults; second stage
        # provides explicit overrides for max_steps, max_turns, and
        # ent_coef.
        stages = [
            _stage("a", patience=2),
            CurriculumStage(
                name="b",
                map_file="maps/1v1/beginner.csv",
                opponent="simple",
                promotion_win_rate=0.9,
                patience=2,
                max_timesteps=10_000,
                n_eval_episodes=2,
                max_steps=800,
                max_turns=40,
                ent_coef=0.10,
            ),
        ]
        program = {"a": [0.95, 0.95], "b": [0.95, 0.95]}

        train_calls: list[dict[str, Any]] = []

        def train_env_factory(stage, cfg_arg):
            # Mirror what the default factory does: resolve via the
            # stage's helpers so the test exercises the same code path.
            train_calls.append(
                {
                    "name": stage.name,
                    "max_steps": stage.resolve_max_steps(cfg_arg.env),
                    "max_turns": stage.resolve_max_turns(cfg_arg.env),
                }
            )
            return _FakeEnv()

        def eval_env_factory(stage, cfg_arg):
            return _FakeEnv()

        cfg = _make_cfg(stages)
        fake_model = _FakeModel(
            win_rate_program=program,
            _stage_names=[s.name for s in stages],
        )

        def model_factory(vec_env, cfg_arg, output_dir):
            return fake_model

        run_curriculum(
            cfg,
            output_dir=tmp_path,
            train_env_factory=train_env_factory,
            eval_env_factory=eval_env_factory,
            model_factory=model_factory,
        )

        # Stage 'a' inherits cfg.env / cfg.ppo defaults.
        assert train_calls[0]["max_steps"] == cfg.env.max_steps
        assert train_calls[0]["max_turns"] == cfg.env.max_turns
        assert fake_model.ent_coef_at_learn[0] == pytest.approx(cfg.ppo.ent_coef)
        # Stage 'b' uses overrides.
        assert train_calls[1]["max_steps"] == 800
        assert train_calls[1]["max_turns"] == 40
        assert fake_model.ent_coef_at_learn[1] == pytest.approx(0.10)

    def test_raises_when_stage_stalls(self, tmp_path):
        stages = [_stage("a", patience=2), _stage("b", "simple", patience=2)]
        program = {
            # Fewer-than-patience successful evals -> the runner should
            # see no promotion and raise CurriculumStalled before reaching
            # stage 'b'.
            "a": [0.95, 0.40, 0.80],
            "b": [0.95, 0.95],
        }
        cfg, mf, tef, eef, _, _, get_model = _setup_run(stages, program, tmp_path)

        with pytest.raises(CurriculumStalled) as excinfo:
            run_curriculum(
                cfg,
                output_dir=tmp_path,
                train_env_factory=tef,
                eval_env_factory=eef,
                model_factory=mf,
            )

        assert excinfo.value.stage_name == "a"
        assert excinfo.value.threshold == pytest.approx(0.9)
        # We never advance to stage 'b'.
        model = get_model()
        assert [c["stage"] for c in model.learn_calls] == ["a"]

        # The exception carries a partial result so notebook
        # diagnostics / replay-video helpers can run after a stall
        # instead of discarding everything.
        partial = excinfo.value.partial_result()
        assert partial["stalled"] is True
        assert partial["stalled_stage"] == "a"
        assert len(partial["history"]) == 1
        assert partial["history"][0]["stage"] == "a"
        assert partial["history"][0]["promoted"] is False
        assert partial["final_model_path"] is not None
        assert (tmp_path / "final_model.zip").exists()
        # The base fake never saves a best_model.zip, so the surfaced
        # best-checkpoint path is None (see the SavesBest variant below
        # for the populated case).
        assert partial["best_model_path"] is None

    def test_stall_surfaces_best_checkpoint(self, tmp_path):
        """On stall, the stage's peak-WR ``best_model.zip`` is surfaced on
        the exception / partial_result / run_status.json alongside the
        collapsed end-of-stage policy: most stalls peak >= threshold
        before crashing, so that checkpoint (not final_model.zip) is the
        one worth retrying from -- it was already on disk but nothing
        pointed at it."""
        stages = [_stage("a", patience=2)]
        program = {"a": [0.95, 0.40, 0.80]}
        cfg = _make_cfg(stages)

        def model_factory(vec_env, cfg_arg, output_dir):
            return _FakeModelSavesBest(
                win_rate_program=program,
                _stage_names=[s.name for s in cfg_arg.curriculum.stages],
            )

        with pytest.raises(CurriculumStalled) as excinfo:
            run_curriculum(
                cfg,
                output_dir=tmp_path,
                train_env_factory=lambda stage, c: _FakeEnv(),
                eval_env_factory=lambda stage, c: _FakeEnv(),
                model_factory=model_factory,
            )

        best_path = excinfo.value.best_model_path
        assert best_path is not None
        assert best_path.endswith("best_model.zip")
        assert Path(best_path).exists()
        assert excinfo.value.partial_result()["best_model_path"] == best_path

        status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
        assert status["status"] == "curriculum_stalled"
        assert status["best_model_path"] == best_path

    def test_installs_entropy_schedule_callback_for_dict_ent_coef(self, tmp_path):
        """When ``stage.ent_coef`` is a ``{start, end, schedule}`` mapping,
        the runner should attach :class:`EntropyScheduleCallback` configured
        against the stage's budget so the coefficient anneals across the
        stage. A constant ``ent_coef`` must NOT install one (we want the
        commitment-phase stages to keep a fixed coefficient).
        """
        stages = [
            # Constant: no schedule callback expected.
            CurriculumStage(
                name="constant",
                map_file="maps/1v1/starter.csv",
                opponent="random",
                promotion_win_rate=0.9,
                patience=2,
                max_timesteps=10_000,
                n_eval_episodes=2,
                ent_coef=0.05,
            ),
            # Schedule mapping: schedule callback expected.
            CurriculumStage(
                name="annealed",
                map_file="maps/1v1/beginner.csv",
                opponent="simple",
                promotion_win_rate=0.9,
                patience=2,
                max_timesteps=20_000,
                n_eval_episodes=2,
                ent_coef={"start": 0.10, "end": 0.03, "schedule": "linear"},
            ),
        ]
        program = {"constant": [0.95, 0.95], "annealed": [0.95, 0.95]}
        cfg, mf, tef, eef, _, _, get_model = _setup_run(stages, program, tmp_path)

        run_curriculum(
            cfg,
            output_dir=tmp_path,
            train_env_factory=tef,
            eval_env_factory=eef,
            model_factory=mf,
        )

        model = get_model()
        # First stage: constant ent_coef -> no schedule callback.
        constant_cbs = model.callbacks_at_learn[0]
        assert not any(isinstance(c, EntropyScheduleCallback) for c in constant_cbs)
        # Second stage: schedule callback installed and configured to span
        # the stage's own budget (not the cumulative timestep counter).
        annealed_cbs = model.callbacks_at_learn[1]
        sched_cb = next((c for c in annealed_cbs if isinstance(c, EntropyScheduleCallback)), None)
        assert sched_cb is not None, "expected EntropyScheduleCallback for dict ent_coef"
        assert sched_cb.start == pytest.approx(0.10)
        assert sched_cb.end == pytest.approx(0.03)
        assert sched_cb.schedule == "linear"
        assert sched_cb.total_timesteps == 20_000
        # The model's ent_coef seeded before learn() should be the
        # schedule's *start* value, so the callback has a sensible
        # initial position to anneal from.
        assert model.ent_coef_at_learn[1] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# EntropyScheduleCallback unit tests
# ---------------------------------------------------------------------------


class _EntropyScheduleHarness:
    """Minimal SB3-compatible model stub for driving EntropyScheduleCallback.

    The real SB3 training loop pokes ``cb.num_timesteps`` directly on the
    callback (``BaseCallback`` keeps it as a plain attribute, not a
    property), and the callback writes ``model.ent_coef`` back. Tests
    set ``cb.num_timesteps`` to drive progress and read
    ``harness.ent_coef`` to verify the schedule.
    """

    def __init__(self) -> None:
        self.ent_coef = 0.0
        self.logger = _StubLogger()


class TestEntropyScheduleCallback:
    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            EntropyScheduleCallback(start=-0.1, end=0.0, total_timesteps=100)
        with pytest.raises(ValueError):
            EntropyScheduleCallback(start=0.1, end=-0.1, total_timesteps=100)
        with pytest.raises(ValueError):
            EntropyScheduleCallback(start=0.1, end=0.0, total_timesteps=0)
        with pytest.raises(ValueError):
            EntropyScheduleCallback(start=0.1, end=0.0, total_timesteps=100, schedule="unknown")

    def test_linear_schedule_endpoints_and_midpoint(self):
        cb = EntropyScheduleCallback(start=0.10, end=0.02, total_timesteps=1000)
        harness = _EntropyScheduleHarness()
        cb.model = harness
        # _on_training_start anchors the stage start step against the
        # cumulative ``num_timesteps`` so progress is per-stage. SB3
        # writes this attribute on the callback during real training;
        # in tests we mutate it directly.
        cb.num_timesteps = 5000  # already non-zero from prior stages
        cb._on_training_start()
        # Step 0 of this stage -> start.
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.10)
        # Halfway through the stage -> linear midpoint.
        cb.num_timesteps = 5000 + 500
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.06)
        # End of stage -> end value.
        cb.num_timesteps = 5000 + 1000
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.02)

    def test_clamps_progress_beyond_budget(self):
        cb = EntropyScheduleCallback(start=0.10, end=0.02, total_timesteps=1000)
        harness = _EntropyScheduleHarness()
        cb.model = harness
        cb.num_timesteps = 0
        cb._on_training_start()
        # Overshoot the budget (can happen if the stage's actual
        # ``learn()`` runs slightly past the announced budget): the
        # coefficient should clamp at ``end``, not extrapolate.
        cb.num_timesteps = 10_000
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.02)

    def test_cosine_schedule_starts_high_ends_low(self):
        cb = EntropyScheduleCallback(start=0.10, end=0.02, total_timesteps=1000, schedule="cosine")
        harness = _EntropyScheduleHarness()
        cb.model = harness
        cb.num_timesteps = 0
        cb._on_training_start()
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.10)
        # Cosine half-cycle midpoint = (start + end) / 2.
        cb.num_timesteps = 500
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.06)
        cb.num_timesteps = 1000
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.02)

    def test_records_value_to_logger(self):
        cb = EntropyScheduleCallback(start=0.10, end=0.02, total_timesteps=1000)
        harness = _EntropyScheduleHarness()
        cb.model = harness
        cb.num_timesteps = 0
        cb._on_training_start()
        cb.num_timesteps = 250
        cb._on_step()
        # Tensorboard plumbing: the live coefficient lands under
        # ``train/ent_coef`` so it shows up alongside SB3's own
        # train/* curves. Lets us see the schedule actually firing
        # in TB without parsing logs.
        assert "train/ent_coef" in harness.logger.records
        assert harness.logger.records["train/ent_coef"] == pytest.approx(harness.ent_coef)


class TestWriteResultsCsv:
    def _read_csv(self, path: Path) -> tuple[list[str], list[dict[str, str]]]:
        import csv as _csv

        with path.open(encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            assert reader.fieldnames is not None
            return list(reader.fieldnames), list(reader)

    def test_writes_one_row_per_eval_with_stage_map_opponent(self, tmp_path):
        # Build a synthetic history shaped like ``run_curriculum`` produces
        # so the test doesn't need the real callbacks / SB3 in the loop.
        history = [
            {
                "stage": "starter_random",
                "map_file": "maps/1v1/starter.csv",
                "opponent": "random",
                "results": [
                    {
                        "timesteps": 50_000,
                        "win_rate": 0.88,
                        "avg_reward": 2276.5,
                        "std_reward": 2716.3,
                        "avg_length": 142.6,
                        "std_length": 30.1,
                        "wins": 53,
                        "losses": 2,
                        "draws": 5,
                        "episodes": 60,
                    },
                    {
                        "timesteps": 100_000,
                        "win_rate": 0.93,
                        "avg_reward": 2893.5,
                        "std_reward": 2122.7,
                        "avg_length": 150.8,
                        "std_length": 32.0,
                        "wins": 56,
                        "losses": 0,
                        "draws": 4,
                        "episodes": 60,
                    },
                ],
            },
            {
                "stage": "starter_simple",
                "map_file": "maps/1v1/starter.csv",
                "opponent": "simple",
                "results": [
                    {
                        "timesteps": 100_004,
                        "win_rate": 1.0,
                        "avg_reward": 1464.1,
                        "std_reward": 0.0,
                        "avg_length": 60.0,
                        "std_length": 0.0,
                        "wins": 60,
                        "losses": 0,
                        "draws": 0,
                        "episodes": 60,
                    },
                ],
            },
        ]
        csv_path = tmp_path / "bootstrap_results.csv"
        _write_results_csv(history, csv_path)
        assert csv_path.is_file()

        fieldnames, rows = self._read_csv(csv_path)
        # Stage / map / opponent columns lead so a glob across runs is
        # self-describing without joining against config.json.
        assert fieldnames[:3] == ["stage", "map_file", "opponent"]
        # One row per eval across both stages.
        assert len(rows) == 3
        assert [r["stage"] for r in rows] == ["starter_random", "starter_random", "starter_simple"]
        assert [r["opponent"] for r in rows] == ["random", "random", "simple"]
        assert all(r["map_file"] == "maps/1v1/starter.csv" for r in rows)
        # Spot-check that the eval payload made it through.
        assert rows[0]["timesteps"] == "50000"
        assert rows[0]["wins"] == "53"
        # Floats are written as Python's default repr (no rounding); the
        # important property is round-trippable, not exact formatting.
        assert float(rows[0]["win_rate"]) == pytest.approx(0.88)

    def test_handles_missing_fields_with_empty_cells(self, tmp_path):
        # Older eval-result schemas lack std / episodes; the writer
        # should leave those cells empty rather than raising.
        history = [
            {
                "stage": "a",
                "map_file": "m.csv",
                "opponent": "random",
                "results": [{"win_rate": 0.5, "avg_reward": 100.0}],
            }
        ]
        csv_path = tmp_path / "out.csv"
        _write_results_csv(history, csv_path)
        _, rows = self._read_csv(csv_path)
        assert len(rows) == 1
        assert rows[0]["std_reward"] == ""
        assert rows[0]["episodes"] == ""

    def test_handles_stage_with_no_results(self, tmp_path):
        # A stage that stalled before its first eval produced no rows;
        # the writer should still emit the header and skip silently.
        history = [{"stage": "a", "map_file": "m.csv", "opponent": "random", "results": []}]
        csv_path = tmp_path / "empty.csv"
        _write_results_csv(history, csv_path)
        fieldnames, rows = self._read_csv(csv_path)
        assert fieldnames[:3] == ["stage", "map_file", "opponent"]
        assert rows == []

    def test_run_curriculum_writes_csv_after_each_stage(self, tmp_path):
        # End-to-end: run_curriculum should drop bootstrap_results.csv at
        # the run root and update it as stages finish.
        stages = [_stage("a", patience=2), _stage("b", "simple", patience=2)]
        program = {"a": [0.95, 0.95], "b": [0.92, 0.93]}
        cfg, mf, tef, eef, _, _, _ = _setup_run(stages, program, tmp_path)
        run_curriculum(
            cfg,
            output_dir=tmp_path,
            train_env_factory=tef,
            eval_env_factory=eef,
            model_factory=mf,
        )
        csv_path = tmp_path / "bootstrap_results.csv"
        assert csv_path.is_file()
        fieldnames, rows = self._read_csv(csv_path)
        assert fieldnames[:3] == ["stage", "map_file", "opponent"]
        # The fake model feeds two evals per stage via the program; both
        # stages promote so the CSV has all four rows.
        assert [r["stage"] for r in rows] == ["a", "a", "b", "b"]
        assert [r["opponent"] for r in rows] == ["random", "random", "simple", "simple"]

    def test_run_curriculum_writes_csv_on_stall(self, tmp_path):
        # If the curriculum stalls partway through, the CSV should still
        # contain rows for every stage that did complete (the stalling
        # stage's evals are appended too because ``history.append`` runs
        # before the stall check).
        stages = [_stage("a", patience=2), _stage("b", "simple", patience=2, threshold=0.99)]
        program = {"a": [0.95, 0.95], "b": [0.5, 0.5]}
        cfg, mf, tef, eef, _, _, _ = _setup_run(stages, program, tmp_path)
        with pytest.raises(CurriculumStalled):
            run_curriculum(
                cfg,
                output_dir=tmp_path,
                train_env_factory=tef,
                eval_env_factory=eef,
                model_factory=mf,
            )
        csv_path = tmp_path / "bootstrap_results.csv"
        assert csv_path.is_file()
        _, rows = self._read_csv(csv_path)
        # Stage 'a' promoted (2 evals); stage 'b' stalled (2 evals but
        # never reached threshold). Both made it into history before the
        # stall raise.
        stages_seen = [r["stage"] for r in rows]
        assert stages_seen.count("a") == 2
        assert stages_seen.count("b") == 2


# ---------------------------------------------------------------------------
# Curriculum-wide pad_to_size resolution
# ---------------------------------------------------------------------------


class TestResolveCurriculumPadSize:
    """Cover the auto-detection that lets a single PPO policy span stages
    with different map sizes (gym_env.set_env rejects shape mismatches)."""

    @staticmethod
    def _stage(name: str, map_file: str) -> CurriculumStage:
        return CurriculumStage(
            name=name,
            map_file=map_file,
            opponent="random",
            promotion_win_rate=0.5,
            patience=1,
            max_timesteps=1,
            n_eval_episodes=1,
        )

    def _cfg(self, stages: list[CurriculumStage], **env_kwargs: Any) -> TrainingConfig:
        env_defaults: dict[str, Any] = {"n_envs": 1, "max_steps": 1, "action_space_type": "flat_discrete"}
        env_defaults.update(env_kwargs)
        return TrainingConfig(
            env=EnvConfig(**env_defaults),
            curriculum=CurriculumConfig(stages=stages),
        )

    def test_returns_none_when_all_maps_same_size(self):
        from reinforcetactics.rl.bootstrap import _resolve_curriculum_pad_size

        cfg = self._cfg(
            [
                self._stage("a", "maps/1v1/starter.csv"),
                self._stage("b", "maps/1v1/starter.csv"),
            ]
        )
        assert _resolve_curriculum_pad_size(cfg) is None

    def test_returns_per_axis_max_for_mixed_sizes(self):
        from reinforcetactics.rl.bootstrap import _resolve_curriculum_pad_size

        # beginner is 6x6, skirmish is 8x8, corner_points is 10x12 (h, w).
        cfg = self._cfg(
            [
                self._stage("a", "maps/1v1/beginner.csv"),
                self._stage("b", "maps/1v1/skirmish.csv"),
                self._stage("c", "maps/1v1/corner_points.csv"),
            ]
        )
        pad = _resolve_curriculum_pad_size(cfg)
        assert pad == (10, 12)

    def test_user_override_is_validated_against_max(self):
        from reinforcetactics.rl.bootstrap import _resolve_curriculum_pad_size

        cfg = self._cfg(
            [
                self._stage("a", "maps/1v1/beginner.csv"),
                self._stage("b", "maps/1v1/corner_points.csv"),
            ],
            pad_to_size=(8, 8),  # too small for corner_points (10, 12)
        )
        with pytest.raises(ValueError, match="smaller than the curriculum's largest map"):
            _resolve_curriculum_pad_size(cfg)

    def test_user_override_with_headroom_is_kept(self):
        from reinforcetactics.rl.bootstrap import _resolve_curriculum_pad_size

        cfg = self._cfg(
            [
                self._stage("a", "maps/1v1/beginner.csv"),
                self._stage("b", "maps/1v1/corner_points.csv"),
            ],
            pad_to_size=(20, 20),
        )
        assert _resolve_curriculum_pad_size(cfg) == (20, 20)

    def test_mixed_sizes_with_multi_discrete_raises(self):
        from reinforcetactics.rl.bootstrap import _resolve_curriculum_pad_size

        cfg = self._cfg(
            [
                self._stage("a", "maps/1v1/beginner.csv"),
                self._stage("b", "maps/1v1/corner_points.csv"),
            ],
            action_space_type="multi_discrete",
        )
        with pytest.raises(ValueError, match="only supported with 'flat_discrete'"):
            _resolve_curriculum_pad_size(cfg)


class TestPaddedEnvAcrossMaps:
    """End-to-end: padded envs on different maps share an obs_space, so
    SB3 ``set_env`` would accept the swap (the actual check that previously
    blocked cross-map curricula at gym_env.py:455)."""

    def test_envs_on_different_maps_share_obs_space_when_padded(self):
        from reinforcetactics.rl.masking import make_maskable_env

        env_small = make_maskable_env(
            map_file="maps/1v1/beginner.csv",
            opponent="noop",
            action_space_type="flat_discrete",
            pad_to_size=(10, 12),
            seed=0,
        )
        env_large = make_maskable_env(
            map_file="maps/1v1/corner_points.csv",
            opponent="noop",
            action_space_type="flat_discrete",
            pad_to_size=(10, 12),
            seed=0,
        )
        assert env_small.observation_space == env_large.observation_space
        # And action_space too — flat_discrete is sized to max_flat_actions,
        # so this should hold even without pad_to_size, but assert it for
        # safety since it's the other half of set_env's check.
        assert env_small.action_space == env_large.action_space

    def test_pad_smaller_than_map_raises(self):
        from reinforcetactics.rl.masking import make_maskable_env

        with pytest.raises(ValueError, match="smaller than the live map"):
            make_maskable_env(
                map_file="maps/1v1/corner_points.csv",
                opponent="noop",
                action_space_type="flat_discrete",
                pad_to_size=(6, 6),
                seed=0,
            )

    def test_pad_with_multi_discrete_raises(self):
        from reinforcetactics.rl.masking import make_maskable_env

        with pytest.raises(NotImplementedError, match="flat_discrete"):
            make_maskable_env(
                map_file="maps/1v1/beginner.csv",
                opponent="noop",
                action_space_type="multi_discrete",
                pad_to_size=(10, 12),
                seed=0,
            )
