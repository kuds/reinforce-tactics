"""The curriculum's promotion gate: what it measures and how it decides.

Review findings covered here:

* rltrain-4 / prior-5: the gate scored the greedy argmax policy
  (``evaluate_model`` defaults to ``deterministic=True`` and
  ``PeriodicEvalCallback`` passed nothing). It now measures
  ``eval.eval_deterministic`` (default False: the stochastic policy PPO
  trains), can evaluate the other mode on the same seeds, and records
  ``win_rate_stochastic`` / ``win_rate_greedy`` plus draws apart from losses.
* rltrain-12 / prior-5: promotion counted consecutive raw point estimates.
  ``PromotionCallback`` now takes ``criterion`` point | wilson | rolling and
  ``score`` win_rate | win_plus_half_draw.
* rltrain-6 / prior-16: ``best_win_rate`` started at -1.0 and only moved when
  a save_dir was set; a stage could promote with best_win_rate -1.0.
* rltrain-19: the promoting eval never reached TensorBoard.
"""

from __future__ import annotations

from typing import Any

import pytest

from reinforcetactics.rl.callbacks import PeriodicEvalCallback, PromotionCallback, gate_statistic
from reinforcetactics.rl.evaluation import evaluate_model, wilson_lower_bound, z_for_confidence

# ---------------------------------------------------------------------------
# Wilson score bound
# ---------------------------------------------------------------------------


class TestWilsonLowerBound:
    @pytest.mark.parametrize(
        ("successes", "n", "expected"),
        [
            # Published 95% (two-sided, z = 1.96) Wilson intervals.
            (8, 10, 0.4902),
            (80, 100, 0.7112),
            (10, 10, 0.7225),
            (0, 10, 0.0),
            (1, 2, 0.0945),
        ],
    )
    def test_known_values_at_z_196(self, successes, n, expected):
        assert wilson_lower_bound(successes, n, 1.959964) == pytest.approx(expected, abs=5e-4)

    def test_one_sided_quantiles(self):
        assert z_for_confidence(0.95) == pytest.approx(1.6449, abs=1e-4)
        assert z_for_confidence(0.975) == pytest.approx(1.95996, abs=1e-4)
        assert z_for_confidence(0.5) == pytest.approx(0.0)
        with pytest.raises(ValueError):
            z_for_confidence(1.0)

    def test_z_zero_is_the_point_estimate_and_bound_grows_with_n(self):
        assert wilson_lower_bound(72, 80, 0.0) == pytest.approx(0.9)
        z = z_for_confidence(0.95)
        assert wilson_lower_bound(18, 20, z) < wilson_lower_bound(72, 80, z) < wilson_lower_bound(720, 800, z) < 0.9

    def test_fractional_successes_and_empty_eval(self):
        # A draw scored as half a win.
        assert wilson_lower_bound(7.5, 10, 0.0) == pytest.approx(0.75)
        assert wilson_lower_bound(0, 0, 1.96) == 0.0


# ---------------------------------------------------------------------------
# Promotion criteria
# ---------------------------------------------------------------------------


def _row(wins: int, episodes: int = 80, draws: int = 0, **extra: Any) -> dict[str, Any]:
    return {
        "win_rate": wins / episodes,
        "wins": wins,
        "draws": draws,
        "losses": episodes - wins - draws,
        "episodes": episodes,
        "avg_reward": 0.0,
        **extra,
    }


def _eval_stub() -> PeriodicEvalCallback:
    stub = PeriodicEvalCallback.__new__(PeriodicEvalCallback)
    stub.results = []  # type: ignore[attr-defined]
    return stub


def _feed(cb: PromotionCallback, rows: list[dict[str, Any]]) -> list[bool]:
    """Append each row and step the callback; the list of 'continue training' returns."""
    out = []
    for i, row in enumerate(rows):
        cb.eval_callback.results.append(row)
        cb.num_timesteps = (i + 1) * 100  # type: ignore[attr-defined]
        out.append(cb._on_step())
        if not out[-1]:
            break
    return out


class TestPromotionCriteria:
    def test_point_is_the_default_and_the_historical_gate(self):
        cb = PromotionCallback(_eval_stub(), threshold=0.9, patience=2, verbose=0)
        assert cb.criterion == "point" and cb.score == "win_rate"
        assert _feed(cb, [_row(76), _row(60), _row(76), _row(76)]) == [True, True, True, False]
        assert cb.promoted

    def test_wilson_needs_the_lower_bound_over_the_threshold(self):
        # 76/80 = 95% clears a 0.9 point gate, but its one-sided 95% Wilson
        # lower bound is ~0.893; 79/80 has a bound of ~0.946.
        z = z_for_confidence(0.95)
        assert wilson_lower_bound(76, 80, z) == pytest.approx(0.893, abs=1e-3)
        assert wilson_lower_bound(79, 80, z) == pytest.approx(0.946, abs=1e-3)

        point = PromotionCallback(_eval_stub(), threshold=0.9, patience=1, verbose=0)
        assert _feed(point, [_row(76)]) == [False]

        wilson = PromotionCallback(_eval_stub(), threshold=0.9, patience=1, verbose=0, criterion="wilson", confidence=0.95)
        assert _feed(wilson, [_row(76), _row(76), _row(79)]) == [True, True, False]
        assert wilson.last_gate_value == pytest.approx(0.946, abs=1e-3)

    def test_rolling_gates_on_the_mean_of_the_last_k_evals(self):
        cb = PromotionCallback(_eval_stub(), threshold=0.9, patience=1, verbose=0, criterion="rolling", rolling_k=3)
        # No pass before k evals, even at 100%; then mean(0.95, 0.80, 0.95) = 0.9.
        assert _feed(cb, [_row(80), _row(64), _row(76)]) == [True, True, False]
        cb = PromotionCallback(_eval_stub(), threshold=0.9, patience=1, verbose=0, criterion="rolling", rolling_k=3)
        # A late dip keeps the mean below the bar: mean(0.95, 0.95, 0.70) = 0.867.
        assert _feed(cb, [_row(76), _row(76), _row(56)]) == [True, True, True]
        assert not cb.promoted

    def test_rolling_window_ignores_pre_window_evals(self):
        cb = PromotionCallback(
            _eval_stub(), threshold=0.9, patience=1, verbose=0, criterion="rolling", rolling_k=2, min_timesteps=250
        )
        # Evals at 100 and 200 are inside min_timesteps: the window starts at 300.
        assert _feed(cb, [_row(80), _row(80), _row(80), _row(80)]) == [True, True, True, False]

    def test_half_draw_score(self):
        row = _row(48, draws=24)  # 60% wins, 30% draws
        assert gate_statistic(row) == pytest.approx(0.6)
        assert gate_statistic(row, score="win_plus_half_draw") == pytest.approx(0.75)
        strict = PromotionCallback(_eval_stub(), threshold=0.7, patience=1, verbose=0)
        assert _feed(strict, [row]) == [True]
        lenient = PromotionCallback(_eval_stub(), threshold=0.7, patience=1, verbose=0, score="win_plus_half_draw")
        assert _feed(lenient, [row]) == [False]

    def test_seat_aggregate_min_takes_the_weaker_seat(self):
        by_seat = {"1": {"wins": 72, "draws": 0, "episodes": 80}, "2": {"wins": 48, "draws": 0, "episodes": 80}}
        row = {**_row(120, episodes=160), "by_seat": by_seat}
        assert gate_statistic(row, seat_aggregate="mean") == pytest.approx(0.75)
        assert gate_statistic(row, seat_aggregate="min") == pytest.approx(0.6)
        z = z_for_confidence(0.9)
        assert gate_statistic(row, criterion="wilson", z=z, seat_aggregate="min") == pytest.approx(
            wilson_lower_bound(48, 80, z)
        )

    def test_row_without_counts_falls_back_to_the_win_rate(self):
        assert gate_statistic({"win_rate": 0.42}, criterion="wilson", z=1.645) == pytest.approx(0.42)
        assert gate_statistic({"win_rate": 0.42, "gate_win_rate": 0.3}) == pytest.approx(0.3)

    def test_rejects_unknown_criterion_and_score(self):
        with pytest.raises(ValueError, match="criterion"):
            PromotionCallback(_eval_stub(), threshold=0.9, criterion="bayes")
        with pytest.raises(ValueError, match="score"):
            PromotionCallback(_eval_stub(), threshold=0.9, score="elo")
        with pytest.raises(ValueError, match="rolling_k"):
            PromotionCallback(_eval_stub(), threshold=0.9, rolling_k=0)

    def test_state_round_trips_for_a_resume(self):
        eval_cb = _eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=3, verbose=0, criterion="rolling", rolling_k=2)
        _feed(cb, [_row(76), _row(76)])
        state = cb.state_dict()
        assert state["streak"] == 1 and len(state["window"]) == 2
        resumed = PromotionCallback(
            eval_cb, threshold=0.9, patience=3, verbose=0, criterion="rolling", rolling_k=2, initial_state=state
        )
        # The two rows already in results are not re-counted; one more pass
        # makes the streak 2, a second makes it 3.
        assert _feed(resumed, [_row(76)]) == [True]
        assert _feed(resumed, [_row(76)]) == [False]


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: dict[str, Any] = {}
        self.dumps: list[int] = []

    def record(self, key: str, value: Any) -> None:
        self.records[key] = value

    def dump(self, step: int = 0) -> None:
        self.dumps.append(step)


class _LoggerModel:
    def __init__(self) -> None:
        self.num_timesteps = 0
        self.logger = _RecordingLogger()
        self.saved: list[str] = []

    def save(self, path: str) -> None:
        self.saved.append(path)
        from pathlib import Path

        Path(path).write_text("fake", encoding="utf-8")


class TestPromotionFlushesTensorBoard:
    def test_promoting_eval_is_dumped_before_learn_returns(self):
        # rltrain-19: SB3 dumps before train(); the step that promotes has no
        # train() after it, so the promoting eval never reached TensorBoard.
        cb = PromotionCallback(_eval_stub(), threshold=0.5, patience=1, verbose=0)
        cb.model = _LoggerModel()  # type: ignore[assignment]
        assert _feed(cb, [_row(60)]) == [False]
        assert cb.model.logger.dumps == [100]
        assert cb.model.logger.records["eval/promoted"] == 1.0

    def test_eval_callback_dumps_at_training_end(self):
        cb = PeriodicEvalCallback(eval_env=object(), eval_freq=100, verbose=0)
        cb.model = _LoggerModel()  # type: ignore[assignment]
        cb.num_timesteps = 700
        cb._on_training_end()
        assert cb.model.logger.dumps == [700]


# ---------------------------------------------------------------------------
# What PeriodicEvalCallback measures and records
# ---------------------------------------------------------------------------


def _fake_evaluate(calls: list[dict[str, Any]], outcomes: dict[bool, tuple[int, int, int]]):
    """A stand-in evaluate_model: (wins, losses, draws) per deterministic flag, per seat when seats are given."""

    def fake(model, env, **kwargs):
        calls.append(kwargs)
        wins, losses, draws = outcomes[kwargs["deterministic"]]
        n = wins + losses + draws
        seats = kwargs.get("seats") or [1]
        by_seat = {
            str(s): {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "episodes": n,
                "win_rate": wins / n,
                "draw_rate": draws / n,
                "loss_rate": losses / n,
                "avg_reward": 0.0,
            }
            for s in seats
        }
        if len(seats) > 1:  # make seat 2 weaker
            by_seat["2"] = {**by_seat["2"], "wins": 0, "losses": n - draws, "win_rate": 0.0}
        total_w = sum(v["wins"] for v in by_seat.values())
        total_d = sum(v["draws"] for v in by_seat.values())
        total_n = n * len(seats)
        return {
            "win_rate": total_w / total_n,
            "avg_reward": float(total_w),
            "std_reward": 0.0,
            "avg_length": 1.0,
            "avg_turns": 1.0,
            "wins": total_w,
            "losses": total_n - total_w - total_d,
            "draws": total_d,
            "episodes": total_n,
            "draw_rate": total_d / total_n,
            "loss_rate": (total_n - total_w - total_d) / total_n,
            "by_seat": by_seat,
        }

    return fake


class TestPeriodicEvalMeasures:
    def _cb(self, monkeypatch, calls, outcomes, **kwargs) -> PeriodicEvalCallback:
        monkeypatch.setattr("reinforcetactics.rl.callbacks.evaluate_model", _fake_evaluate(calls, outcomes))
        cb = PeriodicEvalCallback(eval_env=object(), eval_freq=100, n_eval_episodes=10, verbose=0, **kwargs)
        cb.model = _LoggerModel()  # type: ignore[assignment]
        return cb

    def _eval_at(self, cb: PeriodicEvalCallback, ts: int) -> dict[str, Any]:
        cb.num_timesteps = ts
        cb._last_eval_block = ts // cb.eval_freq
        cb._do_eval()
        return cb.results[-1]

    def test_gate_mode_defaults_to_the_stochastic_policy(self, monkeypatch):
        calls: list[dict[str, Any]] = []
        cb = self._cb(monkeypatch, calls, {False: (6, 3, 1), True: (9, 1, 0)})
        row = self._eval_at(cb, 100)
        assert [c["deterministic"] for c in calls] == [False]
        assert row["deterministic"] is False
        assert row["win_rate"] == pytest.approx(0.6)
        assert row["win_rate_stochastic"] == pytest.approx(0.6)
        assert row["win_rate_greedy"] is None

    def test_both_modes_on_the_same_seeds(self, monkeypatch):
        calls: list[dict[str, Any]] = []
        cb = self._cb(monkeypatch, calls, {False: (6, 3, 1), True: (9, 1, 0)}, eval_both_modes=True, eval_seed_base=7)
        row = self._eval_at(cb, 100)
        assert [(c["deterministic"], c["seed"]) for c in calls] == [(False, 7), (True, 7)]
        # The row itself describes the gate mode; the other mode is alongside.
        assert (row["win_rate"], row["wins"], row["draws"]) == (pytest.approx(0.6), 6, 1)
        assert row["win_rate_stochastic"] == pytest.approx(0.6)
        assert row["win_rate_greedy"] == pytest.approx(0.9)
        assert row["other_mode"]["deterministic"] is True and row["other_mode"]["wins"] == 9
        # Draws are reported on their own, not folded into losses.
        assert row["draw_rate"] == pytest.approx(0.1) and row["loss_rate"] == pytest.approx(0.3)
        records = cb.model.logger.records
        assert records["eval/win_rate_stochastic"] == pytest.approx(0.6)
        assert records["eval/win_rate_greedy"] == pytest.approx(0.9)
        assert records["eval/draw_rate"] == pytest.approx(0.1)

    def test_greedy_gate_when_asked(self, monkeypatch):
        calls: list[dict[str, Any]] = []
        cb = self._cb(monkeypatch, calls, {False: (6, 3, 1), True: (9, 1, 0)}, deterministic=True, eval_both_modes=True)
        row = self._eval_at(cb, 100)
        assert [c["deterministic"] for c in calls] == [True, False]
        assert row["win_rate"] == pytest.approx(0.9)
        assert (row["win_rate_greedy"], row["win_rate_stochastic"]) == (pytest.approx(0.9), pytest.approx(0.6))

    def test_per_seat_rates_and_min_gate(self, monkeypatch):
        calls: list[dict[str, Any]] = []
        cb = self._cb(monkeypatch, calls, {False: (8, 2, 0)}, seats=[1, 2], seat_aggregate="min")
        row = self._eval_at(cb, 100)
        assert calls[0]["seats"] == [1, 2]
        assert row["win_rate_by_seat"] == {"1": pytest.approx(0.8), "2": 0.0}
        assert row["win_rate"] == pytest.approx(0.4)  # pooled
        assert row["gate_win_rate"] == 0.0  # the weaker seat
        assert cb.model.logger.records["eval/win_rate_seat1"] == pytest.approx(0.8)
        assert cb.model.logger.records["eval/win_rate_seat2"] == 0.0

    def test_best_is_none_until_an_eligible_eval_and_tracked_without_save_dir(self, monkeypatch):
        calls: list[dict[str, Any]] = []
        cb = self._cb(monkeypatch, calls, {False: (5, 5, 0)}, best_eligible_after=150)
        assert cb.best_win_rate is None
        cb.num_timesteps = 0
        cb._on_training_start()
        first = self._eval_at(cb, 100)  # carry-in: not eligible
        assert first["best_eligible"] is False
        assert cb.best_win_rate is None
        # ...but the stage's peak counts every eval.
        assert cb.peak_win_rate == pytest.approx(0.5) and cb.peak_timestep == 100
        self._eval_at(cb, 200)
        # No save_dir, and the best is tracked anyway (it used to stay -1.0).
        assert cb.best_win_rate == pytest.approx(0.5) and cb.best_timestep == 200
        assert cb.best_state()["best_win_rate"] == pytest.approx(0.5)

    def test_best_state_carries_across_attempts(self, monkeypatch, tmp_path):
        calls: list[dict[str, Any]] = []
        best = {"best_win_rate": 0.8, "best_reward": 1.0, "best_timestep": 50, "peak_win_rate": 0.9, "peak_timestep": 40}
        cb = self._cb(monkeypatch, calls, {False: (5, 5, 0)}, best_state=best, save_dir=tmp_path)
        row = self._eval_at(cb, 100)
        # A worse eval neither replaces the carried best nor its file.
        assert row["saved_best"] is False and not (tmp_path / "best_model.zip").exists()
        assert (cb.best_win_rate, cb.best_timestep, cb.peak_win_rate) == (0.8, 50, 0.9)

    def test_row_extra_and_hooks(self, monkeypatch):
        calls: list[dict[str, Any]] = []
        cb = self._cb(monkeypatch, calls, {False: (5, 5, 0)}, row_extra={"attempt": 2})
        promote = PromotionCallback(cb, threshold=0.9, verbose=0, criterion="wilson")
        row = self._eval_at(cb, 100)
        assert row["attempt"] == 2
        # The promotion gate stamps its statistic on the row before it is persisted.
        assert row["gate_criterion"] == "wilson"
        assert row["gate_statistic"] == pytest.approx(promote.statistic(row))


# ---------------------------------------------------------------------------
# evaluate_model: draws apart from losses
# ---------------------------------------------------------------------------


class _ScriptedEnv:
    def __init__(self, winners):
        self._winners = list(winners)
        self._idx = -1
        self.agent_player = 1

    def reset(self, seed=None):
        self._idx += 1
        return {"obs": 0}, {}

    def step(self, action):
        return {"obs": 0}, 1.0, True, False, {"episode_stats": {"winner": self._winners[self._idx]}}


class _Model:
    def predict(self, obs, **kwargs):
        return 0, None


def test_evaluate_model_reports_draws_losses_and_outcomes():
    result = evaluate_model(_Model(), _ScriptedEnv([1, None, 2, None]), n_episodes=4)
    assert (result["wins"], result["losses"], result["draws"]) == (1, 1, 2)
    assert (result["draw_rate"], result["loss_rate"]) == (0.5, 0.25)
    assert result["outcomes"] == ["wins", "draws", "losses", "draws"]
    assert result["by_seat"] == {
        "1": {
            "wins": 1,
            "losses": 1,
            "draws": 2,
            "episodes": 4,
            "win_rate": 0.25,
            "loss_rate": 0.25,
            "draw_rate": 0.5,
            "avg_reward": 1.0,
        }
    }
