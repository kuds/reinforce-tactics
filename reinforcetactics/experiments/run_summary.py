"""Read bootstrap run directories and aggregate them across seeds (the §2.1 validation report).

``scripts/eval/summarize_seeds.py`` is the CLI. Everything here reads the raw
JSON / YAML / CSV records a run leaves (no torch, no SB3), so it works on a
laptop with a Drive or GCS copy of the runs, and on the legacy archive:

* **new** run dirs (``resolved_config.yaml``, per-row gate mode, both eval
  modes, ``stage_start_timesteps`` / ``stage_end_timesteps``, retries,
  ``run_status.json`` / ``run_manifest.json``);
* **legacy** run dirs (the 109-run archive: greedy rows without a
  ``deterministic`` key, the source YAML in the run root, per-stage
  ``config.json`` without step bounds, often no ``run_status.json``);
* a bare ``bootstrap_results.csv``, and a row set of the analysis notebook's
  ``runs_per_stage.csv`` (``runs_per_stage.csv:RUN_ID``).

Every reader normalizes to :class:`RunRecord`; :func:`stage_metrics`
computes one run's per-stage numbers (defined in :data:`DEFINITIONS`, which
``summary.json`` repeats), :func:`aggregate` pools them across seeds, and
:func:`compare` sets a group against a baseline run or group.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reinforcetactics.experiments.seed_runs import parse_run_dir_name, steps_per_hour
from reinforcetactics.experiments.stats import describe, finite, median, wilson_interval

SCHEMA_VERSION = 1
REWARD_COMPONENTS = ("action", "shaping_delta", "invalid_penalty", "terminal")
NON_TERMINAL = ("action", "shaping_delta", "invalid_penalty")
CAPTURE_TYPES = ("tower", "building", "hq")
END_REASONS = ("hq_capture", "elimination", "max_turns_draw", "max_steps_truncate")
OUTCOMES = ("wins", "draws", "losses")
# Replicates may differ only here (and in what resume records as nonmaterial).
REPLICATE_IGNORED = ("seed", "ppo.device", "logging", "total_timesteps", "algorithm")
# A legacy record (written before the eval-gate change) that does not set
# these meant the historical values (see bootstrap.LEGACY_RECORD_DEFAULTS).
LEGACY_EVAL_DEFAULTS = {"eval_deterministic": True, "eval_both_modes": False}
# Merge dates of the fixes a pre-change baseline lacks (docs/REVIEW_full_2026-09-26.md status table).
EVAL_GATE_CHANGE_DATE = "2026-09-27"
MAX_STEPS_TRUNCATE_FLAG = 0.05

DEFINITIONS: dict[str, str] = {
    "final_row": "The stage's last eval row by (attempt, timesteps). For a cleared stage it is the promoting eval: "
    "gate-selected, so its win rate is biased upward near the threshold ('at gate').",
    "stochastic / greedy W/D/L": "From the final row: the row's own counts when its 'deterministic' flag is the "
    "requested mode, else its other_mode counts when those are, else missing. A row without 'deterministic' "
    "(legacy) is greedy. Rates carry a two-sided Wilson interval at --confidence.",
    "window": "Pooled counts of the last `patience` rows of the last attempt (secondary).",
    "steps_to_promotion": "Cleared stages: stage_end_timesteps - stage_start_timesteps (every attempt; exact). "
    "Legacy fallback: last row timesteps - first row timesteps (approximate). Stalled or interrupted "
    "stages report trained_steps instead, marked censored.",
    "cum_steps_end": "Cumulative env steps when the stage ended (stage_end_timesteps, else its last eval row).",
    "retries": "retries_used from the stage record; legacy runs had no retry feature (0).",
    "skip_ahead": "Cleared with steps_to_promotion <= patience * eval_freq: the carried-in policy passed the gate "
    "almost at once.",
    "captures_per_ep": "The agent's captures by structure type per episode in the final row's gate-mode eval.",
    "opponent_captures_per_ep": "Structures the opponent seized per episode (neutral / the agent's), new logging only.",
    "reward_per_ep": "reward_components / episodes of the final row (gate mode); flagged when their sum differs "
    "from avg_reward.",
    "shaping_share_abs": "(|A|+|S|+|I|) / (|A|+|S|+|I|+|T|) from the final row's eval-level sums of the action (A), "
    "shaping_delta (S), invalid_penalty (I) and terminal (T) components. Headline; archive-comparable.",
    "shaping_share_signed": "(A+S+I) / (A+S+I+T); None when |A+S+I+T| < 1 per episode.",
    "episode_abs_share": "The abs share from per-episode magnitudes (reward_components_abs; new logging only).",
    "by_outcome": "Non-terminal, terminal and total return per episode for wins / draws / losses "
    "(reward_components_by_outcome; new logging only).",
    "draw_breakeven": "A draw returns >= 0 on average: from by_outcome or the per-episode rewards and outcomes "
    "(new rows); for legacy rows, when even the lowest possible draw total (the row's total return minus its "
    "largest wins+losses episode returns) is >= 0. Counted over every eval row of the stage.",
    "end_reason_rates": "end_reasons / episodes of the final row.",
    "flat_truncated_rate": "Share of decision points whose flat_discrete table was cut to max_flat_actions (final row).",
    "peak_gate_wr": "The stage record's peak_win_rate, else the highest gate (or plain) win rate of its rows.",
    "eval_resampled": "eval_seed varies within one attempt (resample_eval_seeds, as in the archive).",
    "steps_per_hour": "Median of delta-timesteps / delta-wall-time over consecutive rows of one session "
    "(gaps under 1 h); legacy fallback: stage records' created_at.",
    "eval_share": "Sum of the rows' eval_seconds over the wall time they span (new logging only).",
    "seed_sensitive": "A stage at least one seed cleared and at least one seed stalled on.",
}

# Per-stage numbers pooled across seeds (per_stage.csv / the across-seed table).
AGG_METRICS: tuple[str, ...] = (
    "steps_to_promotion",
    "trained_steps",
    "cum_steps_end",
    "retries",
    "stoch_win_rate",
    "stoch_draw_rate",
    "stoch_loss_rate",
    "greedy_win_rate",
    "greedy_draw_rate",
    "greedy_loss_rate",
    "window_stoch_win_rate",
    "window_greedy_win_rate",
    "peak_gate_wr",
    "captures_per_ep_tower",
    "captures_per_ep_building",
    "captures_per_ep_hq",
    "captures_per_ep_total",
    "opponent_captures_per_ep_neutral",
    "opponent_captures_per_ep_owned",
    "reward_per_ep_action",
    "reward_per_ep_shaping_delta",
    "reward_per_ep_invalid_penalty",
    "reward_per_ep_terminal",
    "shaping_share_abs",
    "shaping_share_signed",
    "episode_abs_share",
    "draw_return_per_ep",
    "end_reason_rate_hq_capture",
    "end_reason_rate_elimination",
    "end_reason_rate_max_turns_draw",
    "end_reason_rate_max_steps_truncate",
    "flat_truncated_rate",
    "steps_per_hour",
    "eval_share",
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class StageRecord:
    name: str
    index: int
    settings: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    rows_source: str | None = None
    has_record: bool = False
    outcome_hint: str | None = None


@dataclass
class RunRecord:
    run_id: str
    path: str
    layout: str
    status: str
    seed: int | None = None
    group: str | None = None
    git: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] | None = None
    config_source: str | None = None
    run_status: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    stages: list[StageRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"s{self.seed}" if self.seed is not None else self.run_id


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a last line cut short by a kill
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_yaml(path: Path) -> dict[str, Any] | None:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    return data if isinstance(data, dict) else None


def _num(value: Any) -> Any:
    """A CSV cell as int / float / bool / None (empty), else the string."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text in ("True", "False", "true", "false"):
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError:
        return text
    return number if math.isfinite(number) else None


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for raw in reader:
            row = {(k or "").replace("\\_", "_").strip(): _num(v) for k, v in raw.items()}
            rows.append({k: v for k, v in row.items() if v is not None})
    return rows


def _stage_settings(stage: Mapping[str, Any], config: Mapping[str, Any] | None) -> dict[str, Any]:
    """A stage's identity and gate from a raw config (resolved or source YAML), with env / eval defaults applied."""
    config = config or {}
    env = config.get("env") or {}
    ev = config.get("eval") or {}
    max_turns = stage.get("max_turns") if stage.get("max_turns") is not None else env.get("max_turns")
    n_eval = stage.get("n_eval_episodes") if stage.get("n_eval_episodes") is not None else ev.get("n_eval_episodes")
    return {
        "map_file": stage.get("map_file"),
        "opponent": stage.get("opponent"),
        "opponent_kwargs": stage.get("opponent_kwargs") or {},
        "max_turns": max_turns,
        "promotion_win_rate": stage.get("promotion_win_rate"),
        "patience": stage.get("patience"),
        "max_timesteps": stage.get("max_timesteps"),
        "n_eval_episodes": n_eval,
        "eval_freq": ev.get("eval_freq"),
    }


def _apply_record_settings(settings: dict[str, Any], record: Mapping[str, Any]) -> None:
    """Fill a stage's settings from its config.json (what actually ran), keeping config values it lacks."""
    env_cfg = record.get("env_config") or {}
    extra = record.get("extra") or {}
    if record.get("map_file") or env_cfg.get("map_file"):
        settings["map_file"] = env_cfg.get("map_file") or record.get("map_file")
    if record.get("opponent"):
        settings["opponent"] = record.get("opponent")
    if "opponent_kwargs" in env_cfg:
        settings["opponent_kwargs"] = env_cfg.get("opponent_kwargs") or {}
    if env_cfg.get("max_turns") is not None:
        settings["max_turns"] = env_cfg.get("max_turns")
    for key in ("promotion_win_rate", "patience", "max_timesteps", "n_eval_episodes", "eval_freq"):
        if extra.get(key) is not None:
            settings[key] = extra[key]


def _run_status_kind(status: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> str:
    kind = status.get("status")
    if kind == "completed_curriculum":
        return "completed"
    if kind == "curriculum_stalled":
        return "stalled"
    if kind:
        return str(kind)
    # A killed process cannot report: a run with progress records but no
    # run_status.json was interrupted; one with neither (the archive) aborted.
    return "interrupted" if manifest else "aborted"


def _source_yaml(run_dir: Path) -> Path | None:
    candidates = sorted(
        p
        for p in run_dir.glob("*.y*ml")
        if p.suffix in (".yaml", ".yml") and not p.name.startswith("resolved_config") and not p.name.startswith(".")
    )
    return candidates[0] if candidates else None


def read_run_dir(run_dir: str | Path) -> RunRecord:
    """Read a train_bootstrap.py / notebook run directory (new or legacy layout)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"not a run directory: {run_dir}")
    notes: list[str] = []
    config = _read_yaml(run_dir / "resolved_config.yaml")
    config_source = "resolved_config.yaml" if config is not None else None
    if config is None:
        source_yaml = _source_yaml(run_dir)
        if source_yaml is not None:
            try:
                config = _read_yaml(source_yaml)
                config_source = source_yaml.name
            except Exception as exc:  # noqa: BLE001 - an archived YAML the current loader may not accept
                notes.append(f"could not read {source_yaml.name}: {exc}")
    run_status = _read_json(run_dir / "run_status.json")
    run_status = run_status if isinstance(run_status, dict) else {}
    manifest = _read_json(run_dir / "run_manifest.json")
    manifest = manifest if isinstance(manifest, dict) else {}
    csv_rows: dict[str, list[dict[str, Any]]] = {}
    csv_order: list[str] = []
    if (run_dir / "bootstrap_results.csv").is_file():
        for row in _read_csv_rows(run_dir / "bootstrap_results.csv"):
            name = str(row.get("stage", ""))
            if name not in csv_rows:
                csv_order.append(name)
            csv_rows.setdefault(name, []).append(row)

    config_stages = [s for s in ((config or {}).get("curriculum") or {}).get("stages") or [] if isinstance(s, dict)]
    names = [str(s.get("name")) for s in config_stages if s.get("name")]
    if not names:
        names = list(csv_order) or [str(n) for n in manifest.get("stages") or []]
    # Stage dirs the config does not list (should not happen) go last, in the order they ran.
    extra_dirs = []
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and child.name not in names and (child / "config.json").is_file():
            extra_dirs.append(child.name)
    names += extra_dirs
    by_name = {str(s.get("name")): s for s in config_stages}

    stages: list[StageRecord] = []
    seed = config.get("seed") if config_source == "resolved_config.yaml" and config else None
    git: dict[str, Any] = {}
    for index, name in enumerate(names):
        stage_dir = run_dir / name
        record = _read_json(stage_dir / "config.json") if (stage_dir / "config.json").is_file() else None
        record = record if isinstance(record, dict) else None
        settings = _stage_settings(by_name.get(name, {}), config)
        extra: dict[str, Any] = {}
        meta: dict[str, Any] = {}
        if record is not None:
            _apply_record_settings(settings, record)
            extra = dict(record.get("extra") or {})
            meta = dict(record.get("meta") or {})
            if seed is None and record.get("seed") is not None:
                seed = record.get("seed")
            if not git and meta.get("git"):
                git = dict(meta["git"])
            settings["reward_config"] = (record.get("env_config") or {}).get("reward_config")
        rows: list[dict[str, Any]] = []
        source: str | None = None
        results = _read_json(stage_dir / "eval_results.json") if (stage_dir / "eval_results.json").is_file() else None
        if isinstance(results, list):
            rows, source = [r for r in results if isinstance(r, dict)], "eval_results.json"
        else:
            jsonl = _read_jsonl(stage_dir / "eval_results.jsonl")
            if jsonl:
                rows, source = jsonl, "eval_results.jsonl"
            elif csv_rows.get(name):
                rows, source = csv_rows[name], "bootstrap_results.csv"
        stages.append(
            StageRecord(
                name=name,
                index=index,
                settings=settings,
                extra=extra,
                meta=meta,
                rows=rows,
                rows_source=source,
                has_record=record is not None,
            )
        )
    if seed is None and config is not None and config.get("seed") is not None:
        seed = config.get("seed")
        notes.append("seed taken from the source YAML")
    layout = "new" if config_source == "resolved_config.yaml" else "legacy"
    if layout == "legacy" and any("deterministic" in r for s in stages for r in s.rows):
        layout = "new"
    parsed = parse_run_dir_name(run_dir.name)
    return RunRecord(
        run_id=run_dir.name,
        path=str(run_dir),
        layout=layout,
        status=_run_status_kind(run_status, manifest),
        seed=int(seed) if isinstance(seed, (int, float)) and not isinstance(seed, bool) else None,
        group=parsed[0] if parsed else None,
        git=git,
        config=config,
        config_source=config_source,
        run_status=run_status,
        manifest=manifest,
        stages=stages,
        notes=notes,
    )


def read_results_csv(path: str | Path) -> RunRecord:
    """A run known only from its ``bootstrap_results.csv`` (no config, no stage records)."""
    path = Path(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _read_csv_rows(path):
        grouped.setdefault(str(row.get("stage", "")), []).append(row)
    names = list(grouped)
    stages = []
    for index, name in enumerate(names):
        rows = grouped[name]
        first = rows[0]
        hint = "cleared" if index < len(names) - 1 else None
        if hint is None and rows and isinstance(rows[-1].get("gate_passed"), bool):
            hint = "cleared" if rows[-1]["gate_passed"] else "unknown"
        stages.append(
            StageRecord(
                name=name,
                index=index,
                settings={"map_file": first.get("map_file"), "opponent": first.get("opponent"), "opponent_kwargs": {}},
                rows=rows,
                rows_source="bootstrap_results.csv",
                outcome_hint=hint or "unknown",
            )
        )
    run_id = path.parent.name or path.stem
    return RunRecord(
        run_id=run_id,
        path=str(path),
        layout="csv",
        status="unknown",
        stages=stages,
        notes=["bootstrap_results.csv only: no config, stage records or run status"],
    )


def read_runs_per_stage(path: str | Path, run_id: str) -> RunRecord:
    """One run's rows of the analysis notebook's ``runs_per_stage.csv`` (final / peak win rates, outcome)."""
    path = Path(path)
    rows = [r for r in _read_csv_rows(path) if str(r.get("run_id")) == str(run_id)]
    if not rows:
        raise ValueError(f"{path} has no rows for run_id {run_id!r}")
    stages = []
    for index, r in enumerate(rows):
        kwargs = {
            key: r[col]
            for key, col in (
                ("max_actions", "opponent_max_actions"),
                ("p_hard", "opponent_p_hard"),
                ("easy", "opponent_easy"),
                ("hard", "opponent_hard"),
            )
            if r.get(col) is not None
        }
        outcome = str(r.get("outcome") or "unknown")
        # Only the stage's last eval survives in this table (its win rate,
        # greedy for the archive); a stand-in row carries it.
        final_row = {
            "timesteps": r.get("last_step"),
            "win_rate": r.get("final_win_rate"),
            "avg_reward": r.get("final_avg_reward"),
            "avg_turns": r.get("final_avg_turns"),
        }
        stages.append(
            StageRecord(
                name=str(r.get("stage_name")),
                index=index,
                settings={
                    "map_file": r.get("map_file"),
                    "opponent": r.get("opponent"),
                    "opponent_kwargs": kwargs,
                    "max_turns": r.get("max_turns"),
                    "promotion_win_rate": r.get("promotion_win_rate"),
                    "patience": r.get("patience"),
                    "max_timesteps": r.get("max_timesteps"),
                    "n_eval_episodes": r.get("n_eval_episodes"),
                },
                rows=[final_row] if r.get("final_win_rate") is not None else [],
                rows_source="runs_per_stage.csv",
                outcome_hint={"not_started": "not_reached"}.get(outcome, outcome),
                extra={"peak_win_rate": r.get("peak_win_rate"), "steps_in_stage": r.get("steps_in_stage")},
            )
        )
    return RunRecord(
        run_id=str(run_id),
        path=f"{path}:{run_id}",
        layout="runs_per_stage",
        status="unknown",
        seed=None,
        stages=stages,
        notes=["runs_per_stage.csv: final and peak win rates only (greedy for archive runs); steps approximate"],
    )


def read_run(spec: str | Path) -> RunRecord:
    """A run dir, a ``bootstrap_results.csv``, or ``runs_per_stage.csv:RUN_ID``."""
    text = str(spec)
    base, sep, run_id = text.rpartition(":")
    if sep and base.endswith(".csv") and run_id and Path(base).is_file():
        return read_runs_per_stage(base, run_id)
    path = Path(text)
    if path.is_file() and path.suffix == ".csv":
        return read_results_csv(path)
    return read_run_dir(path)


# ---------------------------------------------------------------------------
# Per-stage metrics of one run
# ---------------------------------------------------------------------------


def _sorted_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(rows, key=lambda r: (int(r.get("attempt") or 0), int(r.get("timesteps") or 0)))


def row_counts(row: Mapping[str, Any] | None, *, stochastic: bool) -> dict[str, Any] | None:
    """W/D/L of ``row`` in the requested mode (the row's own, else its other_mode's), or None.

    A row without ``deterministic`` is a legacy (greedy) row.
    """
    if not row:
        return None
    want_det = not stochastic
    gate_det = row.get("deterministic", True)
    if gate_det is None:
        gate_det = True
    source: Mapping[str, Any] | None = None
    if bool(gate_det) == want_det:
        source = row
    else:
        other = row.get("other_mode")
        if isinstance(other, Mapping) and bool(other.get("deterministic")) == want_det:
            source = other
    if source is None:
        return None
    wins, losses, draws, episodes = (source.get(k) for k in ("wins", "losses", "draws", "episodes"))
    if episodes is None and wins is not None and losses is not None and draws is not None:
        episodes = int(wins) + int(losses) + int(draws)
    if draws is None and wins is not None and losses is not None and episodes is not None:
        draws = int(episodes) - int(wins) - int(losses)
    if losses is None and wins is not None and draws is not None and episodes is not None:
        losses = int(episodes) - int(wins) - int(draws)
    win_rate = source.get("win_rate")
    if win_rate is None and wins is not None and episodes:
        win_rate = int(wins) / int(episodes)
    if win_rate is None:
        return None
    out: dict[str, Any] = {"wins": wins, "draws": draws, "losses": losses, "episodes": episodes, "win_rate": float(win_rate)}
    if episodes:
        out["draw_rate"] = int(draws) / int(episodes) if draws is not None else None
        out["loss_rate"] = int(losses) / int(episodes) if losses is not None else None
    else:
        out["draw_rate"] = source.get("draw_rate")
        out["loss_rate"] = source.get("loss_rate")
    return out


def _pooled(rows: Sequence[Mapping[str, Any]], *, stochastic: bool) -> dict[str, Any] | None:
    counts: list[dict[str, Any]] = []
    for r in rows:
        c = row_counts(r, stochastic=stochastic)
        if c is None or not c.get("episodes") or c.get("wins") is None:
            return None
        counts.append(c)
    if not counts:
        return None
    n = sum(int(c["episodes"]) for c in counts)
    wins = sum(int(c["wins"]) for c in counts)
    return {"wins": wins, "episodes": n, "win_rate": wins / n if n else None}


def _share_abs(components: Mapping[str, Any] | None) -> float | None:
    if not components:
        return None
    nt = sum(abs(float(components.get(c) or 0.0)) for c in NON_TERMINAL)
    total = nt + abs(float(components.get("terminal") or 0.0))
    return nt / total if total > 0 else None


def _share_signed(components: Mapping[str, Any] | None, episodes: int) -> float | None:
    if not components or not episodes:
        return None
    nt = sum(float(components.get(c) or 0.0) for c in NON_TERMINAL)
    total = nt + float(components.get("terminal") or 0.0)
    if abs(total) < 1.0 * episodes:
        return None
    return nt / total


def draw_return(row: Mapping[str, Any]) -> tuple[float | None, bool]:
    """``(mean return of the row's draws, exact)``; for legacy rows the lowest value it can have.

    New rows give it exactly (by_outcome, or the rewards and outcomes lists).
    A legacy row has the per-episode rewards but not which episode drew: the
    draws' total is at least the row's total minus its ``wins + losses``
    largest episode returns, so that bound is returned (``exact`` False).
    """
    by_outcome = row.get("reward_components_by_outcome")
    if isinstance(by_outcome, Mapping):
        draws = by_outcome.get("draws") or {}
        n = int(draws.get("episodes") or 0)
        if n:
            return sum(float(draws.get(c) or 0.0) for c in REWARD_COMPONENTS) / n, True
        return None, True
    rewards, outcomes = row.get("rewards"), row.get("outcomes")
    if isinstance(rewards, list) and isinstance(outcomes, list) and len(rewards) == len(outcomes):
        values = [float(r) for r, o in zip(rewards, outcomes, strict=True) if o == "draws"]
        return (sum(values) / len(values), True) if values else (None, True)
    draws_n = row.get("draws")
    if not draws_n:
        return None, True
    decisive = int(row.get("wins") or 0) + int(row.get("losses") or 0)
    if isinstance(rewards, list) and len(rewards) == decisive + int(draws_n):
        ordered = sorted((float(r) for r in rewards), reverse=True)
        return sum(ordered[decisive:]) / int(draws_n), decisive == 0
    if decisive == 0 and row.get("avg_reward") is not None:
        return float(row["avg_reward"]), True
    return None, False


def _stage_outcome(run: RunRecord, stage: StageRecord, later_reached: bool) -> str:
    promoted = stage.extra.get("promoted")
    if promoted is True:
        return "cleared"
    if promoted is False:
        return "stalled"
    if stage.outcome_hint:
        return stage.outcome_hint
    if not stage.rows:
        return "not_reached"
    if run.status == "stalled" and run.run_status.get("stalled_stage") == stage.name:
        return "stalled"
    if later_reached or run.status == "completed":
        # The curriculum only advances past a stage that promoted.
        return "cleared"
    if run.status in ("interrupted", "aborted"):
        return "interrupted"
    return "unknown"


def _parse_time(text: Any) -> float | None:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def stage_metrics(
    run: RunRecord, stage: StageRecord, *, confidence: float = 0.95, later_reached: bool = False
) -> dict[str, Any]:
    """One run's numbers for one stage (see :data:`DEFINITIONS`)."""
    rows = _sorted_rows(stage.rows)
    outcome = _stage_outcome(run, stage, later_reached)
    settings = stage.settings
    m: dict[str, Any] = {
        "stage": stage.name,
        "index": stage.index,
        "map_file": settings.get("map_file"),
        "opponent": settings.get("opponent"),
        "outcome": outcome,
        "reached": outcome != "not_reached",
        "cleared": outcome == "cleared",
        "n_evals": len(rows),
        "rows_source": stage.rows_source,
        "label": "at gate",
    }
    if not m["reached"]:
        return m
    final = rows[-1] if rows else None
    if final is None and stage.rows:
        final = stage.rows[-1]
    gate_det = final.get("deterministic", True) if final else True
    m["gate_mode"] = "greedy" if gate_det in (True, None) else "stochastic"
    for mode, stochastic in (("stoch", True), ("greedy", False)):
        c = row_counts(final, stochastic=stochastic)
        for key in ("wins", "draws", "losses", "episodes", "win_rate", "draw_rate", "loss_rate"):
            m[f"{mode}_{key}"] = c.get(key) if c else None
        if c and c.get("episodes"):
            lo, hi = wilson_interval(int(c["wins"]), int(c["episodes"]), confidence)
            m[f"{mode}_wr_lo"], m[f"{mode}_wr_hi"] = lo, hi
        else:
            m[f"{mode}_wr_lo"] = m[f"{mode}_wr_hi"] = None
    patience = int(settings.get("patience") or stage.extra.get("patience") or 1)
    last_attempt = max((int(r.get("attempt") or 0) for r in rows), default=0)
    window = [r for r in rows if int(r.get("attempt") or 0) == last_attempt][-patience:]
    for mode, stochastic in (("stoch", True), ("greedy", False)):
        pooled = _pooled(window, stochastic=stochastic) if window else None
        m[f"window_{mode}_win_rate"] = pooled["win_rate"] if pooled else None
        m[f"window_{mode}_episodes"] = pooled["episodes"] if pooled else None

    # Steps.
    extra = stage.extra
    start, end = extra.get("stage_start_timesteps"), extra.get("stage_end_timesteps")
    first_ts = int(rows[0]["timesteps"]) if rows and rows[0].get("timesteps") is not None else None
    last_ts = int(rows[-1]["timesteps"]) if rows and rows[-1].get("timesteps") is not None else None
    if start is not None and end is not None:
        steps, exact = int(end) - int(start), True
    elif extra.get("steps_in_stage") is not None:
        steps, exact = int(extra["steps_in_stage"]), False
    elif first_ts is not None and last_ts is not None:
        steps, exact = last_ts - first_ts, False
    else:
        steps, exact = None, False
    if outcome == "stalled" and run.run_status.get("stalled_stage") == stage.name:
        trained = run.run_status.get("trained_timesteps")
        if trained is not None:
            steps, exact = int(trained), True
    m["steps_exact"] = exact
    m["steps_to_promotion"] = steps if m["cleared"] else None
    m["trained_steps"] = steps
    m["censored"] = not m["cleared"]
    m["cum_steps_end"] = int(end) if end is not None else last_ts
    if "retries_used" in extra:
        m["retries"] = int(extra.get("retries_used") or 0)
    else:
        m["retries"] = 0
        m["retries_note"] = "no retry feature"
    eval_freq = settings.get("eval_freq") or extra.get("eval_freq") or ((run.config or {}).get("eval") or {}).get("eval_freq")
    m["skip_ahead"] = bool(m["cleared"] and steps is not None and eval_freq and steps <= patience * int(eval_freq))

    # Reward and captures of the final row (the gate-mode eval).
    episodes = int(final.get("episodes") or 0) if final else 0
    comps = final.get("reward_components") if final else None
    caps = final.get("captures_by_type") if final else None
    total_caps = 0.0
    for key in CAPTURE_TYPES:
        value = (float(caps.get(key) or 0) / episodes) if (isinstance(caps, Mapping) and episodes) else None
        m[f"captures_per_ep_{key}"] = value
        total_caps += value or 0.0
    m["captures_per_ep_total"] = total_caps if isinstance(caps, Mapping) and episodes else None
    opp = final.get("opponent_captures") if final else None
    for key in ("neutral", "owned"):
        m[f"opponent_captures_per_ep_{key}"] = (
            float(opp.get(key) or 0) / episodes if isinstance(opp, Mapping) and episodes else None
        )
    for key in REWARD_COMPONENTS:
        m[f"reward_per_ep_{key}"] = (
            float(comps.get(key) or 0.0) / episodes if isinstance(comps, Mapping) and episodes else None
        )
    m["reward_sum_mismatch"] = False
    if isinstance(comps, Mapping) and episodes and final and final.get("avg_reward") is not None:
        total = sum(float(comps.get(c) or 0.0) for c in REWARD_COMPONENTS) / episodes
        avg = float(final["avg_reward"])
        m["reward_sum_mismatch"] = abs(total - avg) > max(0.01, 1e-4 * abs(avg))
    m["shaping_share_abs"] = _share_abs(comps) if isinstance(comps, Mapping) else None
    m["shaping_share_signed"] = _share_signed(comps, episodes) if isinstance(comps, Mapping) else None
    m["episode_abs_share"] = _share_abs(final.get("reward_components_abs")) if final else None
    by_outcome = final.get("reward_components_by_outcome") if final else None
    m["by_outcome"] = None
    if isinstance(by_outcome, Mapping):
        m["by_outcome"] = {}
        for outcome_name in OUTCOMES:
            b = by_outcome.get(outcome_name) or {}
            n = int(b.get("episodes") or 0)
            if not n:
                m["by_outcome"][outcome_name] = {"episodes": 0}
                continue
            nt = sum(float(b.get(c) or 0.0) for c in NON_TERMINAL) / n
            term = float(b.get("terminal") or 0.0) / n
            m["by_outcome"][outcome_name] = {
                "episodes": n,
                "non_terminal_per_ep": nt,
                "terminal_per_ep": term,
                "return_per_ep": nt + term,
            }
    draw_ret, exact_draw = draw_return(final) if final else (None, True)
    m["draw_return_per_ep"] = draw_ret
    m["draw_return_exact"] = exact_draw
    m["draw_breakeven"] = bool(draw_ret is not None and draw_ret >= 0)
    breakeven_rows = []
    for r in rows:
        value, _ = draw_return(r)
        if value is not None and value >= 0:
            breakeven_rows.append(int(r.get("timesteps") or 0))
    m["draw_breakeven_evals"] = len(breakeven_rows)
    m["draw_breakeven_at"] = breakeven_rows[:5]
    reasons = final.get("end_reasons") if final else None
    for key in END_REASONS:
        m[f"end_reason_rate_{key}"] = (
            float(reasons.get(key) or 0) / episodes if isinstance(reasons, Mapping) and episodes else None
        )
    m["flat_truncated_rate"] = final.get("flat_truncated_rate") if final else None
    peak = extra.get("peak_win_rate")
    if peak is None:
        gates = finite(r.get("gate_win_rate", r.get("win_rate")) for r in stage.rows)
        peak = max(gates) if gates else None
    m["peak_gate_wr"] = peak
    by_attempt: dict[int, set] = {}
    for r in rows:
        if r.get("eval_seed") is not None:
            by_attempt.setdefault(int(r.get("attempt") or 0), set()).add(r["eval_seed"])
    m["eval_resampled"] = any(len(v) > 1 for v in by_attempt.values())
    m["steps_per_hour"] = steps_per_hour(rows)
    # Eval time over the wall time between consecutive rows of one session
    # (a row's eval ran just before its wall_time stamp).
    timed = sorted(
        (float(r["wall_time"]), float(r.get("eval_seconds") or 0.0))
        for r in rows
        if isinstance(r.get("wall_time"), (int, float))
    )
    spent = span = 0.0
    for (t0, _), (t1, secs) in zip(timed, timed[1:], strict=False):
        if 0 < t1 - t0 < 3600.0:
            spent, span = spent + secs, span + (t1 - t0)
    m["eval_share"] = spent / span if span > 0 else None
    m["created_at"] = stage.meta.get("created_at")
    return m


def run_metrics(run: RunRecord, *, confidence: float = 0.95) -> dict[str, Any]:
    """Every stage's metrics and the run-level summary of one run."""
    reached_flags = [bool(s.rows) or s.extra.get("promoted") is not None or bool(s.outcome_hint) for s in run.stages]
    if run.layout == "runs_per_stage":
        reached_flags = [s.outcome_hint not in (None, "not_reached") for s in run.stages]
    stages = []
    for i, stage in enumerate(run.stages):
        later = any(reached_flags[i + 1 :])
        stages.append(stage_metrics(run, stage, confidence=confidence, later_reached=later))
    reached = [s for s in stages if s["reached"]]
    cleared = [s for s in stages if s["cleared"]]
    stalled = [s for s in stages if s["outcome"] == "stalled"]
    cum = finite(s.get("cum_steps_end") for s in stages)
    # Wall clock: from the eval rows' wall_time, else the stage records' created_at.
    walls = sorted(float(r["wall_time"]) for s in run.stages for r in s.rows if isinstance(r.get("wall_time"), (int, float)))
    wall_h = active_h = None
    wall_source = None
    if len(walls) > 1:
        wall_h = (walls[-1] - walls[0]) / 3600.0
        active_h = sum(b - a for a, b in zip(walls, walls[1:], strict=False) if 0 < b - a < 3600.0) / 3600.0
        wall_source = "eval rows"
    else:
        created = sorted(t for t in (_parse_time(s.meta.get("created_at")) for s in run.stages) if t is not None)
        if len(created) > 1:
            wall_h = (created[-1] - created[0]) / 3600.0
            wall_source = "stage records (approx.; excludes the first stage)"
    # Steps/h per map: a stage's own rate, else from consecutive stage records.
    rates: dict[str, list[float]] = {}
    prev_t = prev_cum = None
    for m, stage in zip(stages, run.stages, strict=True):
        rate = m.get("steps_per_hour")
        t = _parse_time(stage.meta.get("created_at"))
        cum_end = m.get("cum_steps_end")
        if rate is None and t is not None and prev_t is not None and cum_end is not None and prev_cum is not None:
            dt = t - prev_t
            if dt > 0 and cum_end > prev_cum:
                rate = (cum_end - prev_cum) / dt * 3600.0
        if rate is not None and m.get("map_file"):
            rates.setdefault(str(m["map_file"]), []).append(rate)
        if t is not None:
            prev_t, prev_cum = t, cum_end
    status = run.run_status
    manifest = run.manifest
    return {
        "run_id": run.run_id,
        "label": run.label,
        "path": run.path,
        "seed": run.seed,
        "group": run.group,
        "layout": run.layout,
        "status": run.status,
        "gate_mode": next((s["gate_mode"] for s in reached if s.get("gate_mode")), None),
        "stages_total": len(stages),
        "stages_reached": len(reached),
        "stages_cleared": len(cleared),
        "deepest_stage": reached[-1]["stage"] if reached else None,
        "stalled_stage": status.get("stalled_stage") or (stalled[-1]["stage"] if stalled else None),
        "total_steps": int(max(cum)) if cum else None,
        "wall_clock_h": wall_h,
        "active_h": active_h,
        "wall_clock_source": wall_source,
        "resume_count": int(status.get("resume_count", manifest.get("resume_count", 0)) or 0),
        "retries_used": sum(int(s.get("retries") or 0) for s in stages),
        "metadata_write_failures": status.get("metadata_write_failures", manifest.get("metadata_write_failures")),
        "git": run.git.get("short") or (run.git.get("commit") or "")[:7] or None,
        "created_at": next((s.meta.get("created_at") for s in run.stages if s.meta.get("created_at")), None),
        "steps_per_hour_by_map": {k: median(v) for k, v in sorted(rates.items())},
        "notes": list(run.notes),
        "stages": stages,
    }


# ---------------------------------------------------------------------------
# Replicates, aggregation across seeds
# ---------------------------------------------------------------------------


def _diff_paths(a: Any, b: Any, prefix: str, out: list[str]) -> None:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in sorted(set(a) | set(b), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                out.append(path)
            else:
                _diff_paths(a[key], b[key], path, out)
    elif isinstance(a, list) and isinstance(b, list) and all(isinstance(x, Mapping) for x in a + b):
        if len(a) != len(b):
            out.append(f"{prefix} ({len(a)} -> {len(b)} entries)")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _diff_paths(x, y, f"{prefix}[{i}]", out)
    elif a != b:
        out.append(prefix)


def replicate_differences(runs: Sequence[RunRecord]) -> tuple[list[str], list[str]]:
    """``(differences, notes)``: where the runs' resolved configs differ beyond seed, device, logging and labels."""
    notes: list[str] = []
    resolved = [r for r in runs if r.config_source == "resolved_config.yaml" and r.config is not None]
    missing = [r.run_id for r in runs if r not in resolved]
    if missing:
        notes.append(f"no resolved_config.yaml to check: {', '.join(missing)}")
    if len(resolved) < 2:
        return [], notes
    ref = resolved[0]
    diffs: list[str] = []
    for other in resolved[1:]:
        found: list[str] = []
        _diff_paths(ref.config, other.config, "", found)
        for path in found:
            if any(path == p or path.startswith(p + ".") or path.startswith(p + "[") for p in REPLICATE_IGNORED):
                continue
            diffs.append(f"{ref.label} vs {other.label}: {path}")
    return diffs, notes


def _stage_union(orders: Iterable[Sequence[str]]) -> list[str]:
    """Stage names of every run, in curriculum order (a name new to the union goes after its predecessor)."""
    union: list[str] = []
    for order in orders:
        prev: str | None = None
        for name in order:
            if name not in union:
                union.insert(union.index(prev) + 1 if prev is not None and prev in union else len(union), name)
            prev = name
    return union


def aggregate(run_summaries: Sequence[Mapping[str, Any]], *, label: str = "group") -> dict[str, Any]:
    """Per stage across runs: reached / cleared counts and describe() of every :data:`AGG_METRICS` value."""
    order = _stage_union([s["stage"] for s in r["stages"]] for r in run_summaries)
    stages = []
    for index, name in enumerate(order):
        per_run = [(r["label"], next((s for s in r["stages"] if s["stage"] == name), None)) for r in run_summaries]
        reached = [(lab, s) for lab, s in per_run if s is not None and s["reached"]]
        entry: dict[str, Any] = {
            "stage": name,
            "index": index,
            "map_file": next((s.get("map_file") for _, s in reached if s.get("map_file")), None),
            "opponent": next((s.get("opponent") for _, s in reached if s.get("opponent")), None),
            "n_runs": len(run_summaries),
            "n_reached": len(reached),
            "n_cleared": sum(1 for _, s in reached if s["cleared"]),
            "n_stalled": sum(1 for _, s in reached if s["outcome"] == "stalled"),
            "outcomes": {lab: (s["outcome"] if s is not None else "not_reached") for lab, s in per_run},
            "metrics": {},
        }
        for metric in AGG_METRICS:
            values = {lab: s.get(metric) for lab, s in reached}
            if metric == "steps_to_promotion":
                values = {lab: s.get(metric) for lab, s in reached if s["cleared"]}
            stats = describe(list(values.values()))
            stats["values"] = values
            entry["metrics"][metric] = stats
        entry["seed_sensitive"] = entry["n_cleared"] > 0 and entry["n_stalled"] > 0
        stages.append(entry)
    return {"label": label, "n_runs": len(run_summaries), "stage_order": order, "stages": stages}


# ---------------------------------------------------------------------------
# Comparison against a baseline run or group
# ---------------------------------------------------------------------------

COMPARABLE_FIELDS = ("map_file", "opponent", "opponent_kwargs", "max_turns", "promotion_win_rate", "patience")


def _norm_opponent(name: Any) -> Any:
    if not isinstance(name, str):
        return name
    key = name.strip().lower()
    if key == "bot":
        return "simple"
    if key.endswith("bot") and len(key) > 3:
        return key[:-3]
    return key


def _norm_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _norm_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0])) if v is not None}
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return _norm_opponent(value)
    if isinstance(value, list):
        return [_norm_value(v) for v in value]
    return value


def _norm_setting(key: str, value: Any) -> Any:
    if key == "opponent":
        return _norm_opponent(value)
    if key == "opponent_kwargs":
        return _norm_value(value or {})
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def stage_comparability(base: Mapping[str, Any] | None, group: Mapping[str, Any] | None) -> tuple[bool, list[str], list[str]]:
    """``(comparable, differing fields, unverified fields)`` of two stages' settings."""
    base, group = base or {}, group or {}
    differing, unverified = [], []
    for key in COMPARABLE_FIELDS:
        a, b = base.get(key), group.get(key)
        if key != "opponent_kwargs" and (a is None or b is None):
            unverified.append(key)
            continue
        if _norm_setting(key, a) != _norm_setting(key, b):
            differing.append(f"{key}: {a!r} vs {b!r}")
    return not differing, differing, unverified


def _get(d: Mapping[str, Any] | None, *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _gate_mode_of(config: Mapping[str, Any] | None, *, legacy: bool) -> str | None:
    if config is None:
        return None
    ev = config.get("eval") or {}
    det = ev.get("eval_deterministic")
    if det is None:
        det = True if legacy else False
    return "greedy" if det else "stochastic"


def config_deltas(
    base: Mapping[str, Any] | None, group: Mapping[str, Any] | None, *, base_legacy: bool
) -> list[dict[str, Any]]:
    """Setting differences between a baseline config and the group's (raw dicts; ``None`` = unset)."""
    if base is None or group is None:
        return []
    out: list[dict[str, Any]] = []

    def add(path: str, a: Any, b: Any) -> None:
        if _norm_value(a) != _norm_value(b):
            out.append({"field": path, "baseline": a, "group": b})

    rc_a = _get(base, "env", "reward_config") or {}
    rc_b = _get(group, "env", "reward_config") or {}
    for key in sorted(set(rc_a) | set(rc_b)):
        add(f"env.reward_config.{key}", rc_a.get(key), rc_b.get(key))
    for key in ("max_steps", "max_turns", "max_actions_per_turn", "max_flat_actions", "engine_overrides", "action_space_type"):
        add(f"env.{key}", _get(base, "env", key), _get(group, "env", key))
    ppo_a, ppo_b = base.get("ppo") or {}, group.get("ppo") or {}
    for key in sorted(set(ppo_a) | set(ppo_b)):
        if key == "device":
            continue
        add(f"ppo.{key}", ppo_a.get(key), ppo_b.get(key))
    add("gate mode", _gate_mode_of(base, legacy=base_legacy), _gate_mode_of(group, legacy=False))
    for key in ("eval_freq", "n_eval_episodes", "resample_eval_seeds"):
        add(f"eval.{key}", _get(base, "eval", key), _get(group, "eval", key))
    names_a = [s.get("name") for s in (_get(base, "curriculum", "stages") or []) if isinstance(s, Mapping)]
    names_b = [s.get("name") for s in (_get(group, "curriculum", "stages") or []) if isinstance(s, Mapping)]
    only_a = [n for n in names_a if n not in names_b]
    only_b = [n for n in names_b if n not in names_a]
    if only_a:
        out.append({"field": "stages only in the baseline", "baseline": only_a, "group": None})
    if only_b:
        out.append({"field": "stages only in the group", "baseline": None, "group": only_b})
    shared_a = [n for n in names_a if n in names_b]
    shared_b = [n for n in names_b if n in names_a]
    if shared_a != shared_b:
        out.append({"field": "order of the shared stages", "baseline": shared_a, "group": shared_b})
    return out


def _stage_settings_map(runs: Sequence[RunRecord]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for run in runs:
        for stage in run.stages:
            out.setdefault(stage.name, dict(stage.settings))
    return out


@dataclass
class Side:
    """One side of a comparison: a group (or a single run as a group of one)."""

    label: str
    agg: dict[str, Any]
    runs: list[dict[str, Any]]
    settings: dict[str, dict[str, Any]]
    config: dict[str, Any] | None
    legacy: bool
    git: str | None
    created_at: str | None
    source: str


def side_from_runs(label: str, records: Sequence[RunRecord], summaries: Sequence[Mapping[str, Any]], source: str) -> Side:
    first = records[0] if records else None
    return Side(
        label=label,
        agg=aggregate(summaries, label=label),
        runs=[dict(s) for s in summaries],
        settings=_stage_settings_map(records),
        config=first.config if first else None,
        legacy=bool(first and first.layout != "new"),
        git=summaries[0].get("git") if summaries else None,
        created_at=summaries[0].get("created_at") if summaries else None,
        source=source,
    )


def side_from_summary(label: str, summary: Mapping[str, Any], source: str) -> Side:
    group = summary.get("group") or {}
    stages = summary.get("stages") or []
    agg = {"label": label, "n_runs": group.get("n_runs"), "stage_order": [s["stage"] for s in stages], "stages": list(stages)}
    runs = [dict(r) for r in summary.get("runs") or []]
    return Side(
        label=label,
        agg=agg,
        runs=runs,
        settings=dict(group.get("stage_settings") or {}),
        config=group.get("config"),
        legacy=False,
        git=runs[0].get("git") if runs else None,
        created_at=runs[0].get("created_at") if runs else None,
        source=source,
    )


def _metric(stage: Mapping[str, Any] | None, name: str, stat: str = "mean") -> Any:
    if not stage:
        return None
    return ((stage.get("metrics") or {}).get(name) or {}).get(stat)


def compare(baseline: Side, group: Side) -> dict[str, Any]:
    """The comparison block of the report for ``group`` against ``baseline`` (stages matched by name)."""
    base_stages = {s["stage"]: s for s in baseline.agg["stages"]}
    rows = []
    for name in group.agg["stage_order"]:
        if name not in base_stages:
            continue
        b = base_stages[name]
        g = next(s for s in group.agg["stages"] if s["stage"] == name)
        comparable, differing, unverified = stage_comparability(baseline.settings.get(name), group.settings.get(name))
        b_steps = _metric(b, "steps_to_promotion", "median")
        g_steps = _metric(g, "steps_to_promotion", "median")
        b_greedy = _metric(b, "greedy_win_rate")
        g_greedy = _metric(g, "greedy_win_rate")
        b_share = _metric(b, "shaping_share_abs")
        g_share = _metric(g, "shaping_share_abs")
        rows.append(
            {
                "stage": name,
                "comparable": comparable,
                "differing": differing,
                "unverified": unverified,
                "baseline_n_runs": b.get("n_runs"),
                "baseline_n_reached": b.get("n_reached"),
                "baseline_n_cleared": b.get("n_cleared"),
                "baseline_outcomes": b.get("outcomes"),
                "baseline_steps": b_steps,
                "baseline_trained_steps": _metric(b, "trained_steps", "median"),
                "baseline_greedy_wr": b_greedy,
                "baseline_greedy_draw_rate": _metric(b, "greedy_draw_rate"),
                "baseline_captures_per_ep": _metric(b, "captures_per_ep_total"),
                "baseline_shaping_share_abs": b_share,
                "baseline_cum_steps": _metric(b, "cum_steps_end"),
                "group_n_runs": g.get("n_runs"),
                "group_n_reached": g.get("n_reached"),
                "group_n_cleared": g.get("n_cleared"),
                "group_steps_median": g_steps,
                "group_steps_min": _metric(g, "steps_to_promotion", "min"),
                "group_steps_max": _metric(g, "steps_to_promotion", "max"),
                "group_greedy_wr": g_greedy,
                "group_greedy_wr_min": _metric(g, "greedy_win_rate", "min"),
                "group_greedy_wr_max": _metric(g, "greedy_win_rate", "max"),
                "group_stoch_wr": _metric(g, "stoch_win_rate"),
                "group_stoch_wr_min": _metric(g, "stoch_win_rate", "min"),
                "group_stoch_wr_max": _metric(g, "stoch_win_rate", "max"),
                "group_greedy_draw_rate": _metric(g, "greedy_draw_rate"),
                "group_captures_per_ep": _metric(g, "captures_per_ep_total"),
                "group_shaping_share_abs": g_share,
                "delta_greedy_wr": (g_greedy - b_greedy) if g_greedy is not None and b_greedy is not None else None,
                "steps_ratio": (g_steps / b_steps) if g_steps is not None and b_steps else None,
                "delta_shaping_share_abs": (g_share - b_share) if g_share is not None and b_share is not None else None,
            }
        )
    shared = [r for r in rows if r["baseline_n_reached"] and r["group_n_reached"]]
    order = group.agg["stage_order"]
    deepest = max(shared, key=lambda r: order.index(r["stage"]))["stage"] if shared else None
    names = [r["stage"] for r in rows]
    caveats = []
    if baseline.git:
        caveats.append(f"baseline code: git {baseline.git}" + (f" (group: {group.git})" if group.git else ""))
    if baseline.legacy:
        created = (baseline.created_at or "")[:10]
        caveats.append(
            "the baseline is a legacy record (no per-row gate mode): it predates the eval-gate change"
            + (
                ", and the opponent-freeze fix, engine and bot legality and the MDP fixes"
                if created and created < EVAL_GATE_CHANGE_DATE
                else " (check its git hash against the opponent-freeze, legality and MDP fixes)"
            )
            + (f"; created {created}" if created else "")
        )
    base_modes = {r.get("gate_mode") for r in baseline.runs if r.get("gate_mode")}
    if base_modes == {"greedy"}:
        caveats.append("the baseline gated on the greedy policy: compare greedy win rates only")
    if any(st.get("eval_resampled") for r in baseline.runs for st in r.get("stages") or [] if isinstance(st, Mapping)):
        caveats.append("the baseline resampled its eval set every eval block")
    caveats.append("numbers are the promoting (gate-selected) evals, 'at gate'; treat the comparison as qualitative")
    return {
        "label": baseline.label,
        "source": baseline.source,
        "n_baseline_runs": len(baseline.runs),
        "config_deltas": config_deltas(baseline.config, group.config, base_legacy=baseline.legacy),
        "caveats": caveats,
        "stages": rows,
        "shared_stages": names,
        "deepest_shared_stage_reached": deepest,
        "cleared_among_shared": {
            "baseline": {
                r.get("label"): sum(1 for s in r.get("stages") or [] if s.get("cleared") and s.get("stage") in names)
                for r in baseline.runs
            },
            "group": {
                r.get("label"): sum(1 for s in r.get("stages") or [] if s.get("cleared") and s.get("stage") in names)
                for r in group.runs
            },
        },
    }


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def collect_flags(runs: Sequence[Mapping[str, Any]], agg: Mapping[str, Any]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for run in runs:
        if run.get("metadata_write_failures"):
            flags.append(
                {
                    "kind": "metadata_write_failures",
                    "run": run["label"],
                    "stage": None,
                    "detail": str(run["metadata_write_failures"]),
                }
            )
        if run.get("resume_count"):
            flags.append(
                {
                    "kind": "resumed",
                    "run": run["label"],
                    "stage": None,
                    "detail": f"{run['resume_count']} resume(s); not bit-reproducible",
                }
            )
        for s in run["stages"]:
            if not s["reached"]:
                continue
            if s.get("skip_ahead"):
                flags.append(
                    {
                        "kind": "skip_ahead",
                        "run": run["label"],
                        "stage": s["stage"],
                        "detail": f"cleared in {s.get('steps_to_promotion'):,} env steps "
                        "(<= patience x eval_freq: the carried-in policy passed the gate almost at once)",
                    }
                )
            if s.get("draw_breakeven_evals"):
                exact = "" if s.get("draw_return_exact", True) else " (legacy lower bound)"
                flags.append(
                    {
                        "kind": "draw_breakeven",
                        "run": run["label"],
                        "stage": s["stage"],
                        "detail": f"{s['draw_breakeven_evals']} eval(s) where a draw returned >= 0{exact}; e.g. at {s.get('draw_breakeven_at')}",
                    }
                )
            if (s.get("flat_truncated_rate") or 0) > 0:
                flags.append(
                    {
                        "kind": "flat_truncated",
                        "run": run["label"],
                        "stage": s["stage"],
                        "detail": f"flat_truncated_rate {s['flat_truncated_rate']:.4f}",
                    }
                )
            rate = s.get("end_reason_rate_max_steps_truncate")
            if rate is not None and rate > MAX_STEPS_TRUNCATE_FLAG:
                flags.append(
                    {
                        "kind": "max_steps_truncate",
                        "run": run["label"],
                        "stage": s["stage"],
                        "detail": f"{rate:.1%} of final-eval episodes truncated at max_steps",
                    }
                )
            if s.get("reward_sum_mismatch"):
                flags.append(
                    {
                        "kind": "reward_sum_mismatch",
                        "run": run["label"],
                        "stage": s["stage"],
                        "detail": "reward components do not sum to avg_reward",
                    }
                )
    for st in agg["stages"]:
        if st.get("seed_sensitive"):
            detail = ", ".join(f"{lab}: {o}" for lab, o in st["outcomes"].items())
            flags.append({"kind": "seed_sensitive", "run": None, "stage": st["stage"], "detail": detail})
    return flags


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _pct(x: Any, digits: int = 0) -> str:
    return "—" if x is None else f"{100 * float(x):.{digits}f}%"


def _num_fmt(x: Any, digits: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer() and abs(x) >= 1000):
        return f"{int(x):,}"
    return f"{float(x):.{digits}f}"


def _steps(x: Any) -> str:
    if x is None:
        return "—"
    x = float(x)
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def _range(stats: Mapping[str, Any] | None, fmt: Any, centre: str = "mean") -> str:
    if not stats or stats.get(centre) is None:
        return "—"
    if stats.get("n", 0) <= 1 or stats.get("min") == stats.get("max"):
        return fmt(stats[centre])
    return f"{fmt(stats[centre])} [{fmt(stats['min'])}–{fmt(stats['max'])}]"


def _wdl(s: Mapping[str, Any], mode: str) -> str:
    if s.get(f"{mode}_win_rate") is None:
        return "—"
    if s.get(f"{mode}_wins") is None:
        return _pct(s[f"{mode}_win_rate"])
    ci = ""
    if s.get(f"{mode}_wr_lo") is not None:
        ci = f" [{_pct(s[f'{mode}_wr_lo'])}–{_pct(s[f'{mode}_wr_hi'])}]"
    return f"{s[f'{mode}_wins']}/{s[f'{mode}_draws']}/{s[f'{mode}_losses']} {_pct(s[f'{mode}_win_rate'])}{ci}"


def _caps(s: Mapping[str, Any]) -> str:
    if s.get("captures_per_ep_total") is None:
        return "—"
    return "/".join(f"{float(s.get(f'captures_per_ep_{k}') or 0):.1f}" for k in CAPTURE_TYPES)


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def render_report(summary: Mapping[str, Any]) -> str:
    """``report.md`` from a :func:`build_summary` result."""
    group = summary["group"]
    runs = summary["runs"]
    stages = summary["stages"]
    out: list[str] = []
    title = group.get("id") or "seed runs"
    out.append(f"# Seed validation report: {title}")
    out.append("")
    out.append(f"Generated {summary['generated_at']} by scripts/eval/summarize_seeds.py (schema {summary['schema_version']}).")
    out.append("")
    out.append("## Provenance and gate settings")
    out.append("")
    gate = group.get("gate") or {}
    prov = [
        ("runs", ", ".join(f"{r['label']} ({r['run_id']})" for r in runs)),
        ("config", f"{group.get('config_path') or '—'} (digest {str(group.get('config_digest') or '—')[:12]})"),
        ("git", ", ".join(sorted({str(r.get("git")) for r in runs if r.get("git")})) or "—"),
        (
            "gate",
            f"{gate.get('mode') or '—'} policy; both modes recorded: {_num_fmt(gate.get('eval_both_modes'))}; "
            f"criterion {gate.get('promotion_criterion') or '—'}; eval_freq {_num_fmt(gate.get('eval_freq'))}; "
            f"n_eval_episodes {_num_fmt(gate.get('n_eval_episodes'))}; seats {gate.get('eval_seats') or '—'}; "
            f"n_eval_envs {_num_fmt(gate.get('n_eval_envs'))}",
        ),
        ("replicate check", group["replicate_check"]["verdict"]),
        (
            "numbers",
            f"the final (promoting) eval of each stage, 'at gate' (gate-selected, biased upward near the "
            f"threshold); rates with two-sided {int(round(100 * summary['inputs']['confidence']))}% Wilson intervals",
        ),
    ]
    out += _table(["", ""], prov)
    for diff in group["replicate_check"]["differences"][:20]:
        out.append(f"- config difference: {diff}")
    for note in group["replicate_check"]["notes"]:
        out.append(f"- note: {note}")
    out.append("")

    out.append("## 1. Per-seed outcome")
    out.append("")
    out += _table(
        [
            "seed",
            "status",
            "cleared",
            "deepest stage",
            "stalled at",
            "env steps",
            "wall h",
            "resumes",
            "retries",
            "meta fails",
            "gate",
        ],
        (
            [
                r["label"],
                r["status"],
                f"{r['stages_cleared']}/{r['stages_total']}",
                r.get("deepest_stage") or "—",
                r.get("stalled_stage") or "—",
                _steps(r.get("total_steps")),
                _num_fmt(r.get("wall_clock_h"), 1),
                r.get("resume_count", 0),
                r.get("retries_used", 0),
                "—" if r.get("metadata_write_failures") is None else r["metadata_write_failures"],
                r.get("gate_mode") or "—",
            ]
            for r in runs
        ),
    )
    out.append("")

    out.append("## 2. Per stage across seeds")
    out.append("")
    out.append(
        "Mean [min–max] over the seeds that reached the stage; steps: median [min–max] over the seeds that cleared it. "
        "Captures per episode are tower/building/HQ in the gate-mode eval. t-intervals are in per_stage.csv and summary.json."
    )
    out.append("")
    header = [
        "#",
        "stage",
        "cleared/reached",
        "steps to promote",
        "stoch WR",
        "greedy WR",
        "stoch draw",
        "captures/ep",
        "shaping abs",
        "stoch WR per seed",
    ]
    table_rows = []
    for st in stages:
        met = st["metrics"]
        caps = "/".join(
            "—"
            if (met.get(f"captures_per_ep_{k}") or {}).get("mean") is None
            else f"{met[f'captures_per_ep_{k}']['mean']:.1f}"
            for k in CAPTURE_TYPES
        )
        per_seed = " ".join(f"{lab}={_pct(v)}" for lab, v in (met["stoch_win_rate"].get("values") or {}).items())
        table_rows.append(
            [
                st["index"] + 1,
                st["stage"],
                f"{st['n_cleared']}/{st['n_reached']}" + (" ⚠" if st.get("seed_sensitive") else ""),
                _range(met["steps_to_promotion"], _steps, "median"),
                _range(met["stoch_win_rate"], _pct),
                _range(met["greedy_win_rate"], _pct),
                _range(met["stoch_draw_rate"], _pct),
                caps,
                _range(met["shaping_share_abs"], lambda x: f"{x:.2f}"),
                per_seed or "—",
            ]
        )
    out += _table(header, table_rows)
    out.append("")

    out.append("## 3. Per-seed detail")
    for r in runs:
        out.append("")
        out.append(f"### {r['label']} — {r['run_id']} ({r['status']}, {r['layout']} layout)")
        out.append("")
        detail = []
        for s in r["stages"]:
            if not s["reached"]:
                continue
            steps = s.get("steps_to_promotion") if s["cleared"] else s.get("trained_steps")
            steps_txt = _steps(steps) + ("" if s["cleared"] else " (censored)") + ("" if s.get("steps_exact") else " ~")
            detail.append(
                [
                    s["index"] + 1,
                    s["stage"],
                    s["outcome"],
                    steps_txt,
                    _wdl(s, "stoch"),
                    _wdl(s, "greedy"),
                    s.get("retries", 0),
                    _caps(s),
                    "—" if s.get("shaping_share_abs") is None else f"{s['shaping_share_abs']:.2f}",
                    "—"
                    if s.get("draw_return_per_ep") is None
                    else f"{s['draw_return_per_ep']:+.1f}" + ("" if s.get("draw_return_exact", True) else " (≥)"),
                    _steps(s.get("cum_steps_end")),
                ]
            )
        out += _table(
            [
                "#",
                "stage",
                "outcome",
                "steps",
                "stoch W/D/L",
                "greedy W/D/L",
                "retries",
                "captures/ep T/B/H",
                "shaping abs",
                "draw return/ep",
                "cum steps",
            ],
            detail,
        )
        by_map = r.get("steps_per_hour_by_map") or {}
        if by_map:
            out.append("")
            out.append("Steps/h by map: " + ", ".join(f"{Path(k).stem} {_steps(v)}" for k, v in by_map.items()))
        for note in r.get("notes") or []:
            out.append(f"- note: {note}")
    out.append("")

    out.append("## 4. Comparison")
    out.append("")
    if not summary["comparisons"]:
        out.append("No --compare baseline given.")
        out.append("")
    for comp in summary["comparisons"]:
        out.append(f"### Against {comp['label']} ({comp['n_baseline_runs']} run(s))")
        out.append("")
        for c in comp["caveats"]:
            out.append(f"- {c}")
        out.append(
            f"- deepest shared stage reached: {comp['deepest_shared_stage_reached'] or '—'}; stages cleared among the "
            f"{len(comp['shared_stages'])} shared: baseline "
            + ", ".join(f"{k}={v}" for k, v in comp["cleared_among_shared"]["baseline"].items())
            + "; group "
            + ", ".join(f"{k}={v}" for k, v in comp["cleared_among_shared"]["group"].items())
        )
        out.append("")
        if comp["config_deltas"]:
            out.append("Config deltas (baseline → group):")
            out.append("")
            out += _table(
                ["field", "baseline", "group"],
                ([d["field"], json.dumps(d["baseline"]), json.dumps(d["group"])] for d in comp["config_deltas"]),
            )
            out.append("")
        rows = []
        for r in comp["stages"]:
            if r["baseline_n_runs"] == 1:
                b_out = next(iter((r.get("baseline_outcomes") or {}).values()), "—")
            else:
                b_out = f"{r['baseline_n_cleared']}/{r['baseline_n_reached']}"
            b_steps = r["baseline_steps"] if r["baseline_steps"] is not None else r.get("baseline_trained_steps")
            b_steps_txt = _steps(b_steps) + ("" if r["baseline_steps"] is not None or b_steps is None else " (censored)")
            g_steps = (
                "—"
                if r["group_steps_median"] is None
                else (
                    _steps(r["group_steps_median"])
                    + (
                        ""
                        if r["group_steps_min"] == r["group_steps_max"]
                        else f" [{_steps(r['group_steps_min'])}–{_steps(r['group_steps_max'])}]"
                    )
                )
            )
            rows.append(
                [
                    r["stage"] + ("" if r["comparable"] else " ✗"),
                    b_out,
                    b_steps_txt,
                    _pct(r["baseline_greedy_wr"]),
                    _pct(r["baseline_greedy_draw_rate"]),
                    _num_fmt(r["baseline_captures_per_ep"], 1),
                    _num_fmt(r["baseline_shaping_share_abs"]),
                    f"{r['group_n_cleared']}/{r['group_n_runs']}",
                    g_steps,
                    _range(
                        {
                            "mean": r["group_greedy_wr"],
                            "min": r["group_greedy_wr_min"],
                            "max": r["group_greedy_wr_max"],
                            "n": 2,
                        },
                        _pct,
                    ),
                    _range(
                        {"mean": r["group_stoch_wr"], "min": r["group_stoch_wr_min"], "max": r["group_stoch_wr_max"], "n": 2},
                        _pct,
                    ),
                    _pct(r["group_greedy_draw_rate"]),
                    _num_fmt(r["group_captures_per_ep"], 1),
                    _num_fmt(r["group_shaping_share_abs"]),
                    "—" if r["delta_greedy_wr"] is None else f"{100 * r['delta_greedy_wr']:+.0f} pp",
                    "—" if r["steps_ratio"] is None else f"{r['steps_ratio']:.2f}×",
                    "—" if r["delta_shaping_share_abs"] is None else f"{r['delta_shaping_share_abs']:+.2f}",
                ]
            )
        out += _table(
            [
                "stage",
                "base outcome",
                "base steps",
                "base greedy WR",
                "base greedy draw",
                "base captures/ep",
                "base shaping",
                "cleared",
                "steps median",
                "greedy WR",
                "stoch WR",
                "greedy draw",
                "captures/ep",
                "shaping",
                "Δ greedy WR",
                "steps ratio",
                "Δ shaping",
            ],
            rows,
        )
        incomparable = [r for r in comp["stages"] if not r["comparable"]]
        if incomparable:
            out.append("")
            out.append("✗ not comparable (settings differ):")
            for r in incomparable:
                out.append(f"- {r['stage']}: {'; '.join(r['differing'])}")
        unverified = sorted({f for r in comp["stages"] for f in r["unverified"]})
        if unverified:
            out.append(f"- unverified (a side does not record it): {', '.join(unverified)}")
        out.append("")

    out.append("## 5. Flags")
    out.append("")
    if not summary["flags"]:
        out.append("None.")
    for f in summary["flags"]:
        where = " / ".join(x for x in (f.get("run"), f.get("stage")) if x)
        out.append(f"- **{f['kind']}** {where}: {f['detail']}")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The whole summary, and its files
# ---------------------------------------------------------------------------


def _generated_at() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch and epoch.strip().isdigit():
        return datetime.fromtimestamp(int(epoch), UTC).isoformat(timespec="seconds")
    return datetime.now(UTC).isoformat(timespec="seconds")


def _gate_settings(config: Mapping[str, Any] | None, legacy: bool) -> dict[str, Any]:
    config = config or {}
    ev = config.get("eval") or {}
    cur = config.get("curriculum") or {}
    return {
        "mode": _gate_mode_of(config, legacy=legacy) if config else None,
        "eval_both_modes": ev.get("eval_both_modes", LEGACY_EVAL_DEFAULTS["eval_both_modes"] if legacy else None),
        "promotion_criterion": cur.get("promotion_criterion"),
        "eval_freq": ev.get("eval_freq"),
        "n_eval_episodes": ev.get("n_eval_episodes"),
        "eval_seats": ev.get("eval_seats"),
        "n_eval_envs": ev.get("n_eval_envs"),
        "resample_eval_seeds": ev.get("resample_eval_seeds"),
    }


def build_summary(
    records: Sequence[RunRecord],
    *,
    confidence: float = 0.95,
    group_id: str | None = None,
    config_path: str | None = None,
    config_digest: str | None = None,
    baselines: Sequence[Side] = (),
    inputs: Mapping[str, Any] | None = None,
    allow_mixed: bool = False,
) -> dict[str, Any]:
    """Everything the report and the files show (``summary.json``'s content)."""
    ordered = sorted(records, key=lambda r: (r.seed is None, r.seed if r.seed is not None else 0, r.run_id))
    summaries = [run_metrics(r, confidence=confidence) for r in ordered]
    diffs, notes = replicate_differences(ordered)
    if diffs:
        verdict = f"FAILED: {len(diffs)} difference(s) beyond seed/device/logging" + (
            " (allowed by --allow-mixed)" if allow_mixed else ""
        )
    elif len([r for r in ordered if r.config_source == "resolved_config.yaml"]) >= 2:
        verdict = "ok: the runs' resolved configs differ only in seed, device, logging and labels"
    else:
        verdict = "not checked (fewer than two resolved configs)"
    side = side_from_runs(group_id or "group", ordered, summaries, source="inputs")
    agg = side.agg
    first = ordered[0] if ordered else None
    config = dict(first.config) if first and first.config else None
    if config is not None:
        config.pop("seed", None)
    group = {
        "id": group_id,
        "config_path": config_path or (first.config_source if first else None),
        "config_digest": config_digest,
        "n_runs": len(ordered),
        "replicate_check": {"verdict": verdict, "differences": diffs, "notes": notes, "passed": not diffs},
        "gate": _gate_settings(first.config if first else None, legacy=bool(first and first.layout != "new")),
        "stage_order": agg["stage_order"],
        "stage_settings": side.settings,
        "config": config,
    }
    comparisons = [compare(b, side) for b in baselines]
    flags = collect_flags(summaries, agg)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "inputs": {"confidence": confidence, **dict(inputs or {})},
        "group": group,
        "definitions": DEFINITIONS,
        "runs": [{k: v for k, v in s.items()} for s in summaries],
        "stages": agg["stages"],
        "comparisons": comparisons,
        "flags": flags,
    }


_RUN_COLUMNS = (
    "label",
    "run_id",
    "seed",
    "status",
    "layout",
    "gate_mode",
    "stages_total",
    "stages_reached",
    "stages_cleared",
    "deepest_stage",
    "stalled_stage",
    "total_steps",
    "wall_clock_h",
    "active_h",
    "resume_count",
    "retries_used",
    "metadata_write_failures",
    "git",
    "path",
)

_STAGE_COLUMNS = (
    "stage",
    "index",
    "map_file",
    "opponent",
    "outcome",
    "gate_mode",
    "n_evals",
    "steps_to_promotion",
    "trained_steps",
    "steps_exact",
    "censored",
    "cum_steps_end",
    "retries",
    "skip_ahead",
    "stoch_wins",
    "stoch_draws",
    "stoch_losses",
    "stoch_episodes",
    "stoch_win_rate",
    "stoch_wr_lo",
    "stoch_wr_hi",
    "stoch_draw_rate",
    "greedy_wins",
    "greedy_draws",
    "greedy_losses",
    "greedy_episodes",
    "greedy_win_rate",
    "greedy_wr_lo",
    "greedy_wr_hi",
    "greedy_draw_rate",
    "window_stoch_win_rate",
    "window_greedy_win_rate",
    "peak_gate_wr",
    "captures_per_ep_tower",
    "captures_per_ep_building",
    "captures_per_ep_hq",
    "opponent_captures_per_ep_neutral",
    "opponent_captures_per_ep_owned",
    "reward_per_ep_action",
    "reward_per_ep_shaping_delta",
    "reward_per_ep_invalid_penalty",
    "reward_per_ep_terminal",
    "reward_sum_mismatch",
    "shaping_share_abs",
    "shaping_share_signed",
    "episode_abs_share",
    "draw_return_per_ep",
    "draw_return_exact",
    "draw_breakeven",
    "draw_breakeven_evals",
    "end_reason_rate_hq_capture",
    "end_reason_rate_elimination",
    "end_reason_rate_max_turns_draw",
    "end_reason_rate_max_steps_truncate",
    "flat_truncated_rate",
    "eval_resampled",
    "steps_per_hour",
    "eval_share",
    "rows_source",
    "label",
)


def _csv_value(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    return v


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([_csv_value(v) for v in row])


def write_outputs(summary: Mapping[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """report.md, runs.csv, per_seed_stage.csv, per_stage.csv, comparison.csv and summary.json under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        name: out / name
        for name in ("report.md", "runs.csv", "per_seed_stage.csv", "per_stage.csv", "comparison.csv", "summary.json")
    }
    paths["report.md"].write_text(render_report(summary), encoding="utf-8")
    _write_csv(paths["runs.csv"], _RUN_COLUMNS, ([r.get(c) for c in _RUN_COLUMNS] for r in summary["runs"]))
    _write_csv(
        paths["per_seed_stage.csv"],
        ("run_label", "run_id", "seed", *_STAGE_COLUMNS),
        (
            [r["label"], r["run_id"], r["seed"], *[s.get(c) for c in _STAGE_COLUMNS]]
            for r in summary["runs"]
            for s in r["stages"]
        ),
    )
    stat_cols = ("n", "mean", "sd", "min", "max", "median", "ci95_lo", "ci95_hi")
    _write_csv(
        paths["per_stage.csv"],
        ("stage", "index", "n_runs", "n_reached", "n_cleared", "seed_sensitive", "metric", *stat_cols, "values"),
        (
            [
                st["stage"],
                st["index"],
                st["n_runs"],
                st["n_reached"],
                st["n_cleared"],
                st["seed_sensitive"],
                metric,
                *[stats.get(c) for c in stat_cols],
                stats.get("values"),
            ]
            for st in summary["stages"]
            for metric, stats in st["metrics"].items()
        ),
    )
    comp_cols = (
        "stage",
        "comparable",
        "baseline_n_cleared",
        "baseline_n_reached",
        "baseline_steps",
        "baseline_greedy_wr",
        "baseline_greedy_draw_rate",
        "baseline_captures_per_ep",
        "baseline_shaping_share_abs",
        "baseline_cum_steps",
        "group_n_cleared",
        "group_n_runs",
        "group_steps_median",
        "group_steps_min",
        "group_steps_max",
        "group_greedy_wr",
        "group_greedy_wr_min",
        "group_greedy_wr_max",
        "group_stoch_wr",
        "group_stoch_wr_min",
        "group_stoch_wr_max",
        "group_greedy_draw_rate",
        "group_captures_per_ep",
        "group_shaping_share_abs",
        "delta_greedy_wr",
        "steps_ratio",
        "delta_shaping_share_abs",
        "differing",
        "unverified",
    )
    _write_csv(
        paths["comparison.csv"],
        ("baseline", *comp_cols),
        ([c["label"], *[r.get(col) for col in comp_cols]] for c in summary["comparisons"] for r in c["stages"]),
    )
    paths["summary.json"].write_text(json.dumps(summary, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    return paths


def load_summary(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION or "stages" not in data:
        raise ValueError(f"{path} is not a summarize_seeds.py summary.json (schema {SCHEMA_VERSION})")
    return data


def is_summary_json(path: str | Path) -> bool:
    p = Path(path)
    if not (p.is_file() and p.suffix == ".json"):
        return False
    try:
        return load_summary(p) is not None
    except (ValueError, OSError):
        return False


def baseline_side(label: str, spec: str, *, confidence: float = 0.95) -> Side:
    """A ``--compare LABEL=PATH`` baseline: a run dir, a bootstrap_results.csv, a summary.json or runs_per_stage.csv:RUN_ID."""
    if is_summary_json(spec):
        return side_from_summary(label, load_summary(spec), source=spec)
    record = read_run(spec)
    return side_from_runs(label, [record], [run_metrics(record, confidence=confidence)], source=spec)
