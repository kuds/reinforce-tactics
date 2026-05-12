"""Generate 10 entropy-schedule variations of configs/bootstrap.yaml.

Each output is a full, runnable curriculum config (every stage preserved,
identical to the base except for the per-stage ``ent_coef`` field). Variations
sweep only the entropy axis so results can be attributed cleanly.

Usage:
    python scripts/generate_bootstrap_sweep.py

Outputs land in ``configs/bootstrap_sweep/v01_*.yaml`` ... ``v10_*.yaml``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = ROOT / "configs" / "bootstrap.yaml"
OUT_DIR = ROOT / "configs" / "bootstrap_sweep"


def _is_scheduled(ent: Any) -> bool:
    return isinstance(ent, dict)


def _flat(value: float) -> Callable[[dict], dict]:
    def transform(cfg: dict) -> dict:
        for stage in cfg["curriculum"]["stages"]:
            stage["ent_coef"] = value
        return cfg
    return transform


def _rewrite_scheduled(start: float, end: float, kind: str) -> Callable[[dict], dict]:
    """Rewrite every stage that currently has a schedule; leave flat stages alone."""
    def transform(cfg: dict) -> dict:
        for stage in cfg["curriculum"]["stages"]:
            if _is_scheduled(stage.get("ent_coef")):
                stage["ent_coef"] = {"start": start, "end": end, "schedule": kind}
        return cfg
    return transform


def _cosine_all(cfg: dict) -> dict:
    for stage in cfg["curriculum"]["stages"]:
        ent = stage.get("ent_coef")
        if _is_scheduled(ent):
            ent["schedule"] = "cosine"
    return cfg


def _extend_schedule_to_late_flat(cfg: dict) -> dict:
    """Stages currently using a flat ent_coef in the post-starter curriculum get
    a gentle linear 0.05 -> 0.02 schedule added. Starter stages are untouched
    (they're solved trivially today) and already-scheduled stages keep their
    existing schedule."""
    for stage in cfg["curriculum"]["stages"]:
        name = stage["name"]
        if name.startswith("starter_"):
            continue
        ent = stage.get("ent_coef")
        if not _is_scheduled(ent):
            stage["ent_coef"] = {"start": 0.05, "end": 0.02, "schedule": "linear"}
    return cfg


def _baseline_plus_starter_schedule(cfg: dict) -> dict:
    """Baseline curriculum, but the three starter stages also get a schedule."""
    targets = {
        "starter_random":  {"start": 0.10, "end": 0.05, "schedule": "linear"},
        "starter_simple":  {"start": 0.05, "end": 0.03, "schedule": "linear"},
        "starter_medium":  {"start": 0.05, "end": 0.03, "schedule": "linear"},
    }
    for stage in cfg["curriculum"]["stages"]:
        if stage["name"] in targets:
            stage["ent_coef"] = targets[stage["name"]]
    return cfg


VARIATIONS: list[tuple[str, str, Callable[[dict], dict]]] = [
    (
        "v01_flat_low_003",
        "Every stage uses a flat ent_coef=0.03. Null hypothesis: the schedule "
        "adds no value beyond simply running at the schedule's endpoint.",
        _flat(0.03),
    ),
    (
        "v02_flat_mid_005",
        "Every stage uses a flat ent_coef=0.05. Tests whether the curriculum "
        "works fine with the conventional mid value applied uniformly.",
        _flat(0.05),
    ),
    (
        "v03_flat_high_010",
        "Every stage uses a flat ent_coef=0.10. Tests whether sustained high "
        "exploration improves outcomes (or destabilises late stages).",
        _flat(0.10),
    ),
    (
        "v04_aggressive_anneal_010_to_001_linear",
        "Every currently-scheduled stage anneals linear 0.10 -> 0.01. Pushes "
        "the policy to commit harder by end of stage. Flat stages unchanged.",
        _rewrite_scheduled(0.10, 0.01, "linear"),
    ),
    (
        "v05_gentle_anneal_010_to_005_linear",
        "Every currently-scheduled stage anneals linear 0.10 -> 0.05. Less "
        "commit, more residual exploration at end of stage. Flat stages unchanged.",
        _rewrite_scheduled(0.10, 0.05, "linear"),
    ),
    (
        "v06_extend_schedule_to_late_stages",
        "Adds a gentle linear 0.05 -> 0.02 schedule to currently-flat post-starter "
        "stages (simple/medium/advanced/mixed_50). Starter stages and already-"
        "scheduled stages unchanged. Tests whether flat 0.05 in late stages is "
        "locking in suboptimal policies.",
        _extend_schedule_to_late_flat,
    ),
    (
        "v07_cosine_all",
        "Every linear schedule switched to cosine (endpoints unchanged). Tests "
        "schedule shape: slow-then-fast cooling vs uniform.",
        _cosine_all,
    ),
    (
        "v08_high_start_aggressive_015_to_001_linear",
        "Every currently-scheduled stage anneals linear 0.15 -> 0.01. Widens "
        "the exploration sweep on both ends. Flat stages unchanged.",
        _rewrite_scheduled(0.15, 0.01, "linear"),
    ),
    (
        "v09_low_start_007_to_003_linear",
        "Every currently-scheduled stage anneals linear 0.07 -> 0.03. Tests "
        "whether the 0.10 start was overshooting on warm-started stages.",
        _rewrite_scheduled(0.07, 0.03, "linear"),
    ),
    (
        "v10_baseline_plus_starter_schedule",
        "Baseline curriculum, but starter_random/simple/medium also get linear "
        "schedules (0.10->0.05 / 0.05->0.03 / 0.05->0.03). Tests whether the "
        "starter plateau seen in runs 2-7 stems from holding entropy flat.",
        _baseline_plus_starter_schedule,
    ),
]


HEADER_TEMPLATE = """\
# Auto-generated by scripts/generate_bootstrap_sweep.py
# Variation: {name}
#
# {description_wrapped}
#
# Sweep dimension: entropy schedule (per-stage ent_coef). Every other field is
# identical to configs/bootstrap.yaml so result deltas can be attributed
# cleanly to the entropy change.
"""


def _wrap(text: str, width: int = 76, indent: str = "# ") -> str:
    import textwrap
    return textwrap.fill(
        text,
        width=width,
        initial_indent="",
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def main() -> None:
    base = yaml.safe_load(BASE_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, description, transform in VARIATIONS:
        cfg = transform(copy.deepcopy(base))
        header = HEADER_TEMPLATE.format(
            name=name,
            description_wrapped=_wrap(description),
        )
        body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
        (OUT_DIR / f"{name}.yaml").write_text(header + "\n" + body)
        print(f"wrote {OUT_DIR / f'{name}.yaml'}")


if __name__ == "__main__":
    main()
