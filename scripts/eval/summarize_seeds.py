#!/usr/bin/env python3
"""Aggregate bootstrap runs across seeds into the §2.1 validation report.

Reads each run's raw records (JSON, YAML, CSV; no torch or SB3 needed) and
writes, under ``--out-dir``:

    report.md            provenance and gate settings; 1 per-seed outcome;
                         2 per stage across seeds; 3 per-seed detail;
                         4 comparison with each --compare baseline; 5 flags
    runs.csv             one row per run
    per_seed_stage.csv   one row per (run, stage)
    per_stage.csv        one row per (stage, metric): n, mean, sd, min, max,
                         median, 95% t-interval, per-seed values
    comparison.csv       one row per (baseline, shared stage)
    summary.json         all of it, plus the metric definitions (schema 1)

Inputs: run directories (new or legacy layout), ``--group-manifest`` (a
``seed_group.json`` written by scripts/train/run_seeds.py; its runs are
read from the manifest's root), and/or ``--glob``. ``--compare LABEL=PATH``
adds a baseline: a run dir, a ``bootstrap_results.csv``, a ``summary.json``
of this tool (group against group), or ``runs_per_stage.csv:RUN_ID``.

Exit codes: 0 ok; 1 the replicate check failed (the runs' resolved configs
differ beyond seed, device and logging; ``--allow-mixed`` accepts it) or an
input was unreadable; 2 no runs, or a usage error.

Runs that share a seed (the legacy archive is all seed 42) are labelled
``s<seed>@<run id>``. With ``--group-manifest`` a run's active hours come
from its launch sessions.

Examples:
    python3 scripts/eval/summarize_seeds.py \\
        --group-manifest benchmarks/bootstrap/_groups/20260928_120000_val/seed_group.json \\
        --compare v52a=/content/drive/MyDrive/reinforce-tactics/benchmarks/bootstrap/20260601_172412
    # Off Colab: download that Drive folder (its YAML, bootstrap_results.csv and
    # stage folders) and pass the local path instead.
    python3 scripts/eval/summarize_seeds.py benchmarks/bootstrap/run_a benchmarks/bootstrap/run_b --out-dir /tmp/report
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from reinforcetactics.experiments import run_summary as rs  # noqa: E402
from reinforcetactics.experiments.seed_runs import read_group  # noqa: E402

EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate bootstrap runs across seeds (the §2.1 validation report).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:", 1)[1] if __doc__ and "Examples:" in __doc__ else None,
    )
    p.add_argument("run_dirs", nargs="*", help="Run directories (or a bootstrap_results.csv)")
    p.add_argument("--group-manifest", help="A seed_group.json from scripts/train/run_seeds.py")
    p.add_argument("--glob", dest="glob_pattern", help="Also read the run directories this glob matches")
    p.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="A baseline: a run dir, a bootstrap_results.csv, a summary.json, or runs_per_stage.csv:RUN_ID (repeatable)",
    )
    p.add_argument("--out-dir", help="Output directory (default: <root>/_groups/<group>/report, else ./seed_report)")
    p.add_argument("--confidence", type=float, default=0.95, help="Two-sided Wilson interval level for win rates")
    p.add_argument("--allow-mixed", action="store_true", help="Report even when the runs' configs differ")
    p.add_argument("--quiet", action="store_true", help="Do not print report.md")
    return p


def _manifest_runs(path: Path) -> tuple[dict, list[Path], list[str], dict[str, list]]:
    """``(manifest, run dirs, missing seeds, {run dir name: its launch sessions})``."""
    manifest = read_group(path)
    # <root>/_groups/<group>/seed_group.json: the root is where the manifest
    # sits, so the group can be summarized wherever it was copied to.
    root = path.resolve().parent.parent.parent
    dirs: list[Path] = []
    missing: list[str] = []
    sessions: dict[str, list] = {}
    for seed in manifest.get("seeds", []):
        run = (manifest.get("runs") or {}).get(str(seed)) or {}
        run_dir = root / str(run.get("run_dir"))
        sessions[run_dir.name] = list(run.get("sessions") or [])
        if run_dir.is_dir():
            dirs.append(run_dir)
        else:
            missing.append(f"s{seed} ({run_dir.name})")
    return manifest, dirs, missing, sessions


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.5 <= args.confidence < 1.0:
        print(f"--confidence must be in [0.5, 1), got {args.confidence}", file=sys.stderr)
        return EXIT_USAGE
    inputs: list[str] = list(args.run_dirs)
    group_id = config_path = digest = None
    sessions: dict[str, list] = {}
    default_out = Path("seed_report")
    if args.group_manifest:
        path = Path(args.group_manifest)
        try:
            manifest, dirs, missing, sessions = _manifest_runs(path)
        except (OSError, ValueError) as exc:
            print(f"cannot read {path}: {exc}", file=sys.stderr)
            return EXIT_BAD_INPUT
        for name in missing:
            print(f"  note: {name} has no run directory yet (not started, or not fetched); left out")
        inputs += [str(d) for d in dirs]
        group_id, config_path, digest = manifest.get("group"), manifest.get("config"), manifest.get("config_digest")
        default_out = path.parent / "report"
    if args.glob_pattern:
        inputs += sorted(p for p in glob.glob(args.glob_pattern) if Path(p).is_dir())
    unique: dict[str, str] = {}
    for p in inputs:
        unique.setdefault(os.path.realpath(p), p)
    inputs = list(unique.values())
    if not inputs:
        print("no runs to summarize (give run directories, --group-manifest or --glob)", file=sys.stderr)
        return EXIT_USAGE

    records = []
    for spec in inputs:
        try:
            records.append(rs.read_run(spec))
        except Exception as exc:  # noqa: BLE001 - any unreadable input is reported the same way
            print(f"cannot read {spec}: {exc}", file=sys.stderr)
            return EXIT_BAD_INPUT
    baselines = []
    compare_inputs = {}
    for item in args.compare:
        label, sep, spec = item.partition("=")
        if not sep or not label or not spec:
            print(f"--compare expects LABEL=PATH, got {item!r}", file=sys.stderr)
            return EXIT_USAGE
        try:
            baselines.append(rs.baseline_side(label, spec, confidence=args.confidence))
        except Exception as exc:  # noqa: BLE001
            print(f"cannot read --compare {label}={spec}: {exc}", file=sys.stderr)
            return EXIT_BAD_INPUT
        compare_inputs[label] = spec

    summary = rs.build_summary(
        records,
        confidence=args.confidence,
        group_id=group_id,
        config_path=config_path,
        config_digest=digest,
        baselines=baselines,
        inputs={"runs": inputs, "group_manifest": args.group_manifest, "compare": compare_inputs},
        allow_mixed=args.allow_mixed,
        sessions=sessions,
    )
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    paths = rs.write_outputs(summary, out_dir)
    if not args.quiet:
        print(paths["report.md"].read_text(encoding="utf-8"))
    print(f"Wrote {', '.join(p.name for p in paths.values())} -> {out_dir}")
    check = summary["group"]["replicate_check"]
    if not check["passed"]:
        print(f"replicate check: {check['verdict']}", file=sys.stderr)
        for diff in check["differences"][:20]:
            print(f"  {diff}", file=sys.stderr)
        if not args.allow_mixed:
            return EXIT_BAD_INPUT
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
