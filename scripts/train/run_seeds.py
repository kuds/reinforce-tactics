#!/usr/bin/env python3
"""Launch, continue and inspect a group of seeds of the bootstrap curriculum (review §2.1).

One ``train_bootstrap.py`` per seed, each in ``<root>/<group>_s<seed>/``,
with the group recorded in ``<root>/_groups/<group>/seed_group.json``
(``<group>`` = ``<YYYYmmdd_HHMMSS>_<tag>``). Every seed runs the same
command, ``--resume-if-exists`` included, so continuing the group with
``--group <id>`` (or resubmitting a Vertex job) after any interruption
continues where each seed stopped: finished and stalled seeds are left
alone, interrupted ones resume, and running ones are not touched.

    run_seeds.py [launch] --config C (--seeds LIST | --n-seeds N [--seed-stride 1000]) --tag T
        [--group GROUP] [--root benchmarks/bootstrap] [--parallel K] [--cpu-sets "0-9,10-19" | --no-pin]
        [--gpus 0,1 | --device cpu|cuda|auto] [--backend local|vertex] [--retry-failed]
        [--allow-seed-overlap] [--force] [--new-group] [--dry-run] [--tee | --no-tee] -- <train_bootstrap.py args>
    run_seeds.py status (--group-manifest M | --group G [--root R]) [--jobs]
    run_seeds.py fetch (--group-manifest M | --group G [--root R]) [--include-checkpoints]

Launch rules:

* Seeds default to a stride of 1000 from the config's seed (42, 1042, 2042);
  seeds whose random streams overlap (training envs or gate-eval episodes)
  are refused unless ``--allow-seed-overlap``.
* Continue a group with ``--group <id>`` alone: the config, seeds,
  train_bootstrap.py args and scheduling options (``--parallel``,
  ``--cpu-sets``, ``--no-pin``, ``--gpus``, ``--device``, ``--tee``; on
  Vertex the bucket, image and ``MACHINE_TYPE`` / ``ACCELERATOR_*`` /
  ``REPLICA_COUNT`` / ``SERVICE_ACCOUNT`` / ``RESTART_ON_WORKER_RESTART``)
  default to the group's. A different config, seeds, args, bucket or image
  is refused unless ``--force`` (recorded in the manifest's history);
  scheduling options given again replace the stored ones.
* Re-running a launch command without ``--group`` would start a new group
  from step 0, so it is refused while an unfinished group with the same tag
  and config exists under ``--root`` (the message names it); ``--new-group``
  starts another one anyway.
* The args after ``--`` go to every seed's train_bootstrap.py; ``--config``,
  ``--output-dir``, ``--resume``, ``--resume-if-exists``, ``--seed`` and
  ``--set seed=`` are the launcher's. ``--build-bc`` gets ``--bc-seed <seed>``.
* Local: seeds that are pending or interrupted are queued (failed ones only
  with ``--retry-failed``), at most ``--parallel`` at a time; a seed whose
  trainer is still running (it holds its run dir's lock, e.g. an orphan of
  a launcher that lost its terminal) is skipped and shown as running. With
  K > 1 each gets its own contiguous CPU set (``os.sched_setaffinity``) and
  ``--torch-threads`` of its size. Each session's output goes to
  ``<run dir>/logs/train.<UTC stamp>.log`` (and the console with ``--tee``,
  the default for K = 1). SIGINT / SIGTERM / SIGHUP reach every child as
  that signal (SIGHUP as SIGTERM), and each checkpoints and exits; after
  600 s the rest are killed. Run a multi-day launch under tmux, screen or
  nohup all the same: a launcher killed outright (SIGKILL) leaves its
  children training with no one recording them.
* Vertex (``--backend vertex``): one custom job per seed through
  scripts/cloud/submit_vertex_job.sh (needs BUCKET and IMAGE_URI or TAG;
  PROJECT_ID / REGION / MACHINE_TYPE / ... pass through). All seeds write
  under ``gs://<BUCKET>/jobs/<group>/<run dir>/``; a resubmitted job
  restores its run dir from there first. A seed whose job is still active,
  or whose run in GCS finished or stalled, is never resubmitted (reading
  run_status.json needs google-cloud-storage; without it name the seeds to
  resubmit with ``--only-seeds``). A job that ended FAILED is resubmitted
  only when its exit status was an interrupt (130 / 137 / 143: preempted,
  cancelled); any other failure (exit 1, a timeout) needs
  ``--retry-failed``, as a failed local seed does. Each job gets
  ``RT_GIT_COMMIT`` when the image tag names this checkout's commit, so its
  run records name the commit (the image has no .git). ``fetch`` downloads
  the runs into ``<root>/`` (without checkpoints, traces, videos or
  tensorboard by default; ``final_model.zip`` always, so a finished run
  shows as completed).

Exit codes: 0 every seed completed; 3 some stalled (none failed or
interrupted); 1 a seed failed or is not finished (pending, still running
under another launcher, a submission failed); 129 / 130 / 143 interrupted;
2 a usage error (a refused relaunch or seed overlap included).

Examples:
    python3 scripts/train/run_seeds.py --config configs/ppo/bootstrap.yaml --n-seeds 3 --tag val \\
        --parallel 3 -- --strict --skip-videos
    python3 scripts/train/run_seeds.py --group 20260928_120000_val            # continue it, same settings
    python3 scripts/train/run_seeds.py status --group 20260928_120000_val
    BUCKET=my-bucket TAG=abc1234 python3 scripts/train/run_seeds.py --backend vertex \\
        --config configs/ppo/bootstrap.yaml --n-seeds 3 --tag val -- --strict --skip-videos
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from reinforcetactics.experiments import seed_runs as sr  # noqa: E402

SUBCOMMANDS = ("launch", "status", "fetch")
DEFAULT_KILL_TIMEOUT = 600.0
# final_model.zip is fetched: without it a finished run reads as interrupted.
FETCH_EXCLUDE = r"(?<!^final_model)\.zip$|(^|/)(traces|videos|tensorboard)/"
FETCH_EXCLUDE_KEEP_ZIPS = r"(^|/)(traces|videos|tensorboard)/"

# Seams for the tests: the command runner (gcloud, the submit script) and the
# GCS run-state reader of the Vertex backend.
RUNNER: Callable[..., Any] | None = None
GCS_STATE: Callable[[str, str], str | None] | None = None


class UsageError(Exception):
    """A refused command line (exit 2)."""


def _runner() -> Callable[..., Any]:
    return RUNNER or subprocess.run


def split_passthrough(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    argv = list(argv)
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def _add_group_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--group-manifest", help="The group's seed_group.json")
    p.add_argument("--group", help="The group id (<YYYYmmdd_HHMMSS>_<tag>); its manifest is under --root")
    p.add_argument("--root", default=sr.DEFAULT_ROOT, help="Directory the run dirs and _groups/ live in")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_seeds.py",
        description="Launch, continue and inspect a group of seeds of the bootstrap curriculum.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Arguments after -- are passed to every seed's train_bootstrap.py.",
    )
    sub = p.add_subparsers(dest="command")
    launch = sub.add_parser("launch", help="Start or continue the group (the default command)")
    launch.add_argument("--config", help="The bootstrap YAML (default when continuing: the group's)")
    seeds = launch.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", help="Comma-separated seeds, e.g. 42,1042,2042")
    seeds.add_argument("--n-seeds", type=int, help="N seeds from the config's seed, --seed-stride apart")
    launch.add_argument("--seed-stride", type=int, default=sr.DEFAULT_SEED_STRIDE, help="Distance between --n-seeds seeds")
    launch.add_argument("--tag", help="Group tag ([a-z0-9][a-z0-9-]{0,30}); required for a new group")
    launch.add_argument("--group", help="Continue (or create with this id) the group <YYYYmmdd_HHMMSS>_<tag>")
    launch.add_argument("--root", default=sr.DEFAULT_ROOT, help="Where the run dirs and _groups/ live")
    launch.add_argument(
        "--parallel", type=int, default=None, help="Seeds run at once (local; default 1, or the group's when continuing)"
    )
    launch.add_argument(
        "--cpu-sets", help='CPU set per parallel slot, e.g. "0-9,10-19" (default: an even split, or the group\'s)'
    )
    pin = launch.add_mutually_exclusive_group()
    pin.add_argument(
        "--no-pin", dest="no_pin", action="store_const", const=True, default=None, help="Do not pin children to CPU sets"
    )
    pin.add_argument("--pin", dest="no_pin", action="store_const", const=False, help="Pin them (undoes a group's --no-pin)")
    launch.add_argument(
        "--gpus", help="GPU ids, one per slot round-robin, e.g. 0,1 (implies --device cuda; default: the group's)"
    )
    launch.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        help="train_bootstrap.py --device (default: the group's, else auto; cuda on Vertex)",
    )
    launch.add_argument("--backend", choices=("local", "vertex"), help="Where the seeds run (default: local, or the group's)")
    launch.add_argument("--retry-failed", action="store_true", help="Also relaunch seeds whose last session failed")
    launch.add_argument("--allow-seed-overlap", action="store_true", help="Launch even if seeds share random streams")
    launch.add_argument(
        "--force", action="store_true", help="Continue the group with a different config / seeds / args / bucket / image"
    )
    launch.add_argument(
        "--new-group",
        action="store_true",
        help="Start a new group even though an unfinished one with this tag and config exists under --root",
    )
    launch.add_argument("--dry-run", action="store_true", help="Print what would run; change nothing")
    launch.add_argument(
        "--tee",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also copy child output to the console (default: with --parallel 1)",
    )
    launch.add_argument(
        "--only-seeds", help="Vertex: (re)submit only these seeds (needed to resubmit without google-cloud-storage)"
    )
    launch.add_argument("--train-script", help=argparse.SUPPRESS)
    launch.add_argument("--kill-timeout", type=float, default=DEFAULT_KILL_TIMEOUT, help=argparse.SUPPRESS)
    status = sub.add_parser("status", help="Per-seed progress of a group")
    _add_group_args(status)
    status.add_argument("--jobs", action="store_true", help="Vertex: also ask gcloud for each job's state")
    fetch = sub.add_parser("fetch", help="Vertex: download the group's runs from GCS into --root")
    _add_group_args(fetch)
    fetch.add_argument("--include-checkpoints", action="store_true", help="Also download the .zip checkpoints")
    return p


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    own, passthrough = split_passthrough(argv)
    if not own or own[0] not in SUBCOMMANDS and own[0] not in ("-h", "--help"):
        own = ["launch", *own]
    args = build_parser().parse_args(own)
    if args.command is None:
        build_parser().print_help()
        raise SystemExit(sr.EXIT_USAGE)
    if args.command != "launch" and passthrough:
        raise UsageError(f"'{args.command}' takes no train_bootstrap.py arguments")
    return args, passthrough


def _passthrough_sets(passthrough: Sequence[str]) -> dict[str, Any]:
    """The ``--set KEY=VALUE`` overrides among the passthrough args, parsed as train_bootstrap.py does."""
    import yaml

    items: list[str] = []
    args = list(passthrough)
    for i, arg in enumerate(args):
        if arg == "--set" and i + 1 < len(args):
            items.append(args[i + 1])
        elif arg.startswith("--set="):
            items.append(arg.split("=", 1)[1])
    overrides: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if sep:
            value = yaml.safe_load(raw)
            overrides[key.strip()] = "null" if value is None else value
    return overrides


def _load_config(config: str, passthrough: Sequence[str]) -> tuple[Any, Any]:
    """``(the config, the config each seed actually runs: with the passthrough --set applied)``."""
    from reinforcetactics.rl.config import apply_overrides, load_config

    try:
        cfg = load_config(config)
        sets = _passthrough_sets(passthrough)
        return cfg, (apply_overrides(cfg, sets) if sets else cfg)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise UsageError(f"cannot load {config} with the given --set values: {exc}") from None


def _resolve_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.group_manifest:
        path = Path(args.group_manifest)
    elif args.group:
        path = sr.group_manifest_path(args.root, args.group)
    else:
        raise UsageError("give --group-manifest, or --group (with --root)")
    if not path.is_file():
        raise UsageError(f"no group manifest at {path}")
    return path, _read_manifest(path)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        return sr.read_group(path)
    except (OSError, ValueError) as exc:
        raise UsageError(f"cannot read {path}: {exc}") from None


def _root_of(manifest_path: Path) -> Path:
    return manifest_path.parent.parent.parent


def _fmt(value: Any, kind: str = "") -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{100 * float(value):.0f}%"
    if kind == "steps":
        v = float(value)
        return f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.0f}k" if v >= 1e3 else f"{v:.0f}"
    return str(value)


def _print_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    cells = [list(map(str, header)), *[[str(c) for c in row] for row in rows]]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    for n, row in enumerate(cells):
        print("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))
        if n == 0:
            print("  " + "  ".join("-" * w for w in widths))


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


# The local scheduling options a group records (seed_group.json "launch") and
# a continuation without them reuses.
LAUNCH_KEYS = ("parallel", "cpu_sets", "no_pin", "gpus", "device", "tee")
_LAUNCH_DEFAULTS: dict[str, Any] = {
    "parallel": 1,
    "cpu_sets": None,
    "no_pin": False,
    "gpus": None,
    "device": None,
    "tee": None,
}
_LAUNCH_FLAGS = {
    "parallel": "--parallel",
    "cpu_sets": "--cpu-sets",
    "no_pin": "--no-pin",
    "gpus": "--gpus",
    "device": "--device",
}


def _resolve_launch(args: argparse.Namespace, stored: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``(the scheduling options this launch uses, the keys taken from the group)``: given > the group's > default."""
    launch: dict[str, Any] = {}
    reused: list[str] = []
    for key in LAUNCH_KEYS:
        given = getattr(args, key)
        if given is not None:
            launch[key] = given
        elif stored.get(key) is not None:
            launch[key] = stored[key]
            reused.append(key)
        else:
            launch[key] = _LAUNCH_DEFAULTS[key]
    return launch, reused


def _unfinished_groups(root: Path, tag: str, digest: str, exclude: str) -> list[tuple[str, dict[int, str]]]:
    """Groups under ``root`` with this tag and config digest that have a seed not finished: ``[(group, {seed: state})]``."""
    found = []
    for path in sorted((root / sr.GROUPS_DIR).glob(f"*_{tag}/{sr.GROUP_MANIFEST}")):
        try:
            other = sr.read_group(path)
        except (OSError, ValueError):
            continue
        if other.get("group") == exclude or other.get("tag") != tag or other.get("config_digest") != digest:
            continue
        runs = other.get("runs") or {}
        states = {}
        for seed in other.get("seeds") or []:
            record = runs.get(str(seed)) or {}
            states[int(seed)] = sr.seed_state(root / str(record.get("run_dir")), record)
        unfinished = {s: st for s, st in states.items() if st not in sr.DONE_STATES}
        if unfinished:
            found.append((str(other.get("group")), unfinished))
    return found


def _prepare_group(args: argparse.Namespace, passthrough: list[str]) -> dict[str, Any]:
    """Resolve the group, its manifest (new or continued), the config, the seeds, the train args and the scheduling."""
    try:
        sr.check_passthrough(passthrough)
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    root = Path(args.root)
    manifest: dict[str, Any] | None = None
    if args.group:
        try:
            _, group_tag = sr.parse_group_id(args.group)
        except ValueError as exc:
            raise UsageError(str(exc)) from None
        if args.tag and args.tag != group_tag:
            raise UsageError(f"--tag {args.tag} does not match the group's tag {group_tag}")
        group = args.group
        path = sr.group_manifest_path(root, group)
        if path.is_file():
            manifest = _read_manifest(path)
    else:
        if not args.tag:
            raise UsageError("--tag is required for a new group (or --group <id> to continue one)")
        try:
            group = sr.make_group_id(args.tag)
        except ValueError as exc:
            raise UsageError(str(exc)) from None
        path = sr.group_manifest_path(root, group)
        while path.exists():  # another group of this tag started within the same second: never reuse its id
            time.sleep(0.2)
            group = sr.make_group_id(args.tag)
            path = sr.group_manifest_path(root, group)
    config = args.config or (manifest or {}).get("config")
    if not config:
        raise UsageError("--config is required for a new group")
    continuing = manifest is not None
    train_args = list(passthrough) if (passthrough or not continuing) else list(manifest["train_args"])  # type: ignore[index]
    cfg, effective = _load_config(config, train_args)
    try:
        if args.seeds:
            seeds = sr.parse_seeds(args.seeds)
        elif args.n_seeds:
            seeds = sr.seeds_from_stride(int(cfg.seed), args.n_seeds, args.seed_stride)
        elif continuing:
            seeds = [int(s) for s in manifest["seeds"]]  # type: ignore[index]
        else:
            raise UsageError("give --seeds or --n-seeds")
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    collisions = sr.seed_collisions(seeds, effective)
    if collisions:
        for line in collisions:
            print(f"  seed overlap: {line}", file=sys.stderr)
        if not args.allow_seed_overlap:
            raise UsageError(
                "these seeds share random streams; choose seeds further apart (--seed-stride) or pass --allow-seed-overlap"
            )
    backend = args.backend or (manifest or {}).get("backend") or "local"
    if continuing and manifest.get("backend") != backend:  # type: ignore[union-attr]
        raise UsageError(f"the group runs on {manifest.get('backend')}; it cannot continue on {backend}")  # type: ignore[union-attr]
    digest = sr.config_digest(cfg)
    git = sr.git_info()
    if not continuing and not args.new_group:
        # Re-running a launch command without --group would start every seed
        # over in a new group; refuse while one with this tag and config is
        # unfinished, and name it.
        others = _unfinished_groups(root, sr.parse_group_id(group)[1], digest, group)
        if others:
            for other, states in others:
                shown = ", ".join(f"s{s} {st}" for s, st in states.items())
                print(f"  unfinished group {other}: {shown}", file=sys.stderr)
            newest = others[-1][0]
            raise UsageError(
                f"an unfinished group with this tag and config exists under {root}; continue it with "
                f"`{Path(sys.argv[0]).name} --group {newest} --root {root}` (its config, seeds, args and "
                "scheduling are reused), or pass --new-group to start another group from scratch"
            )
    stored_launch = dict((manifest or {}).get("launch") or {})
    launch, reused = _resolve_launch(args, stored_launch)
    if continuing:
        assert manifest is not None
        problems = sr.manifest_mismatches(manifest, digest=digest, seeds=seeds, train_args=train_args)
        if problems:
            for line in problems:
                print(f"  differs from the group: {line}", file=sys.stderr)
            if not args.force:
                raise UsageError(f"{path}: this launch differs from the group it continues; pass --force to continue anyway")
            manifest["history"].append(
                {"at": sr.utc_now(), "event": "forced relaunch", "differences": problems, "git": git.get("short")}
            )
            manifest.update(config=str(config), config_digest=digest, seeds=seeds, train_args=train_args)
        for s in seeds:
            manifest["runs"].setdefault(
                str(s),
                {
                    "run_dir": sr.run_dir_name(group, s),
                    "job_name": sr.vertex_job_name(group, s) if backend == "vertex" else None,
                    "job_resource": None,
                    "sessions": [],
                    "last_state": sr.PENDING,
                },
            )
        recorded = (manifest.get("git") or {}).get("commit")
        if recorded and git.get("commit") and recorded != git["commit"]:
            print(f"  warning: the group started at git {recorded[:7]}; this checkout is {git['short']}", file=sys.stderr)
        changed = {
            k: [stored_launch.get(k), launch[k]] for k in LAUNCH_KEYS if k in stored_launch and stored_launch[k] != launch[k]
        }
        if changed:
            manifest["history"].append({"at": sr.utc_now(), "event": "scheduling changed", "changes": changed})
        manifest["launch"] = launch
    else:
        manifest = sr.new_group_manifest(
            group=group,
            root=root,
            backend=backend,
            config=config,
            digest=digest,
            seeds=seeds,
            train_args=train_args,
            git=git,
            launch=launch,
        )
    return {
        "group": group,
        "root": root,
        "manifest_path": path,
        "manifest": manifest,
        "config": str(config),
        "cfg": cfg,
        "effective": effective,
        "seeds": seeds,
        "train_args": train_args,
        "backend": backend,
        "continuing": continuing,
        "launch": launch,
        "launch_reused": reused,
    }


def _say(*parts: Any, **kwargs: Any) -> None:
    """print() that survives a console gone away (EIO after the terminal hung up)."""
    try:
        print(*parts, **kwargs)
    except OSError:
        pass


def _local(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    root, manifest, seeds = plan["root"], plan["manifest"], plan["seeds"]
    runs = manifest["runs"]
    launch = plan["launch"]
    states = {s: sr.seed_state(root / runs[str(s)]["run_dir"], runs[str(s)]) for s in seeds}
    queue = [s for s in seeds if states[s] in (sr.PENDING, sr.INTERRUPTED) or (states[s] == sr.FAILED and args.retry_failed)]
    parallel = max(1, int(launch["parallel"]))
    cpu_sets: list[list[int]] | None = None
    if launch["cpu_sets"] and not launch["no_pin"]:
        try:
            cpu_sets = sr.parse_cpu_sets(launch["cpu_sets"])
        except ValueError as exc:
            raise UsageError(f"--cpu-sets: {exc}") from None
        if len(cpu_sets) < parallel:
            raise UsageError(f"--cpu-sets gives {len(cpu_sets)} set(s) for --parallel {parallel}")
        missing = sorted({c for cs in cpu_sets for c in cs} - set(sr.available_cpus()))
        if missing and "cpu_sets" in plan["launch_reused"]:
            raise UsageError(
                f"the group's --cpu-sets {launch['cpu_sets']} name CPUs this machine lacks ({missing}); "
                "pass --cpu-sets or --no-pin"
            )
    elif parallel > 1 and not launch["no_pin"]:
        try:
            cpu_sets = sr.split_cpus(sr.available_cpus(), parallel)
        except ValueError as exc:
            raise UsageError(str(exc)) from None
    for warning in sr.cpu_warnings(cpu_sets or [], plan["effective"]):
        print(f"  warning: {warning}", file=sys.stderr)
    if cpu_sets and not hasattr(os, "sched_setaffinity"):
        print("  warning: CPU pinning needs Linux (os.sched_setaffinity); children run unpinned", file=sys.stderr)
    gpus = [g.strip() for g in launch["gpus"].split(",") if g.strip()] if launch["gpus"] else []
    device = launch["device"] or ("cuda" if gpus else "auto")
    config = str(Path(plan["config"]).resolve())

    def make_spec(seed: int, slot: int) -> sr.ChildSpec:
        run_dir = (root / runs[str(seed)]["run_dir"]).resolve()
        cpu_set = cpu_sets[slot] if cpu_sets else None
        gpu = gpus[slot % len(gpus)] if gpus else ("" if device == "cpu" else None)
        cmd = sr.build_child_command(
            config,
            seed,
            run_dir,
            device,
            plan["train_args"],
            torch_threads=len(cpu_set) if cpu_set else None,
            script=args.train_script,
        )
        return sr.ChildSpec(
            seed=seed, run_dir=run_dir, cmd=cmd, env=sr.build_child_env(cpu_set, gpu), cpu_set=cpu_set, gpu=gpu
        )

    print(
        f"Group {plan['group']} ({'continued' if plan['continuing'] else 'new'}): {len(seeds)} seed(s), local, --parallel {parallel}"
    )
    reused = [k for k in plan["launch_reused"] if k in _LAUNCH_FLAGS]
    if reused:
        shown = " ".join(
            _LAUNCH_FLAGS[k] if launch[k] is True else f"{_LAUNCH_FLAGS[k]} {launch[k]}"
            for k in reused
            if launch[k] is not False
        )
        print(f"  the group's scheduling: {shown}")
    _print_table(["seed", "state", "run dir"], [[s, states[s], runs[str(s)]["run_dir"]] for s in seeds])
    skipped_failed = [s for s in seeds if states[s] == sr.FAILED and not args.retry_failed]
    if skipped_failed:
        print(f"  failed seed(s) {skipped_failed} are not relaunched; read their logs, fix, then pass --retry-failed")
    live = [s for s in seeds if states[s] == sr.RUNNING]
    for seed in live:
        holder = sr.lock_holder(root / runs[str(seed)]["run_dir"])
        where = f" (pid {holder.get('pid')} on {holder.get('host')})" if holder.get("pid") else ""
        print(f"  seed {seed} is still running{where}; left alone")
    if args.dry_run:
        for i, seed in enumerate(queue):
            spec = make_spec(seed, i % parallel)
            pin = f" [cpus {sr.format_cpu_set(spec.cpu_set)}]" if spec.cpu_set else ""
            gpu = f" [CUDA_VISIBLE_DEVICES={spec.gpu!r}]" if spec.gpu is not None else ""
            print(f"  would run{pin}{gpu}: {shlex.join(spec.cmd)}")
        if not queue:
            print("  nothing to run")
        return sr.EXIT_OK
    path = plan["manifest_path"]
    sr.write_group(path, manifest)
    git_short = sr.git_info().get("short")

    def on_start(spec: sr.ChildSpec, session: dict[str, Any]) -> None:
        session["git"] = git_short
        record = runs[str(spec.seed)]
        record["sessions"].append(session)
        record["last_state"] = sr.RUNNING
        sr.write_group(path, manifest)
        _say(f"  started seed {spec.seed} (pid {session['pid']}); log: {session['log']}")

    def on_exit(spec: sr.ChildSpec, session: dict[str, Any]) -> None:
        record = runs[str(spec.seed)]
        record["last_state"] = sr.state_after_exit(spec.run_dir, int(session["exit_code"]))
        sr.write_group(path, manifest)
        _say(f"  seed {spec.seed} exited {session['exit_code']}: {record['last_state']}")

    result = sr.run_local(
        queue,
        make_spec,
        parallel=parallel,
        tee=launch["tee"] if launch["tee"] is not None else parallel == 1,
        on_start=on_start,
        on_exit=on_exit,
        kill_timeout=args.kill_timeout,
    )
    final = {s: sr.seed_state(root / runs[str(s)]["run_dir"], runs[str(s)]) for s in seeds}
    try:
        print(f"Group {plan['group']}:")
        _print_table(["seed", "state", "exit"], [[s, final[s], result.exit_codes.get(s, "—")] for s in seeds])
        print(f"  continue with: {Path(sys.argv[0]).name} --group {plan['group']} --root {root}  (same settings)")
    except OSError:
        pass
    return sr.launcher_exit_code(final, result.signalled, codes=result.exit_codes)


# ---------------------------------------------------------------------------
# Vertex
# ---------------------------------------------------------------------------


# submit_vertex_job.sh's machine settings: recorded with the group and reused
# by a continuation that does not set them (the script's defaults are a
# smaller machine than the runbook's).
MACHINE_KEYS = (
    "MACHINE_TYPE",
    "ACCELERATOR_TYPE",
    "ACCELERATOR_COUNT",
    "REPLICA_COUNT",
    "SERVICE_ACCOUNT",
    "RESTART_ON_WORKER_RESTART",
    "SYNC_INTERVAL",
)


def _image_identity(env: dict[str, str]) -> str | None:
    """The image a job runs: ``IMAGE_URI``, else ``TAG=<tag>``."""
    if env.get("IMAGE_URI"):
        return env["IMAGE_URI"]
    return f"TAG={env['TAG']}" if env.get("TAG") else None


def _stored_image_env(stored: dict[str, Any]) -> dict[str, str]:
    image_env = dict(stored.get("image_env") or {})
    if not image_env and stored.get("image_uri"):  # a manifest written before image_env was recorded
        uri = str(stored["image_uri"])
        image_env = {"TAG": uri[len("TAG=") :]} if uri.startswith("TAG=") else {"IMAGE_URI": uri}
    return {k: str(v) for k, v in image_env.items() if v}


def _vertex_env(stored: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """``(the job settings, the machine settings)``: from the environment, else the group's (``stored``)."""
    image_env = {k: os.environ[k].strip() for k in ("IMAGE_URI", "TAG") if os.environ.get(k, "").strip()}
    if not image_env:
        image_env = _stored_image_env(stored)
    env = {
        "BUCKET": os.environ.get("BUCKET", "").strip() or str(stored.get("bucket") or ""),
        "PROJECT_ID": os.environ.get("PROJECT_ID", "").strip() or str(stored.get("project") or ""),
        "REGION": os.environ.get("REGION", "").strip() or str(stored.get("region") or ""),
        **image_env,
    }
    env = {k: v for k, v in env.items() if v}
    if not env.get("BUCKET"):
        raise UsageError("--backend vertex needs BUCKET (the GCS bucket the jobs write to)")
    if not env.get("IMAGE_URI") and not env.get("TAG"):
        raise UsageError("--backend vertex needs IMAGE_URI, or TAG=<short sha> of an image built from this commit")
    stored_machine = dict(stored.get("machine") or {})
    machine = {}
    for key in MACHINE_KEYS:
        value = os.environ.get(key, "").strip() or str(stored_machine.get(key) or "")
        if value:
            machine[key] = value
    return env, machine


def _image_commit(env: dict[str, str]) -> str | None:
    """The commit the image's tag names, when it names one of this checkout (``TAG=<short sha>``, the runbook's rule)."""
    uri = env.get("IMAGE_URI") or ""
    tag = env.get("TAG") or (uri.rsplit(":", 1)[1] if ":" in uri.rsplit("/", 1)[-1] and "@" not in uri else "")
    return sr.resolve_commit(tag) if tag else None


def _repo_relative(config: str) -> str:
    path = Path(config).resolve()
    try:
        return path.relative_to(sr.REPO_ROOT.resolve()).as_posix()
    except ValueError:
        raise UsageError(f"--config {config} is outside the repository, so it is not in the training image") from None


def _gcs_state(output_uri: str, run: str) -> str | None:
    if GCS_STATE is not None:
        return GCS_STATE(output_uri, run)
    return sr.gcs_run_state(output_uri, run)


def _vertex(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    manifest, group = plan["manifest"], plan["group"]
    stored = dict(manifest.get("vertex") or {}) if plan["continuing"] else {}
    image_env, machine = _vertex_env(stored)
    config = _repo_relative(plan["config"])
    device = plan["launch"]["device"] or "cuda"
    bucket = image_env["BUCKET"]
    output_uri = sr.group_output_uri(bucket, group)
    image = _image_identity(image_env)
    if stored:
        # A different bucket leaves the group's runs behind (the jobs restore
        # nothing and start over); a different image runs seeds of one group on
        # different code. Neither happens silently.
        differences = []
        if stored.get("bucket") and stored["bucket"] != sr.bucket_name(bucket):
            differences.append(f"bucket {sr.bucket_name(bucket)} != the group's {stored['bucket']}")
        stored_image = _image_identity(_stored_image_env(stored))
        if stored_image and stored_image != image:
            differences.append(f"image {image} != the group's {stored_image}")
        if differences:
            for line in differences:
                print(f"  differs from the group: {line}", file=sys.stderr)
            if not args.force:
                raise UsageError(
                    f"{plan['manifest_path']}: this launch uses another bucket or image than the group; "
                    "unset BUCKET / IMAGE_URI / TAG to use the group's, or pass --force"
                )
            manifest["history"].append({"at": sr.utc_now(), "event": "forced Vertex change", "differences": differences})
    commit = _image_commit(image_env)
    if commit is None:
        print(
            f"  note: the image tag of {image} names no commit of this checkout, so the jobs' run records "
            "will name none (build the image with TAG=$(git rev-parse --short HEAD))",
            file=sys.stderr,
        )
    job_env = {**image_env, **machine}
    manifest["vertex"] = {
        "project": image_env.get("PROJECT_ID"),
        "region": image_env.get("REGION"),
        "bucket": sr.bucket_name(bucket),
        "output_uri": output_uri,
        "image_uri": image,
        "image_env": {k: image_env[k] for k in ("IMAGE_URI", "TAG") if image_env.get(k)},
        "image_commit": commit,
        "machine": machine,
    }
    try:
        only = set(sr.parse_seeds(args.only_seeds)) if args.only_seeds else None
    except ValueError as exc:
        raise UsageError(f"--only-seeds: {exc}") from None
    runner = _runner()
    path = plan["manifest_path"]
    failed = False
    print(f"Group {group}: {len(plan['seeds'])} seed(s) on Vertex -> {output_uri}")
    if machine:
        print("  machine: " + " ".join(f"{k}={v}" for k, v in machine.items()))
    for seed in plan["seeds"]:
        if only is not None and seed not in only:
            continue
        record = manifest["runs"][str(seed)]
        run = record["run_dir"]
        resource = record.get("job_resource")
        if resource:
            state, message = sr.describe_job(resource, runner=runner)
            if state not in sr.TERMINAL_JOB_STATES:
                print(f"  s{seed}: job {resource} is {state or 'in an unknown state'}; not resubmitting")
                continue
            if only is None:
                try:
                    run_state = _gcs_state(output_uri, run)
                except ImportError:
                    print(
                        f"  s{seed}: job {state}; cannot read its run_status.json without google-cloud-storage: "
                        "pass --only-seeds to resubmit it"
                    )
                    continue
                if run_state in sr.DONE_STATES:
                    record["last_state"] = run_state
                    print(f"  s{seed}: {run_state} ({state}); nothing to resubmit")
                    continue
                code = sr.job_exit_status(message)
                interrupted = code in sr.INTERRUPT_CODES | sr.KILLED_CODES
                if state == "JOB_STATE_FAILED" and not interrupted and not args.retry_failed:
                    # Exit 1 would fail again the same way, and a failure that
                    # names no exit status (a timeout, say) wants a look first;
                    # an interrupt (preemption, a cancel) resumes like a local one.
                    record["last_state"] = sr.FAILED
                    failed = True
                    why = f"exit {code}" if code is not None else (message[:160] or "no exit status reported")
                    print(
                        f"  s{seed}: job failed ({why}); read its log (gcloud ai custom-jobs stream-logs {resource}), "
                        "fix, then pass --retry-failed"
                    )
                    continue
        env, argv = sr.vertex_submission(
            group=group,
            seed=seed,
            config=config,
            device=device,
            passthrough=plan["train_args"],
            bucket=bucket,
            image_env=job_env,
            git_commit=commit,
        )
        shown = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
        if args.dry_run:
            print(f"  s{seed}: would submit: {shown} {shlex.join(argv)}")
            continue
        print(f"  s{seed}: submitting {env['JOB_NAME']}")
        proc = runner(argv, env={**os.environ, **env}, cwd=str(sr.REPO_ROOT), capture_output=True, text=True, check=False)
        output = (getattr(proc, "stdout", "") or "") + (getattr(proc, "stderr", "") or "")
        sys.stdout.write(output)
        job = sr.parse_job_resource(output)
        session = {
            "submitted_at": sr.utc_now(),
            "job_name": env["JOB_NAME"],
            "job_resource": job,
            "exit_code": getattr(proc, "returncode", None),
            "cmd": argv,
            "git": sr.git_info().get("short"),
            "image": image,
            "image_commit": commit,
            "machine": machine,
        }
        record["sessions"].append(session)
        if proc.returncode != 0 or not job:
            failed = True
            record["last_state"] = sr.FAILED
            print(f"  s{seed}: submission failed (exit {proc.returncode}{', no job resource printed' if not job else ''})")
        else:
            record["job_resource"] = job
            record["last_state"] = sr.SUBMITTED
        sr.write_group(path, manifest)
    if not args.dry_run:
        sr.write_group(path, manifest)
        print(f"  status: {Path(sys.argv[0]).name} status --group {group} --root {plan['root']} --jobs")
        print(f"  fetch:  {Path(sys.argv[0]).name} fetch --group {group} --root {plan['root']}")
    return sr.EXIT_FAILED if failed else sr.EXIT_OK


def cmd_launch(args: argparse.Namespace, passthrough: list[str]) -> int:
    plan = _prepare_group(args, passthrough)
    if plan["backend"] == "vertex":
        if args.dry_run:
            print(f"(dry run: {plan['manifest_path']} is not written)")
        return _vertex(args, plan)
    return _local(args, plan)


# ---------------------------------------------------------------------------
# status / fetch
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    path, manifest = _resolve_manifest(args)
    root = _root_of(path)
    rows = []
    for seed in manifest["seeds"]:
        record = manifest["runs"][str(seed)]
        run_dir = root / record["run_dir"]
        progress = sr.run_progress(run_dir) if run_dir.is_dir() else {}
        sessions = record.get("sessions") or []
        last_exit = sessions[-1].get("exit_code") if sessions else None
        stage = progress.get("stage")
        where = f"{stage} ({progress.get('stage_index')}/{progress.get('n_stages')})" if stage else "—"
        row = [
            seed,
            sr.seed_state(run_dir, record),
            where,
            _fmt(progress.get("timesteps"), "steps"),
            _fmt(progress.get("last_gate_wr"), "pct"),
            f"{progress.get('stages_cleared', 0)}/{progress.get('n_stages', 0)}",
            progress.get("resume_count", 0),
            len(sessions),
            _fmt(last_exit),
            _fmt(progress.get("steps_per_hour"), "steps"),
        ]
        if args.jobs:
            resource = record.get("job_resource")
            row.append((sr.describe_job_state(resource, runner=_runner()) or "?") if resource else "—")
        rows.append(row)
    header = ["seed", "state", "stage", "steps", "last gate WR", "cleared", "resumes", "sessions", "last exit", "steps/h"]
    if args.jobs:
        header.append("job")
    print(
        f"Group {manifest['group']} ({manifest.get('backend')}, config {manifest.get('config')}, git {(manifest.get('git') or {}).get('short')})"
    )
    _print_table(header, rows)
    return sr.EXIT_OK


def cmd_fetch(args: argparse.Namespace) -> int:
    from reinforcetactics.cloud.storage import download_tree

    path, manifest = _resolve_manifest(args)
    vertex = manifest.get("vertex") or {}
    output_uri = vertex.get("output_uri")
    if not output_uri:
        raise UsageError(f"{path}: not a Vertex group (no vertex.output_uri)")
    root = _root_of(path)
    exclude = re.compile(FETCH_EXCLUDE_KEEP_ZIPS if args.include_checkpoints else FETCH_EXCLUDE)
    total = 0
    for seed in manifest["seeds"]:
        run = manifest["runs"][str(seed)]["run_dir"]
        count = download_tree(f"{output_uri}/{run}", root / run, exclude=exclude)
        total += count
        print(f"  s{seed}: {count} file(s) -> {root / run}")
    print(f"Fetched {total} file(s)")
    return sr.EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args, passthrough = parse_args(list(sys.argv[1:] if argv is None else argv))
        if args.command == "status":
            return cmd_status(args)
        if args.command == "fetch":
            return cmd_fetch(args)
        return cmd_launch(args, passthrough)
    except UsageError as exc:
        print(f"run_seeds.py: {exc}", file=sys.stderr)
        return sr.EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
