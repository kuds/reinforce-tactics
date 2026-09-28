#!/usr/bin/env python3
"""Launch, continue and inspect a group of seeds of the bootstrap curriculum (review §2.1).

One ``train_bootstrap.py`` per seed, each in ``<root>/<group>_s<seed>/``,
with the group recorded in ``<root>/_groups/<group>/seed_group.json``
(``<group>`` = ``<YYYYmmdd_HHMMSS>_<tag>``). Every seed runs the same
command, ``--resume-if-exists`` included, so re-running the launcher (or
resubmitting a Vertex job) after any interruption continues where each seed
stopped: finished and stalled seeds are left alone, interrupted ones resume.

    run_seeds.py [launch] --config C (--seeds LIST | --n-seeds N [--seed-stride 1000]) --tag T
        [--group GROUP] [--root benchmarks/bootstrap] [--parallel K] [--cpu-sets "0-9,10-19" | --no-pin]
        [--gpus 0,1 | --device cpu|cuda|auto] [--backend local|vertex] [--retry-failed]
        [--allow-seed-overlap] [--force] [--dry-run] [--tee | --no-tee] -- <train_bootstrap.py args>
    run_seeds.py status (--group-manifest M | --group G [--root R]) [--jobs]
    run_seeds.py fetch (--group-manifest M | --group G [--root R]) [--include-checkpoints]

Launch rules:

* Seeds default to a stride of 1000 from the config's seed (42, 1042, 2042);
  seeds whose random streams overlap (training envs or gate-eval episodes)
  are refused unless ``--allow-seed-overlap``.
* Continue a group with ``--group <id>``; the config, seeds and
  train_bootstrap.py args default to the group's, and different ones are
  refused unless ``--force`` (which is recorded in the manifest's history).
* The args after ``--`` go to every seed's train_bootstrap.py; ``--config``,
  ``--output-dir``, ``--resume``, ``--resume-if-exists``, ``--seed`` and
  ``--set seed=`` are the launcher's. ``--build-bc`` gets ``--bc-seed <seed>``.
* Local: seeds that are pending or interrupted are queued (failed ones only
  with ``--retry-failed``), at most ``--parallel`` at a time; with K > 1 each
  gets its own contiguous CPU set (``os.sched_setaffinity``) and
  ``--torch-threads`` of its size. Each session's output goes to
  ``<run dir>/logs/train.<UTC stamp>.log`` (and the console with ``--tee``,
  the default for K = 1). SIGINT / SIGTERM is forwarded to every child,
  which checkpoints and exits; after 600 s the rest are killed.
* Vertex (``--backend vertex``): one custom job per seed through
  scripts/cloud/submit_vertex_job.sh (needs BUCKET and IMAGE_URI or TAG;
  PROJECT_ID / REGION / MACHINE_TYPE / ... pass through). All seeds write
  under ``gs://<BUCKET>/jobs/<group>/<run dir>/``; a resubmitted job
  restores its run dir from there first. A seed whose job is still active,
  or whose run in GCS finished or stalled, is never resubmitted (reading
  run_status.json needs google-cloud-storage; without it name the seeds to
  resubmit with ``--only-seeds``). ``fetch`` downloads the runs into
  ``<root>/`` (without checkpoints, traces, videos or tensorboard by default).

Exit codes: 0 every seed completed; 3 some stalled (none failed or
interrupted); 1 a seed failed (or a submission did); 130 / 143 interrupted;
2 a usage error (a refused relaunch or seed overlap included).

Examples:
    python3 scripts/train/run_seeds.py --config configs/ppo/bootstrap.yaml --n-seeds 3 --tag val \\
        --parallel 3 -- --strict --skip-videos
    python3 scripts/train/run_seeds.py --group 20260928_120000_val            # continue it
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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from reinforcetactics.experiments import seed_runs as sr  # noqa: E402

SUBCOMMANDS = ("launch", "status", "fetch")
DEFAULT_KILL_TIMEOUT = 600.0
FETCH_EXCLUDE = r"\.zip$|(^|/)(traces|videos|tensorboard)/"
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
    launch.add_argument("--parallel", type=int, default=1, help="Seeds run at once (local)")
    launch.add_argument("--cpu-sets", help='CPU set per parallel slot, e.g. "0-9,10-19" (default: an even split)')
    launch.add_argument("--no-pin", action="store_true", help="Do not pin children to CPU sets")
    launch.add_argument("--gpus", help="GPU ids, one per slot round-robin, e.g. 0,1 (implies --device cuda)")
    launch.add_argument(
        "--device", choices=("cpu", "cuda", "auto"), help="train_bootstrap.py --device (default: auto; cuda on Vertex)"
    )
    launch.add_argument("--backend", choices=("local", "vertex"), help="Where the seeds run (default: local, or the group's)")
    launch.add_argument("--retry-failed", action="store_true", help="Also relaunch seeds whose last session failed")
    launch.add_argument("--allow-seed-overlap", action="store_true", help="Launch even if seeds share random streams")
    launch.add_argument("--force", action="store_true", help="Continue the group with a different config / seeds / args")
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


def _prepare_group(args: argparse.Namespace, passthrough: list[str]) -> dict[str, Any]:
    """Resolve the group, its manifest (new or continued), the config, the seeds and the train args."""
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
    else:
        manifest = sr.new_group_manifest(
            group=group, root=root, backend=backend, config=config, digest=digest, seeds=seeds, train_args=train_args, git=git
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
    }


def _local(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    root, manifest, seeds = plan["root"], plan["manifest"], plan["seeds"]
    runs = manifest["runs"]
    states = {s: sr.seed_state(root / runs[str(s)]["run_dir"], runs[str(s)]) for s in seeds}
    queue = [s for s in seeds if states[s] in (sr.PENDING, sr.INTERRUPTED) or (states[s] == sr.FAILED and args.retry_failed)]
    parallel = max(1, int(args.parallel))
    cpu_sets: list[list[int]] | None = None
    if args.cpu_sets and not args.no_pin:
        cpu_sets = sr.parse_cpu_sets(args.cpu_sets)
        if len(cpu_sets) < parallel:
            raise UsageError(f"--cpu-sets gives {len(cpu_sets)} set(s) for --parallel {parallel}")
    elif parallel > 1 and not args.no_pin:
        try:
            cpu_sets = sr.split_cpus(sr.available_cpus(), parallel)
        except ValueError as exc:
            raise UsageError(str(exc)) from None
    for warning in sr.cpu_warnings(cpu_sets or [], plan["effective"]):
        print(f"  warning: {warning}", file=sys.stderr)
    if cpu_sets and not hasattr(os, "sched_setaffinity"):
        print("  warning: CPU pinning needs Linux (os.sched_setaffinity); children run unpinned", file=sys.stderr)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else []
    device = args.device or ("cuda" if gpus else "auto")
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
    _print_table(["seed", "state", "run dir"], [[s, states[s], runs[str(s)]["run_dir"]] for s in seeds])
    skipped_failed = [s for s in seeds if states[s] == sr.FAILED and not args.retry_failed]
    if skipped_failed:
        print(f"  failed seed(s) {skipped_failed} are not relaunched; read their logs, fix, then pass --retry-failed")
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
        record["last_state"] = "running"
        sr.write_group(path, manifest)
        print(f"  started seed {spec.seed} (pid {session['pid']}); log: {session['log']}")

    def on_exit(spec: sr.ChildSpec, session: dict[str, Any]) -> None:
        record = runs[str(spec.seed)]
        record["last_state"] = sr.state_after_exit(spec.run_dir, int(session["exit_code"]))
        sr.write_group(path, manifest)
        print(f"  seed {spec.seed} exited {session['exit_code']}: {record['last_state']}")

    result = sr.run_local(
        queue,
        make_spec,
        parallel=parallel,
        tee=args.tee if args.tee is not None else parallel == 1,
        on_start=on_start,
        on_exit=on_exit,
        kill_timeout=args.kill_timeout,
    )
    final = {s: sr.seed_state(root / runs[str(s)]["run_dir"], runs[str(s)]) for s in seeds}
    print(f"Group {plan['group']}:")
    _print_table(["seed", "state", "exit"], [[s, final[s], result.exit_codes.get(s, "—")] for s in seeds])
    print(f"  continue with: {Path(sys.argv[0]).name} --group {plan['group']} --root {root}")
    return sr.launcher_exit_code(final, result.signalled, codes=result.exit_codes)


# ---------------------------------------------------------------------------
# Vertex
# ---------------------------------------------------------------------------


def _vertex_env() -> dict[str, str]:
    bucket = os.environ.get("BUCKET", "").strip()
    if not bucket:
        raise UsageError("--backend vertex needs BUCKET (the GCS bucket the jobs write to)")
    if not os.environ.get("IMAGE_URI") and not os.environ.get("TAG"):
        raise UsageError("--backend vertex needs IMAGE_URI, or TAG=<short sha> of an image built from this commit")
    return {k: os.environ[k] for k in ("BUCKET", "PROJECT_ID", "REGION", "IMAGE_URI", "TAG") if os.environ.get(k)}


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
    image_env = _vertex_env()
    manifest, group = plan["manifest"], plan["group"]
    config = _repo_relative(plan["config"])
    device = args.device or "cuda"
    bucket = image_env["BUCKET"]
    output_uri = sr.group_output_uri(bucket, group)
    manifest["vertex"] = {
        "project": image_env.get("PROJECT_ID"),
        "region": image_env.get("REGION"),
        "bucket": sr.bucket_name(bucket),
        "output_uri": output_uri,
        "image_uri": image_env.get("IMAGE_URI") or f"TAG={image_env.get('TAG')}",
    }
    try:
        only = set(sr.parse_seeds(args.only_seeds)) if args.only_seeds else None
    except ValueError as exc:
        raise UsageError(f"--only-seeds: {exc}") from None
    runner = _runner()
    path = plan["manifest_path"]
    failed = False
    print(f"Group {group}: {len(plan['seeds'])} seed(s) on Vertex -> {output_uri}")
    for seed in plan["seeds"]:
        if only is not None and seed not in only:
            continue
        record = manifest["runs"][str(seed)]
        run = record["run_dir"]
        resource = record.get("job_resource")
        if resource:
            state = sr.describe_job_state(resource, runner=runner)
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
        env, argv = sr.vertex_submission(
            group=group,
            seed=seed,
            config=config,
            device=device,
            passthrough=plan["train_args"],
            bucket=bucket,
            image_env=image_env,
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
        }
        record["sessions"].append(session)
        if proc.returncode != 0 or not job:
            failed = True
            record["last_state"] = sr.FAILED
            print(f"  s{seed}: submission failed (exit {proc.returncode}{', no job resource printed' if not job else ''})")
        else:
            record["job_resource"] = job
            record["last_state"] = "submitted"
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
        if args.jobs and record.get("job_resource"):
            row.append(sr.describe_job_state(record["job_resource"], runner=_runner()) or "?")
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
