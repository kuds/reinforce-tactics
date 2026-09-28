"""Seed replication of the bootstrap curriculum: names, seeds, run states and the launcher's plumbing.

The review's §2.1 validation run is ``configs/ppo/bootstrap.yaml`` at three
seeds. ``scripts/train/run_seeds.py`` launches one ``train_bootstrap.py``
per seed (locally, sequentially or in parallel, or as one Vertex AI job per
seed) and records the group in ``<root>/_groups/<group>/seed_group.json``;
``scripts/eval/summarize_seeds.py`` aggregates it. This module holds their
logic.

Pure logic: the stdlib (and yaml) only at import, no torch, so the launcher
and the tests import it without the training stack. Configs are taken as the
plain ``TrainingConfig.to_dict()`` mapping (or any object with a
``to_dict``).

Layout (flat, so ``train_bootstrap.py``'s GCS upload, ``vertex_train.py``'s
sync of ``benchmarks/bootstrap`` and the analysis notebook's run discovery
all keep working)::

    <root>/<group>_s<seed>/              one run dir per seed
    <root>/<group>_s<seed>/logs/train.<UTC stamp>.log   one per launcher session
    <root>/_groups/<group>/seed_group.json
    <root>/_groups/<group>/report/       summarize_seeds.py output

with ``<group> = <YYYYmmdd_HHMMSS>_<tag>``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = "scripts/train/train_bootstrap.py"
SUBMIT_SCRIPT = "scripts/cloud/submit_vertex_job.sh"
DEFAULT_ROOT = "benchmarks/bootstrap"
GROUPS_DIR = "_groups"
GROUP_MANIFEST = "seed_group.json"
MANIFEST_VERSION = 1
DEFAULT_SEED_STRIDE = 1000
PARTIAL_SUFFIX = ".partial"

# Exit codes shared with train_bootstrap.py (0 / 1 / 3 / 130 / 143) plus 2
# for a usage error.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_STALLED = 3
EXIT_SIGINT = 128 + signal.SIGINT
EXIT_SIGTERM = 128 + signal.SIGTERM
INTERRUPT_CODES = frozenset({EXIT_SIGINT, EXIT_SIGTERM})
# A child killed from outside without a chance to checkpoint: a hangup
# (SIGHUP, 129) or SIGKILL (137: ``kill -9``, the kernel's OOM killer). The
# run dir still resumes from its last rolling checkpoint, so these count as
# interrupted, not failed (numeric: SIGHUP and SIGKILL do not exist on Windows).
EXIT_SIGHUP = 129
EXIT_SIGKILL = 137
KILLED_CODES = frozenset({EXIT_SIGHUP, EXIT_SIGKILL})

# Run states (classify_run), plus FAILED from a session's exit code, RUNNING
# when a live process holds the run dir's lock (run_is_live), and SUBMITTED
# for a Vertex seed whose job was submitted and whose run dir is not here.
PENDING = "pending"
INTERRUPTED = "interrupted"
COMPLETED = "completed"
STALLED = "stalled"
FAILED = "failed"
RUNNING = "running"
SUBMITTED = "submitted"
DONE_STATES = frozenset({COMPLETED, STALLED})

# train_bootstrap.py holds an exclusive flock on this file in its output
# directory for as long as it runs (acquire_run_lock), so a second trainer
# never resumes a run dir that is still being written, and the launcher can
# tell a live run from an interrupted one.
RUN_LOCK = ".train_bootstrap.lock"

TAG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")
GROUP_RE = re.compile(r"^(\d{8}_\d{6})_([a-z0-9][a-z0-9-]{0,30})$")
RUN_RE = re.compile(r"^(\d{8}_\d{6}_[a-z0-9][a-z0-9-]{0,30})_s(\d+)$")

# train_bootstrap.py options the launcher sets itself (per seed), so a
# passthrough may not carry them.
_RESERVED_OPTIONS = ("--config", "--output-dir", "--resume", "--resume-if-exists", "--seed")


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def check_tag(tag: str) -> str:
    if not isinstance(tag, str) or not TAG_RE.match(tag):
        raise ValueError(f"tag must match {TAG_RE.pattern} (lowercase letters, digits, '-'; at most 31), got {tag!r}")
    return tag


def make_group_id(tag: str, now: datetime | None = None) -> str:
    """``<YYYYmmdd_HHMMSS>_<tag>``: a group of seeds launched together."""
    check_tag(tag)
    return f"{(now or datetime.now()).strftime('%Y%m%d_%H%M%S')}_{tag}"


def parse_group_id(group: str) -> tuple[str, str]:
    """``(timestamp, tag)`` of a group id; a malformed id is a ValueError."""
    m = GROUP_RE.match(group or "")
    if not m:
        raise ValueError(f"not a seed group id (<YYYYmmdd_HHMMSS>_<tag>): {group!r}")
    return m.group(1), m.group(2)


def run_dir_name(group: str, seed: int) -> str:
    """``<group>_s<seed>``: the run directory of one seed (flat under the root)."""
    parse_group_id(group)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    return f"{group}_s{seed}"


def parse_run_dir_name(name: str) -> tuple[str, int] | None:
    """``(group, seed)`` of a seed run dir name, or None for any other name."""
    m = RUN_RE.match(name)
    return (m.group(1), int(m.group(2))) if m else None


def group_dir(root: str | Path, group: str) -> Path:
    return Path(root) / GROUPS_DIR / group


def group_manifest_path(root: str | Path, group: str) -> Path:
    return group_dir(root, group) / GROUP_MANIFEST


def vertex_job_name(group: str, seed: int) -> str:
    """``rt-<tag>-<YYYYmmdd-HHMMSS>-s<seed>``: the display name of a seed's Vertex job (same on resubmission)."""
    stamp, tag = parse_group_id(group)
    return f"rt-{tag}-{stamp.replace('_', '-')}-s{seed}"


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def _check_distinct(seeds: Sequence[int]) -> list[int]:
    repeated = sorted({s for s in seeds if list(seeds).count(s) > 1})
    if repeated:
        raise ValueError(f"duplicate seed(s): {repeated}")
    if any(s < 0 for s in seeds):
        raise ValueError(f"seeds must be >= 0, got {list(seeds)}")
    return list(seeds)


def parse_seeds(text: str) -> list[int]:
    """``"42,1042,2042"`` -> ``[42, 1042, 2042]``; duplicates, negatives and junk are ValueErrors."""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError("no seeds given")
    try:
        seeds = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"seeds must be comma-separated integers, got {text!r}") from None
    return _check_distinct(seeds)


def seeds_from_stride(base: int, n: int, stride: int = DEFAULT_SEED_STRIDE) -> list[int]:
    """``n`` seeds ``base, base + stride, ...`` (42, 1042, 2042 for base 42)."""
    if n < 1:
        raise ValueError(f"need at least one seed, got n={n}")
    if stride < 1:
        raise ValueError(f"seed stride must be >= 1, got {stride}")
    return _check_distinct([int(base) + i * int(stride) for i in range(n)])


def _as_dict(cfg: Any) -> Mapping[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if not isinstance(cfg, Mapping):
        raise TypeError(f"expected a config mapping or TrainingConfig, got {type(cfg).__name__}")
    return cfg


@dataclass(frozen=True)
class SeedStreams:
    """What a run's seed drives, from the config (see :func:`seed_streams`)."""

    n_envs: int
    n_eval: int
    n_seats: int
    seed_offset: int
    resample: bool
    max_blocks: int


def seed_streams(cfg: Any) -> SeedStreams:
    """The random streams one run's ``seed`` (s) drives:

    * training env rank r resets with ``s + r`` (r < ``env.n_envs``), at
      every stage (the vec env is rebuilt);
    * gate-eval episode i resets with ``s + eval.seed_offset + i``
      (i < the largest per-stage ``n_eval_episodes``), the same seeds for
      each seat; with ``eval.resample_eval_seeds``, plus ``1000 * block``
      where block = cumulative steps // ``eval.eval_freq`` (bounded by the
      whole curriculum's budget with every retry used);
    * the stochastic eval's policy stream is derived from each episode seed.

    A key the mapping leaves out takes the ``TrainingConfig`` default (a raw
    YAML works as well as ``to_dict()``).
    """
    data = _as_dict(cfg)
    env = data.get("env") or {}
    ev = data.get("eval") or {}
    cur = data.get("curriculum") or {}
    stages = cur.get("stages") or []

    def pick(section: Mapping[str, Any], key: str, default: Any) -> Any:
        value = section.get(key)
        return default if value is None else value

    default_eval = int(pick(ev, "n_eval_episodes", _DEFAULTS["n_eval_episodes"]))
    n_eval = max([int(pick(s, "n_eval_episodes", default_eval)) for s in stages] or [default_eval])
    default_retries = int(pick(cur, "max_retries", _DEFAULTS["max_retries"]))
    budget = sum(
        int(pick(s, "max_timesteps", _DEFAULTS["max_timesteps"])) * (1 + int(pick(s, "max_retries", default_retries)))
        for s in stages
    )
    eval_freq = max(1, int(pick(ev, "eval_freq", _DEFAULTS["eval_freq"])))
    seats = ev.get("eval_seats")
    if seats is None:
        seats = [1, 2] if env.get("agent_seat") == "random" else [1]
    return SeedStreams(
        n_envs=max(1, int(pick(env, "n_envs", _DEFAULTS["n_envs"]))),
        n_eval=max(1, n_eval),
        n_seats=len(seats),
        seed_offset=int(pick(ev, "seed_offset", _DEFAULTS["seed_offset"])),
        resample=bool(ev.get("resample_eval_seeds")),
        max_blocks=math.ceil(budget / eval_freq),
    )


# TrainingConfig defaults of the fields seed_streams reads (rl/config.py;
# repeated here so this module needs no torch). test_seed_runs checks them.
_DEFAULTS: dict[str, int] = {
    "n_envs": 4,
    "n_eval_episodes": 10,
    "eval_freq": 10000,
    "seed_offset": 1_000_000,
    "max_retries": 1,
    "max_timesteps": 1_000_000,
}


def _hits(delta: int, lo: int, hi: int, step: int, kmin: int, kmax: int) -> list[int]:
    """The integers k in [kmin, kmax] with lo < delta - step * k < hi."""
    if step == 0:
        return [0] if lo < delta < hi and kmin <= 0 <= kmax else []
    k_lo = (delta - hi) // step + 1
    k_hi = -((lo - delta) // step) - 1
    return list(range(max(k_lo, kmin), min(k_hi, kmax) + 1)) if max(k_lo, kmin) <= min(k_hi, kmax) else []


def seed_collisions(seeds: Sequence[int], cfg: Any) -> list[str]:
    """Why the runs of ``seeds`` would share random streams (empty: they would not).

    For seeds s1 < s2, d = s2 - s1: the training envs share streams when
    d < n_envs, and the gate evals share episodes (and their policy-sampling
    streams) when d < n_eval -- or, with ``resample_eval_seeds``, whenever
    d mod 1000 falls outside [n_eval, 1000 - n_eval] within the block range.
    A run's training envs meeting another run's eval seeds (d near
    ``seed_offset``) is reported too. Consecutive seeds (42, 43) share
    n_envs - 1 training streams and n_eval - 1 eval episode seeds per seat
    (under bootstrap.yaml, 7 of 8 and 59 of 60); the default stride of 1000
    shares none.
    """
    st = seed_streams(cfg)
    eval_blocks = st.max_blocks if st.resample else 0
    step = 1000 if st.resample else 0
    out: list[str] = []
    ordered = sorted(int(s) for s in seeds)
    for s in ordered:
        if _hits(-st.seed_offset, -st.n_envs, st.n_eval, step, 0, eval_blocks):
            out.append(
                f"seed {s}: its training envs reuse its own eval seeds (eval.seed_offset {st.seed_offset} is too small)"
            )
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            d = b - a
            if d < st.n_envs:
                out.append(
                    f"seeds {a} and {b}: the training envs share {st.n_envs - d} of {st.n_envs} env seed streams "
                    f"(difference {d} < env.n_envs {st.n_envs})"
                )
            if st.resample:
                ks = _hits(d, -st.n_eval, st.n_eval, 1000, -st.max_blocks, st.max_blocks)
                if ks:
                    out.append(
                        f"seeds {a} and {b}: with eval.resample_eval_seeds, eval blocks {ks[0]} apart replay the same "
                        f"episodes (difference {d}; d mod 1000 = {d % 1000} is outside [{st.n_eval}, {1000 - st.n_eval}])"
                    )
            elif d < st.n_eval:
                out.append(
                    f"seeds {a} and {b}: the gate evals share {st.n_eval - d} of {st.n_eval} episode seeds per seat, "
                    f"policy-sampling streams included (difference {d} < n_eval_episodes {st.n_eval})"
                )
            for x, y, dd in ((a, b, d - st.seed_offset), (b, a, -d - st.seed_offset)):
                if _hits(dd, -st.n_envs, st.n_eval, step, 0, eval_blocks):
                    out.append(f"seeds {x} and {y}: the training envs of seed {y} reuse eval episode seeds of seed {x}")
    return out


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def classify_run(run_dir: str | Path) -> str:
    """The state of a train_bootstrap.py run directory.

    * ``completed``: run_status.json says completed_curriculum and
      final_model.zip exists;
    * ``stalled``: run_status.json says curriculum_stalled;
    * ``interrupted``: resolved_config.yaml exists but the run did not end
      (no run_status.json -- or a completed one whose final_model.zip is
      missing, which ``--resume-if-exists`` rebuilds);
    * ``pending``: nothing was trained yet.
    """
    run_dir = Path(run_dir)
    status = _read_json(run_dir / "run_status.json")
    kind = status.get("status") if isinstance(status, dict) else None
    if kind == "completed_curriculum" and (run_dir / "final_model.zip").is_file():
        return COMPLETED
    if kind == "curriculum_stalled":
        return STALLED
    if (run_dir / "resolved_config.yaml").is_file():
        return INTERRUPTED
    return PENDING


# ---------------------------------------------------------------------------
# The run-dir lock: one trainer per run directory
# ---------------------------------------------------------------------------


class RunLockedError(RuntimeError):
    """Another live process holds the run directory's lock."""


def _fcntl() -> Any:
    try:
        import fcntl
    except ImportError:  # Windows: no flock; the lock is skipped
        return None
    return fcntl


def lock_holder(run_dir: str | Path) -> dict[str, Any]:
    """What the lock file says about its holder (``pid``, ``host``, ``started_at``); {} if unreadable."""
    data = _read_json(Path(run_dir) / RUN_LOCK)
    return data if isinstance(data, dict) else {}


# Locks this process holds: {lock path: (pid, fd)}. flock treats every open()
# of the file separately, so a second acquire in the same process (a test
# calling train_bootstrap.main twice) must reuse the first descriptor.
_HELD_LOCKS: dict[str, tuple[int, int]] = {}


def _held_fd(path: Path) -> int | None:
    held = _HELD_LOCKS.get(str(path))
    if held is None:
        return None
    pid, fd = held
    try:
        same = pid == os.getpid() and os.fstat(fd).st_ino == os.stat(path).st_ino
    except OSError:
        same = False
    if same:
        return fd
    _HELD_LOCKS.pop(str(path), None)
    if pid == os.getpid():
        try:
            os.close(fd)
        except OSError:
            pass
    return None


def acquire_run_lock(run_dir: str | Path, *, attempts: int = 5, wait: float = 0.2) -> int | None:
    """Take an exclusive lock on ``<run_dir>/RUN_LOCK`` and keep it for the life of this process.

    Returns the lock's file descriptor (it stays open; the lock ends when
    the process does, however it ends, or with :func:`release_run_lock`), or
    None where file locks do not work (no ``fcntl``, or a filesystem without
    flock support): the run goes on unlocked. Raises :class:`RunLockedError`
    when another process holds the lock -- a trainer still running in this
    directory. ``attempts`` covers a status probe holding it for an instant.
    """
    fcntl = _fcntl()
    if fcntl is None:
        return None
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_LOCK
    held = _held_fd(path)
    if held is not None:
        return held
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:  # a read-only or odd filesystem: no lock
        return None
    for attempt in range(max(1, attempts)):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if attempt + 1 >= max(1, attempts):
                os.close(fd)
                holder = lock_holder(run_dir)
                who = ", ".join(f"{k} {holder[k]}" for k in ("pid", "host", "started_at") if holder.get(k))
                raise RunLockedError(
                    f"{run_dir} is in use by another process ({who or 'holder unknown'}); a second trainer "
                    "would write the same run dir. Wait for it to finish, or stop it first."
                ) from None
            time.sleep(wait)
        except OSError:
            # A filesystem without flock (some network / FUSE mounts): no lock.
            os.close(fd)
            return None
    info = {"pid": os.getpid(), "host": _hostname(), "started_at": utc_now()}
    try:
        os.ftruncate(fd, 0)
        os.write(fd, (json.dumps(info) + "\n").encode("utf-8"))
    except OSError:
        pass
    _HELD_LOCKS[str(path)] = (os.getpid(), fd)
    return fd


def release_run_lock(run_dir: str | Path) -> None:
    """Release a lock this process took with :func:`acquire_run_lock` (no-op otherwise)."""
    path = Path(run_dir).resolve() / RUN_LOCK
    held = _HELD_LOCKS.pop(str(path), None)
    if held is not None and held[0] == os.getpid():
        try:
            os.close(held[1])
        except OSError:
            pass


def run_lock_held(run_dir: str | Path) -> bool | None:
    """Whether a live process holds the run dir's lock: None when there is no lock file or locks do not work.

    Read-only: the probe takes a shared lock for an instant and never
    creates the file.
    """
    fcntl = _fcntl()
    path = Path(run_dir) / RUN_LOCK
    if fcntl is None or not path.is_file():
        return None
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return None
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _hostname() -> str:
    import socket

    try:
        return socket.gethostname()
    except OSError:
        return ""


def _pid_runs(pid: Any, run_dir: Path) -> bool:
    """Whether process ``pid`` is alive here and (where /proc shows it) its command names ``run_dir``."""
    try:
        pid = int(pid)
        os.kill(pid, 0)
    except (TypeError, ValueError, ProcessLookupError, OverflowError):
        return False
    except PermissionError:
        pass  # alive, another user's
    except OSError:
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.parent.parent.is_dir():  # Linux: a reused pid runs some other command
        try:
            args = cmdline.read_bytes().split(b"\0")
        except OSError:
            return False
        return any(os.fsencode(run_dir.name) in a for a in args)
    return True


def run_is_live(run_dir: str | Path, record: Mapping[str, Any] | None = None) -> bool:
    """Whether a process is still training in ``run_dir``.

    The run dir's lock answers when there is one (train_bootstrap.py holds it
    while it runs). Otherwise -- a trainer that takes no lock, a filesystem
    without flock -- the last session the launcher recorded as running on
    this host counts while its pid is alive and still runs this seed.
    """
    run_dir = Path(run_dir)
    held = run_lock_held(run_dir)
    if held is not None:
        return held
    record = record or {}
    sessions = record.get("sessions") or []
    if record.get("last_state") != RUNNING or not sessions:
        return False
    last = sessions[-1]
    if last.get("ended_at") or not last.get("pid"):
        return False
    if last.get("host") and last["host"] != _hostname():
        return False
    return _pid_runs(last["pid"], run_dir)


# ---------------------------------------------------------------------------
# CPUs and GPUs
# ---------------------------------------------------------------------------


def available_cpus() -> list[int]:
    """The CPUs this process may run on (``os.sched_getaffinity(0)`` where it exists)."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        return sorted(getter(0))
    return list(range(os.cpu_count() or 1))


def split_cpus(cpus: Sequence[int], k: int) -> list[list[int]]:
    """``k`` contiguous, even chunks of ``cpus`` (sizes differ by at most one; the first ones get the extra)."""
    ordered = sorted(int(c) for c in cpus)
    if k < 1:
        raise ValueError(f"need at least one chunk, got {k}")
    if k > len(ordered):
        raise ValueError(f"cannot split {len(ordered)} CPU(s) into {k} sets")
    size, extra = divmod(len(ordered), k)
    chunks, start = [], 0
    for i in range(k):
        end = start + size + (1 if i < extra else 0)
        chunks.append(ordered[start:end])
        start = end
    return chunks


def parse_cpu_sets(text: str) -> list[list[int]]:
    """``"0-9,10-19"`` -> two sets; each comma-separated item is ``a-b`` or ``a``."""
    sets: list[list[int]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        lo, sep, hi = item.partition("-")
        try:
            first, last = int(lo), int(hi) if sep else int(lo)
        except ValueError:
            raise ValueError(f"bad CPU set {item!r} (expected a-b or a)") from None
        if first < 0 or last < first:
            raise ValueError(f"bad CPU set {item!r}")
        sets.append(list(range(first, last + 1)))
    if not sets:
        raise ValueError("no CPU sets given")
    return sets


def cpus_needed(cfg: Any) -> int:
    """CPUs one run keeps busy: the main process, its env workers and its eval workers."""
    data = _as_dict(cfg)
    env = data.get("env") or {}
    ev = data.get("eval") or {}
    n = 1
    if env.get("use_subprocess", True):
        n += int(env.get("n_envs") or _DEFAULTS["n_envs"])
    if ev.get("eval_use_subprocess") and int(ev.get("n_eval_envs") or 1) > 1:
        n += int(ev.get("n_eval_envs") or 1)
    return n


def cpu_warnings(cpu_sets: Sequence[Sequence[int]], cfg: Any) -> list[str]:
    need = cpus_needed(cfg)
    return [
        f"CPU set {i} ({len(cs)} CPU(s)) is smaller than the {need} one run keeps busy (main + env/eval workers); "
        "its processes will share cores"
        for i, cs in enumerate(cpu_sets)
        if len(cs) < need
    ]


def format_cpu_set(cpus: Sequence[int] | None) -> str | None:
    if not cpus:
        return None
    ordered = sorted(cpus)
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}" if len(ordered) > 1 else str(ordered[0])
    return ",".join(str(c) for c in ordered)


# ---------------------------------------------------------------------------
# Child command and environment
# ---------------------------------------------------------------------------


def check_passthrough(args: Sequence[str]) -> None:
    """Refuse train_bootstrap.py options the launcher sets per seed (ValueError)."""
    args = list(args)
    for i, arg in enumerate(args):
        name = arg.split("=", 1)[0]
        if name in _RESERVED_OPTIONS:
            raise ValueError(f"{name} is set per seed by the launcher; remove it from the train_bootstrap.py arguments")
        if name == "--set":
            value = arg.split("=", 1)[1] if "=" in arg else (args[i + 1] if i + 1 < len(args) else "")
            if value.split("=", 1)[0].strip() == "seed":
                raise ValueError("--set seed=... conflicts with the launcher's per-seed --seed")
        if name == "--bc-seed":
            raise ValueError("--bc-seed is set per seed by the launcher (with --build-bc)")


def build_child_command(
    config: str | Path,
    seed: int,
    run_dir: str | Path,
    device: str | None,
    passthrough: Sequence[str] = (),
    *,
    torch_threads: int | None = None,
    script: str | Path | None = None,
    python: str | None = None,
) -> list[str]:
    """``[python, train_bootstrap.py, --config C, --seed S, --output-dir D, --resume-if-exists, --device X, ...]``.

    The same command starts a seed and continues it after any interruption.
    A passed-through ``--build-bc`` gets ``--bc-seed <seed>``, so each seed's
    warm start differs too; ``torch_threads`` (the size of a pinned CPU set)
    becomes ``--torch-threads``.
    """
    check_passthrough(passthrough)
    cmd = [
        python or sys.executable,
        str(script or (REPO_ROOT / TRAIN_SCRIPT)),
        "--config",
        str(config),
        "--seed",
        str(int(seed)),
        "--output-dir",
        str(run_dir),
        "--resume-if-exists",
    ]
    if device:
        cmd += ["--device", str(device)]
    cmd += list(passthrough)
    if "--build-bc" in passthrough:
        cmd += ["--bc-seed", str(int(seed))]
    if torch_threads is not None:
        cmd += ["--torch-threads", str(int(torch_threads))]
    return cmd


def build_child_env(cpu_set: Sequence[int] | None, gpu: str | None, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The child's environment: one BLAS/OMP thread per process, headless SDL/matplotlib, unbuffered output.

    ``gpu`` becomes ``CUDA_VISIBLE_DEVICES`` (``""`` hides every GPU, for a
    CPU run); ``None`` leaves the variable as inherited. ``cpu_set`` is only
    recorded here -- pinning happens in :func:`run_local`.
    """
    env = dict(os.environ if base is None else base)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        SDL_VIDEODRIVER="dummy",
        MPLBACKEND="Agg",
    )
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


# ---------------------------------------------------------------------------
# The group manifest
# ---------------------------------------------------------------------------


def config_digest(cfg: Any) -> str:
    """sha256 of the config's canonical JSON, without ``seed`` and ``ppo.device`` (the same for every seed)."""
    data = json.loads(json.dumps(_as_dict(cfg), default=str))
    data.pop("seed", None)
    if isinstance(data.get("ppo"), dict):
        data["ppo"].pop("device", None)
    canon = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def git_info(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """``{commit, short, dirty}`` of the checkout (Nones outside a git checkout)."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "short": None, "dirty": None}
    return {"commit": commit, "short": commit[:7], "dirty": dirty}


def resolve_commit(ref: str, repo_root: str | Path = REPO_ROOT) -> str | None:
    """The full sha of commit ``ref`` (a short sha, a tag) in the checkout, or None when it names no commit here."""
    ref = str(ref or "").strip()
    if not ref or ref.startswith("-"):
        return None
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_group_manifest(
    *,
    group: str,
    root: str | Path,
    backend: str,
    config: str | Path,
    digest: str,
    seeds: Sequence[int],
    train_args: Sequence[str],
    git: Mapping[str, Any] | None = None,
    vertex: Mapping[str, Any] | None = None,
    launch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A new ``seed_group.json``.

    ``launch`` holds the local scheduling options (``parallel``,
    ``cpu_sets``, ``no_pin``, ``gpus``, ``device``, ``tee``) that a
    continuation without them reuses, as it reuses the config, seeds and
    train args.
    """
    _, tag = parse_group_id(group)
    return {
        "version": MANIFEST_VERSION,
        "group": group,
        "tag": tag,
        "created_at": utc_now(),
        "root": str(root),
        "backend": backend,
        "config": str(config),
        "config_digest": digest,
        "git": dict(git or {}),
        "seeds": [int(s) for s in seeds],
        "train_args": list(train_args),
        "vertex": dict(vertex) if vertex else None,
        "launch": dict(launch or {}),
        "runs": {
            str(int(s)): {
                "run_dir": run_dir_name(group, int(s)),
                "job_name": vertex_job_name(group, int(s)) if backend == "vertex" else None,
                "job_resource": None,
                "sessions": [],
                "last_state": PENDING,
            }
            for s in seeds
        },
        "history": [],
    }


def read_group(path: str | Path) -> dict[str, Any]:
    """Read a ``seed_group.json`` (ValueError if it is not one)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "group" not in data or "runs" not in data:
        raise ValueError(f"{path} is not a seed_group.json")
    if int(data.get("version", 0)) > MANIFEST_VERSION:
        raise ValueError(f"{path} has version {data.get('version')}; this tool reads version {MANIFEST_VERSION}")
    return data


def write_group(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write the manifest via a ``.partial`` sibling, so a kill never leaves it half-written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    try:
        partial.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def manifest_mismatches(
    manifest: Mapping[str, Any], *, digest: str, seeds: Sequence[int], train_args: Sequence[str]
) -> list[str]:
    """How a relaunch differs from the group it continues (empty: it does not)."""
    problems = []
    if manifest.get("config_digest") != digest:
        problems.append(f"config digest {digest[:12]} != the group's {str(manifest.get('config_digest'))[:12]}")
    if [int(s) for s in manifest.get("seeds", [])] != [int(s) for s in seeds]:
        problems.append(f"seeds {list(seeds)} != the group's {manifest.get('seeds')}")
    if list(manifest.get("train_args", [])) != list(train_args):
        problems.append(f"train_bootstrap.py args {list(train_args)} != the group's {manifest.get('train_args')}")
    return problems


def seed_state(run_dir: str | Path, record: Mapping[str, Any] | None) -> str:
    """A seed's state.

    Its run dir's (:func:`classify_run`) when finished; ``running`` while a
    process still trains in it (:func:`run_is_live`); ``failed`` when its
    last session failed; ``submitted`` for a Vertex seed whose job was
    submitted and whose run dir here (a fetched copy, or none yet) is not
    finished -- only the job knows whether it still runs (``status
    --jobs``); else the run dir's state.
    """
    state = classify_run(run_dir)
    if state in DONE_STATES:
        return state
    record = record or {}
    if record.get("last_state") == SUBMITTED:
        return SUBMITTED
    if run_is_live(run_dir, record):
        return RUNNING
    if record.get("last_state") == FAILED:
        return FAILED
    return state


def state_after_exit(run_dir: str | Path, exit_code: int) -> str:
    """A seed's state after a session ended with ``exit_code``.

    The run dir decides completed / stalled; otherwise 130 / 143 (a signal
    the child handled) and 129 / 137 (a hangup, ``kill -9`` or the OOM
    killer) mean interrupted -- the run resumes from its last rolling
    checkpoint -- and anything else failed (an exit 0 or 3 whose run dir
    does not say so included: the record and the exit code disagree).
    """
    state = classify_run(run_dir)
    if state in DONE_STATES:
        return state
    return INTERRUPTED if exit_code in INTERRUPT_CODES | KILLED_CODES else FAILED


def launcher_exit_code(states: Mapping[Any, str], signalled: int | None = None, codes: Mapping[Any, int] | None = None) -> int:
    """0 all completed; 3 some stalled (none failed or interrupted); 1 any failed or not finished; 130/143 interrupted.

    Not finished: pending, or still running under another launcher.
    """
    if signalled is not None:
        return 128 + int(signalled)
    values = list(states.values())
    if FAILED in values or PENDING in values or RUNNING in values or SUBMITTED in values:
        return EXIT_FAILED
    interrupted = [k for k, v in states.items() if v == INTERRUPTED]
    if interrupted:
        code = (codes or {}).get(interrupted[0])
        return code if code in INTERRUPT_CODES else EXIT_SIGTERM
    if STALLED in values:
        return EXIT_STALLED
    return EXIT_OK


# ---------------------------------------------------------------------------
# Progress of one run (``run_seeds.py status``)
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def stage_names_of(run_dir: str | Path) -> list[str]:
    """The run's stage order: its resolved_config.yaml, else its run_manifest.json."""
    import yaml

    run_dir = Path(run_dir)
    try:
        raw = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
        stages = ((raw.get("curriculum") or {}).get("stages")) or []
        names = [str(s["name"]) for s in stages if isinstance(s, dict) and s.get("name")]
        if names:
            return names
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        pass
    manifest = _read_json(run_dir / "run_manifest.json")
    return [str(n) for n in (manifest or {}).get("stages") or []] if isinstance(manifest, dict) else []


def steps_per_hour(rows: Iterable[Mapping[str, Any]], max_gap_s: float = 3600.0) -> float | None:
    """Median env steps per hour between consecutive eval rows of one session (from their ``wall_time``).

    Pairs further apart than ``max_gap_s`` (a resume, a pause) or out of
    order are skipped. None without two rows that carry ``wall_time``.
    """
    points = sorted(
        (float(r["wall_time"]), int(r["timesteps"]))
        for r in rows
        if isinstance(r.get("wall_time"), (int, float)) and r.get("timesteps") is not None
    )
    rates = []
    for (t0, s0), (t1, s1) in zip(points, points[1:], strict=False):
        dt, ds = t1 - t0, s1 - s0
        if 0 < dt < max_gap_s and ds > 0:
            rates.append(ds / dt * 3600.0)
    if not rates:
        return None
    rates.sort()
    mid = len(rates) // 2
    return rates[mid] if len(rates) % 2 else (rates[mid - 1] + rates[mid]) / 2


def run_progress(run_dir: str | Path) -> dict[str, Any]:
    """Where a run is: its state, current stage, steps, last gate win rate, stages cleared, resumes, steps/h."""
    run_dir = Path(run_dir)
    names = stage_names_of(run_dir)
    manifest = _read_json(run_dir / "run_manifest.json")
    manifest = manifest if isinstance(manifest, dict) else {}
    status = _read_json(run_dir / "run_status.json")
    status = status if isinstance(status, dict) else {}
    cleared: list[str] = []
    for name in names:
        record = _read_json(run_dir / name / "config.json")
        promoted = ((record or {}).get("extra") or {}).get("promoted") if isinstance(record, dict) else None
        if promoted is None:
            promoted = any(
                isinstance(e, dict) and e.get("stage") == name and e.get("promoted") for e in manifest.get("completed") or []
            )
        if not promoted:
            break
        cleared.append(name)
    current = manifest.get("current") or {}
    stage = current.get("stage") or status.get("stalled_stage")
    if not stage and len(cleared) < len(names):
        stage = names[len(cleared)]
    rows = _read_jsonl(run_dir / stage / "eval_results.jsonl") if stage else []
    all_rows = [r for name in names for r in _read_jsonl(run_dir / name / "eval_results.jsonl")]
    timesteps = current.get("latest_timesteps")
    if timesteps is None and all_rows:
        timesteps = max(int(r.get("timesteps", 0)) for r in all_rows)
    last = rows[-1] if rows else None
    gate_wr = None
    if last is not None:
        gate_wr = last.get("gate_win_rate", last.get("win_rate"))
    return {
        "state": classify_run(run_dir),
        "stage": stage,
        "stage_index": (names.index(stage) + 1) if stage in names else None,
        "n_stages": len(names),
        "timesteps": timesteps,
        "last_gate_wr": gate_wr,
        "stages_cleared": len(cleared),
        "resume_count": int(status.get("resume_count", manifest.get("resume_count", 0)) or 0),
        "steps_per_hour": steps_per_hour(all_rows),
    }


# ---------------------------------------------------------------------------
# The local scheduler
# ---------------------------------------------------------------------------


@dataclass
class ChildSpec:
    """One launch of one seed's train_bootstrap.py."""

    seed: int
    run_dir: Path
    cmd: list[str]
    env: dict[str, str]
    cpu_set: list[int] | None = None
    gpu: str | None = None
    log_path: Path | None = None
    # train_bootstrap.py resolves the config's map paths against the working
    # directory, so the children run from the repository root.
    cwd: Path | None = REPO_ROOT


@dataclass
class _Running:
    spec: ChildSpec
    proc: subprocess.Popen
    session: dict[str, Any]
    log: IO[str]
    pump: threading.Thread | None = None


@dataclass
class LocalRunResult:
    exit_codes: dict[int, int] = field(default_factory=dict)
    signalled: int | None = None


def session_log_path(run_dir: str | Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return Path(run_dir) / "logs" / f"train.{stamp}.log"


def _normalize_returncode(rc: int) -> int:
    """A child killed by signal N reports -N; report it as the shell's 128 + N."""
    return 128 - rc if rc < 0 else rc


def _pump(stream: IO[str], log: IO[str], console: IO[str], prefix: str) -> None:
    """Copy a child's output to its log and the console, line by line, until the pipe closes."""
    for line in iter(stream.readline, ""):
        for target, text in ((log, line), (console, prefix + line)):
            try:
                target.write(text)
                target.flush()
            except (OSError, ValueError):  # a closed log (the launcher gave up waiting) or console
                pass
    stream.close()


def _preexec(cpu_set: Sequence[int] | None) -> Callable[[], None] | None:
    if not cpu_set:
        return None
    setter = getattr(os, "sched_setaffinity", None)
    if setter is None:
        return None
    cpus = set(cpu_set)

    def pin() -> None:
        setter(0, cpus)

    return pin


def run_local(
    seeds: Sequence[int],
    make_spec: Callable[[int, int], ChildSpec],
    *,
    parallel: int = 1,
    tee: bool = True,
    on_start: Callable[[ChildSpec, dict[str, Any]], None] | None = None,
    on_exit: Callable[[ChildSpec, dict[str, Any]], None] | None = None,
    kill_timeout: float = 600.0,
    poll_interval: float = 0.2,
    console: IO[str] | None = None,
    handle_signals: bool = True,
) -> LocalRunResult:
    """Run one child per seed, at most ``parallel`` at a time, and record each as a session.

    ``make_spec(seed, slot)`` builds a child for a free slot (slot i gets CPU
    set i and GPU i mod len). Each child's output goes to its
    ``spec.log_path``, and with ``tee`` to ``console`` as well. A child is
    started in its own session and pinned to ``spec.cpu_set`` (Linux).
    ``on_start`` / ``on_exit`` get the session record (``started_at``,
    ``ended_at``, ``exit_code``, ``cmd``, ``cpu_set``, ``gpu``, ``pid``,
    ``log``) so the caller can persist it.

    SIGINT / SIGTERM to this process (with ``handle_signals``) is forwarded
    to every child, no new child starts, and after ``kill_timeout`` seconds
    (or a second signal) the children left are killed; the result's
    ``signalled`` is the signal. SIGHUP (the terminal or SSH session went
    away) is handled the same way, and the children get SIGTERM, which
    train_bootstrap.py turns into a checkpoint and exit 143: they run in
    their own sessions and would otherwise train on unrecorded.
    """
    console = console or sys.stdout
    queue = [int(s) for s in seeds]
    free_slots = list(range(max(1, int(parallel))))
    running: dict[int, _Running] = {}
    result = LocalRunResult()
    received: list[int] = []

    def on_signal(signum: int, _frame: Any) -> None:
        received.append(signum)

    previous: dict[int, Any] = {}
    if handle_signals and threading.current_thread() is threading.main_thread():
        handled = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            handled.append(signal.SIGHUP)
        for signum in handled:
            previous[signum] = signal.signal(signum, on_signal)

    def finish(slot: int, rc: int) -> None:
        item = running.pop(slot)
        if item.pump is not None:
            item.pump.join(timeout=10)
        item.log.close()
        code = _normalize_returncode(rc)
        item.session.update(ended_at=utc_now(), exit_code=code)
        result.exit_codes[item.spec.seed] = code
        free_slots.append(slot)
        free_slots.sort()
        if on_exit is not None:
            on_exit(item.spec, item.session)

    def start(seed: int, slot: int) -> None:
        spec = make_spec(seed, slot)
        log_path = spec.log_path or session_log_path(spec.run_dir)
        spec.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        log.write(f"# {utc_now()} {shlex.join(spec.cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            spec.cmd,
            env=spec.env,
            cwd=spec.cwd,
            stdout=subprocess.PIPE if tee else log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=True,
            preexec_fn=_preexec(spec.cpu_set),
        )
        pump = None
        if tee:
            assert proc.stdout is not None
            prefix = f"[s{seed}] " if parallel > 1 else ""
            pump = threading.Thread(target=_pump, args=(proc.stdout, log, console, prefix), daemon=True)
            pump.start()
        session = {
            "started_at": utc_now(),
            "ended_at": None,
            "exit_code": None,
            "cmd": list(spec.cmd),
            "cpu_set": format_cpu_set(spec.cpu_set),
            "gpu": spec.gpu,
            "pid": proc.pid,
            "host": _hostname(),
            "log": str(log_path),
        }
        running[slot] = _Running(spec=spec, proc=proc, session=session, log=log, pump=pump)
        if on_start is not None:
            on_start(spec, session)

    try:
        while queue or running:
            if received:
                break
            while queue and free_slots:
                start(queue.pop(0), free_slots.pop(0))
            for slot, item in list(running.items()):
                rc = item.proc.poll()
                if rc is not None:
                    finish(slot, rc)
            time.sleep(poll_interval)
        if received:
            first = received[0]
            result.signalled = first
            # A hangup is forwarded as SIGTERM, the signal the trainer checkpoints on.
            forward = signal.SIGTERM if first == getattr(signal, "SIGHUP", None) else first
            for item in running.values():
                try:
                    item.proc.send_signal(forward)
                except OSError:
                    pass
            deadline = time.monotonic() + kill_timeout
            while running and time.monotonic() < deadline and len(received) < 2:
                for slot, item in list(running.items()):
                    rc = item.proc.poll()
                    if rc is not None:
                        finish(slot, rc)
                time.sleep(poll_interval)
            for slot, item in list(running.items()):
                try:
                    os.killpg(item.proc.pid, signal.SIGKILL)
                except (OSError, AttributeError):
                    item.proc.kill()
                finish(slot, item.proc.wait())
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    return result


# ---------------------------------------------------------------------------
# Vertex AI: one custom job per seed through scripts/cloud/submit_vertex_job.sh
# ---------------------------------------------------------------------------

TERMINAL_JOB_STATES = frozenset(
    {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
    }
)
_JOB_RESOURCE_RE = re.compile(r"projects/[^/\s\]]+/locations/[^/\s\]]+/customJobs/\d+")


def bucket_name(bucket: str) -> str:
    """``gs://b/`` or ``b`` -> ``b``."""
    name = str(bucket).strip()
    if name.startswith("gs://"):
        name = name[len("gs://") :]
    name = name.strip("/").split("/", 1)[0]
    if not name:
        raise ValueError(f"not a bucket: {bucket!r}")
    return name


def group_output_uri(bucket: str, group: str) -> str:
    """``gs://<bucket>/jobs/<group>``: every seed of the group lands under it (``<uri>/<run dir>/``)."""
    return f"gs://{bucket_name(bucket)}/jobs/{group}"


def vertex_submission(
    *,
    group: str,
    seed: int,
    config: str,
    device: str,
    passthrough: Sequence[str],
    bucket: str,
    image_env: Mapping[str, str] | None = None,
    root: str = DEFAULT_ROOT,
    git_commit: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """``(env, argv)`` of the ``submit_vertex_job.sh`` call for one seed.

    The same spec serves the first submission and every resubmission: the
    job restores ``<OUTPUT_URI>/<run>/`` into ``benchmarks/bootstrap/<run>``
    before it starts (``RESTORE_DIRS``), and ``--resume-if-exists`` then
    starts, resumes, or finds the run done. ``git_commit`` (the commit the
    image was built from) reaches the job as ``RT_GIT_COMMIT``: the image
    has no ``.git``, so the run's records take the commit from there.
    """
    run = run_dir_name(group, seed)
    local = f"{root.rstrip('/')}/{run}"
    env = {
        **dict(image_env or {}),
        "BUCKET": bucket_name(bucket),
        "JOB_NAME": vertex_job_name(group, seed),
        "OUTPUT_URI": group_output_uri(bucket, group),
        "RESTORE_DIRS": f"{local}={run}",
    }
    if git_commit:
        env["RT_GIT_COMMIT"] = str(git_commit)
    train = ["python3", TRAIN_SCRIPT, "--config", str(config), "--seed", str(int(seed)), "--output-dir", local]
    train += ["--resume-if-exists", "--device", device, *passthrough]
    if "--build-bc" in passthrough:
        train += ["--bc-seed", str(int(seed))]
    return env, ["bash", SUBMIT_SCRIPT, *train]


def parse_job_resource(output: str) -> str | None:
    """The ``projects/.../customJobs/<id>`` a submission printed (its ``JOB_RESOURCE=`` line first)."""
    for line in output.splitlines():
        if line.startswith("JOB_RESOURCE="):
            value = line.split("=", 1)[1].strip()
            if value:
                return value
    m = _JOB_RESOURCE_RE.search(output)
    return m.group(0) if m else None


def job_region(resource: str) -> str | None:
    m = re.search(r"/locations/([^/]+)/", resource)
    return m.group(1) if m else None


Runner = Callable[..., Any]


def describe_job(resource: str, runner: Runner = subprocess.run) -> tuple[str | None, str]:
    """``(state, error message)`` of a job from ``gcloud ai custom-jobs describe``; ``(None, "")`` if unknown."""
    cmd = ["gcloud", "ai", "custom-jobs", "describe", resource, "--format=value(state,error.message)"]
    region = job_region(resource)
    if region:
        cmd.append(f"--region={region}")
    try:
        proc = runner(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return None, ""
    if getattr(proc, "returncode", 1) != 0:
        return None, ""
    lines = [line for line in (getattr(proc, "stdout", "") or "").strip().splitlines() if line.strip()]
    if not lines:
        return None, ""
    state, _, message = lines[-1].partition("\t")
    return state.strip() or None, message.strip()


def describe_job_state(resource: str, runner: Runner = subprocess.run) -> str | None:
    """The job's state (``JOB_STATE_RUNNING``, ...) from ``gcloud ai custom-jobs describe``; None if unknown."""
    return describe_job(resource, runner=runner)[0]


_EXIT_STATUS_RE = re.compile(r"non-zero status of (\d+)")


def job_exit_status(message: str) -> int | None:
    """The container's exit status in a failed job's error message (Vertex writes "... exited with a
    non-zero status of N ..."), else None."""
    m = _EXIT_STATUS_RE.search(message or "")
    return int(m.group(1)) if m else None


def gcs_run_state(output_uri: str, run: str, client: Any = None) -> str | None:
    """The run's state from ``<output_uri>/<run>/run_status.json`` in GCS (completed / stalled), else None.

    Raises ImportError when google-cloud-storage is missing and no
    ``client`` is given.
    """
    from reinforcetactics.cloud.storage import parse_gcs_uri

    if client is None:
        from google.cloud import storage  # noqa: F401 - raises ImportError without the library

        client = storage.Client()
    bucket, prefix = parse_gcs_uri(output_uri)
    base = f"{prefix}/{run}" if prefix else run
    handle = client.bucket(bucket)
    try:
        status = json.loads(handle.blob(f"{base}/run_status.json").download_as_text())
    except Exception:  # noqa: BLE001 - missing object or unreadable: not finished
        return None
    kind = status.get("status") if isinstance(status, dict) else None
    if kind == "curriculum_stalled":
        return STALLED
    if kind == "completed_curriculum":
        try:
            if handle.blob(f"{base}/final_model.zip").exists():
                return COMPLETED
        except Exception:  # noqa: BLE001
            return None
    return None
