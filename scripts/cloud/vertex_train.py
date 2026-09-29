#!/usr/bin/env python3
"""Container entrypoint that runs a training command and syncs artifacts to GCS.

This is the ``ENTRYPOINT`` of the training Docker image. It runs whatever
training command it is given (everything after the script name) as a child
process, and around that run it periodically — and once more on exit — uploads
the local output directories (``models/``, ``checkpoints/``, ``tensorboard/``,
``logs/``, plus ``benchmarks/bootstrap/`` where ``train_bootstrap.py`` writes
its runs) to Google Cloud Storage. That is what makes a Vertex AI custom job
useful: the machine is torn down when the job finishes, so anything not pushed
to GCS is lost.

The destination is taken from ``GCS_OUTPUT_URI`` (preferred) or, failing that,
Vertex's ``AIP_MODEL_DIR``. When neither is set the command still runs normally
and nothing is uploaded, so the same image works locally.

Environment variables:
    GCS_OUTPUT_URI     gs:// base for outputs (overrides AIP_MODEL_DIR).
    GCS_SYNC_INTERVAL  Seconds between periodic syncs (default 300; <=0 disables).
    GCS_SYNC_DIRS      Extra directories to sync, comma-separated. ``dir`` goes to
                       ``<base>/dir/``; ``dir=prefix`` goes to ``<base>/prefix/``;
                       ``dir=`` puts the directory's contents straight under
                       ``<base>/``. A file under two entries goes to both places.
    GCS_RESTORE_DIRS   Directories to download from the output base BEFORE the
                       command starts, comma-separated ``dir=prefix`` entries
                       (the GCS_SYNC_DIRS syntax): ``<base>/prefix/`` lands in
                       ``dir/``. Used by resubmitted seed jobs to continue
                       their run; charts/, videos/, checkpoints/, traces/ and
                       tensorboard/ are not restored. A failed restore fails
                       the job without running the command (a fresh start
                       would overwrite the stored run).
    GCS_CREDENTIALS    Optional path to a service-account JSON file.

The child gets ``GCS_WRAPPER_SYNC`` describing what the final sync uploads, so
``train_bootstrap.py`` can leave its run directory to that sync instead of
re-uploading every file of it (``reinforcetactics.cloud.storage.synced_by_wrapper``).

Usage:
    python3 scripts/cloud/vertex_train.py python3 main.py --mode train --timesteps 1000000
"""

import logging
import os
import posixpath
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from types import FrameType

# Make the package importable when the image is run from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from reinforcetactics.cloud.storage import (  # noqa: E402
    DEFAULT_OUTPUT_DIRS,
    WRAPPER_SYNC_ENV,
    download_tree,
    resolve_output_base,
    sync_directories,
    wrapper_sync_env,
)

logger = logging.getLogger("vertex_train")

DEFAULT_SYNC_INTERVAL = 300

# What a restore leaves in GCS: output a resumed run does not read.
RESTORE_EXCLUDE = re.compile(r"(^|/)(videos|charts|checkpoints|traces|tensorboard)/")

# scripts/train/train_bootstrap.py writes each run to benchmarks/bootstrap/<run_id>/
# by default, outside every DEFAULT_OUTPUT_DIRS entry, so a cancelled or
# preempted bootstrap job used to lose its whole run. The root is synced with
# an empty remote prefix: <run_id>/... lands at <base>/<run_id>/..., exactly
# where train_bootstrap's own final upload puts it, so the periodic copies and
# that final upload write the same objects rather than two copies of every
# checkpoint.
BOOTSTRAP_RUNS_DIR = "benchmarks/bootstrap"


def resolve_sync_dirs(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Map each local directory to sync onto its prefix under the GCS base.

    The defaults are ``DEFAULT_OUTPUT_DIRS`` (each to a same-named folder) and
    ``BOOTSTRAP_RUNS_DIR`` (to the base itself). ``GCS_SYNC_DIRS`` adds more,
    for runs launched with a custom output directory: a comma-separated list
    of ``dir`` (uploaded to ``<base>/dir/``), ``dir=prefix`` (to
    ``<base>/prefix/``) or ``dir=`` (straight under ``<base>/``) entries. An
    entry naming a default directory replaces its prefix.
    """
    resolved = os.environ if env is None else env
    dirs = {name: name for name in DEFAULT_OUTPUT_DIRS}
    dirs[BOOTSTRAP_RUNS_DIR] = ""
    dirs.update(_parse_dir_entries(resolved.get("GCS_SYNC_DIRS", ""), "GCS_SYNC_DIRS"))
    _warn_about_overlaps(dirs)
    return dirs


def _parse_dir_entries(value: str, variable: str) -> dict[str, str]:
    """``dir``, ``dir=prefix`` and ``dir=`` entries of a comma-separated list, normalized."""
    dirs: dict[str, str] = {}
    for entry in value.split(","):
        local, sep, remote = entry.partition("=")
        local = local.strip()
        if not local:
            continue
        local = os.path.normpath(local)
        # normpath folds "a/./b" and "a/../b". What is left of "." (from
        # GCS_SYNC_DIRS=. or dir=.) means the base itself: kept as ".", it
        # became a literal path segment in every object name (jobs/<name>/./...).
        prefix = posixpath.normpath((remote if sep else local).strip().replace(os.sep, "/")).strip("/")
        if prefix == ".":
            prefix = ""
        if prefix == ".." or prefix.startswith("../"):
            logger.warning("Ignoring %s entry %r: prefix %r points above the output base", variable, entry, prefix)
            continue
        dirs[local] = prefix
    return dirs


def resolve_restore_dirs(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Map each local directory to restore onto its prefix under the GCS base (``GCS_RESTORE_DIRS``).

    Same syntax as ``GCS_SYNC_DIRS`` (``dir``, ``dir=prefix``, ``dir=``); no
    defaults. scripts/train/run_seeds.py --backend vertex sets
    ``benchmarks/bootstrap/<run>=<run>``, so a resubmitted seed continues the
    run its earlier job synced.
    """
    resolved = os.environ if env is None else env
    return _parse_dir_entries(resolved.get("GCS_RESTORE_DIRS", ""), "GCS_RESTORE_DIRS")


def restore_directories(
    base_uri: str,
    restore_dirs: Mapping[str, str],
    credentials_file: str | None = None,
    client: object = None,
) -> dict[str, int]:
    """Download ``<base>/<prefix>/`` into each local directory before the training command starts.

    Charts, videos, the flattened checkpoints/ copies, traces and tensorboard
    are left out (the run does not need them to continue); the per-stage
    zips are restored, since the resume plan and the final checkpoint
    snapshot load them.
    """
    restored: dict[str, int] = {}
    for local, prefix in restore_dirs.items():
        source = f"{base_uri.rstrip('/')}/{prefix}" if prefix else base_uri.rstrip("/")
        restored[local] = download_tree(
            source, local, exclude=RESTORE_EXCLUDE, credentials_file=credentials_file, client=client
        )
    return restored


def _warn_about_overlaps(dirs: Mapping[str, str]) -> None:
    """Log each synced directory that sits inside another synced directory.

    Both entries upload the shared files, each to its own prefix, so they are
    stored twice (GCS_SYNC_DIRS=benchmarks also puts every bootstrap run under
    <base>/benchmarks/bootstrap/). Allowed, since the docs promise ``dir`` goes
    to ``<base>/dir/``, but rarely what was meant.
    """
    absolute = {local: Path(os.path.abspath(local)) for local in dirs}
    for inner, inner_path in absolute.items():
        for outer, outer_path in absolute.items():
            if inner != outer and inner_path.is_relative_to(outer_path):
                logger.warning(
                    "Synced directory %s is inside %s; its files are uploaded under both %r and %r",
                    inner,
                    outer,
                    dirs[inner],
                    dirs[outer],
                )


def _default_command() -> list[str]:
    """Fallback training command when none is supplied (matches the image CMD)."""
    return ["python3", "main.py", "--mode", "train"]


def _sync(
    base_uri: str | None,
    credentials_file: str | None,
    lock: threading.Lock,
    manifest: dict,
    sync_dirs: Mapping[str, str],
) -> None:
    """Run one sync pass under ``lock`` so periodic and final syncs don't overlap.

    ``manifest`` is shared across passes so unchanged files are not re-uploaded.
    ``sync_dirs`` maps local directories to remote prefixes (``resolve_sync_dirs``).
    """
    with lock:
        uploaded = sync_directories(
            base_uri,
            dirs=sync_dirs,
            credentials_file=credentials_file,
            manifest=manifest,
            remote_prefixes=sync_dirs,
        )
    if uploaded:
        summary = ", ".join(f"{name}={count}" for name, count in uploaded.items())
        logger.info("Synced to %s (%s)", base_uri, summary)


def _periodic_sync_loop(
    base_uri: str,
    credentials_file: str | None,
    interval: int,
    stop_event: threading.Event,
    lock: threading.Lock,
    manifest: dict,
    sync_dirs: Mapping[str, str],
) -> None:
    """Sync every ``interval`` seconds until ``stop_event`` is set."""
    while not stop_event.wait(interval):
        try:
            _sync(base_uri, credentials_file, lock, manifest, sync_dirs)
        except Exception as e:  # pragma: no cover - background best-effort
            logger.warning("Periodic GCS sync failed: %s", e)


def main() -> int:
    # Configured here rather than at import so importing this module (the
    # tests do) leaves the root logger alone.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [vertex_train] %(message)s")
    command = sys.argv[1:] or _default_command()

    base_uri = resolve_output_base()
    credentials_file = os.environ.get("GCS_CREDENTIALS") or None
    try:
        interval = int(os.environ.get("GCS_SYNC_INTERVAL", DEFAULT_SYNC_INTERVAL))
    except ValueError:
        interval = DEFAULT_SYNC_INTERVAL
    sync_dirs = resolve_sync_dirs()

    # Ensure the output directories exist so a final sync has something to find
    # even if training stops early.
    for name in DEFAULT_OUTPUT_DIRS:
        os.makedirs(name, exist_ok=True)

    if base_uri:
        logger.info("Output sync target: %s (every %ss) for %s", base_uri, interval, ", ".join(sync_dirs))
    else:
        logger.info("No GCS_OUTPUT_URI / AIP_MODEL_DIR set — running locally, outputs will not be uploaded.")

    # Tell the child what the final sync below will upload, so train_bootstrap
    # can leave its run directory to it rather than re-upload every file of it
    # (the periodic syncs already stored most) inside Vertex's shutdown grace
    # period. Without a sync, make sure the child sees no such promise, even
    # one inherited from an outer process.
    child_env = {key: value for key, value in os.environ.items() if key != WRAPPER_SYNC_ENV}
    if base_uri:
        child_env[WRAPPER_SYNC_ENV] = wrapper_sync_env(base_uri, sync_dirs)

    # Restore before the child starts and before the first periodic sync: the
    # command then finds the run it continues, and nothing is uploaded over
    # the stored copy first.
    restore_dirs = resolve_restore_dirs()
    if restore_dirs:
        if not base_uri:
            logger.warning("GCS_RESTORE_DIRS is set but there is no output base to restore from; ignoring it")
        else:
            try:
                restored = restore_directories(base_uri, restore_dirs, credentials_file)
            except Exception as e:
                logger.error(
                    "Restoring %s from %s failed (%s); not starting the command", ", ".join(restore_dirs), base_uri, e
                )
                return 1
            logger.info("Restored from %s: %s", base_uri, ", ".join(f"{k}={v}" for k, v in restored.items()))

    logger.info("Running: %s", " ".join(command))
    proc = subprocess.Popen(command, env=child_env)

    # Forward termination signals (Vertex sends SIGTERM on cancel/preemption)
    # to the training process so it can checkpoint before we do a final sync.
    def _forward(signum: int, _frame: FrameType | None) -> None:
        logger.info("Received signal %s; forwarding to training process.", signum)
        proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    sync_lock = threading.Lock()
    stop_event = threading.Event()
    manifest: dict = {}  # shared across syncs so unchanged files aren't re-uploaded
    sync_thread: threading.Thread | None = None
    if base_uri and interval > 0:
        sync_thread = threading.Thread(
            target=_periodic_sync_loop,
            args=(base_uri, credentials_file, interval, stop_event, sync_lock, manifest, sync_dirs),
            daemon=True,
        )
        sync_thread.start()

    try:
        returncode = proc.wait()
    finally:
        stop_event.set()
        if sync_thread is not None:
            sync_thread.join(timeout=30)
        if base_uri:
            logger.info("Performing final GCS sync...")
            try:
                _sync(base_uri, credentials_file, sync_lock, manifest, sync_dirs)
            except Exception as e:
                logger.warning("Final GCS sync failed: %s", e)

    logger.info("Training process exited with code %s", returncode)
    # A child killed by signal N reports returncode -N; surface it as the
    # conventional 128+N so orchestration can tell a preemption (SIGTERM -> 143)
    # from a genuine non-zero failure.
    return returncode if returncode >= 0 else 128 - returncode


if __name__ == "__main__":
    sys.exit(main())
