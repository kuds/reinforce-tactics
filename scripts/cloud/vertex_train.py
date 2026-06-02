#!/usr/bin/env python3
"""Container entrypoint that runs a training command and syncs artifacts to GCS.

This is the ``ENTRYPOINT`` of the training Docker image. It runs whatever
training command it is given (everything after the script name) as a child
process, and around that run it periodically — and once more on exit — uploads
the local output directories (``models/``, ``checkpoints/``, ``tensorboard/``,
``logs/``) to Google Cloud Storage. That is what makes a Vertex AI custom job
useful: the machine is torn down when the job finishes, so anything not pushed
to GCS is lost.

The destination is taken from ``GCS_OUTPUT_URI`` (preferred) or, failing that,
Vertex's ``AIP_MODEL_DIR``. When neither is set the command still runs normally
and nothing is uploaded, so the same image works locally.

Environment variables:
    GCS_OUTPUT_URI     gs:// base for outputs (overrides AIP_MODEL_DIR).
    GCS_SYNC_INTERVAL  Seconds between periodic syncs (default 300; <=0 disables).
    GCS_CREDENTIALS    Optional path to a service-account JSON file.

Usage:
    python3 scripts/cloud/vertex_train.py python3 main.py --mode train --timesteps 1000000
"""

import logging
import os
import signal
import subprocess
import sys
import threading
from types import FrameType
from typing import List, Optional

# Make the package importable when the image is run from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from reinforcetactics.cloud.storage import (  # noqa: E402
    DEFAULT_OUTPUT_DIRS,
    resolve_output_base,
    sync_directories,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [vertex_train] %(message)s")
logger = logging.getLogger("vertex_train")

DEFAULT_SYNC_INTERVAL = 300


def _default_command() -> List[str]:
    """Fallback training command when none is supplied (matches the image CMD)."""
    return ["python3", "main.py", "--mode", "train"]


def _sync(base_uri: Optional[str], credentials_file: Optional[str], lock: threading.Lock, manifest: dict) -> None:
    """Run one sync pass under ``lock`` so periodic and final syncs don't overlap.

    ``manifest`` is shared across passes so unchanged files are not re-uploaded.
    """
    with lock:
        uploaded = sync_directories(base_uri, credentials_file=credentials_file, manifest=manifest)
    if uploaded:
        summary = ", ".join(f"{name}={count}" for name, count in uploaded.items())
        logger.info("Synced to %s (%s)", base_uri, summary)


def _periodic_sync_loop(
    base_uri: str,
    credentials_file: Optional[str],
    interval: int,
    stop_event: threading.Event,
    lock: threading.Lock,
    manifest: dict,
) -> None:
    """Sync every ``interval`` seconds until ``stop_event`` is set."""
    while not stop_event.wait(interval):
        try:
            _sync(base_uri, credentials_file, lock, manifest)
        except Exception as e:  # pragma: no cover - background best-effort
            logger.warning("Periodic GCS sync failed: %s", e)


def main() -> int:
    command = sys.argv[1:] or _default_command()

    base_uri = resolve_output_base()
    credentials_file = os.environ.get("GCS_CREDENTIALS") or None
    try:
        interval = int(os.environ.get("GCS_SYNC_INTERVAL", DEFAULT_SYNC_INTERVAL))
    except ValueError:
        interval = DEFAULT_SYNC_INTERVAL

    # Ensure the output directories exist so a final sync has something to find
    # even if training stops early.
    for name in DEFAULT_OUTPUT_DIRS:
        os.makedirs(name, exist_ok=True)

    if base_uri:
        logger.info("Output sync target: %s (every %ss)", base_uri, interval)
    else:
        logger.info("No GCS_OUTPUT_URI / AIP_MODEL_DIR set — running locally, outputs will not be uploaded.")

    logger.info("Running: %s", " ".join(command))
    proc = subprocess.Popen(command)

    # Forward termination signals (Vertex sends SIGTERM on cancel/preemption)
    # to the training process so it can checkpoint before we do a final sync.
    def _forward(signum: int, _frame: Optional[FrameType]) -> None:
        logger.info("Received signal %s; forwarding to training process.", signum)
        proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    sync_lock = threading.Lock()
    stop_event = threading.Event()
    manifest: dict = {}  # shared across syncs so unchanged files aren't re-uploaded
    sync_thread: Optional[threading.Thread] = None
    if base_uri and interval > 0:
        sync_thread = threading.Thread(
            target=_periodic_sync_loop,
            args=(base_uri, credentials_file, interval, stop_event, sync_lock, manifest),
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
                _sync(base_uri, credentials_file, sync_lock, manifest)
            except Exception as e:
                logger.warning("Final GCS sync failed: %s", e)

    logger.info("Training process exited with code %s", returncode)
    # A child killed by signal N reports returncode -N; surface it as the
    # conventional 128+N so orchestration can tell a preemption (SIGTERM -> 143)
    # from a genuine non-zero failure.
    return returncode if returncode >= 0 else 128 - returncode


if __name__ == "__main__":
    sys.exit(main())
