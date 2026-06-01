"""Google Cloud Storage helpers for persisting training artifacts.

Training runs (whether ``main.py`` or the scripts under ``scripts/train/``)
write models, checkpoints, and logs to the local filesystem. On an ephemeral
runner such as a Vertex AI custom job those files vanish when the job ends, so
this module provides small, dependency-light helpers to sync the local output
directories up to a ``gs://`` location.

Nothing here imports ``google-cloud-storage`` at module load time; the client is
created lazily so the rest of the package (and the test suite) can import this
module without the optional dependency installed.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# Local output directories produced by the training entry points. These are all
# git-ignored at the repo root; see ``.gitignore``.
DEFAULT_OUTPUT_DIRS: Tuple[str, ...] = ("models", "checkpoints", "tensorboard", "logs")


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    """Split a ``gs://bucket/prefix`` URI into ``(bucket, prefix)``.

    The returned prefix has no leading or trailing slash. Raises ``ValueError``
    for anything that is not a ``gs://`` URI with a bucket.
    """
    if not uri or not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI (expected gs://...): {uri!r}")

    path = uri[len("gs://") :]
    bucket, _, prefix = path.partition("/")
    if not bucket:
        raise ValueError(f"GCS URI is missing a bucket name: {uri!r}")
    return bucket, prefix.strip("/")


def resolve_output_base(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Determine the GCS base URI to sync outputs to, or ``None`` if unset.

    Resolution order:

    1. ``GCS_OUTPUT_URI`` — explicit ``gs://`` location (our preferred contract).
    2. ``AIP_MODEL_DIR`` — set automatically by Vertex AI to
       ``<baseOutputDirectory>/model``; we strip the trailing ``model`` segment
       to recover the base directory.

    Returns the base URI without a trailing slash, or ``None`` when neither is
    configured (in which case callers should skip uploading).
    """
    resolved = os.environ if env is None else env

    explicit = resolved.get("GCS_OUTPUT_URI", "").strip()
    if explicit:
        return explicit.rstrip("/")

    model_dir = resolved.get("AIP_MODEL_DIR", "").strip()
    if model_dir:
        # Vertex sets AIP_MODEL_DIR = <base>/model
        return model_dir.rstrip("/").rsplit("/", 1)[0]

    return None


def is_available() -> bool:
    """Return ``True`` if the ``google-cloud-storage`` package is importable."""
    try:
        from google.cloud import storage  # noqa: F401
    except Exception:  # pragma: no cover - exercised only without the dep
        return False
    return True


class GCSUploader:
    """Uploads files and directory trees to a Google Cloud Storage bucket.

    Uploads are best-effort: failures are logged and reported via return values
    rather than raised, so a transient storage error never takes down a training
    run. A pre-built ``client`` may be injected (used by the tests); otherwise a
    client is created lazily on first use.
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "",
        credentials_file: Optional[str] = None,
        client: Any = None,
    ):
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self.credentials_file = credentials_file
        self._client: Any = client
        self._bucket: Any = None

    def _get_bucket(self) -> Any:
        """Lazily initialise the client/bucket and return the bucket handle."""
        if self._client is None:
            from google.cloud import storage

            if self.credentials_file and os.path.exists(self.credentials_file):
                self._client = storage.Client.from_service_account_json(self.credentials_file)
                logger.info("GCS client initialised from %s", self.credentials_file)
            else:
                self._client = storage.Client()
                logger.info("GCS client initialised with default credentials")
        if self._bucket is None:
            self._bucket = self._client.bucket(self.bucket_name)
        return self._bucket

    def upload_file(self, local_path: str, remote_path: Optional[str] = None) -> Optional[str]:
        """Upload a single file, returning its ``gs://`` URI or ``None`` on failure."""
        try:
            bucket = self._get_bucket()
            if remote_path is None:
                remote_path = os.path.basename(local_path)
            full_remote_path = f"{self.prefix}{remote_path}"
            bucket.blob(full_remote_path).upload_from_filename(local_path)
            return f"gs://{self.bucket_name}/{full_remote_path}"
        except Exception as e:  # pragma: no cover - defensive, network dependent
            logger.warning("Failed to upload %s to GCS: %s", local_path, e)
            return None

    def upload_directory(self, local_dir: str, remote_prefix: Optional[str] = None) -> int:
        """Recursively upload every file under ``local_dir``; return the count uploaded."""
        local_path = Path(local_dir)
        if not local_path.is_dir():
            return 0

        uploaded = 0
        for file_path in sorted(local_path.rglob("*")):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(local_path).as_posix()
            remote_path = f"{remote_prefix}/{relative}" if remote_prefix else relative
            if self.upload_file(str(file_path), remote_path):
                uploaded += 1
        return uploaded


def sync_directories(
    base_uri: Optional[str],
    dirs: Iterable[str] = DEFAULT_OUTPUT_DIRS,
    root: str = ".",
    credentials_file: Optional[str] = None,
    client: Any = None,
) -> Dict[str, int]:
    """Upload local output directories to ``base_uri/<dir>/``.

    Each entry in ``dirs`` is uploaded (if it exists locally) to a same-named
    folder under ``base_uri``. Returns a mapping of directory name to the number
    of files uploaded. A no-op (empty dict) when ``base_uri`` is falsy or the
    ``google-cloud-storage`` dependency is missing — callers can treat this as
    "ran locally, nothing synced".
    """
    if not base_uri:
        return {}

    try:
        bucket, prefix = parse_gcs_uri(base_uri)
    except ValueError as e:
        logger.warning("Skipping GCS sync: %s", e)
        return {}

    if client is None and not is_available():
        logger.warning("google-cloud-storage not installed; skipping GCS sync to %s", base_uri)
        return {}

    uploader = GCSUploader(bucket, prefix, credentials_file=credentials_file, client=client)
    results: Dict[str, int] = {}
    for name in dirs:
        local_dir = os.path.join(root, name)
        if not os.path.isdir(local_dir):
            continue
        count = uploader.upload_directory(local_dir, remote_prefix=name)
        if count:
            results[name] = count
    return results


def upload_tree(
    local_dir: str,
    dest_uri: Optional[str],
    credentials_file: Optional[str] = None,
    client: Any = None,
) -> int:
    """Upload an entire local directory tree to a ``gs://`` destination.

    Unlike :func:`sync_directories` (which maps several named top-level dirs),
    this uploads everything under ``local_dir`` to ``dest_uri``, preserving the
    relative layout. Returns the number of files uploaded — ``0`` when
    ``dest_uri`` is falsy, the directory is missing, or ``google-cloud-storage``
    is unavailable. Used to persist a bootstrap run directory (charts/, videos/,
    checkpoints/, …) to GCS.
    """
    if not dest_uri:
        return 0

    try:
        bucket, prefix = parse_gcs_uri(dest_uri)
    except ValueError as e:
        logger.warning("Skipping upload: %s", e)
        return 0

    if client is None and not is_available():
        logger.warning("google-cloud-storage not installed; skipping upload to %s", dest_uri)
        return 0

    uploader = GCSUploader(bucket, prefix, credentials_file=credentials_file, client=client)
    return uploader.upload_directory(local_dir)
