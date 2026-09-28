"""Google Cloud Storage helpers for persisting training artifacts.

Training runs (whether ``main.py`` or the scripts under ``scripts/train/``)
write models, checkpoints, and logs to the local filesystem. On an ephemeral
runner such as a Vertex AI custom job those files vanish when the job ends, so
this module provides small, dependency-light helpers to sync the local output
directories up to a ``gs://`` location, and (:func:`download_tree`) to bring a
run back down: a resubmitted job restoring its run directory, or a copy for
analysis.

Nothing here imports ``google-cloud-storage`` at module load time; the client is
created lazily so the rest of the package (and the test suite) can import this
module without the optional dependency installed.
"""

import json
import logging
import os
import re
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Local output directories produced by the training entry points. These are all
# git-ignored at the repo root; see ``.gitignore``.
DEFAULT_OUTPUT_DIRS: tuple[str, ...] = ("models", "checkpoints", "tensorboard", "logs")

# A manifest maps (local file path, destination object name) to the file's
# (mtime, size) signature when it was last uploaded there, so repeated syncs
# can skip files that have not changed since. The destination is part of the
# key because one local file can be due at two destinations (overlapping
# synced directories); keyed by the local path alone, the first upload marked
# the second as done and it never happened.
Manifest = MutableMapping[tuple[str, str], tuple[float, int]]

# Suffix of a file that is still being written and gets renamed into place
# once complete (``reinforcetactics.rl.callbacks.save_model_atomically``).
# Uploads skip these: a sync that runs mid-write would store a truncated copy.
PARTIAL_SUFFIX = ".partial"

# Environment variable through which scripts/cloud/vertex_train.py tells the
# training command it wraps which local directories its final sync uploads,
# and where to (``wrapper_sync_env`` writes it, ``synced_by_wrapper`` reads it).
WRAPPER_SYNC_ENV = "GCS_WRAPPER_SYNC"


def parse_gcs_uri(uri: str) -> tuple[str, str]:
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


def resolve_output_base(env: Mapping[str, str] | None = None) -> str | None:
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
        # Vertex sets AIP_MODEL_DIR = <base>/model. removesuffix keeps this
        # robust for a bucket-root base (``gs://bucket/model`` -> ``gs://bucket``)
        # and for a stray trailing slash, neither of which a blind rsplit handles.
        return model_dir.rstrip("/").removesuffix("/model")

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
        credentials_file: str | None = None,
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

    def upload_file(self, local_path: str, remote_path: str | None = None) -> str | None:
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

    def upload_directory(
        self,
        local_dir: str,
        remote_prefix: str | None = None,
        manifest: Manifest | None = None,
    ) -> int:
        """Recursively upload files under ``local_dir``; return the count uploaded.

        When ``manifest`` is provided, files whose ``(mtime, size)`` signature is
        unchanged since their last upload to the same destination are skipped —
        so a periodic sync does not re-upload gigabytes of unchanged checkpoints
        every cycle. Files still being written (``PARTIAL_SUFFIX``) are never
        uploaded.
        """
        local_path = Path(local_dir)
        if not local_path.is_dir():
            return 0

        uploaded = 0
        for file_path in sorted(local_path.rglob("*")):
            if not file_path.is_file() or file_path.name.endswith(PARTIAL_SUFFIX):
                continue

            relative = file_path.relative_to(local_path).as_posix()
            remote_path = f"{remote_prefix}/{relative}" if remote_prefix else relative
            key = (str(file_path), f"gs://{self.bucket_name}/{self.prefix}{remote_path}")
            signature = _file_signature(file_path)
            if manifest is not None and signature is not None and manifest.get(key) == signature:
                continue

            if self.upload_file(str(file_path), remote_path):
                uploaded += 1
                if manifest is not None and signature is not None:
                    manifest[key] = signature
        return uploaded


def _file_signature(path: Path) -> tuple[float, int] | None:
    """Return a cheap ``(mtime, size)`` change-signature for ``path``."""
    try:
        st = path.stat()
    except OSError:  # pragma: no cover - file vanished between rglob and stat
        return None
    return (st.st_mtime, st.st_size)


def _make_uploader(
    base_uri: str | None,
    credentials_file: str | None,
    client: Any,
) -> GCSUploader | None:
    """Build a :class:`GCSUploader` for ``base_uri`` or return ``None`` to skip.

    Returns ``None`` (a no-op for callers) when ``base_uri`` is falsy, not a valid
    ``gs://`` URI, or when ``google-cloud-storage`` is unavailable and no client
    was injected.
    """
    if not base_uri:
        return None
    try:
        bucket, prefix = parse_gcs_uri(base_uri)
    except ValueError as e:
        logger.warning("Skipping GCS upload: %s", e)
        return None
    if client is None and not is_available():
        logger.warning("google-cloud-storage not installed; skipping upload to %s", base_uri)
        return None
    return GCSUploader(bucket, prefix, credentials_file=credentials_file, client=client)


def sync_directories(
    base_uri: str | None,
    dirs: Iterable[str] = DEFAULT_OUTPUT_DIRS,
    root: str = ".",
    credentials_file: str | None = None,
    client: Any = None,
    manifest: Manifest | None = None,
    remote_prefixes: Mapping[str, str] | None = None,
) -> dict[str, int]:
    """Upload local output directories to ``base_uri/<dir>/``.

    Each entry in ``dirs`` is uploaded (if it exists locally) to a same-named
    folder under ``base_uri``, or to ``remote_prefixes[dir]`` when given (``""``
    uploads the directory's contents straight under ``base_uri``). Returns a
    mapping of directory name to the number of files uploaded. A no-op (empty
    dict) when ``base_uri`` is falsy or the ``google-cloud-storage`` dependency
    is missing — callers can treat this as "ran locally, nothing synced". Pass
    a shared ``manifest`` across repeated calls to skip unchanged files; a file
    under two overlapping entries is uploaded to both destinations.
    """
    uploader = _make_uploader(base_uri, credentials_file, client)
    if uploader is None:
        return {}

    prefixes = remote_prefixes or {}
    results: dict[str, int] = {}
    for name in dirs:
        local_dir = os.path.join(root, name)
        if not os.path.isdir(local_dir):
            continue
        remote_prefix = prefixes.get(name, name)
        count = uploader.upload_directory(local_dir, remote_prefix=remote_prefix, manifest=manifest)
        if count:
            results[name] = count
    return results


def upload_tree(
    local_dir: str,
    dest_uri: str | None,
    credentials_file: str | None = None,
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
    uploader = _make_uploader(dest_uri, credentials_file, client)
    if uploader is None:
        return 0
    return uploader.upload_directory(local_dir)


def download_tree(
    src_uri: str | None,
    local_dir: str | os.PathLike[str],
    *,
    exclude: re.Pattern[str] | None = None,
    credentials_file: str | None = None,
    client: Any = None,
) -> int:
    """Download every object under ``src_uri`` into ``local_dir``, keeping the relative layout.

    The inverse of :func:`upload_tree`: ``gs://b/p/run/a/x.json`` lands at
    ``<local_dir>/a/x.json``. Objects still being written (``PARTIAL_SUFFIX``)
    are skipped, and so is every relative path ``exclude`` matches (``search``).
    Each file is written to a ``.partial`` sibling and renamed into place, so
    a kill never leaves a truncated file behind. Returns the number of files
    downloaded: 0 for an empty prefix, a bad URI, or a missing
    ``google-cloud-storage`` (with no ``client`` injected).
    """
    if not src_uri:
        return 0
    try:
        bucket_name, prefix = parse_gcs_uri(src_uri)
    except ValueError as e:
        logger.warning("Skipping GCS download: %s", e)
        return 0
    if client is None:
        if not is_available():
            logger.warning("google-cloud-storage not installed; skipping download from %s", src_uri)
            return 0
        from google.cloud import storage

        if credentials_file and os.path.exists(credentials_file):
            client = storage.Client.from_service_account_json(credentials_file)
        else:
            client = storage.Client()
    base = f"{prefix}/" if prefix else ""
    target = Path(local_dir)
    count = 0
    for blob in client.list_blobs(bucket_name, prefix=base):
        name = str(blob.name)
        relative = name[len(base) :]
        if not relative or relative.endswith("/") or relative.endswith(PARTIAL_SUFFIX):
            continue
        if exclude is not None and exclude.search(relative):
            continue
        destination = target / relative
        # An object name like "../x" must not escape the target directory.
        if not destination.resolve().is_relative_to(target.resolve()):
            logger.warning("Skipping %s: it would land outside %s", name, target)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + PARTIAL_SUFFIX)
        try:
            blob.download_to_filename(str(partial))
            os.replace(partial, destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        count += 1
    return count


def wrapper_sync_env(base_uri: str, sync_dirs: Mapping[str, str], root: str = ".") -> str:
    """Encode a wrapper's sync targets as the value of ``WRAPPER_SYNC_ENV``.

    ``sync_dirs`` maps local directories (relative to ``root``) to remote
    prefixes under ``base_uri``, the way ``sync_directories``' ``dirs`` and
    ``remote_prefixes`` do. Local paths are resolved so the child can compare
    them whatever its working directory.
    """
    dirs = {os.path.realpath(os.path.join(root, local)): prefix for local, prefix in sync_dirs.items()}
    return json.dumps({"base": base_uri.rstrip("/"), "dirs": dirs})


def synced_by_wrapper(local_dir: str | os.PathLike[str], dest_uri: str, env: Mapping[str, str] | None = None) -> bool:
    """Whether the wrapping entrypoint's final sync uploads ``local_dir`` to ``dest_uri``.

    True only when ``WRAPPER_SYNC_ENV`` is set and one of its directories
    contains ``local_dir`` such that every file lands on the object
    ``upload_tree(local_dir, dest_uri)`` would write. The caller can then
    leave the upload to the wrapper, whose final sync runs once the child has
    exited and skips what its periodic syncs already stored; ``upload_tree``
    would re-upload the whole tree first, inside the same shutdown grace
    period.
    """
    resolved = os.environ if env is None else env
    raw = resolved.get(WRAPPER_SYNC_ENV)
    if not raw:
        return False
    try:
        spec = json.loads(raw)
        base = str(spec["base"]).rstrip("/")
        dirs = {str(local): str(prefix) for local, prefix in spec["dirs"].items()}
    except (ValueError, KeyError, TypeError, AttributeError):
        logger.warning("Ignoring malformed %s: %r", WRAPPER_SYNC_ENV, raw)
        return False

    target = Path(os.path.realpath(local_dir))
    dest = dest_uri.rstrip("/")
    for local, prefix in dirs.items():
        try:
            relative = target.relative_to(local).as_posix()
        except ValueError:
            continue
        parts = (base, prefix.strip("/"), "" if relative == "." else relative)
        if "/".join(part for part in parts if part) == dest:
            return True
    return False
