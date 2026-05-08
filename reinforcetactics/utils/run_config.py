"""Build and write a self-describing run config snapshot.

Captures the resolved hyperparameters, env settings, git commit (with a
dirty-tree flag), key library versions, and hardware info into a single
``config.json`` next to a saved model. Shared across notebooks so
bootstrap, PPO training, and future trainers all emit consistent
metadata when YAML defaults or library versions drift.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

PathLike = Union[str, Path]


def _git_meta() -> Dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"commit": None, "short": None, "dirty": None}
    try:
        rc = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        dirty: Optional[bool] = bool(rc)
    except (FileNotFoundError, OSError):
        dirty = None
    return {"commit": sha, "short": sha[:7] if sha else None, "dirty": dirty}


def _lib_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in (
        "reinforcetactics",
        "torch",
        "numpy",
        "gymnasium",
        "stable_baselines3",
        "sb3_contrib",
    ):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def _hardware_meta() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_gb": None,
        "torch_cuda_available": None,
        "torch_cuda_version": None,
        "gpu_name": None,
        "gpu_count": 0,
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        # psutil is optional; absence shouldn't break a training run.
        pass
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def build_run_config(
    *,
    run_type: str,
    map_file: Optional[str],
    opponent: Optional[str],
    hyperparams: Mapping[str, Any],
    env_config: Mapping[str, Any],
    seed: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a self-describing config dict.

    ``hyperparams`` and ``env_config`` should be the *resolved* values
    actually used by training (after defaults / per-stage overrides
    merge), so the snapshot is reproducible without re-deriving the
    merge logic.
    """
    return {
        "run_type": run_type,
        "map_file": map_file,
        "opponent": opponent,
        "seed": seed,
        "hyperparams": dict(hyperparams),
        "env_config": dict(env_config),
        "extra": dict(extra) if extra else {},
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git": _git_meta(),
            "libraries": _lib_versions(),
            "hardware": _hardware_meta(),
        },
    }


def write_run_config(config: Mapping[str, Any], path: PathLike) -> Path:
    """Serialise ``config`` to ``path`` as pretty-printed JSON.

    Parent directories are created as needed. Non-JSON-serialisable
    values (``Path`` etc.) are coerced via ``str``.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    return p
