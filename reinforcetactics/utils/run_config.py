"""Build and write a self-describing run config snapshot.

Captures the resolved hyperparameters, env settings, git commit (with a
dirty-tree flag), key library versions, the non-YAML engine economy
constants (starting gold, income rates, per-unit stat block), and
hardware info into a single ``config.json`` next to a saved model.
Shared across notebooks so bootstrap, PPO training, and future trainers
all emit consistent metadata when YAML defaults, library versions, or
engine constants drift.
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


def _engine_economy() -> Dict[str, Any]:
    """Snapshot the engine economy constants that are NOT YAML-settable.

    STARTING_GOLD / *_INCOME and the per-unit stat block live in
    constants.py and silently change across commits (e.g. f4dc50e
    bumped Knight defence, 6f64745 cut HQ income, a596c15 nerfed
    Warrior + cut starting gold). None of these were captured in any
    run artifact, so historical runs had hidden economy confounds
    only recoverable via ``git show <sha>:reinforcetactics/constants.py``.
    Recording them here makes every future run's economy auditable
    from its own config.json -- the same gap that was closed for
    enabled_units.
    """
    try:
        from reinforcetactics import constants as _c

        return {
            "starting_gold": getattr(_c, "STARTING_GOLD", None),
            "headquarters_income": getattr(_c, "HEADQUARTERS_INCOME", None),
            "building_income": getattr(_c, "BUILDING_INCOME", None),
            "tower_income": getattr(_c, "TOWER_INCOME", None),
            # Per-unit stat block: cost/health/attack/defence/movement
            # for every unit code. Attack may be an int or a dict
            # ({adjacent, range}) for ranged units -- stored as-is.
            "unit_data": {
                code: {
                    k: spec.get(k)
                    for k in ("cost", "health", "attack", "defence", "movement")
                }
                for code, spec in (getattr(_c, "UNIT_DATA", {}) or {}).items()
            },
        }
    except Exception:
        # Metadata capture must never break a training run.
        return {}


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
            "engine_economy": _engine_economy(),
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
