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

import hashlib
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
                code: {k: spec.get(k) for k in ("cost", "health", "attack", "defence", "movement")}
                for code, spec in (getattr(_c, "UNIT_DATA", {}) or {}).items()
            },
        }
    except Exception:
        # Metadata capture must never break a training run.
        return {}


def _map_meta(map_file: Optional[str]) -> Dict[str, Any]:
    """Fingerprint the map CSV by content, not just path.

    Maps are referenced by path; a silent terrain edit to e.g.
    ``beginner.csv`` makes runs before/after incomparable and nothing
    records it -- the same confound class as enabled_units. The
    sha256 of the raw bytes is the load-bearing fingerprint; dims are
    a convenience derived from the CSV grid (rows x first-row cols).
    Missing/unreadable file -> null hash, never raises.
    """
    if not map_file:
        return {"map_file": None, "map_sha256": None, "map_dims": None}
    out: Dict[str, Any] = {"map_file": map_file, "map_sha256": None, "map_dims": None}
    try:
        raw = Path(map_file).read_bytes()
        out["map_sha256"] = hashlib.sha256(raw).hexdigest()
        rows = [ln for ln in raw.decode("utf-8", "replace").splitlines() if ln.strip()]
        if rows:
            out["map_dims"] = [len(rows), len(rows[0].split(","))]  # [h, w]
    except Exception:
        pass
    return out


def _full_engine_constants_hash() -> Optional[str]:
    """Verbatim hash of the *entire* engine constant surface.

    ``_engine_economy`` only enumerates 5 unit fields + 4 economy
    scalars; an ability magnitude or any other ``constants.py`` value
    could still drift unrecorded. This hashes the COMPLETE ``UNIT_DATA``
    (every key, not the projection) plus the economy scalars, so *any*
    engine-constant change flips one auditable field with zero
    enumeration to maintain. sha256, first 16 hex; None on failure.
    """
    try:
        from reinforcetactics import constants as _c

        blob = {
            "STARTING_GOLD": getattr(_c, "STARTING_GOLD", None),
            "HEADQUARTERS_INCOME": getattr(_c, "HEADQUARTERS_INCOME", None),
            "BUILDING_INCOME": getattr(_c, "BUILDING_INCOME", None),
            "TOWER_INCOME": getattr(_c, "TOWER_INCOME", None),
            "UNIT_DATA": getattr(_c, "UNIT_DATA", {}),
        }
        canon = json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


def _apply_engine_overrides(economy: Mapping[str, Any], overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return the *effective* economy = captured defaults + overlay.

    Mirrors ``GameState._resolve_engine_overrides`` so config.json
    records exactly what the env played with, not just the engine
    defaults. Pure dict math (no engine import); unknown keys are
    ignored here since GameState already validated them.
    """
    eff: Dict[str, Any] = json.loads(json.dumps(economy, default=str)) if economy else {}
    if not overrides:
        return eff
    for k_ov, k_econ in (
        ("starting_gold", "starting_gold"),
        ("headquarters_income", "headquarters_income"),
        ("building_income", "building_income"),
        ("tower_income", "tower_income"),
    ):
        if k_ov in overrides:
            eff[k_econ] = overrides[k_ov]
    unit_ov = overrides.get("unit_data") or {}
    if unit_ov and isinstance(eff.get("unit_data"), dict):
        for code, fields in unit_ov.items():
            if code in eff["unit_data"] and isinstance(fields, Mapping):
                eff["unit_data"][code].update(fields)
    return eff


def _balance_profile_hash(engine_economy: Mapping[str, Any]) -> Optional[str]:
    """Short stable hash of the engine economy, so runs can be grouped
    by balance era in one column instead of diffing constants. Derived
    from the already-captured engine_economy block (canonical JSON ->
    sha1, first 12 hex). Empty economy -> None.
    """
    if not engine_economy:
        return None
    try:
        canon = json.dumps(engine_economy, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return None


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
    _econ = _engine_economy()
    _overrides = env_config.get("engine_overrides")
    _eff_econ = _apply_engine_overrides(_econ, _overrides)
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
            # Engine *defaults* (constants.py as imported).
            "engine_economy": _econ,
            "balance_profile_hash": _balance_profile_hash(_econ),
            # Sparse overlay from config + the resolved economy the env
            # actually played with (defaults + overlay). When no overlay
            # is set these are None / equal to engine_economy.
            "engine_overrides": dict(_overrides) if _overrides else None,
            "effective_engine_economy": _eff_econ,
            "effective_balance_profile_hash": _balance_profile_hash(_eff_econ),
            # Verbatim hash of the ENTIRE constant surface (full
            # UNIT_DATA, not the 5-field projection) -- catches drift
            # in fields engine_economy doesn't enumerate.
            "engine_constants_hash": _full_engine_constants_hash(),
            "map": _map_meta(map_file),
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
