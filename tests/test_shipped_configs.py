"""Every config file under ``configs/`` loads, validates, and is read by its entry point.

The config layer now coerces and range-checks every value, checks reward
keys and opponents against the env and the bot registry, and entry points
report config fields they do not read (review rltrain-9 / rltrain-10). This
walks every shipped YAML/JSON file:

* training configs load through ``load_config`` (which validates), and every
  curriculum resolves (its maps exist, the padding is consistent);
* behaviour-cloning scenario files load through ``load_scenarios_from_yaml``;
* each training config is checked against the fields its entry point
  declares it reads. The non-sweep configs set nothing their entry point
  ignores, so they run under ``--strict``. The sweep archive
  (``ppo/bootstrap_sweep``) still carries an informational top-level
  ``total_timesteps`` that the curriculum runner ignores; that is the only
  field it may set that train_bootstrap.py reports;
* the configs whose ``warm_start_path`` is a placeholder (or a Colab Drive
  path) load, and fail the pre-run file check.
"""

import functools
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from reinforcetactics.rl import bootstrap
from reinforcetactics.rl.config import TrainingConfig, ignored_config_fields, load_config
from reinforcetactics.rl.imitation import load_scenarios_from_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
CONFIG_FILES = sorted(p for p in CONFIGS.rglob("*") if p.suffix.lower() in {".yaml", ".yml", ".json"})

# Shipped configs whose warm_start_path must be supplied before a run: two
# placeholders for a BC checkpoint and one Colab Drive path.
NEEDS_WARM_START_FILE = {
    "ppo/skirmish_bc_selfplay.yaml",
    "ppo/bootstrap_sweep/v30_random15_warmstart_probe.yaml",
    "ppo/bootstrap_sweep/v33_production_bc_warmstart.yaml",
}


def _rel(path: Path) -> str:
    return path.relative_to(CONFIGS).as_posix()


@functools.cache
def _raw(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return (json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)) or {}


@functools.cache
def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_shipped_configs", REPO_ROOT / "scripts" / "train" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _consumed(path: Path, cfg: TrainingConfig):
    """``(entry point, consumed fields, accepted algorithms)`` for a shipped training config."""
    rel = _rel(path)
    if cfg.curriculum.stages:
        return "train_bootstrap.py", bootstrap.CONSUMED_CONFIG_FIELDS, bootstrap.CONSUMED_ALGORITHMS
    if rel.startswith("self_play/"):
        # What the run itself reads: its mode and whether the pool is on.
        sp = _script("train_self_play")
        mode = "mixed" if cfg.self_play.mixed_training else "self-play"
        consumed = sp.consumed_config_fields(mode, cfg.self_play.use_opponent_pool)
        return f"train_self_play.py --mode {mode}", consumed, sp.consumed_algorithms(mode)
    if rel.startswith("feudal/"):
        fd = _script("train_feudal_rl")
        consumed = fd.consumed_config_fields("feudal", cfg.env.opponent)
        return "train_feudal_rl.py --mode feudal", consumed, fd.CONSUMED_ALGORITHMS["feudal"]
    if rel in ("ppo/maskable_ppo.yaml", "ppo/ppo_baseline.yaml"):  # configs/README: train_feudal_rl.py (flat)
        fd = _script("train_feudal_rl")
        return "train_feudal_rl.py --mode flat", fd.CONSUMED_CONFIG_FIELDS["flat"], fd.CONSUMED_ALGORITHMS["flat"]
    if rel.startswith("alphazero/"):
        az = _script("train_alphazero")
        return "train_alphazero.py", az.CONSUMED_CONFIG_FIELDS, az.CONSUMED_ALGORITHMS
    pytest.fail(f"{rel}: no entry point known for this config; add it to _consumed()")


def test_configs_are_found():
    assert len(CONFIG_FILES) >= 70


@pytest.mark.parametrize("path", CONFIG_FILES, ids=_rel)
def test_shipped_config_loads_validates_and_is_read(path):
    if "scenarios" in _raw(path):  # behaviour-cloning scenario list, not a TrainingConfig
        assert load_scenarios_from_yaml(str(path))
        return
    cfg = load_config(path)
    cfg.validate()

    entry_point, consumed, algorithms = _consumed(path, cfg)
    ignored = ignored_config_fields(cfg, consumed, algorithms=algorithms)
    allowed = ["total_timesteps"] if _rel(path).startswith("ppo/bootstrap_sweep/") else []
    assert set(ignored) <= set(allowed), f"{entry_point} ignores {ignored} in {_rel(path)}"

    if _rel(path) in NEEDS_WARM_START_FILE:
        with pytest.raises(FileNotFoundError, match="warm_start_path"):
            cfg.validate(check_files=True)
        return
    cfg.validate(check_files=True)
    if cfg.curriculum.stages:
        resolved = bootstrap.resolve_config(cfg)
        assert resolved.env.flat_action_version is not None


_ENTRY_POINT_ARGV = {
    "self_play/self_play.yaml": ("train_self_play", []),
    "feudal/feudal_rl.yaml": ("train_feudal_rl", ["--mode", "feudal"]),
    "ppo/maskable_ppo.yaml": ("train_feudal_rl", ["--mode", "flat"]),
    "ppo/ppo_baseline.yaml": ("train_feudal_rl", ["--mode", "flat"]),
    "alphazero/alphazero.yaml": ("train_alphazero", []),
}


@pytest.mark.parametrize("rel", sorted(_ENTRY_POINT_ARGV))
def test_non_curriculum_config_parses_strict_through_its_entry_point(rel):
    """The script itself: its mode-aware report and the validation of the values the run uses."""
    script, extra = _ENTRY_POINT_ARGV[rel]
    args = _script(script).parse_args(["--config", str(CONFIGS / rel), "--strict", *extra])
    assert args.strict


def test_canonical_bootstrap_config_is_strict_clean():
    cfg = load_config(CONFIGS / "ppo" / "bootstrap.yaml")
    assert ignored_config_fields(cfg, bootstrap.CONSUMED_CONFIG_FIELDS, algorithms=bootstrap.CONSUMED_ALGORITHMS) == []
