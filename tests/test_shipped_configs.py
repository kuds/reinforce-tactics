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

import dataclasses
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


# ---------------------------------------------------------------------------
# The validation-run configs (docs/validation_run_config.md)
# ---------------------------------------------------------------------------

BOOTSTRAP = CONFIGS / "ppo" / "bootstrap.yaml"
BOOTSTRAP_SLICE = CONFIGS / "ppo" / "bootstrap_validation.yaml"
SELF_PLAY = CONFIGS / "self_play" / "self_play.yaml"


def _max_steps_floor(max_turns: int, max_actions_per_turn: int) -> int:
    """The most steps a mask-following policy can spend before the max_turns clock ends the game.

    Each game turn it takes at most ``max_actions_per_turn`` actions (the mask
    then offers end_turn alone) plus the end_turn.
    """
    return max_turns * (max_actions_per_turn + 1) + max_actions_per_turn


@pytest.mark.parametrize("path", [BOOTSTRAP, BOOTSTRAP_SLICE], ids=_rel)
def test_curriculum_max_steps_never_truncates_before_the_clock(path):
    """Truncation pays 0 plus the bootstrapped value: it must not be a cheaper exit than a max-turns draw."""
    cfg = load_config(path)
    cap = cfg.env.max_actions_per_turn
    assert cap is not None, "the bound needs env.max_actions_per_turn"
    for stage in cfg.curriculum.stages:
        max_turns = stage.resolve_max_turns(cfg.env)
        assert max_turns is not None, stage.name
        floor = _max_steps_floor(max_turns, cap)
        max_steps = stage.resolve_max_steps(cfg.env)
        # At or above the bound, rounded up to the next 100 (no slack beyond it).
        assert floor <= max_steps < floor + 100, f"{stage.name}: max_steps {max_steps}, bound {floor}"


def test_self_play_max_steps_never_truncates_before_the_clock():
    cfg = load_config(SELF_PLAY)
    assert cfg.env.max_turns is not None and cfg.env.max_actions_per_turn is not None
    floor = _max_steps_floor(cfg.env.max_turns, cfg.env.max_actions_per_turn)
    assert floor <= cfg.env.max_steps < floor + 100


def _without(obj, *names: str) -> dict:
    out = dataclasses.asdict(obj)
    for name in names:
        out.pop(name)
    return out


def test_validation_slice_mirrors_the_canonical_curriculum():
    """bootstrap_validation.yaml is bootstrap.yaml's first stages with shorter budgets, and nothing else.

    Allowed differences: per-stage max_timesteps / anneal_horizon (no larger
    than the canonical ones), eval.eval_freq, and curriculum.max_retries.
    Compared as the runner resolves them (``bootstrap.resolve_config``): the
    padding is derived from the stage maps, and the slice's maps are all 6x6,
    so two identical env blocks could still train on different observations.
    """
    canonical = bootstrap.resolve_config(load_config(BOOTSTRAP))
    probe = bootstrap.resolve_config(load_config(BOOTSTRAP_SLICE))
    assert (probe.algorithm, probe.seed) == (canonical.algorithm, canonical.seed)
    assert dataclasses.asdict(probe.env) == dataclasses.asdict(canonical.env)
    assert probe.env.pad_to_size == canonical.env.pad_to_size == (10, 12)
    assert dataclasses.asdict(probe.ppo) == dataclasses.asdict(canonical.ppo)
    assert _without(probe.eval, "eval_freq") == _without(canonical.eval, "eval_freq")
    assert probe.eval.eval_freq <= canonical.eval.eval_freq
    assert _without(probe.curriculum, "stages", "max_retries") == _without(canonical.curriculum, "stages", "max_retries")
    n = len(probe.curriculum.stages)
    assert 0 < n < len(canonical.curriculum.stages)
    for mine, theirs in zip(probe.curriculum.stages, canonical.curriculum.stages[:n], strict=True):
        budget = ("max_timesteps", "anneal_horizon")
        assert _without(mine, *budget) == _without(theirs, *budget), mine.name
        assert mine.max_timesteps <= theirs.max_timesteps, mine.name
        assert mine.resolve_horizon() <= theirs.resolve_horizon(), mine.name


def test_self_play_trains_the_bootstrap_mdp():
    """self_play.yaml's env and policy match the curriculum's, so a bootstrap checkpoint continues there."""
    sp, boot = load_config(SELF_PLAY), load_config(BOOTSTRAP)
    for name in (
        "reward_config",
        "action_space_type",
        "max_flat_actions",
        "flat_action_version",
        "max_actions_per_turn",
        "enabled_units",
        "fog_of_war",
        "engine_overrides",
        "gold_scale",
        "turn_scale",
        "unit_count_scale",
    ):
        assert getattr(sp.env, name) == getattr(boot.env, name), f"env.{name}"
    assert sp.env.pad_to_size == bootstrap._resolve_curriculum_pad_size(boot)
    assert sp.ppo.policy_kwargs == boot.ppo.policy_kwargs
    # Potential shaping is policy-invariant only for the trainer's own gamma.
    assert sp.ppo.gamma == boot.ppo.gamma
    # The map and its game clock are the curriculum's.
    clocks = {s.map_file: s.resolve_max_turns(boot.env) for s in boot.curriculum.stages}
    assert sp.env.map_file in clocks and sp.env.max_turns == clocks[sp.env.map_file]
    # The pool actually fills: a snapshot's gate counts wins against a near copy of itself.
    assert sp.self_play.use_opponent_pool and 0.0 < sp.self_play.latest_opponent_prob < 1.0
    assert sp.self_play.min_win_rate_for_pool < 0.5
