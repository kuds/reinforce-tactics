"""Every shipped config still loads and passes the env's own validation.

``StrategyGameEnv`` now rejects unknown opponents (review rlenv-10 /
rulebots-7), opponent kwargs the opponent's constructor does not take, and
unknown or non-numeric reward_config keys (rlenv-16). This walks every
YAML/JSON file under ``configs/`` and checks that nothing shipped relies on
what is now rejected: training configs load and validate through
``load_config``, every env block and curriculum stage passes the env-level
validators, and one env per distinct setting is actually constructed.
Behaviour-cloning scenario files (``scenarios:``) only name bots, which must
resolve through the registry.
"""

import functools
import json
from pathlib import Path

import pytest
import yaml

from reinforcetactics.game.bot_registry import canonical_name
from reinforcetactics.rl.config import load_config
from reinforcetactics.rl.gym_env import (
    StrategyGameEnv,
    resolve_opponent,
    validate_opponent_kwargs,
    validate_reward_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = sorted(p for p in (REPO_ROOT / "configs").rglob("*") if p.suffix.lower() in {".yaml", ".yml", ".json"})


@functools.cache
def _read(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    return data or {}


@functools.cache
def _env_settings(path: Path) -> list[tuple[str, dict]]:
    """``(where, env_kwargs)`` for the config's env block and each curriculum stage (``[]`` for BC scenario files)."""
    if "scenarios" in _read(path):
        return []
    cfg = load_config(path)
    env = cfg.env
    settings = []
    base = {
        "action_space_type": env.action_space_type,
        "max_flat_actions": env.max_flat_actions,
        "max_actions_per_turn": env.max_actions_per_turn,
        "enabled_units": env.enabled_units,
        "engine_overrides": env.engine_overrides,
    }
    settings.append(
        (
            "env",
            {
                **base,
                "map_file": env.map_file,
                "opponent": env.opponent,
                "opponent_kwargs": env.opponent_kwargs,
                "reward_config": env.reward_config,
            },
        )
    )
    for stage in cfg.curriculum.stages:
        settings.append(
            (
                f"stage {stage.name}",
                {
                    **base,
                    "map_file": stage.map_file,
                    "opponent": stage.opponent,
                    "opponent_kwargs": stage.opponent_kwargs,
                    "reward_config": stage.resolve_reward_config(env),
                },
            )
        )
    return settings


def test_configs_are_found():
    assert len(CONFIG_FILES) > 60


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_shipped_config_passes_env_validation(path):
    raw = _read(path)
    if "scenarios" in raw:  # behaviour-cloning scenario list, not a TrainingConfig
        for scenario in raw["scenarios"]:
            resolve_opponent(scenario["opponent"])
            canonical_name(scenario["demonstrator"])
        return
    for where, kwargs in _env_settings(path):
        try:
            validate_reward_config(kwargs["reward_config"])
            validate_opponent_kwargs(kwargs["opponent"], kwargs["opponent_kwargs"])
        except (ValueError, TypeError, KeyError) as exc:  # pragma: no cover - the failure message
            pytest.fail(f"{path.relative_to(REPO_ROOT)} {where}: {exc}")


def test_every_distinct_shipped_env_setting_constructs():
    # Keyed without reward_config (validated per stage above; hundreds of
    # tuning variants would otherwise dominate the run time).
    seen: set[str] = set()
    for path in CONFIG_FILES:
        for where, kwargs in _env_settings(path):
            key = json.dumps({k: v for k, v in kwargs.items() if k != "reward_config"}, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            try:
                StrategyGameEnv(**kwargs).close()
            except Exception as exc:  # pragma: no cover - the failure message
                pytest.fail(f"{path.relative_to(REPO_ROOT)} {where}: {type(exc).__name__}: {exc}")
    assert len(seen) > 20
