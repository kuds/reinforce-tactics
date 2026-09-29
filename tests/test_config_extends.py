"""Config inheritance and stage addressing (review rltrain-15 / consolidate-14).

* ``extends: <path>``: a file as a diff against another (deep merge; stages by
  name; ``__replace__``; ``drop_stages`` / ``stage_order``);
* ``curriculum.stage_defaults``: fields every stage inherits unless it sets
  them;
* ``--set curriculum.stages[<name>|*].<field>`` and dotted tails below
  mapping-valued fields (``env.reward_config.turn_penalty``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from reinforcetactics.rl import bootstrap
from reinforcetactics.rl.config import (
    EXTENDS_MAX_DEPTH,
    _read_config_file,
    _read_config_tree,
    apply_overrides,
    config_from_dict,
    load_config,
    merge_config_dicts,
    save_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP = "maps/1v1/beginner.csv"


def _stage(name: str, **kw: Any) -> dict[str, Any]:
    return {"name": name, "map_file": MAP, "opponent": "random", "max_timesteps": 1000, **kw}


BASE: dict[str, Any] = {
    "seed": 42,
    "env": {"n_envs": 1, "use_subprocess": False, "reward_config": {"win": 50.0, "draw": -10.0}},
    "ppo": {"learning_rate": 3e-4, "policy_kwargs": {"net_arch": [32, 32]}},
    "eval": {"eval_freq": 100, "n_eval_episodes": 4},
    "curriculum": {
        "stages": [
            _stage("a", patience=2),
            _stage("b", opponent="simple", reward_config={"draw": -20.0}),
            _stage("c", opponent="medium"),
        ]
    },
}


def _write(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _names(cfg) -> list[str]:
    return [s.name for s in cfg.curriculum.stages]


class TestMerge:
    def test_mappings_merge_everything_else_replaces(self):
        base = {"a": {"x": 1, "y": {"p": 1, "q": 2}}, "l": [1, 2], "s": "keep", "n": 5}
        overlay = {"a": {"y": {"q": 3}, "z": 4}, "l": [9], "n": None}
        merged = merge_config_dicts(base, overlay)
        assert merged == {"a": {"x": 1, "y": {"p": 1, "q": 3}, "z": 4}, "l": [9], "s": "keep", "n": None}
        assert base == {"a": {"x": 1, "y": {"p": 1, "q": 2}}, "l": [1, 2], "s": "keep", "n": 5}  # untouched

    def test_replace_marker(self):
        base = {"a": {"x": 1, "y": 2}}
        assert merge_config_dicts(base, {"a": {"__replace__": True, "z": 3}}) == {"a": {"z": 3}}
        assert merge_config_dicts(base, {"a": {"__replace__": False, "z": 3}}) == {"a": {"x": 1, "y": 2, "z": 3}}
        # A marker with nothing to replace is dropped.
        assert merge_config_dicts({}, {"b": {"__replace__": True, "k": 1}}) == {"b": {"k": 1}}
        with pytest.raises(TypeError, match="__replace__"):
            merge_config_dicts(base, {"a": {"__replace__": "yes"}})

    def test_stages_merge_by_name(self):
        merged = merge_config_dicts(
            BASE,
            {"curriculum": {"stages": [{"name": "b", "patience": 5, "reward_config": {"win": 1.0}}, _stage("d")]}},
        )
        stages = merged["curriculum"]["stages"]
        assert [s["name"] for s in stages] == ["a", "b", "c", "d"]
        assert stages[1]["patience"] == 5 and stages[1]["opponent"] == "simple"
        assert stages[1]["reward_config"] == {"draw": -20.0, "win": 1.0}
        replaced = merge_config_dicts(
            BASE, {"curriculum": {"stages": [{"__replace__": True, **_stage("b", opponent="noop")}]}}
        )
        assert replaced["curriculum"]["stages"][1] == _stage("b", opponent="noop")

    @pytest.mark.parametrize(
        "stages, match",
        [([{"patience": 3}], "has no name"), ([_stage("x"), _stage("x")], "duplicate stage name 'x'")],
    )
    def test_stage_merge_errors(self, stages, match):
        with pytest.raises(ValueError, match=match):
            merge_config_dicts(BASE, {"curriculum": {"stages": stages}})


class TestExtends:
    def test_relative_chain_and_absolute_paths(self, tmp_path):
        _write(tmp_path / "base.yaml", BASE)
        _write(tmp_path / "sub" / "mid.yaml", {"extends": "../base.yaml", "ppo": {"learning_rate": 1e-4}})
        leaf = _write(
            tmp_path / "sub" / "deeper" / "leaf.yaml",
            {"extends": "../mid.yaml", "env": {"reward_config": {"turn_penalty": -0.5}}, "seed": 7},
        )
        cfg = load_config(leaf)
        assert cfg.seed == 7 and cfg.ppo.learning_rate == pytest.approx(1e-4)
        assert cfg.env.reward_config == {"win": 50.0, "draw": -10.0, "turn_penalty": -0.5}
        assert cfg.ppo.policy_kwargs == {"net_arch": [32, 32]} and _names(cfg) == ["a", "b", "c"]
        absolute = _write(tmp_path / "abs.yaml", {"extends": str(tmp_path / "base.yaml"), "seed": 1})
        assert load_config(absolute).seed == 1

    def test_cycle_missing_base_and_depth(self, tmp_path):
        _write(tmp_path / "a.yaml", {"extends": "b.yaml"})
        _write(tmp_path / "b.yaml", {"extends": "a.yaml"})
        with pytest.raises(ValueError, match=r"'extends' cycle: .*a\.yaml -> .*b\.yaml -> .*a\.yaml"):
            load_config(tmp_path / "a.yaml")
        _write(tmp_path / "orphan.yaml", {"extends": "nowhere.yaml"})
        with pytest.raises(FileNotFoundError, match="nowhere.yaml"):
            load_config(tmp_path / "orphan.yaml")
        _write(tmp_path / "chain" / "c0.yaml", BASE)
        for i in range(1, EXTENDS_MAX_DEPTH + 2):
            _write(tmp_path / "chain" / f"c{i}.yaml", {"extends": f"c{i - 1}.yaml", "seed": i})
        assert load_config(tmp_path / "chain" / f"c{EXTENDS_MAX_DEPTH}.yaml").seed == EXTENDS_MAX_DEPTH
        with pytest.raises(ValueError, match="deeper than"):
            load_config(tmp_path / "chain" / f"c{EXTENDS_MAX_DEPTH + 1}.yaml")

    def test_directives(self, tmp_path):
        _write(tmp_path / "base.yaml", BASE)
        child = _write(
            tmp_path / "child.yaml",
            {
                "extends": "base.yaml",
                "curriculum": {"stages": [_stage("d")], "drop_stages": ["b"], "stage_order": ["d", "a", "c"]},
            },
        )
        cfg = load_config(child)
        assert _names(cfg) == ["d", "a", "c"]
        # The directives were applied at their level: a grandchild sees the result.
        grandchild = _write(tmp_path / "g.yaml", {"extends": "child.yaml", "curriculum": {"drop_stages": ["d"]}})
        assert _names(load_config(grandchild)) == ["a", "c"]

    @pytest.mark.parametrize(
        "curriculum, match",
        [
            ({"drop_stages": ["zzz"]}, r"unknown stage\(s\) \['zzz'\]"),
            ({"stage_order": ["a", "b"]}, r"missing \['c'\]"),
            ({"stage_order": ["a", "b", "c", "x"]}, r"unknown \['x'\]"),
            ({"stage_order": ["a", "b", "c", "a"]}, r"repeated \['a'\]"),
            ({"drop_stages": ["a"], "stage_order": ["a", "b", "c"]}, r"unknown \['a'\]"),
            ({"drop_stages": "a"}, "list of stage names"),
        ],
    )
    def test_directive_errors(self, tmp_path, curriculum, match):
        _write(tmp_path / "base.yaml", BASE)
        child = _write(tmp_path / "child.yaml", {"extends": "base.yaml", "curriculum": curriculum})
        with pytest.raises((ValueError, TypeError), match=match):
            load_config(child)

    def test_dicts_take_directives_but_not_extends(self):
        data = {**BASE, "curriculum": {**BASE["curriculum"], "drop_stages": ["a"]}}
        assert _names(config_from_dict(data)) == ["b", "c"]
        assert "drop_stages" in data["curriculum"]  # the caller's dict is not modified
        with pytest.raises(ValueError, match="extends"):
            config_from_dict({**BASE, "extends": "x.yaml"})

    def test_saved_config_has_no_directives_and_round_trips(self, tmp_path):
        _write(tmp_path / "base.yaml", BASE)
        child = _write(
            tmp_path / "child.yaml",
            {"extends": "base.yaml", "curriculum": {"stage_defaults": {"max_turns": 9}, "drop_stages": ["c"]}},
        )
        cfg = load_config(child)
        save_config(cfg, tmp_path / "resolved.yaml")
        text = (tmp_path / "resolved.yaml").read_text()
        for word in ("extends", "stage_defaults", "drop_stages", "stage_order", "__replace__"):
            assert word not in text
        assert load_config(tmp_path / "resolved.yaml").to_dict() == cfg.to_dict()
        assert [s.max_turns for s in cfg.curriculum.stages] == [9, 9]

    def test_a_run_started_from_an_extends_config_resumes_with_zero_differences(self, tmp_path):
        _write(tmp_path / "base.yaml", BASE)
        child = _write(
            tmp_path / "child.yaml",
            {
                "extends": "base.yaml",
                "curriculum": {"stage_defaults": {"patience": 3}, "stages": [{"name": "c", "promotion_win_rate": 0.5}]},
            },
        )
        cfg = bootstrap.resolve_config(load_config(child))
        run = tmp_path / "run"
        run.mkdir()
        save_config(cfg, run / "resolved_config.yaml")  # what train_bootstrap.py records
        assert bootstrap.resume_config_differences(load_config(child), run) == []
        assert bootstrap.resume_mismatches(load_config(child), run) == []


class TestStageDefaults:
    def test_fills_absent_and_null_fields_only(self):
        data = {
            **BASE,
            "curriculum": {
                "stage_defaults": {"patience": 4, "max_turns": 30, "reward_config": {"win": 1.0}},
                "stages": [_stage("a", patience=2), _stage("b", max_turns=None), _stage("c", reward_config={"draw": -1.0})],
            },
        }
        cfg = config_from_dict(data)
        a, b, c = cfg.curriculum.stages
        assert (a.patience, b.patience, c.patience) == (2, 4, 4)
        assert (a.max_turns, b.max_turns, c.max_turns) == (30, 30, 30)
        # Field by field: a stage's own mapping is kept whole.
        assert (a.reward_config, c.reward_config) == ({"win": 1.0}, {"draw": -1.0})
        assert "stage_defaults" not in cfg.to_dict()["curriculum"]

    @pytest.mark.parametrize("defaults, match", [({"name": "x"}, "cannot set 'name'"), ({"bogus": 1}, r"\['bogus'\]")])
    def test_errors(self, defaults, match):
        with pytest.raises(ValueError, match=match):
            config_from_dict({**BASE, "curriculum": {**BASE["curriculum"], "stage_defaults": defaults}})
        with pytest.raises(TypeError, match="stage_defaults"):
            config_from_dict({**BASE, "curriculum": {**BASE["curriculum"], "stage_defaults": [1]}})

    def test_values_are_validated_per_stage(self):
        with pytest.raises(ValueError, match="patience must be >= 1"):
            config_from_dict({**BASE, "curriculum": {**BASE["curriculum"], "stage_defaults": {"patience": 0}}})


def _training_configs() -> list[Path]:
    out = []
    for path in sorted((REPO_ROOT / "configs").rglob("*")):
        if path.suffix.lower() not in (".yaml", ".yml", ".json"):
            continue
        if "scenarios" in (_read_config_file(path) or {}):
            continue
        out.append(path)
    return out


@pytest.mark.parametrize("path", _training_configs(), ids=lambda p: p.relative_to(REPO_ROOT / "configs").as_posix())
def test_shipped_configs_load_as_before(path):
    """No shipped config uses the new keys, so the tree reader changes nothing about them."""
    assert _read_config_tree(path) == _read_config_file(path)
    assert load_config(path).to_dict() == config_from_dict(_read_config_file(path)).to_dict()


class TestSetStages:
    def test_one_stage_every_stage_and_coercion(self):
        cfg = config_from_dict(BASE)
        new = apply_overrides(cfg, {"curriculum.stages[*].patience": "3", "curriculum.stages[b].max_turns": 40})
        assert [s.patience for s in new.curriculum.stages] == [3, 3, 3]
        assert [s.max_turns for s in new.curriculum.stages] == [None, 40, None]
        assert [s.patience for s in cfg.curriculum.stages] == [2, 2, 2]  # a copy
        # Applied in order: every stage, then one.
        new = apply_overrides(
            cfg, {"curriculum.stages[*].promotion_win_rate": 0.5, "curriculum.stages[a].promotion_win_rate": 0.8}
        )
        assert [s.promotion_win_rate for s in new.curriculum.stages] == [0.8, 0.5, 0.5]
        new = apply_overrides(cfg, {"curriculum.stages[b].max_turns": "null"})
        assert new.curriculum.stages[1].max_turns is None

    def test_unknown_stage_or_field(self):
        cfg = config_from_dict(BASE)
        with pytest.raises(KeyError, match="No curriculum stage named 'zzz'.*Stages: a, b, c"):
            apply_overrides(cfg, {"curriculum.stages[zzz].patience": 1})
        with pytest.raises(KeyError, match="'bogus' is not a field of CurriculumStage"):
            apply_overrides(cfg, {"curriculum.stages[a].bogus": 1})
        with pytest.raises(KeyError, match="only curriculum.stages takes"):
            apply_overrides(cfg, {"env[a].n_envs": 1})
        with pytest.raises(KeyError, match="name a stage field"):
            apply_overrides(cfg, {"curriculum.stages[a]": 1})
        with pytest.raises(KeyError, match="stage_defaults is applied when a config file is loaded"):
            apply_overrides(cfg, {"curriculum.stage_defaults.patience": 1})

    def test_values_are_validated(self):
        cfg = config_from_dict(BASE)
        with pytest.raises(ValueError, match="patience must be >= 1"):
            apply_overrides(cfg, {"curriculum.stages[*].patience": 0})
        with pytest.raises(ValueError, match="must be an integer"):
            apply_overrides(cfg, {"curriculum.stages[a].patience": "two"})

    def test_mapping_tails(self):
        cfg = config_from_dict(BASE)
        new = apply_overrides(cfg, {"env.reward_config.turn_penalty": -0.5, "env.reward_config.draw": "null"})
        assert new.env.reward_config == {"win": 50.0, "turn_penalty": -0.5}
        new = apply_overrides(cfg, {"ppo.policy_kwargs.features_extractor_kwargs.pool": "flatten"})
        assert new.ppo.policy_kwargs == {"net_arch": [32, 32], "features_extractor_kwargs": {"pool": "flatten"}}
        new = apply_overrides(
            cfg, {"curriculum.stages[a].reward_config.win": "7", "curriculum.stages[b].reward_config.win": 3}
        )
        assert new.curriculum.stages[0].reward_config == {"win": 7.0}
        assert new.curriculum.stages[1].reward_config == {"draw": -20.0, "win": 3.0}
        # Deleting from an unset mapping leaves it unset.
        assert (
            apply_overrides(cfg, {"curriculum.stages[c].reward_config.win": "null"}).curriculum.stages[2].reward_config is None
        )
        with pytest.raises(ValueError, match="Unknown reward_config keys"):
            apply_overrides(cfg, {"env.reward_config.not_a_reward": 1})
        with pytest.raises(ValueError, match="Unknown reward_config keys"):
            apply_overrides(cfg, {"curriculum.stages[*].reward_config.nope": 1})
        with pytest.raises(KeyError, match="does not point to a config section"):
            apply_overrides(cfg, {"seed.x": 1})
