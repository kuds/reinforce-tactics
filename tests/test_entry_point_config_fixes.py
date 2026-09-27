"""Config and training-entry-point fixes from the review of the MDP/config package.

One class per finding; each docstring says what used to happen.
"""

import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path

import pytest
import yaml

from reinforcetactics.game.bot import MixedBot, RandomBot
from reinforcetactics.game.bot_registry import accepted_names, validate_scripted_kwargs
from reinforcetactics.rl import bootstrap
from reinforcetactics.rl.config import (
    IgnoredConfigFieldError,
    IgnoredConfigFieldWarning,
    TrainingConfig,
    check_ignored_config_fields,
    config_from_dict,
    effective_config,
)
from reinforcetactics.rl.gym_env import StrategyGameEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP = "maps/1v1/beginner.csv"


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "train" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_review_fixes_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def self_play_script():
    return _load_script("train_self_play")


@pytest.fixture(scope="module")
def feudal_script():
    return _load_script("train_feudal_rl")


@pytest.fixture(scope="module")
def alphazero_script():
    return _load_script("train_alphazero")


def _yaml(path: Path, data: dict) -> str:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def _stage(**kwargs) -> dict:
    return {"name": "s1", "map_file": MAP, "opponent": "random", "max_timesteps": 1_000, **kwargs}


# ---------------------------------------------------------------------------
# train_feudal_rl --mode feudal --opponent self with n_envs > 1
# ---------------------------------------------------------------------------


class TestFeudalSelfPlayVecEnvs:
    """The snapshot factory was registered on the single env only.

    ``collect_rollout_vec`` (n_envs > 1) steps separate envs built with
    opponent='self' and no factory, so every vectorized env trained against
    no opponent at all.
    """

    def test_every_rollout_env_plays_a_snapshot(self, feudal_script, monkeypatch, tmp_path):
        from reinforcetactics.rl import feudal_rl

        class _Stop(Exception):
            pass

        seen = {}

        def first_vec_rollout(agent, envs, **kwargs):
            seen["opponents"] = [None if e.opponent is None else type(e.opponent).__name__ for e in envs]
            raise _Stop

        monkeypatch.setattr(feudal_rl.FeudalRLAgent, "collect_rollout_vec", first_vec_rollout)
        argv = [
            "--mode", "feudal", "--opponent", "self", "--n-envs", "2", "--map-file", MAP, "--max-steps", "20",
            "--n-steps", "8", "--total-timesteps", "16", "--device", "cpu", "--log-dir", str(tmp_path),
        ]  # fmt: skip
        with pytest.raises(_Stop):
            feudal_script.main(argv)
        assert seen["opponents"] == ["ModelBot", "ModelBot"]


# ---------------------------------------------------------------------------
# train_bootstrap --set KEY=null
# ---------------------------------------------------------------------------


class TestSetNull:
    """``--set env.max_turns=null`` was dropped: apply_overrides skips None."""

    BASE = {
        "seed": 1,
        "env": {"n_envs": 1, "use_subprocess": False, "max_turns": 30, "max_actions_per_turn": 30},
        "curriculum": {"stages": [_stage()]},
    }

    @pytest.fixture
    def run(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "path", [*sys.path])
        script = _load_script("train_bootstrap")
        seen: list[TrainingConfig] = []

        def fake_run(cfg, output_dir):
            seen.append(cfg)
            return {"history": [], "final_model_path": None}

        monkeypatch.setattr(bootstrap, "run_curriculum", fake_run)
        config = _yaml(tmp_path / "c.yaml", self.BASE)

        def _run(*sets: str) -> TrainingConfig:
            argv = ["--config", config, "--output-dir", str(tmp_path / f"out{len(seen)}"), "--device", "cpu"]
            argv += ["--no-gcs", "--skip-plots", "--skip-videos", "--sanity-episodes", "0", "--strict"]
            for item in sets:
                argv += ["--set", item]
            assert script.main(argv) == 0
            return seen[-1]

        return _run

    @pytest.mark.parametrize("null", ["null", "~", "Null", ""])
    def test_a_yaml_null_unsets_the_field(self, run, null):
        cfg = run(f"env.max_turns={null}", f"env.max_actions_per_turn={null}")
        assert cfg.env.max_turns is None and cfg.env.max_actions_per_turn is None

    def test_other_values_still_apply(self, run):
        assert run("env.max_turns=12").env.max_turns == 12
        assert run().env.max_turns == 30

    def test_null_for_a_required_field_is_an_error(self, run):
        with pytest.raises(TypeError, match="ppo.gamma must be float, got null"):
            run("ppo.gamma=null")


# ---------------------------------------------------------------------------
# The ignored-field report follows the mode a run is in
# ---------------------------------------------------------------------------


class TestModeAwareIgnoredFieldReport:
    """The consumed sets were per script, so fields a mode never reads passed --strict."""

    @pytest.mark.parametrize(
        ("data", "fields"),
        [
            # Pure self-play: no bot workers.
            ({"env": {"opponent": "medium"}, "self_play": {"bot_ratio": 0.5}}, ["env.opponent", "self_play.bot_ratio"]),
            # No opponent pool: its knobs do nothing.
            (
                {"self_play": {"latest_opponent_prob": 0.7, "pool_size": 3, "pool_strategy": "recent"}},
                ["self_play.pool_size", "self_play.pool_strategy", "self_play.latest_opponent_prob"],
            ),
            # algorithm: mixed on a pure self-play run.
            ({"algorithm": "mixed"}, ["algorithm"]),
        ],
    )
    def test_self_play_mode_reports_what_it_does_not_read(self, self_play_script, tmp_path, data, fields):
        config = _yaml(tmp_path / "sp.yaml", data)
        with pytest.raises(IgnoredConfigFieldError) as excinfo:
            self_play_script.parse_args(["--config", config, "--strict"])
        for path in fields:
            assert f"  {path} = " in str(excinfo.value)

    def test_the_same_fields_are_read_in_the_mode_that_uses_them(self, self_play_script, tmp_path):
        config = _yaml(
            tmp_path / "sp.yaml",
            {
                "algorithm": "mixed",
                "env": {"opponent": "medium"},
                "self_play": {
                    "mixed_training": True,
                    "bot_ratio": 0.5,
                    "use_opponent_pool": True,
                    "latest_opponent_prob": 0.7,
                    "pool_size": 3,
                },
            },
        )
        args = self_play_script.parse_args(["--config", config, "--strict"])
        assert (args.mode, args.bot_opponent, args.latest_opponent_prob) == ("mixed", "medium", 0.7)
        # A flag that turns the mode on counts as well.
        config = _yaml(tmp_path / "sp2.yaml", {"env": {"opponent": "medium"}})
        assert self_play_script.parse_args(["--config", config, "--strict", "--mode", "mixed"]).bot_opponent == "medium"

    def test_feudal_self_play_fields_need_opponent_self(self, feudal_script, tmp_path):
        data = {"algorithm": "feudal", "self_play": {"eval_opponent": "medium", "snapshot_freq": 5, "pool_size": 3}}
        config = _yaml(tmp_path / "f.yaml", data)
        with pytest.raises(IgnoredConfigFieldError, match="(?s)self_play.snapshot_freq.*self_play.eval_opponent"):
            feudal_script.parse_args(["--config", config, "--strict", "--mode", "feudal"])
        args = feudal_script.parse_args(["--config", config, "--strict", "--mode", "feudal", "--opponent", "self"])
        assert (args.eval_opponent, args.opponent_snapshot_freq, args.opponent_pool_size) == ("medium", 5, 3)


# ---------------------------------------------------------------------------
# train_self_play reads self_play.eval_opponent
# ---------------------------------------------------------------------------


class TestSelfPlayEvalOpponent:
    """The field was unmapped: the run evaluated against --eval-opponent's default ("bot")."""

    def test_the_config_value_is_the_eval_opponent(self, self_play_script, tmp_path):
        config = _yaml(tmp_path / "sp.yaml", {"self_play": {"eval_opponent": "random"}})
        assert self_play_script.parse_args(["--config", config, "--strict"]).eval_opponent == "random"
        config = _yaml(tmp_path / "sp2.yaml", {"self_play": {"eval_opponent": "medium"}})
        assert self_play_script.parse_args(["--config", config, "--strict"]).eval_opponent == "medium"

    def test_the_flag_still_overrides_and_the_no_config_default_is_unchanged(self, self_play_script, tmp_path):
        config = _yaml(tmp_path / "sp.yaml", {"self_play": {"eval_opponent": "medium"}})
        assert self_play_script.parse_args(["--config", config, "--eval-opponent", "master"]).eval_opponent == "master"
        assert self_play_script.parse_args([]).eval_opponent == "bot"

    def test_the_shipped_config_keeps_evaluating_against_simple_bot(self, self_play_script):
        args = self_play_script.parse_args(["--config", str(REPO_ROOT / "configs/self_play/self_play.yaml"), "--strict"])
        assert args.eval_opponent == "bot"


# ---------------------------------------------------------------------------
# Scripted-bot kwarg values
# ---------------------------------------------------------------------------


class TestBotKwargValues:
    """Only the keys were checked: RandomBot max_actions '3' / 2.5 crashed on the bot's
    first turn, -1 / 0 ran a bot that never acts, and none of it was caught up front."""

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "3", None, True])
    def test_random_bot_rejects_a_bad_max_actions(self, bad):
        with pytest.raises(ValueError, match="max_actions must be an integer >= 1"):
            RandomBot.validate_config(max_actions=bad)
        with pytest.raises(ValueError, match="max_actions"):
            validate_scripted_kwargs("random", {"max_actions": bad})
        with pytest.raises(ValueError, match="max_actions"):
            RandomBot(None, player=2, max_actions=bad)

    def test_mixed_bot_checks_the_inner_values(self):
        with pytest.raises(ValueError, match="easy_kwargs: RandomBot max_actions"):
            MixedBot.validate_config(easy="random", easy_kwargs={"max_actions": -5})

    def test_the_env_rejects_them_at_construction(self):
        with pytest.raises(ValueError, match="max_actions"):
            StrategyGameEnv(map_file=MAP, opponent="random", opponent_kwargs={"max_actions": -1})

    @pytest.mark.parametrize(("raw", "coerced"), [("3", 3), ("1e1", 10), (5.0, 5)])
    def test_config_values_are_coerced_like_every_other_field(self, raw, coerced):
        cfg = config_from_dict({"curriculum": {"stages": [_stage(opponent_kwargs={"max_actions": raw})]}})
        assert cfg.curriculum.stages[0].opponent_kwargs == {"max_actions": coerced}
        cfg = config_from_dict({"env": {"opponent": "random", "opponent_kwargs": {"max_actions": raw}}})
        assert cfg.env.opponent_kwargs == {"max_actions": coerced}

    def test_mixed_inner_values_are_coerced_without_adding_keys(self):
        kwargs = {"easy": "random", "p_hard": "1e-1", "easy_kwargs": {"max_actions": "5"}}
        cfg = config_from_dict({"env": {"opponent": "mixed", "opponent_kwargs": kwargs}})
        assert cfg.env.opponent_kwargs == {"easy": "random", "p_hard": 0.1, "easy_kwargs": {"max_actions": 5}}

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_actions": 2.5}, r"stage 's1': opponent_kwargs\['max_actions'\] must be an integer"),
            ({"max_actions": None}, r"stage 's1': opponent_kwargs\['max_actions'\] must be int, got null"),
            ({"max_actions": -1}, "max_actions must be an integer >= 1"),
            ({"max_actions": 0}, "max_actions must be an integer >= 1"),
        ],
    )
    def test_bad_values_fail_validation(self, kwargs, match):
        with pytest.raises((TypeError, ValueError), match=match):
            config_from_dict({"curriculum": {"stages": [_stage(opponent_kwargs=kwargs)]}})

    def test_bad_inner_mixed_value_fails_validation(self):
        stage = _stage(opponent="mixed", opponent_kwargs={"easy": "random", "easy_kwargs": {"max_actions": -5}})
        with pytest.raises(ValueError, match="easy_kwargs: RandomBot max_actions"):
            config_from_dict({"curriculum": {"stages": [stage]}})


# ---------------------------------------------------------------------------
# CLI flags are validated like config values
# ---------------------------------------------------------------------------


class TestCliFlagsAreValidated:
    """Only the loaded config was validated; flags overriding it reached the trainer unchecked."""

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["--gamma", "1.5"], "ppo.gamma must be in"),
            (["--eval-freq", "0"], "eval.eval_freq must be > 0"),
            (["--n-envs", "0"], "env.n_envs must be positive"),
            (["--reward-config", '{"wn": 1}'], r"Unknown reward_config keys \['wn'\]"),
            (["--engine-overrides", '{"starting_gld": 5}'], "env.engine_overrides"),
            (["--enabled-units", "W,c"], "env.enabled_units"),
            (["--bot-opponent", "random", "--bot-opponent-kwargs", '{"max_actions": 0}'], "max_actions"),
            (["--latest-opponent-prob", "1.5"], "latest_opponent_prob"),
        ],
    )
    def test_train_self_play(self, self_play_script, capsys, argv, message):
        config = str(REPO_ROOT / "configs/self_play/self_play.yaml")
        with pytest.raises(SystemExit) as excinfo:
            self_play_script.parse_args(["--config", config, "--strict", *argv])
        assert excinfo.value.code == 2
        assert re.search(message, capsys.readouterr().err)

    def test_train_self_play_without_a_config(self, self_play_script, capsys):
        with pytest.raises(SystemExit):
            self_play_script.parse_args(["--gamma", "1.5"])
        assert "ppo.gamma" in capsys.readouterr().err
        assert self_play_script.parse_args(["--gamma", "0.997"]).gamma == 0.997

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["--gamma", "1.5"], "ppo.gamma must be in"),
            (["--n-steps", "0"], "ppo.n_steps must be positive"),
            (["--manager-horizon", "0"], "feudal.manager_horizon"),
            (["--opponent", "self", "--mode", "feudal", "--opponent-pool-size", "0"], "self_play.pool_size"),
        ],
    )
    def test_train_feudal_rl(self, feudal_script, capsys, argv, message):
        with pytest.raises(SystemExit):
            feudal_script.parse_args(argv)
        assert re.search(message, capsys.readouterr().err)

    def test_train_alphazero(self, alphazero_script, capsys):
        with pytest.raises(SystemExit):
            alphazero_script.parse_args(["--lr", "-1"])
        assert "alphazero.lr must be > 0" in capsys.readouterr().err
        assert alphazero_script.parse_args(["--lr", "0.01"]).lr == 0.01

    def test_effective_config_writes_every_mapped_value(self, self_play_script):
        args = self_play_script.parse_args(["--gamma", "0.9", "--enabled-units", "W,M", "--max-turns", "7"])
        cfg = effective_config(
            None, args, self_play_script._ARG_TO_CONFIG_PATH, convert={"enabled_units": self_play_script._enabled_units_list}
        )
        assert (cfg.ppo.gamma, cfg.env.enabled_units, cfg.env.max_turns) == (0.9, ["W", "M"], 7)

    def test_engine_overrides_are_checked_by_the_config_layer(self):
        config_from_dict({"env": {"engine_overrides": {"starting_gold": 300}}})
        with pytest.raises(ValueError, match="env.engine_overrides"):
            config_from_dict({"env": {"engine_overrides": {"starting_gld": 300}}})
        with pytest.raises(ValueError, match="env.engine_overrides"):
            config_from_dict({"env": {"engine_overrides": {"unit_data": {"Z": {"attack": 1}}}}})


class TestSelfPlayRunsWhatItValidated:
    """Flags were validated on a coerced copy of the config and then used raw.

    ``--bot-opponent-kwargs '{"max_actions": 10.0}'`` and ``--reward-config
    '{"win": "1e3"}'`` passed parse_args (the copy held 10 and 1000.0) and
    the envs then rejected the raw values; ``env.opponent: self`` in mixed
    mode passed parse_args even with --strict, and the vec-env builder
    rejected it. Each failed only after the run had created its log
    directory.
    """

    def test_bot_kwargs_run_as_validated(self, self_play_script):
        from reinforcetactics.rl.self_play import make_self_play_vec_env

        argv = ["--mode", "mixed", "--n-envs", "2", "--bot-ratio", "0.5", "--bot-opponent", "random", "--map-file", MAP]
        args = self_play_script.parse_args([*argv, "--bot-opponent-kwargs", '{"max_actions": 10.0}'])
        assert args.bot_opponent_kwargs == {"max_actions": 10}
        assert type(args.bot_opponent_kwargs["max_actions"]) is int
        vec_env = make_self_play_vec_env(
            n_envs=args.n_envs,
            use_subprocess=False,
            bot_ratio=args.bot_ratio,
            bot_opponent=args.bot_opponent,
            bot_opponent_kwargs=args.bot_opponent_kwargs,
            **self_play_script.build_env_kwargs(args),
        )
        vec_env.close()

    def test_reward_config_runs_as_validated(self, self_play_script):
        from reinforcetactics.rl.masking import make_maskable_env

        args = self_play_script.parse_args(["--map-file", MAP, "--reward-config", '{"win": "1e3", "loss": -5}'])
        assert args.reward_config == {"win": 1000.0, "loss": -5.0}
        env = make_maskable_env(opponent="simple", **self_play_script.build_env_kwargs(args))
        assert env.unwrapped.reward_config["win"] == 1000.0
        env.close()

    def test_every_written_flag_takes_the_validated_value(self, self_play_script):
        args = self_play_script.parse_args(
            ["--enabled-units", "W,M", "--pad-to-size", "8", "9", "--wandb-entity", "null", "--max-turns", "7"]
        )
        assert args.enabled_units == ["W", "M"]  # the config field's shape
        assert args.pad_to_size == (8, 9)
        assert args.wandb_entity is None  # the "null" sentinel, as validated
        assert args.max_turns == 7
        # An unset optional flag stays unset rather than taking the config default.
        assert args.flat_action_version is None and args.engine_overrides is None

    @pytest.mark.parametrize("strict", [[], ["--strict"]])
    def test_env_opponent_self_in_mixed_mode_is_a_usage_error(self, self_play_script, capsys, tmp_path, strict):
        config = _yaml(
            tmp_path / "sp.yaml",
            {"env": {"opponent": "self"}, "self_play": {"mixed_training": True, "bot_ratio": 0.5}},
        )
        log_dir = tmp_path / "logs"
        with pytest.raises(SystemExit) as excinfo:
            self_play_script.main(["--config", config, *strict, "--log-dir", str(log_dir), "--no-subprocess"])
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "--mode mixed: bot_opponent must be a scripted bot" in err and "got 'self'" in err
        assert "env.opponent" in err
        assert not log_dir.exists()  # rejected before the run created any output

    def test_env_opponent_self_is_fine_where_no_bot_worker_reads_it(self, self_play_script, tmp_path):
        config = _yaml(tmp_path / "sp.yaml", {"env": {"opponent": "self"}})
        with pytest.warns(IgnoredConfigFieldWarning, match="env.opponent = 'self'"):
            assert self_play_script.parse_args(["--config", config]).mode == "self-play"

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["--n-envs", "1", "--bot-ratio", "0.3"], "gives 0 bot workers"),
            (["--n-envs", "2", "--bot-ratio", "0.9"], "gives 2 bot workers"),
        ],
    )
    def test_the_worker_split_is_checked_at_parse_time(self, self_play_script, capsys, argv, message):
        with pytest.raises(SystemExit) as excinfo:
            self_play_script.parse_args(["--mode", "mixed", *argv])
        assert excinfo.value.code == 2
        assert message in capsys.readouterr().err

    def test_a_missing_resume_checkpoint_is_a_usage_error(self, self_play_script, capsys, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            self_play_script.parse_args(["--resume-from", str(tmp_path / "nope.zip")])
        assert excinfo.value.code == 2
        assert "--resume-from" in capsys.readouterr().err

    def test_feudal_and_alphazero_run_with_the_validated_values_too(self, feudal_script, alphazero_script):
        assert feudal_script.parse_args(["--wandb-entity", "null"]).wandb_entity is None
        assert alphazero_script.parse_args(["--map-file", "null"]).map_file is None


# ---------------------------------------------------------------------------
# train_feudal_rl forwards ppo.gamma to the env
# ---------------------------------------------------------------------------


class TestFeudalShapingGamma:
    """The env's potential-based shaping used 0.99 whatever the learner's discount."""

    def test_config_gamma_reaches_every_env(self, feudal_script, tmp_path):
        config = _yaml(tmp_path / "f.yaml", {"algorithm": "feudal", "env": {"map_file": MAP}, "ppo": {"gamma": 0.997}})
        args = feudal_script.parse_args(["--config", config, "--strict", "--mode", "feudal"])
        assert feudal_script._make_strategy_env(args).gamma == 0.997
        assert feudal_script._env_kwargs_from_cfg(args._cfg.env, args, include_render=False)["gamma"] == 0.997

    def test_the_flag_reaches_the_env_too(self, feudal_script):
        args = feudal_script.parse_args(["--gamma", "0.95", "--map-file", MAP])
        assert feudal_script._make_strategy_env(args).gamma == 0.95


# ---------------------------------------------------------------------------
# train_self_play's config.json records the resolved flat_action_version
# ---------------------------------------------------------------------------


class TestSelfPlayRunRecord:
    """``vars(args)`` recorded ``flat_action_version: null`` for a version-1 resume."""

    def test_a_version_1_resume_is_recorded_as_version_1(self, self_play_script, monkeypatch, tmp_path):
        sb3_contrib = pytest.importorskip("sb3_contrib")
        from reinforcetactics.rl.masking import make_maskable_env

        env = make_maskable_env(
            map_file=MAP, opponent="noop", action_space_type="flat_discrete", max_flat_actions=32, flat_action_version=1
        )
        ckpt = tmp_path / "v1.zip"
        sb3_contrib.MaskablePPO("MultiInputPolicy", env, n_steps=8, batch_size=8, policy_kwargs={"net_arch": [8]}).save(ckpt)
        env.close()

        monkeypatch.setattr(sb3_contrib.MaskablePPO, "learn", lambda self, *args, **kwargs: self)
        args = self_play_script.parse_args(
            [
                "--map-file", MAP, "--action-space", "flat_discrete", "--max-flat-actions", "32",
                "--resume-from", str(ckpt), "--n-envs", "1", "--no-subprocess", "--device", "cpu",
                "--total-timesteps", "8", "--log-dir", str(tmp_path / "logs"), "--no-progress-bar",
            ]
        )  # fmt: skip
        assert args.flat_action_version is None  # resolved from the checkpoint
        log_dir = self_play_script.train_self_play(args)
        record = json.loads((log_dir / "config.json").read_text())
        assert record["flat_action_version"] == 1
        assert record["env_kwargs"]["flat_action_version"] == 1
        assert record["env_kwargs"]["max_flat_actions"] == 32


# ---------------------------------------------------------------------------
# run_curriculum reports ignored fields itself
# ---------------------------------------------------------------------------


class TestRunCurriculumReportsIgnoredFields:
    """Only the CLI scripts reported them; the notebook calls run_curriculum directly."""

    def test_run_curriculum_warns_before_building_anything(self, tmp_path):
        # (eval.checkpoint_freq was the example here until the curriculum
        # started reading it for its rolling resume checkpoint.)
        cfg = config_from_dict(
            {"ppo": {"use_action_masking": False}, "logging": {"wandb": True}, "curriculum": {"stages": [_stage()]}}
        )

        class _Built(Exception):
            pass

        def train_env_factory(stage, cfg):
            raise _Built

        with pytest.warns(IgnoredConfigFieldWarning, match="(?s)run_curriculum.*ppo.use_action_masking.*logging.wandb"):
            with pytest.raises(_Built):
                bootstrap.run_curriculum(cfg, tmp_path / "out", train_env_factory=train_env_factory)

    def test_train_bootstrap_reports_once(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "path", [*sys.path])
        script = _load_script("train_bootstrap")

        def reporting_run(cfg, output_dir):
            # What the real run_curriculum does first.
            check_ignored_config_fields(cfg, bootstrap.CONSUMED_CONFIG_FIELDS, entry_point="run_curriculum")
            return {"history": [], "final_model_path": None}

        monkeypatch.setattr(bootstrap, "run_curriculum", reporting_run)
        config = _yaml(
            tmp_path / "c.yaml",
            {
                "env": {"n_envs": 1, "use_subprocess": False},
                "logging": {"log_dir": "elsewhere"},
                "curriculum": {"stages": [_stage()]},
            },
        )
        argv = ["--config", config, "--output-dir", str(tmp_path / "out"), "--device", "cpu", "--no-gcs"]
        argv += ["--skip-plots", "--skip-videos", "--sanity-episodes", "0"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert script.main(argv) == 0
        reports = [w for w in caught if issubclass(w.category, IgnoredConfigFieldWarning)]
        assert len(reports) == 1 and "train_bootstrap.py" in str(reports[0].message)


# ---------------------------------------------------------------------------
# env.enabled_units
# ---------------------------------------------------------------------------


class TestEnabledUnits:
    """A typo crashed on the first action mask; [] meant "none" to one trainer and "all" to another."""

    @pytest.mark.parametrize("bad", [["Z"], ["W", "M", "c"], []])
    def test_config_rejects(self, bad):
        with pytest.raises(ValueError, match="env.enabled_units"):
            config_from_dict({"env": {"enabled_units": bad}})

    def test_config_accepts_valid_codes(self):
        assert config_from_dict({"env": {"enabled_units": ["W", "A"]}}).env.enabled_units == ["W", "A"]
        assert config_from_dict({"env": {"enabled_units": None}}).env.enabled_units is None

    def test_env_rejects_unknown_codes_at_construction(self):
        with pytest.raises(ValueError, match=r"Unknown enabled_units \['c'\]"):
            StrategyGameEnv(map_file=MAP, opponent="noop", enabled_units=["W", "c"])


# ---------------------------------------------------------------------------
# Opponent fields: one acceptance rule, useful messages
# ---------------------------------------------------------------------------


class TestOpponentFieldConsistency:
    """``env.opponent: null`` was a bare type error; 'SimpleBot' passed for env.opponent and
    eval_opponent but not for a stage; env.opponent_kwargs on a curriculum config got an
    error about the unused env.opponent."""

    def test_null_env_opponent_lists_the_names(self):
        with pytest.raises(ValueError, match="env.opponent must be set.*Expected one of: self, advanced"):
            config_from_dict({"env": {"opponent": None}})

    @pytest.mark.parametrize("form", ["SimpleBot", "Simple", " simple "])
    def test_non_canonical_forms_are_rejected_everywhere(self, form):
        for data in (
            {"env": {"opponent": form}},
            {"self_play": {"eval_opponent": form}},
            {"curriculum": {"stages": [_stage(opponent=form)]}},
        ):
            with pytest.raises(ValueError, match="nknown opponent"):
                config_from_dict(data)

    def test_every_listed_name_is_accepted_everywhere(self):
        for name in accepted_names():
            config_from_dict({"env": {"opponent": name}})
            config_from_dict({"self_play": {"eval_opponent": name}})
            config_from_dict({"curriculum": {"stages": [_stage(opponent=name)]}})

    def test_curriculum_config_with_env_opponent_kwargs_points_at_the_stage(self):
        data = {"env": {"opponent_kwargs": {"max_actions": 10}}, "curriculum": {"stages": [_stage()]}}
        with pytest.raises(ValueError, match="set opponent_kwargs on the curriculum stage"):
            config_from_dict(data)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def test_readme_main_py_examples_use_accepted_opponents():
    """README documented ``main.py ... --opponent self``, which main.py's choices reject."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    opponents = re.findall(r"main\.py[^\n]*--opponent\s+(\S+)", readme)
    assert opponents, "expected at least one main.py --opponent example"
    assert set(opponents) <= set(accepted_names())
