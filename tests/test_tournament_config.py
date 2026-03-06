"""Tests for TournamentConfig and config parsing."""

import os
import tempfile

import pytest

from reinforcetactics.tournament.bots import BotType
from reinforcetactics.tournament.config import TournamentConfig, parse_bots_from_config
from reinforcetactics.tournament.schedule import MapConfig


class TestTournamentConfigBasics:
    """Tests for basic TournamentConfig creation."""

    def test_default_config(self):
        """Default config has sensible defaults."""
        config = TournamentConfig()
        assert config.name == "Tournament"
        assert config.games_per_side == 2
        assert config.max_turns == 500
        assert config.save_replays is True
        assert config.concurrent_games == 1

    def test_config_with_string_maps(self):
        """String maps are converted to MapConfig."""
        config = TournamentConfig(maps=["maps/1v1/beginner.csv"])
        assert len(config.maps) == 1
        assert isinstance(config.maps[0], MapConfig)
        assert config.maps[0].path == "maps/1v1/beginner.csv"

    def test_config_with_dict_maps(self):
        """Dict maps are converted to MapConfig."""
        config = TournamentConfig(
            maps=[{"path": "maps/test.csv", "max_turns": 200}],
            max_turns=500,
        )
        assert len(config.maps) == 1
        assert config.maps[0].path == "maps/test.csv"

    def test_config_with_mapconfig(self):
        """MapConfig objects are preserved."""
        mc = MapConfig(path="maps/test.csv", max_turns=100)
        config = TournamentConfig(maps=[mc])
        assert config.maps[0] is mc

    def test_invalid_map_config_raises(self):
        """Invalid map config type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid map config"):
            TournamentConfig(maps=[42])

    def test_replay_dir_default(self):
        """Replay dir defaults to output_dir/replays."""
        config = TournamentConfig(output_dir="results")
        assert "replays" in config.replay_dir

    def test_conversation_log_dir_default(self):
        """Conversation log dir set when log_conversations is True."""
        config = TournamentConfig(log_conversations=True, output_dir="results")
        assert config.conversation_log_dir is not None
        assert "llm_conversations" in config.conversation_log_dir


class TestTournamentConfigFromDict:
    """Tests for TournamentConfig.from_dict()."""

    def test_from_flat_dict(self):
        """from_dict handles flat config format."""
        data = {
            "name": "Test Tournament",
            "maps": ["maps/test.csv"],
            "games_per_side": 3,
            "max_turns": 300,
            "enabled_units": ["W", "M"],
        }
        config = TournamentConfig.from_dict(data)
        assert config.name == "Test Tournament"
        assert config.games_per_side == 3
        assert config.max_turns == 300
        assert config.enabled_units == ["W", "M"]

    def test_from_flat_dict_string_maps(self):
        """from_dict handles single string map."""
        data = {"maps": "maps/test.csv"}
        config = TournamentConfig.from_dict(data)
        assert len(config.maps) == 1

    def test_from_nested_dict(self):
        """from_dict handles nested 'tournament' key format."""
        data = {
            "tournament": {
                "name": "Docker Tournament",
                "games_per_matchup": 4,
                "max_turns": 200,
                "save_replays": False,
                "log_conversations": True,
                "should_reason": True,
                "llm_api_delay": 2.0,
                "concurrent_games": 2,
                "enabled_units": ["W", "C"],
            },
            "maps": [{"path": "maps/test.csv"}],
            "output": {
                "results_dir": "custom_results",
                "replay_dir": "custom_replays",
                "conversation_log_dir": "custom_logs",
            },
        }
        config = TournamentConfig.from_dict(data)
        assert config.name == "Docker Tournament"
        assert config.games_per_side == 4
        assert config.max_turns == 200
        assert config.save_replays is False
        assert config.log_conversations is True
        assert config.should_reason is True
        assert config.llm_api_delay == 2.0
        assert config.concurrent_games == 2
        assert config.output_dir == "custom_results"
        assert config.replay_dir == "custom_replays"
        assert config.conversation_log_dir == "custom_logs"
        assert config.enabled_units == ["W", "C"]


class TestTournamentConfigSerialization:
    """Tests for to_dict/to_json/from_json."""

    def test_to_dict(self):
        """to_dict produces serializable dict."""
        config = TournamentConfig(name="Test", maps=["maps/test.csv"])
        d = config.to_dict()
        assert d["name"] == "Test"
        assert isinstance(d["maps"], list)
        assert "games_per_side" in d
        assert "enabled_units" in d

    def test_to_json_and_from_json(self):
        """Round-trip through JSON file preserves config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "config.json")
            config = TournamentConfig(
                name="Round Trip",
                maps=["maps/test.csv"],
                games_per_side=5,
                enabled_units=["W", "M", "C"],
            )
            config.to_json(filepath)
            loaded = TournamentConfig.from_json(filepath)
            assert loaded.name == "Round Trip"
            assert loaded.games_per_side == 5
            assert loaded.enabled_units == ["W", "M", "C"]


class TestTournamentConfigValidation:
    """Tests for config validation."""

    def test_validate_no_maps(self):
        """Validation catches missing maps."""
        config = TournamentConfig(maps=[])
        errors = config.validate()
        assert any("map" in e.lower() for e in errors)

    def test_validate_invalid_games_per_side(self):
        """Validation catches games_per_side < 1."""
        config = TournamentConfig(games_per_side=0)
        errors = config.validate()
        assert any("games_per_side" in e for e in errors)

    def test_validate_invalid_max_turns(self):
        """Validation catches max_turns < 1."""
        config = TournamentConfig(max_turns=0)
        errors = config.validate()
        assert any("max_turns" in e for e in errors)

    def test_validate_invalid_map_pool_mode(self):
        """Validation catches invalid map_pool_mode."""
        config = TournamentConfig(map_pool_mode="invalid")
        errors = config.validate()
        assert any("map_pool_mode" in e for e in errors)

    def test_validate_invalid_concurrent_games(self):
        """Validation catches concurrent_games < 1."""
        config = TournamentConfig(concurrent_games=0)
        errors = config.validate()
        assert any("concurrent_games" in e for e in errors)

    def test_validate_invalid_unit_types(self):
        """Validation catches invalid unit types."""
        config = TournamentConfig(enabled_units=["W", "X", "Z"])
        errors = config.validate()
        assert any("Invalid unit types" in e for e in errors)

    def test_validate_empty_enabled_units(self):
        """Validation catches empty enabled_units list."""
        config = TournamentConfig(enabled_units=[])
        errors = config.validate()
        assert any("empty" in e.lower() for e in errors)

    def test_validate_none_enabled_units_ok(self):
        """None enabled_units passes validation (means all enabled)."""
        config = TournamentConfig(maps=["maps/1v1/beginner.csv"])
        errors = config.validate()
        # Only map-not-found errors expected, not unit errors
        assert not any("enabled_units" in e for e in errors)


class TestTournamentConfigAddMaps:
    """Tests for map addition methods."""

    def test_add_map(self):
        """add_map appends a map and returns self for chaining."""
        config = TournamentConfig()
        result = config.add_map("maps/test.csv", max_turns=200)
        assert result is config
        assert len(config.maps) == 1
        assert config.maps[0].path == "maps/test.csv"

    def test_add_maps_from_directory(self):
        """add_maps_from_directory finds CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some CSV files
            for name in ["map1.csv", "map2.csv", "readme.txt"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("test")
            config = TournamentConfig()
            result = config.add_maps_from_directory(tmpdir)
            assert result is config
            assert len(config.maps) == 2  # Only CSV files

    def test_add_maps_from_nonexistent_directory_raises(self):
        """add_maps_from_directory raises for missing dir."""
        config = TournamentConfig()
        with pytest.raises(ValueError, match="Directory not found"):
            config.add_maps_from_directory("/nonexistent/dir")


class TestParseBots:
    """Tests for parse_bots_from_config."""

    def test_parse_simple_bot(self):
        """Parse a simple bot config."""
        data = {"bots": [{"name": "Easy Bot", "type": "simple"}]}
        bots = parse_bots_from_config(data)
        assert len(bots) == 1
        assert bots[0].name == "Easy Bot"
        assert bots[0].bot_type == BotType.SIMPLE

    def test_parse_medium_bot(self):
        """Parse a medium bot config."""
        data = {"bots": [{"name": "Med Bot", "type": "medium"}]}
        bots = parse_bots_from_config(data)
        assert bots[0].bot_type == BotType.MEDIUM

    def test_parse_advanced_bot(self):
        """Parse an advanced bot config."""
        data = {"bots": [{"name": "Hard Bot", "type": "advanced"}]}
        bots = parse_bots_from_config(data)
        assert bots[0].bot_type == BotType.ADVANCED

    def test_parse_llm_bot(self):
        """Parse an LLM bot config."""
        data = {
            "bots": [
                {
                    "name": "Claude Bot",
                    "type": "llm",
                    "provider": "anthropic",
                    "model": "claude-3-haiku",
                    "temperature": 0.5,
                    "max_tokens": 4000,
                }
            ]
        }
        bots = parse_bots_from_config(data)
        assert len(bots) == 1
        assert bots[0].bot_type == BotType.LLM
        assert bots[0].provider == "anthropic"
        assert bots[0].model == "claude-3-haiku"
        assert bots[0].temperature == 0.5
        assert bots[0].max_tokens == 4000

    def test_parse_model_bot(self):
        """Parse a model bot config."""
        data = {"bots": [{"name": "RL Bot", "type": "model", "model_path": "models/best.zip"}]}
        bots = parse_bots_from_config(data)
        assert bots[0].bot_type == BotType.MODEL
        assert bots[0].model_path == "models/best.zip"

    def test_parse_unknown_bot_type_raises(self):
        """Unknown bot type raises ValueError."""
        data = {"bots": [{"name": "Bad Bot", "type": "unknown"}]}
        with pytest.raises(ValueError, match="Unknown bot type"):
            parse_bots_from_config(data)

    def test_parse_empty_bots(self):
        """Empty bots list returns empty."""
        data = {"bots": []}
        assert parse_bots_from_config(data) == []

    def test_parse_no_bots_key(self):
        """Missing bots key returns empty."""
        data = {}
        assert parse_bots_from_config(data) == []

    def test_parse_multiple_bots(self):
        """Parse multiple bots of different types."""
        data = {
            "bots": [
                {"name": "Easy", "type": "simple"},
                {"name": "Medium", "type": "medium"},
                {"name": "LLM", "type": "llm", "provider": "openai"},
            ]
        }
        bots = parse_bots_from_config(data)
        assert len(bots) == 3
