"""
Tests for the tournament library (reinforcetactics.tournament).

Tests cover:
- EloRatingSystem calculations
- BotDescriptor creation and serialization
- Scheduling (round-robin generation)
- TournamentConfig validation
- TournamentResults tracking
"""

import json
import os
import tempfile
import pytest

from reinforcetactics.tournament import (
    EloRatingSystem,
    BotDescriptor,
    BotType,
    MapConfig,
    TournamentConfig,
    ScheduledGame,
    GameResult,
    TournamentResults,
    generate_round_robin_schedule,
)


class TestEloRatingSystemLibrary:
    """Tests for EloRatingSystem from the library."""

    def test_initialization(self):
        """Test EloRatingSystem initialization."""
        elo = EloRatingSystem()
        assert elo.starting_elo == 1500
        assert elo.k_factor == 32

    def test_custom_initialization(self):
        """Test EloRatingSystem with custom values."""
        elo = EloRatingSystem(starting_elo=1200, k_factor=16)
        assert elo.starting_elo == 1200
        assert elo.k_factor == 16

    def test_initialize_bot(self):
        """Test bot initialization."""
        elo = EloRatingSystem()
        elo.initialize_bot("TestBot")

        assert "TestBot" in elo.ratings
        assert elo.ratings["TestBot"] == 1500.0
        assert elo.initial_ratings["TestBot"] == 1500.0
        assert elo.rating_history["TestBot"] == [1500.0]

    def test_duplicate_initialization(self):
        """Test that duplicate initialization is a no-op."""
        elo = EloRatingSystem()
        elo.initialize_bot("TestBot")
        elo.ratings["TestBot"] = 1600.0
        elo.initialize_bot("TestBot")  # Should not reset

        assert elo.ratings["TestBot"] == 1600.0

    def test_expected_score_equal(self):
        """Test expected score for equal-rated players."""
        elo = EloRatingSystem()
        expected = elo.calculate_expected_score(1500, 1500)
        assert abs(expected - 0.5) < 0.001

    def test_expected_score_higher_rated(self):
        """Test expected score for higher-rated player."""
        elo = EloRatingSystem()
        expected = elo.calculate_expected_score(1600, 1400)
        assert expected > 0.5
        assert expected < 1.0

    def test_update_ratings_win(self):
        """Test rating update when bot1 wins."""
        elo = EloRatingSystem()
        elo.initialize_bot("Bot1")
        elo.initialize_bot("Bot2")

        bot1_change, bot2_change = elo.update_ratings("Bot1", "Bot2", 1)

        assert bot1_change > 0
        assert bot2_change < 0
        assert abs(bot1_change + bot2_change) < 0.001  # Zero-sum

    def test_update_ratings_draw(self):
        """Test rating update on draw."""
        elo = EloRatingSystem()
        elo.initialize_bot("Bot1")
        elo.initialize_bot("Bot2")

        bot1_change, bot2_change = elo.update_ratings("Bot1", "Bot2", 0)

        # Draw between equal-rated players should result in minimal change
        assert abs(bot1_change) < 1.0
        assert abs(bot2_change) < 1.0

    def test_rating_history(self):
        """Test that rating history is recorded."""
        elo = EloRatingSystem()
        elo.initialize_bot("Bot1")
        elo.initialize_bot("Bot2")

        elo.update_ratings("Bot1", "Bot2", 1)
        elo.update_ratings("Bot1", "Bot2", 1)

        assert len(elo.rating_history["Bot1"]) == 3  # Initial + 2 updates
        assert len(elo.rating_history["Bot2"]) == 3

    def test_get_rankings(self):
        """Test get_rankings returns sorted list."""
        elo = EloRatingSystem()
        elo.initialize_bot("Bot1")
        elo.initialize_bot("Bot2")
        elo.initialize_bot("Bot3")

        # Bot1 wins all
        elo.update_ratings("Bot1", "Bot2", 1)
        elo.update_ratings("Bot1", "Bot3", 1)

        rankings = elo.get_rankings()
        assert rankings[0][0] == "Bot1"  # Bot1 should be first
        assert rankings[0][1] > rankings[1][1]  # Higher rating

    def test_to_dict_and_from_dict(self):
        """Test serialization/deserialization."""
        elo1 = EloRatingSystem()
        elo1.initialize_bot("Bot1")
        elo1.initialize_bot("Bot2")
        elo1.update_ratings("Bot1", "Bot2", 1)

        data = elo1.to_dict()
        elo2 = EloRatingSystem.from_dict(data)

        assert elo2.ratings == elo1.ratings
        assert elo2.rating_history == elo1.rating_history


class TestBotDescriptorLibrary:
    """Tests for BotDescriptor from the library."""

    def test_simple_bot(self):
        """Test SimpleBot descriptor creation."""
        bot = BotDescriptor.simple_bot("TestSimple")
        assert bot.name == "TestSimple"
        assert bot.bot_type == BotType.SIMPLE

    def test_medium_bot(self):
        """Test MediumBot descriptor creation."""
        bot = BotDescriptor.medium_bot("TestMedium")
        assert bot.name == "TestMedium"
        assert bot.bot_type == BotType.MEDIUM

    def test_advanced_bot(self):
        """Test AdvancedBot descriptor creation."""
        bot = BotDescriptor.advanced_bot("TestAdvanced")
        assert bot.name == "TestAdvanced"
        assert bot.bot_type == BotType.ADVANCED

    def test_llm_bot(self):
        """Test LLM bot descriptor creation."""
        bot = BotDescriptor.llm_bot(
            name="TestLLM",
            provider="openai",
            model="gpt-4",
            temperature=0.7,
        )
        assert bot.name == "TestLLM"
        assert bot.bot_type == BotType.LLM
        assert bot.provider == "openai"
        assert bot.model == "gpt-4"
        assert bot.temperature == 0.7

    def test_model_bot(self):
        """Test model bot descriptor creation."""
        bot = BotDescriptor.model_bot("TestModel", "/path/to/model.zip")
        assert bot.name == "TestModel"
        assert bot.bot_type == BotType.MODEL
        assert bot.model_path == "/path/to/model.zip"

    def test_from_dict(self):
        """Test creating BotDescriptor from dictionary."""
        data = {
            "name": "TestBot",
            "type": "simple",
        }
        bot = BotDescriptor.from_dict(data)
        assert bot.name == "TestBot"
        assert bot.bot_type == BotType.SIMPLE

    def test_to_dict(self):
        """Test converting BotDescriptor to dictionary."""
        bot = BotDescriptor.llm_bot(
            name="TestLLM",
            provider="openai",
            model="gpt-4",
        )
        data = bot.to_dict()

        assert data["name"] == "TestLLM"
        assert data["type"] == "llm"
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4"

    def test_get_display_info(self):
        """Test display info generation."""
        simple_bot = BotDescriptor.simple_bot("TestSimple")
        assert "Simple" in simple_bot.get_display_info()

        llm_bot = BotDescriptor.llm_bot(
            name="TestLLM",
            provider="openai",
            model="gpt-4",
        )
        display = llm_bot.get_display_info()
        assert "openai" in display
        assert "gpt-4" in display


class TestMapConfigLibrary:
    """Tests for MapConfig from the library."""

    def test_from_string(self):
        """Test creating MapConfig from string."""
        config = MapConfig.from_config("maps/test.csv", default_max_turns=500)
        assert config.path == "maps/test.csv"
        assert config.max_turns == 500

    def test_from_dict(self):
        """Test creating MapConfig from dictionary."""
        config = MapConfig.from_config(
            {"path": "maps/test.csv", "max_turns": 300},
            default_max_turns=500,
        )
        assert config.path == "maps/test.csv"
        assert config.max_turns == 300

    def test_name_property(self):
        """Test name property."""
        config = MapConfig(path="maps/1v1/test_map.csv", max_turns=500)
        assert config.name == "test_map.csv"

    def test_stem_property(self):
        """Test stem property."""
        config = MapConfig(path="maps/1v1/test_map.csv", max_turns=500)
        assert config.stem == "test_map"

    def test_to_dict(self):
        """Test serialization."""
        config = MapConfig(path="maps/test.csv", max_turns=300)
        data = config.to_dict()
        assert data["path"] == "maps/test.csv"
        assert data["max_turns"] == 300


class TestSchedulingLibrary:
    """Tests for tournament scheduling from the library."""

    def test_round_robin_two_bots(self):
        """Test round-robin with 2 bots."""
        bots = [
            BotDescriptor.simple_bot("Bot1"),
            BotDescriptor.simple_bot("Bot2"),
        ]
        maps = [MapConfig(path="maps/test.csv", max_turns=500)]

        schedule, skipped = generate_round_robin_schedule(
            bots=bots,
            map_configs=maps,
            games_per_side=1,
            map_pool_mode="all",
        )

        assert skipped == 0
        assert len(schedule) == 1  # One round (one map)
        assert len(schedule[0]) == 2  # 2 games (1 per side)

    def test_round_robin_three_bots(self):
        """Test round-robin with 3 bots."""
        bots = [
            BotDescriptor.simple_bot("Bot1"),
            BotDescriptor.simple_bot("Bot2"),
            BotDescriptor.simple_bot("Bot3"),
        ]
        maps = [MapConfig(path="maps/test.csv", max_turns=500)]

        schedule, skipped = generate_round_robin_schedule(
            bots=bots,
            map_configs=maps,
            games_per_side=1,
            map_pool_mode="all",
        )

        # 3 matchups * 2 games per matchup = 6 games
        total_games = sum(len(r) for r in schedule)
        assert total_games == 6

    def test_multiple_maps_all_mode(self):
        """Test with multiple maps in 'all' mode."""
        bots = [
            BotDescriptor.simple_bot("Bot1"),
            BotDescriptor.simple_bot("Bot2"),
        ]
        maps = [
            MapConfig(path="maps/map1.csv", max_turns=500),
            MapConfig(path="maps/map2.csv", max_turns=500),
        ]

        schedule, skipped = generate_round_robin_schedule(
            bots=bots,
            map_configs=maps,
            games_per_side=1,
            map_pool_mode="all",
        )

        # 2 rounds (one per map)
        assert len(schedule) == 2
        # 2 games per round
        assert len(schedule[0]) == 2
        assert len(schedule[1]) == 2

    def test_scheduled_game_properties(self):
        """Test ScheduledGame properties."""
        bot1 = BotDescriptor.simple_bot("Bot1")
        bot2 = BotDescriptor.simple_bot("Bot2")
        map_config = MapConfig(path="maps/1v1/test.csv", max_turns=300)

        game = ScheduledGame(
            game_id=1,
            bot1=bot1,
            bot2=bot2,
            map_config=map_config,
            round_index=0,
            game_index=0,
        )

        assert game.map_name == "test.csv"
        assert game.max_turns == 300


class TestTournamentConfigLibrary:
    """Tests for TournamentConfig from the library."""

    def test_default_config(self):
        """Test default configuration."""
        config = TournamentConfig(maps=["maps/test.csv"])

        assert config.games_per_side == 2
        assert config.max_turns == 500
        assert config.map_pool_mode == "all"
        assert config.save_replays is True

    def test_map_conversion(self):
        """Test that string maps are converted to MapConfig."""
        config = TournamentConfig(maps=["maps/test.csv"])

        assert len(config.maps) == 1
        assert isinstance(config.maps[0], MapConfig)
        assert config.maps[0].path == "maps/test.csv"

    def test_validation_no_maps(self):
        """Test validation fails with no maps."""
        config = TournamentConfig(maps=[])
        errors = config.validate()

        assert len(errors) > 0
        assert any("map" in e.lower() for e in errors)

    def test_from_dict_flat(self):
        """Test creating config from flat dictionary."""
        data = {
            "name": "Test Tournament",
            "maps": ["maps/test.csv"],
            "games_per_side": 3,
            "max_turns": 300,
        }
        config = TournamentConfig.from_dict(data)

        assert config.name == "Test Tournament"
        assert config.games_per_side == 3
        assert config.max_turns == 300

    def test_from_dict_nested(self):
        """Test creating config from nested dictionary (docker format)."""
        data = {
            "tournament": {
                "name": "Docker Tournament",
                "games_per_matchup": 4,
                "max_turns": 200,
            },
            "maps": ["maps/test.csv"],
            "output": {
                "results_dir": "/output/results",
            }
        }
        config = TournamentConfig.from_dict(data)

        assert config.name == "Docker Tournament"
        assert config.games_per_side == 4
        assert config.max_turns == 200

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TournamentConfig(
            name="Test",
            maps=["maps/test.csv"],
            games_per_side=3,
        )
        data = config.to_dict()

        assert data["name"] == "Test"
        assert data["games_per_side"] == 3

    def test_add_map(self):
        """Test adding maps to config."""
        config = TournamentConfig(maps=["maps/map1.csv"])
        config.add_map("maps/map2.csv")

        assert len(config.maps) == 2
        assert config.maps[1].path == "maps/map2.csv"


class TestTournamentResultsLibrary:
    """Tests for TournamentResults from the library."""

    def test_add_game_result(self):
        """Test adding game results."""
        results = TournamentResults()

        result = GameResult(
            game_id=1,
            bot1_name="Bot1",
            bot2_name="Bot2",
            winner=1,
            winner_name="Bot1",
            turns=50,
            map_name="test.csv",
        )
        results.add_game_result(result)

        assert len(results.game_results) == 1
        assert results.bot_stats["Bot1"]["wins"] == 1
        assert results.bot_stats["Bot2"]["losses"] == 1

    def test_get_standings(self):
        """Test getting standings."""
        results = TournamentResults()

        # Add some results
        results.add_game_result(GameResult(
            game_id=1, bot1_name="Bot1", bot2_name="Bot2",
            winner=1, winner_name="Bot1", turns=50, map_name="test.csv"
        ))
        results.add_game_result(GameResult(
            game_id=2, bot1_name="Bot1", bot2_name="Bot2",
            winner=1, winner_name="Bot1", turns=50, map_name="test.csv"
        ))

        standings = results.get_standings()

        # Bot1 should be first (higher Elo from 2 wins)
        assert standings[0].bot_name == "Bot1"
        assert standings[0].wins == 2
        assert standings[1].bot_name == "Bot2"
        assert standings[1].losses == 2

    def test_get_matchups(self):
        """Test getting matchup results."""
        results = TournamentResults()

        results.add_game_result(GameResult(
            game_id=1, bot1_name="Bot1", bot2_name="Bot2",
            winner=1, winner_name="Bot1", turns=50, map_name="test.csv"
        ))
        results.add_game_result(GameResult(
            game_id=2, bot1_name="Bot2", bot2_name="Bot1",
            winner=2, winner_name="Bot1", turns=50, map_name="test.csv"
        ))

        matchups = results.get_matchups()

        assert len(matchups) == 1
        matchup = matchups[0]
        assert matchup.bot1 == "Bot1"
        assert matchup.bot2 == "Bot2"
        assert matchup.bot1_wins == 2
        assert matchup.bot2_wins == 0

    def test_draw_handling(self):
        """Test handling of draws."""
        results = TournamentResults()

        results.add_game_result(GameResult(
            game_id=1, bot1_name="Bot1", bot2_name="Bot2",
            winner=0, winner_name="Draw", turns=100, map_name="test.csv"
        ))

        assert results.bot_stats["Bot1"]["draws"] == 1
        assert results.bot_stats["Bot2"]["draws"] == 1

        matchups = results.get_matchups()
        assert matchups[0].draws == 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        results = TournamentResults()
        results.start()

        results.add_game_result(GameResult(
            game_id=1, bot1_name="Bot1", bot2_name="Bot2",
            winner=1, winner_name="Bot1", turns=50, map_name="test.csv"
        ))

        results.finish()
        data = results.to_dict()

        assert "timestamp" in data
        assert "standings" in data
        assert "matchups" in data
        assert "elo_history" in data
        assert "games" in data

    def test_per_map_stats(self):
        """Test per-map statistics tracking."""
        results = TournamentResults()

        results.add_game_result(GameResult(
            game_id=1, bot1_name="Bot1", bot2_name="Bot2",
            winner=1, winner_name="Bot1", turns=50, map_name="map1.csv"
        ))
        results.add_game_result(GameResult(
            game_id=2, bot1_name="Bot1", bot2_name="Bot2",
            winner=2, winner_name="Bot2", turns=50, map_name="map2.csv"
        ))

        standings = results.get_standings()

        # Bot1 should have per-map stats
        bot1 = next(s for s in standings if s.bot_name == "Bot1")
        assert "map1.csv" in bot1.per_map_stats
        assert bot1.per_map_stats["map1.csv"]["wins"] == 1
        assert "map2.csv" in bot1.per_map_stats
        assert bot1.per_map_stats["map2.csv"]["losses"] == 1


class TestGameResultLibrary:
    """Tests for GameResult from the library."""

    def test_game_result_creation(self):
        """Test GameResult creation."""
        result = GameResult(
            game_id=1,
            bot1_name="Bot1",
            bot2_name="Bot2",
            winner=1,
            winner_name="Bot1",
            turns=50,
            map_name="test.csv",
            replay_path="/path/to/replay.json",
        )

        assert result.game_id == 1
        assert result.winner == 1
        assert result.replay_path == "/path/to/replay.json"

    def test_game_result_to_dict(self):
        """Test GameResult serialization."""
        result = GameResult(
            game_id=1,
            bot1_name="Bot1",
            bot2_name="Bot2",
            winner=1,
            winner_name="Bot1",
            turns=50,
            map_name="test.csv",
        )

        data = result.to_dict()
        assert data["game_id"] == 1
        assert data["bot1"] == "Bot1"
        assert data["winner"] == 1

    def test_game_result_with_error(self):
        """Test GameResult with error."""
        result = GameResult(
            game_id=1,
            bot1_name="Bot1",
            bot2_name="Bot2",
            winner=0,
            winner_name="Error",
            turns=10,
            map_name="test.csv",
            error="Game crashed",
        )

        assert result.error == "Game crashed"
        data = result.to_dict()
        assert "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
