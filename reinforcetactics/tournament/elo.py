"""
Elo Rating System for tournament rankings.

This module provides a standard Elo rating implementation for tracking
bot performance across tournament games.
"""

import json
from typing import Dict, List, Optional


class EloRatingSystem:
    """
    Manages Elo ratings for tournament participants.

    The Elo rating system is a method for calculating the relative skill levels
    of players in zero-sum games. After each game, players exchange rating points
    based on the expected vs actual outcome.

    Attributes:
        starting_elo: Initial rating for new players (default: 1500)
        k_factor: Maximum rating change per game (default: 32)
        ratings: Current ratings for all players
        initial_ratings: Ratings at initialization (for tracking changes)
        rating_history: Complete rating history for each player
    """

    def __init__(self, starting_elo: int = 1500, k_factor: int = 32):
        """
        Initialize Elo rating system.

        Args:
            starting_elo: Initial Elo rating for all bots (default: 1500)
            k_factor: K-factor for rating changes (default: 32)
                     Higher values = more volatile ratings
        """
        self.starting_elo = starting_elo
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {}
        self.initial_ratings: Dict[str, float] = {}
        self.rating_history: Dict[str, List[float]] = {}

    def initialize_bot(self, bot_name: str) -> None:
        """
        Initialize a bot with starting Elo rating.

        If the bot already exists, this is a no-op.

        Args:
            bot_name: Name of the bot
        """
        if bot_name not in self.ratings:
            self.ratings[bot_name] = float(self.starting_elo)
            self.initial_ratings[bot_name] = float(self.starting_elo)
            self.rating_history[bot_name] = [float(self.starting_elo)]

    def calculate_expected_score(
        self, player_elo: float, opponent_elo: float
    ) -> float:
        """
        Calculate expected score for a player against an opponent.

        Uses the standard Elo formula:
        E = 1 / (1 + 10^((opponent_elo - player_elo) / 400))

        Args:
            player_elo: Player's current Elo rating
            opponent_elo: Opponent's current Elo rating

        Returns:
            Expected score between 0.0 and 1.0
            - 0.5 means equal expected performance
            - Higher values indicate player is favored
        """
        return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400.0))

    def update_ratings(
        self, bot1_name: str, bot2_name: str, result: int
    ) -> tuple[float, float]:
        """
        Update Elo ratings after a game.

        Args:
            bot1_name: Name of first bot
            bot2_name: Name of second bot
            result: Game result
                - 1 = bot1 wins
                - 2 = bot2 wins
                - 0 = draw

        Returns:
            Tuple of (bot1_rating_change, bot2_rating_change)
        """
        # Initialize bots if needed
        self.initialize_bot(bot1_name)
        self.initialize_bot(bot2_name)

        # Get current ratings
        bot1_elo = self.ratings[bot1_name]
        bot2_elo = self.ratings[bot2_name]

        # Calculate expected scores
        bot1_expected = self.calculate_expected_score(bot1_elo, bot2_elo)
        bot2_expected = self.calculate_expected_score(bot2_elo, bot1_elo)

        # Determine actual scores
        if result == 1:  # bot1 wins
            bot1_actual, bot2_actual = 1.0, 0.0
        elif result == 2:  # bot2 wins
            bot1_actual, bot2_actual = 0.0, 1.0
        else:  # draw
            bot1_actual, bot2_actual = 0.5, 0.5

        # Calculate new ratings
        bot1_change = self.k_factor * (bot1_actual - bot1_expected)
        bot2_change = self.k_factor * (bot2_actual - bot2_expected)

        bot1_new = bot1_elo + bot1_change
        bot2_new = bot2_elo + bot2_change

        # Update ratings
        self.ratings[bot1_name] = bot1_new
        self.ratings[bot2_name] = bot2_new

        # Record history
        self.rating_history[bot1_name].append(bot1_new)
        self.rating_history[bot2_name].append(bot2_new)

        return bot1_change, bot2_change

    def get_rating(self, bot_name: str) -> float:
        """
        Get current Elo rating for a bot.

        Args:
            bot_name: Name of the bot

        Returns:
            Current Elo rating, or starting_elo if bot not found
        """
        return self.ratings.get(bot_name, float(self.starting_elo))

    def get_rating_change(self, bot_name: str) -> float:
        """
        Get Elo rating change since initialization.

        Args:
            bot_name: Name of the bot

        Returns:
            Rating change (positive or negative)
        """
        initial = self.initial_ratings.get(bot_name, float(self.starting_elo))
        current = self.ratings.get(bot_name, float(self.starting_elo))
        return current - initial

    def get_rankings(self) -> List[tuple[str, float]]:
        """
        Get all bots ranked by Elo rating.

        Returns:
            List of (bot_name, rating) tuples, sorted by rating descending
        """
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def save_ratings(self, filepath: str) -> None:
        """
        Save ratings to a JSON file.

        Args:
            filepath: Path to save ratings
        """
        data = {
            "ratings": self.ratings,
            "rating_history": self.rating_history,
            "starting_elo": self.starting_elo,
            "k_factor": self.k_factor,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_ratings(self, filepath: str) -> None:
        """
        Load ratings from a JSON file.

        When loading, initial_ratings is set to current ratings so that
        rating changes track from this load point forward.

        Args:
            filepath: Path to load ratings from
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.ratings = data["ratings"]
        self.rating_history = data["rating_history"]
        self.starting_elo = data.get("starting_elo", 1500)
        self.k_factor = data.get("k_factor", 32)
        # Set initial ratings to current ratings when loading
        self.initial_ratings = self.ratings.copy()

    def to_dict(self) -> Dict:
        """
        Convert rating system state to dictionary.

        Returns:
            Dictionary with all rating data
        """
        return {
            "ratings": self.ratings.copy(),
            "initial_ratings": self.initial_ratings.copy(),
            "rating_history": {
                name: history.copy()
                for name, history in self.rating_history.items()
            },
            "starting_elo": self.starting_elo,
            "k_factor": self.k_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EloRatingSystem":
        """
        Create EloRatingSystem from dictionary.

        Args:
            data: Dictionary with rating data

        Returns:
            EloRatingSystem instance
        """
        system = cls(
            starting_elo=data.get("starting_elo", 1500),
            k_factor=data.get("k_factor", 32),
        )
        system.ratings = data.get("ratings", {})
        system.initial_ratings = data.get("initial_ratings", {})
        system.rating_history = data.get("rating_history", {})
        return system

    def merge_from(
        self, other: "EloRatingSystem", strategy: str = "latest"
    ) -> None:
        """
        Merge ratings from another EloRatingSystem.

        Useful for combining results from resumed tournaments.

        Args:
            other: Another EloRatingSystem to merge from
            strategy: How to handle conflicts
                - "latest": Use the other system's values
                - "average": Average the ratings
        """
        for bot_name, rating in other.ratings.items():
            if bot_name in self.ratings:
                if strategy == "average":
                    self.ratings[bot_name] = (
                        self.ratings[bot_name] + rating
                    ) / 2
                else:  # latest
                    self.ratings[bot_name] = rating
            else:
                self.ratings[bot_name] = rating
                self.initial_ratings[bot_name] = other.initial_ratings.get(
                    bot_name, float(self.starting_elo)
                )

            # Extend history
            if bot_name in other.rating_history:
                if bot_name not in self.rating_history:
                    self.rating_history[bot_name] = []
                self.rating_history[bot_name].extend(other.rating_history[bot_name])
