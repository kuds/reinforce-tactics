"""The game modes the New Game flow can start, and how many seats each has.

Every screen that reasons about modes reads this one table: the mode picker
offers only the modes listed here, the player-config screen builds one row
per seat, the game loop sizes the ``GameState`` from it, and the map editor
files a new map under the folder for its player count. Before this table
existed each of those sites hard-coded ``"1v1"``/``"2v2"``, so the bundled
``maps/1v1v1`` folder was offered by the mode picker and then rejected by the
player-config screen with an uncaught ``ValueError`` that closed the app.
"""

# Mode name (also the ``maps/<mode>/`` folder name) -> number of player seats.
GAME_MODE_PLAYER_COUNTS: dict[str, int] = {
    "1v1": 2,
    "1v1v1": 3,
    "2v2": 4,
}

# Teams a mode plays with when its map declares none (a random map, or a
# hand-made map without ``type_player_team`` HQ codes). Seats alternate
# teams so turn order alternates too: players 1 and 3 against 2 and 4, as
# the bundled 2v2 maps declare. Modes not listed are free-for-all.
GAME_MODE_DEFAULT_TEAMS: dict[str, dict[int, int]] = {
    "2v2": {1: 1, 2: 2, 3: 1, 4: 2},
}


def players_for_mode(game_mode: str) -> int:
    """Return the number of player seats for ``game_mode``.

    Raises:
        ValueError: If ``game_mode`` is not a supported mode.
    """
    try:
        return GAME_MODE_PLAYER_COUNTS[game_mode]
    except KeyError:
        supported = ", ".join(f"'{mode}'" for mode in GAME_MODE_PLAYER_COUNTS)
        raise ValueError(f"Invalid game_mode: {game_mode}. Must be one of {supported}") from None


def mode_for_player_count(num_players: int) -> str | None:
    """Return the mode (and map folder) for ``num_players`` seats, if any."""
    for mode, count in GAME_MODE_PLAYER_COUNTS.items():
        if count == num_players:
            return mode
    return None


def default_teams_for_mode(game_mode: str) -> dict[int, int] | None:
    """The ``{player: team}`` map for ``game_mode`` when its map declares no teams, if any."""
    teams = GAME_MODE_DEFAULT_TEAMS.get(game_mode)
    return dict(teams) if teams else None
