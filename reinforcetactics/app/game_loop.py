"""
Game Loop and Session Management for Reinforce Tactics.

This module manages the main game loop, game session, and game modes.
"""
# pylint: disable=cyclic-import

import logging
from datetime import datetime

import pygame

from reinforcetactics.app.bot_factory import create_bots_from_config
from reinforcetactics.app.input_handler import InputHandler
from reinforcetactics.core.game_state import GameState
from reinforcetactics.ui import theme
from reinforcetactics.ui.menus import (
    GameOverMenu,
    LoadGameMenu,
    MapSelectionMenu,
    PauseMenu,
    ReplaySelectionMenu,
    SaveGameMenu,
)
from reinforcetactics.ui.menus.game_setup.modes import GAME_MODE_PLAYER_COUNTS, default_teams_for_mode
from reinforcetactics.ui.renderer import Renderer
from reinforcetactics.ui.widgets.text import ellipsize
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.utils.replay_player import ReplayPlayer
from reinforcetactics.utils.settings import get_settings

logger = logging.getLogger(__name__)


class GameSession:  # pylint: disable=too-few-public-methods
    """
    Manages a game session including initialization and game loop.

    Attributes:
        game: The GameState instance
        renderer: The Renderer instance
        bots: Dictionary mapping player numbers to bot instances
        input_handler: InputHandler instance
        clock: pygame.Clock for frame timing
        running: Whether the game loop is running
    """

    def __init__(self, game, renderer, bots, num_players):
        """
        Initialize a GameSession.

        Args:
            game: The GameState instance
            renderer: The Renderer instance
            bots: Dictionary mapping player numbers to bot instances
            num_players: Total number of players
        """
        self.game = game
        self.renderer = renderer
        self.bots = bots
        self.input_handler = InputHandler(game, renderer, bots, num_players)
        self.clock = pygame.time.Clock()
        self.running = True
        # (turn_number, current_player) at which a bot neither ended its turn
        # nor raised; bots aren't re-run until the state moves on.
        self._stalled_bot_state = None

        # Every move walks its unit's sprite along its path, and a bot's
        # move plays out before the bot goes on (see _on_unit_moved).
        game.move_listeners.append(self._on_unit_moved)
        # Set when the player pauses, saves or closes the window during a
        # bot's walk: the bot moves still to come this frame aren't waited for.
        self._skip_bot_walks = False
        # The human seat whose fog of war the board shows while a bot moves
        # (see _update_view); None when every seat is a bot.
        self._last_human_seat = min((p for p in range(1, num_players + 1) if p not in bots), default=None)

    def run(self):
        """
        Run the main game loop.

        Returns:
            'new_game', 'main_menu', or 'quit' based on game over menu selection
        """
        # Track why the loop exited (for mid-game exits)
        self._exit_reason = "quit"

        print("\n🎮 Game started!")
        print("Controls:")
        print("  - Click units to select")
        print("  - Click buildings to create units")
        print("  - Click tiles to move")
        print("  - Right-click and hold on a unit to preview attack range")
        print("  - Press SPACE to end turn")
        print("  - Press S to save game")
        print("  - Press ESC to open pause menu")
        print()

        while self.running and not self.game.game_over:
            self._skip_bot_walks = False

            # Bots also move without a human ending a turn first: a bot in seat
            # 1, a save loaded on a bot's turn, or an all-bot game. Bot turns
            # used to run only from the End Turn handlers, so these games sat
            # at turn 0 and the human could play the bot's units (pygame-5).
            # One bot turn per frame, then this frame's events, drawing and
            # clock tick as usual: running bot turns instead of the frame left
            # a game with only bots to move (all-bot, or after the human
            # resigned or was eliminated) unable to be paused or closed.
            if self._bot_should_act():
                self._render_frame()  # show the board before a possibly slow bot turn
                self._run_pending_bot_turns()
                if self.game.game_over:
                    break

            # Get mouse position once per frame
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Window close -> open pause menu for save/quit options
                    pause_result = self._handle_pause()
                    if pause_result:
                        return pause_result

                elif self.game.current_player in self.bots:
                    # A bot's seat is to move: the board is not the human's
                    # to play. Only pausing and saving answer.
                    if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_s):
                        result = self.input_handler.handle_keyboard_event(event)
                        if result == "pause":
                            pause_result = self._handle_pause()
                            if pause_result:
                                return pause_result
                        elif result == "save":
                            self._handle_save_game()

                elif event.type == pygame.KEYDOWN:
                    result = self.input_handler.handle_keyboard_event(event)
                    if result == "pause":
                        pause_result = self._handle_pause()
                        if pause_result:
                            return pause_result
                    elif result == "save":
                        self._handle_save_game()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        result = self.input_handler.handle_mouse_click(mouse_pos)
                        if result == "continue":
                            continue
                    elif event.button == 3:  # Right click
                        self.input_handler.handle_right_click_press(mouse_pos)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3:  # Right click release
                        self.input_handler.handle_right_click_release()

                elif event.type == pygame.MOUSEMOTION:
                    self.input_handler.handle_mouse_motion(mouse_pos)

            self.input_handler.update(pygame.time.get_ticks())

            # Rendering
            self._render_frame()

            # Frame timing
            self.clock.tick(60)

        # Handle game over
        if self.game.game_over:
            return self._handle_game_over()

        # Auto-save replay on mid-game quit
        if self.game.action_history:
            replay_path = self.game.save_replay_to_file()
            if replay_path:
                print(f"Replay saved to {replay_path}")

        return self._exit_reason

    def _bot_should_act(self):
        """Whether the seat to move is a bot that hasn't stalled at this state."""
        if self.game.game_over or self.game.current_player not in self.bots:
            return False
        return (self.game.turn_number, self.game.current_player) != self._stalled_bot_state

    def _run_pending_bot_turns(self):
        """Play one bot turn; the frame loop calls this once per frame while a bot is to move.

        A bot that returns without ending its turn (and without raising,
        which the input handler already contains) has its turn ended for it,
        as a crashed bot's is, and the human is told. Its seat used to be
        left current and handed to the human's input, who could then move
        the bot's units, spend its gold and end its turn. Only if even ending
        the turn fails is the state marked stalled, so the bot is not re-run
        every frame.
        """
        before = (self.game.turn_number, self.game.current_player)
        self.input_handler._process_bot_turns(max_turns=1)
        if not self.game.game_over and (self.game.turn_number, self.game.current_player) == before:
            self.input_handler.show_notice(f"Player {before[1]}'s bot did not finish its turn, so it was ended")
            if not self.input_handler._end_crashed_bot_turn(before[1]):
                self._stalled_bot_state = before

    def _on_unit_moved(self, unit, path):
        """Walk a moved unit's sprite along ``path`` (a ``GameState.move_listeners`` callback).

        A human's move only starts the walk (the input handler waits for it
        before opening the action menu). A bot's move is played out here,
        before the bot's next action, so its turn can be watched one move at
        a time; a move the viewer can't see any of (under fog of war) isn't
        waited for.
        """
        self._update_view()
        shown = self.renderer.queue_movement_path_animation(unit, path)
        if shown and self.game.current_player in self.bots and not self._skip_bot_walks:
            self._play_out_walk(unit)

    def _play_out_walk(self, unit):
        """Render frames until ``unit``'s sprite has walked its move.

        Runs inside a bot's turn, so of the input that arrives meanwhile
        only a pause, a save or a window close is kept, re-posted for the
        main loop. Any of them also ends the waiting, so it isn't held up by
        the rest of the bot's moves. Clicks and other keys are dropped: the
        board isn't the human's while a bot moves.
        """
        held = []
        while self.renderer.is_unit_moving(unit) and not held:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_s)):
                    held.append(event)
            self._render_frame()
            self.clock.tick(60)
        if held:
            self._skip_bot_walks = True
            for event in held:
                pygame.event.post(event)

    def _update_view(self):
        """Show the board from the human side of the fog of war while a bot moves.

        The renderer draws the fog of the player to move, which made a bot's
        turn (now played out on screen) show everything the bot sees. While a
        bot's seat is to move, the board keeps the view of the last human
        seat to move instead; with no human seat left in the game, each
        bot's own view.
        """
        if self.game.current_player in self.bots:
            human = self._last_human_seat
            self.renderer.viewing_player = None if human is None or self.game.is_eliminated(human) else human
        else:
            self._last_human_seat = self.game.current_player
            self.renderer.viewing_player = None

    def _handle_pause(self):
        """
        Show pause menu and handle the result.

        Returns:
            'main_menu' or 'quit' if the game session should end, None to resume.
        """
        pause_menu = PauseMenu(self.renderer.screen, self.game)
        result = pause_menu.run()
        pygame.event.clear()

        if result == "resume":
            return None
        elif result == "save_quit":
            self._handle_save_game()
            self.running = False
            self._exit_reason = "quit"
            return "quit"
        elif result == "quit":
            self.running = False
            self._exit_reason = "quit"
            return "quit"
        elif result == "main_menu":
            self.running = False
            self._exit_reason = "main_menu"
            return "main_menu"

        return None

    def _handle_save_game(self):
        """Handle save game request."""
        save_menu = SaveGameMenu(self.game)
        result = save_menu.run()
        if result:
            print(f"✅ Game saved to {result}")

    def _render_frame(self):
        """Render a single frame."""
        self._update_view()
        self.renderer.render()

        # Draw movement overlay and pulsing highlight if unit selected
        if self.input_handler.selected_unit:
            self.renderer.draw_movement_overlay(self.input_handler.selected_unit)
            self.renderer.draw_selected_unit_highlight(self.input_handler.selected_unit)

        # Draw attack range preview if right-clicking on a unit
        if self.input_handler.right_click_preview_active and self.input_handler.preview_positions:
            self.renderer.draw_attack_range_overlay(self.input_handler.preview_positions)

        # Draw target overlay if in target selection mode
        if self.input_handler.target_selection_mode and self.input_handler.target_selection_action:
            self.renderer.draw_target_overlay(self.input_handler.target_selection_action["targets"])

        # Draw unit tooltip when hovering (only if no menu is open)
        if not self.input_handler.active_menu:
            mouse_pos = pygame.mouse.get_pos()
            self.renderer.draw_unit_tooltip(mouse_pos)

        # Draw active menu last (on top)
        if self.input_handler.active_menu:
            self.input_handler.active_menu.draw(self.renderer.screen)

        self._draw_notice()

        pygame.display.flip()

    def _draw_notice(self):
        """Draw the input handler's transient notice, if one is showing."""
        handler = self.input_handler
        if not handler.notice_text:
            return
        if pygame.time.get_ticks() >= handler.notice_expires_at:
            handler.notice_text = None
            return

        screen = self.renderer.screen
        font = get_font(theme.FONT_SIZE_BODY)
        padding = 12
        text = ellipsize(handler.notice_text, font, max(1, screen.get_width() - 4 * padding))
        text_surface = font.render(text, True, theme.STATUS_WARNING)
        box = text_surface.get_rect().inflate(2 * padding, 2 * padding)
        box.midtop = (screen.get_width() // 2, padding)
        backdrop = pygame.Surface(box.size, pygame.SRCALPHA)
        backdrop.fill((0, 0, 0, 200))
        screen.blit(backdrop, box)
        screen.blit(text_surface, text_surface.get_rect(center=box.center))

    def _handle_game_over(self):
        """
        Handle game over state.

        Returns:
            'new_game', 'main_menu', or 'quit'
        """
        print(f"\n🎉 Game Over! {_winner_text(self.game)}")

        # Automatically save replay
        replay_path = self.game.save_replay_to_file()
        if replay_path:
            print(f"📼 Replay saved to {replay_path}")

        # Show game over screen
        game_over_menu = GameOverMenu(self.game.winner, self.game, self.renderer.screen)
        result = game_over_menu.run()

        return result if result else "quit"


def _winner_text(game):
    """Console line naming who won (the whole team in a team game)."""
    if game.winner is None:
        return "The game is a draw."
    team = game.team_of(game.winner)
    players = [p for p in sorted(game.teams) if game.teams[p] == team]
    if len(players) > 1:
        return f"Team {team} (players {', '.join(str(p) for p in players)}) wins!"
    return f"Player {game.winner} wins!"


def _save_crash_artifacts(game):
    """
    Best-effort autosave of a game whose session crashed.

    Writes the replay (as a normal mid-game quit does) and, for an unfinished
    game, a crash save the player can resume from Load Game. Each write is
    attempted on its own and a failure is only logged: this runs while
    another exception is being handled, and must not replace it.

    Args:
        game: The GameState of the crashed session

    Returns:
        List of the file paths that were written
    """
    written = []
    if game.action_history:
        try:
            replay_path = game.save_replay_to_file()
        except Exception:
            logger.exception("Could not save the replay of the crashed game")
        else:
            if replay_path:
                print(f"📼 Replay saved to {replay_path}")
                written.append(replay_path)

    if not game.game_over:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            save_path = game.save_to_file(f"saves/crash_{timestamp}.json")
        except Exception:
            logger.exception("Could not write a crash save")
        else:
            if save_path:
                print(f"💾 Crash save written to {save_path} (open it from Load Game)")
                written.append(save_path)

    return written


def _run_session(session):
    """
    Run a game session, containing any exception that escapes it.

    An exception in the game loop used to unwind into start_new_game's or
    load_saved_game's blanket handler, which printed it and returned None:
    the replay was never written and the game in progress was lost. Now the
    replay and a crash save are written first, and the player goes back to
    the main menu. KeyboardInterrupt and SystemExit are not caught.

    Args:
        session: The GameSession to run

    Returns:
        The session's result, or 'main_menu' after a crash
    """
    try:
        return session.run()
    except Exception:
        logger.exception("Game session crashed")
        print("❌ The game hit an unexpected error. Saving what we can and returning to the main menu.")
        _save_crash_artifacts(session.game)
        return "main_menu"


def start_new_game(mode="human_vs_computer", selected_map=None, player_configs=None, fog_of_war=False, num_players=None):
    """
    Start a new game with the specified mode, map, and player configurations.

    Args:
        mode: Game mode string
        selected_map: Map file path or 'random'
        player_configs: List of player configuration dictionaries
        fog_of_war: Whether to enable fog of war
        num_players: Number of player seats. The New Game flow passes the
            seat count of the chosen mode. When omitted it is the number of
            player_configs, else the seat count of ``mode``, else 2.

    Returns:
        'new_game', 'main_menu' or 'quit' for the caller's navigation, or
        None if map selection was cancelled

    Raises:
        ValueError: If player_configs does not have one entry per seat
    """
    print(f"\n🎮 Starting new game: {mode}\n")

    # Use provided map or show map selection
    if selected_map is None:
        map_menu = MapSelectionMenu()
        selected_map = map_menu.run()

    if not selected_map:
        print("Map selection cancelled")
        return

    # Determine number of players. The caller now passes the seat count of
    # the chosen mode; it used to be guessed here from the mode name (only
    # "2v2" was special-cased). A config list of a different length is a
    # caller bug, so fail loudly rather than seat the wrong number of players.
    if num_players is None:
        num_players = len(player_configs) if player_configs else GAME_MODE_PLAYER_COUNTS.get(mode, 2)
    if player_configs and len(player_configs) != num_players:
        raise ValueError(f"Got {len(player_configs)} player configs for a {num_players}-player game")

    try:
        # Load or generate map
        if selected_map == "random":
            print("Generating random map...")
            map_data = FileIO.generate_random_map(20, 20, num_players=num_players)
            map_file_used = None
        else:
            print(f"Loading map: {selected_map}")
            map_data = FileIO.load_map(selected_map, for_ui=True, border_size=2)
            map_file_used = selected_map

        if map_data is None:
            print("Failed to load map")
            return "main_menu"

        # Get settings for enabled units
        settings = get_settings()
        enabled_units = settings.get_enabled_units()

        # Teams (review core-4): a map declares its own on its HQ codes, and
        # GameState derives them; for a map that declares none (e.g. a random
        # map) the mode's default applies, so 2v2 is always played in teams.
        teams = default_teams_for_mode(mode)
        if teams and GameState.map_team_declarations(map_data, num_players):
            teams = None

        # Create game state with enabled units from settings
        game = GameState(map_data, num_players=num_players, enabled_units=enabled_units, fog_of_war=fog_of_war, teams=teams)

        # GameState computes fog-of-war visibility itself
        if fog_of_war:
            print("Fog of war enabled!")

        # Store map file for saving
        if map_file_used:
            game.map_file_used = map_file_used

        # Set player configurations
        if player_configs:
            game.player_configs = player_configs
        else:
            # Generate default player configs
            game.player_configs = []
            for i in range(num_players):
                if i == 0:
                    game.player_configs.append({"type": "human", "bot_type": None})
                else:
                    game.player_configs.append({"type": "computer", "bot_type": "SimpleBot"})

        # Create renderer
        renderer = Renderer(game)
        bot_notices = []
        bots = create_bots_from_config(game, game.player_configs, settings, notices=bot_notices)

        # Legacy mode: Ensure bot for player 2 in human_vs_computer
        if mode == "human_vs_computer" and 2 not in bots:
            from reinforcetactics.game.bot import SimpleBot

            bots[2] = SimpleBot(game, player=2)
            print("Bot created for Player 2")

        # Create and run game session
        session = GameSession(game, renderer, bots, num_players)
        if bot_notices:
            session.input_handler.show_notice("; ".join(bot_notices))

        # Return result to let caller handle navigation
        return _run_session(session)

    except Exception as e:
        # Setup failed before any turn was played, so there is nothing to
        # autosave; the session itself is guarded by _run_session.
        logger.exception("Could not start the game")
        print(f"❌ Could not start the game: {e}")
        return "main_menu"

    finally:
        # Restore the display: the renderer sized the window to the map, and
        # play_mode re-initialises pygame for the main menu.
        pygame.quit()


def _map_data_for_save(save_data):
    """
    Rebuild the map a save's unit and structure coordinates refer to.

    The terrain a save records (``GameState.to_dict`` writes ``map_data``) is
    used first: it is exactly the grid the game was played on. Older saves
    only name their map file, which is reloaded with the UI padding
    start_new_game applies. The load check used to be ``"map_file" in
    save_data``, which is always true (random-map saves write ``null``), so
    every random-map save failed to load.

    Args:
        save_data: Parsed save dictionary

    Returns:
        The map data, or None if the save records neither terrain nor a map
        file (a random-map save written before the terrain was recorded).
        A freshly generated random map would put the saved units on
        different terrain, so there is no such fallback.
    """
    map_data = GameState.saved_map_data(save_data)
    if map_data is not None:
        return map_data

    map_file = save_data.get("map_file")
    if map_file:
        return FileIO.load_map(map_file, for_ui=True, border_size=2)

    return None


def restore_saved_game(save_data):
    """
    Rebuild the GameState of a parsed save, the way Load Game does.

    Args:
        save_data: Parsed save dictionary (as returned by LoadGameMenu)

    Returns:
        The restored GameState, or None if the save's map can't be rebuilt
    """
    map_data = _map_data_for_save(save_data)
    if map_data is None:
        return None
    return GameState.from_dict(save_data, map_data)


def load_saved_game(save_data=None):
    """
    Load and play a saved game.

    Args:
        save_data: Parsed save dictionary, as returned by LoadGameMenu. When
            None, the load menu is shown first.

    Returns:
        'new_game', 'main_menu' or 'quit' for the caller's navigation, or
        None if loading was cancelled
    """
    print("\n💾 Loading saved game...\n")

    # Show load menu unless the caller (the main menu) already picked a save
    if save_data is None:
        load_menu = LoadGameMenu()
        save_data = load_menu.run()

    if not save_data:
        print("Load cancelled")
        return

    try:
        # Restore game state
        game = restore_saved_game(save_data)
        if game is None:
            if save_data.get("map_file"):
                print(f"❌ This save can't be loaded: its map file {save_data['map_file']} could not be read.")
            else:
                print("❌ This save can't be loaded: it was made on a random map before saves recorded their terrain.")
            return "main_menu"

        # from_dict restores the saved fog of war (or rebuilds it for an old save)
        if game.fog_of_war:
            print("Fog of war enabled!")

        # Create renderer
        renderer = Renderer(game)

        # If game is already over, show game over screen directly
        if game.game_over:
            print(f"Loaded a completed game. Winner: Player {game.winner}")
            game_over_menu = GameOverMenu(game.winner, game, renderer.screen)
            result = game_over_menu.run()
            return result if result else "quit"

        # Create bots
        settings = get_settings()
        bots = {}
        bot_notices = []
        if game.player_configs:
            bots = create_bots_from_config(game, game.player_configs, settings, notices=bot_notices)
        else:
            # Fallback for old saves
            from reinforcetactics.game.bot import SimpleBot

            for player_num in range(2, game.num_players + 1):
                bots[player_num] = SimpleBot(game, player=player_num)
                print(f"Bot created for Player {player_num} (loaded game - legacy)")

        print(f"\n✅ Game loaded! Turn {game.turn_number}, Player {game.current_player}'s turn")
        print("\nControls:")
        print("  - Click units to select")
        print("  - Click tiles to move")
        print("  - Right-click and hold on a unit to preview attack range")
        print("  - Press SPACE to end turn")
        print("  - Press S to save game")
        print("  - Press ESC to open pause menu")
        print()

        # Create and run game session
        session = GameSession(game, renderer, bots, game.num_players)
        if bot_notices:
            session.input_handler.show_notice("; ".join(bot_notices))

        # Return result to let caller handle navigation
        return _run_session(session)

    except Exception as e:
        logger.exception("Could not load the saved game")
        print(f"❌ Error loading game: {e}")
        return "main_menu"

    finally:
        # Restore the display for the main menu (see start_new_game).
        pygame.quit()


def watch_replay(replay_path=None):
    """
    Watch a replay.

    Args:
        replay_path: Path to replay file. If None, shows replay selection menu.

    Returns:
        'main_menu' once the replay closes, or None if selection was cancelled
    """
    print("\n📼 Loading replay...\n")

    # Show replay selection menu if path not provided
    if not replay_path:
        replay_menu = ReplaySelectionMenu()
        replay_path = replay_menu.run()

    if not replay_path:
        print("Replay selection cancelled")
        return

    try:
        # Load replay data
        replay_data = FileIO.load_replay(replay_path)

        if not replay_data:
            print("Failed to load replay")
            return "main_menu"

        # Build the starting map from the replay itself: its recorded
        # initial_map (random-map games included), else its map file. This
        # used to fall back to a freshly generated random map, which replayed
        # the actions on the wrong terrain.
        game_info = replay_data.get("game_info", {})
        map_data = FileIO.load_replay_map(game_info)

        # Create and run replay player
        player = ReplayPlayer(replay_data, map_data)
        player.run()

        return "main_menu"  # Return to main menu after watching replay

    except Exception as e:
        logger.exception("Could not play the replay")
        print(f"❌ Error playing replay: {e}")
        return "main_menu"

    finally:
        # Restore the display for the main menu (see start_new_game).
        pygame.quit()
