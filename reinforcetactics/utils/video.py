"""
Headless video recording utilities for Jupyter notebooks and CI.

Provides functions to record game replays and agent evaluations to MP4 video
without requiring a display server. Works on Google Colab and headless Linux.
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from reinforcetactics.utils.replay_actions import (
    execute_replay_action as _execute_replay_action,
)
from reinforcetactics.utils.replay_actions import (
    get_schema_version,
)

logger = logging.getLogger(__name__)


def _ensure_headless_pygame():
    """Ensure pygame is initialised with a dummy video driver for headless use."""
    if not os.environ.get("SDL_VIDEODRIVER"):
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    import pygame

    if not pygame.get_init():
        pygame.init()
    return pygame


def record_game_to_video(
    game_states: list[Any],
    output_path: str = "game_replay.mp4",
    fps: int = 4,
    map_file: str | None = None,
    scale: int | None = None,
    use_pixel_art: bool | None = False,
    show_hud: bool = True,
    progress_callback: Any = None,
) -> str:
    """
    Record a sequence of game state snapshots to an MP4 video.

    Args:
        game_states: List of game state snapshot dicts (from GameState.to_dict())
        output_path: Path for the output video file
        fps: Frames per second for the output video
        map_file: Path to the map CSV file, used as the map fallback for
               snapshots that don't embed their own ``map_data``
        scale: Integer upscale factor (nearest-neighbour). ``None`` (default)
               auto-picks the smallest factor that makes the short side at
               least ~480 px so small boards stay readable.
        use_pixel_art: True/False force pixel-art sprites on/off; ``None``
               follows ``settings.json``.
        show_hud: Stamp a compact turn/player badge onto each frame.
        progress_callback: Optional ``callback(done, total)`` invoked after
               each encoded snapshot.

    Returns:
        Path to the saved video file
    """
    _ensure_headless_pygame()
    from reinforcetactics.core.game_state import GameState
    from reinforcetactics.ui.renderer import Renderer

    if not game_states:
        raise ValueError("No game states to record")

    map_df = None
    if map_file:
        from reinforcetactics.utils.file_io import FileIO

        map_df = FileIO.load_map(map_file)

    def _frames():
        # One renderer reused across all frames: constructing a Renderer
        # reloads every tile/unit sprite from disk, which used to happen
        # once per snapshot.
        renderer = None
        total = len(game_states)
        for i, state_dict in enumerate(game_states):
            gs = GameState.from_dict(state_dict, state_dict.get("map_data") if "map_data" in state_dict else map_df)
            if renderer is None:
                renderer = Renderer(gs, replay_mode=True, headless=True, pixel_art=use_pixel_art)
            else:
                renderer.game_state = gs
            renderer.render()
            if show_hud:
                _draw_video_hud(renderer.screen, gs)
            yield renderer.get_rgb_array()
            if progress_callback is not None:
                progress_callback(i + 1, total)

    return _write_frames_to_video(_frames(), output_path, fps, scale=scale)


def record_evaluation_to_video(
    env,
    model,
    output_path: str = "agent_replay.mp4",
    fps: int = 4,
    max_steps: int = 500,
    deterministic: bool = True,
    scale: int | None = None,
    use_pixel_art: bool | None = False,
    show_hud: bool = True,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """
    Run one episode of a trained model and record it to video.

    Creates a headless renderer, runs the model through one episode,
    captures a frame after each action, and writes to MP4.

    Supports both standard SB3 models (PPO, A2C, DQN) and MaskablePPO
    from sb3-contrib. When the environment exposes ``action_masks()``,
    masks are automatically forwarded to ``model.predict()``.

    Args:
        env: A StrategyGameEnv or ActionMaskedEnv instance (unwrapped,
             not vectorised).
        model: A trained Stable-Baselines3 model (or any model with .predict())
        output_path: Path for the output video file
        fps: Frames per second for the output video
        max_steps: Maximum steps before stopping
        deterministic: Whether to use deterministic actions
        scale: Integer upscale factor applied with nearest-neighbour
               interpolation before encoding. The native render is
               TILE_SIZE px per tile (e.g. 192x192 on a 6x6 map).
               ``None`` (default) auto-picks the smallest factor that
               makes the short side at least ~480 px so small boards
               stay readable; pass 1 to force the native resolution.
        use_pixel_art: If True, render with the bundled pixel-art tile
               and unit sprites from ``assets/sprites/`` instead of the
               default fallback (coloured rects + unit letters). Falls
               back silently if the assets directory can't be located.
               ``None`` follows ``settings.json``, matching the
               interactive renderer.
        show_hud: Stamp a compact turn/player badge onto each frame so
               viewers can follow whose move it is (headless frames
               otherwise carry no HUD at all).
        progress_callback: Optional ``callback(done, total)`` invoked
               after each recorded step for long-render progress
               reporting (total is ``max_steps``).

    Returns:
        Dict with keys:
            - video_path: Path to the saved video
            - winner: Winner of the game (1, 2, or None)
            - total_reward: Total episode reward
            - steps: Number of steps taken
    """
    _ensure_headless_pygame()
    from reinforcetactics.ui.renderer import Renderer

    # Determine the unwrapped game_state accessor
    _inner = env.unwrapped if hasattr(env, "unwrapped") else env

    def _get_gs():
        return _inner.game_state

    # Agent player id (defaults to 1 if the env doesn't expose it)
    agent_player = getattr(_inner, "agent_player", 1)
    opponent_player = 3 - agent_player

    def _snapshot(action_type=None, unit_type=None, reward=None, reward_breakdown=None, valid_action=None):
        gs = _get_gs()
        agent_units = [u for u in gs.units if u.player == agent_player]
        opp_units = [u for u in gs.units if u.player == opponent_player]
        return {
            "turn": gs.turn_number,
            "current_player": gs.current_player,
            "action_type": action_type,
            "unit_type": unit_type,
            "reward": reward,
            "reward_breakdown": dict(reward_breakdown) if reward_breakdown else None,
            "valid_action": valid_action,
            "agent_units": len(agent_units),
            "opponent_units": len(opp_units),
            "agent_gold": gs.player_gold.get(agent_player, 0),
            "opponent_gold": gs.player_gold.get(opponent_player, 0),
            "agent_hp_total": sum(getattr(u, "health", 0) for u in agent_units),
            "opponent_hp_total": sum(getattr(u, "health", 0) for u in opp_units),
            # Capturable tiles owned by each player (towers + buildings + HQ).
            # Reads via the same helper used by the env's structure_control
            # potential term, so the chart and reward signal stay in sync.
            "agent_structures": len(gs.grid.get_capturable_tiles(player=agent_player)),
            "opponent_structures": len(gs.grid.get_capturable_tiles(player=opponent_player)),
        }

    # Check whether the env supports action masking (ActionMaskedEnv)
    _has_masks = hasattr(env, "action_masks") and callable(env.action_masks)

    obs, info = env.reset()
    # Create headless renderer after reset so game_state is fresh
    renderer = Renderer(_get_gs(), replay_mode=True, headless=True, pixel_art=use_pixel_art)

    step_stats: list[dict[str, Any]] = [_snapshot()]

    total_reward = 0.0
    steps = 0
    done = False

    def _capture_frame():
        renderer.render()
        if show_hud:
            _draw_video_hud(renderer.screen, _get_gs())
        return renderer.get_rgb_array()

    with _VideoWriter(output_path, fps, scale=scale) as writer:
        # Capture initial state
        last_frame = _capture_frame()
        writer.write(last_frame)

        while not done and steps < max_steps:
            predict_kwargs = dict(deterministic=deterministic)
            if _has_masks:
                predict_kwargs["action_masks"] = env.action_masks()
            action, _ = model.predict(obs, **predict_kwargs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            step_stats.append(
                _snapshot(
                    action_type=info.get("action_type"),
                    unit_type=info.get("unit_type"),
                    reward=float(reward),
                    reward_breakdown=info.get("reward_breakdown"),
                    valid_action=info.get("valid_action"),
                )
            )

            # Capture frame after each action
            renderer.game_state = _get_gs()
            last_frame = _capture_frame()
            writer.write(last_frame)
            # Same beat as replay videos: hold one extra frame on end_turn
            # (action_type 5) so the handover to the opponent reads as a
            # pause instead of a jump-cut.
            if info.get("action_type") == 5:
                writer.write(last_frame)
            if progress_callback is not None:
                progress_callback(steps, max_steps)

        # Hold last frame with a game-over caption so viewers can see who
        # won (or why the episode ended) instead of staring at an unmarked
        # mid-board freeze. Recaptured without the turn badge: the post-game
        # turn counter would contradict the overlay's turn count.
        gs_final = _get_gs()
        eval_game_info = {
            "winner": info.get("winner") or (gs_final.winner if gs_final.game_over else None),
            "total_turns": gs_final.turn_number,
            "max_turns": gs_final.max_turns,
            "end_reason": info.get("end_reason"),
        }
        renderer.render()
        final_frame = _overlay_game_over(renderer.get_rgb_array(), eval_game_info, gs_final)
        for _ in range(3 * fps):
            writer.write(final_frame)

    video_path = output_path

    gs = _get_gs()
    winner = info.get("winner") or (gs.winner if gs.game_over else None)
    episode_stats = info.get("episode_stats", {})

    # Save the recorded game's action history as a standard replay JSON
    # next to the MP4. GameState.save_replay_to_file builds the canonical
    # replay format (game_info + actions) and writes it via FileIO; the
    # same JSON is loadable by FileIO.load_replay and replayable through
    # record_replay_to_video / the in-game replay viewer. Matches the
    # video filename so .mp4 and .json stay paired (agent_replay_best.mp4
    # ↔ agent_replay_best.json).
    #
    # GameState's game_info records winner / total_turns / game_over but
    # not the env's end_reason classification (hq_capture / elimination /
    # max_turns_draw / max_steps_truncate). Patch it in here so a reader
    # can see *how* the game ended without re-deriving it from actions[].
    replay_path: str | None = None
    try:
        replay_path = gs.save_replay_to_file(str(Path(output_path).with_suffix(".json")))
        if replay_path:
            import json as _json

            with open(replay_path, encoding="utf-8") as _f:
                _replay_data = _json.load(_f)
            _replay_data.setdefault("game_info", {})["end_reason"] = info.get("end_reason")
            with open(replay_path, "w", encoding="utf-8") as _f:
                _json.dump(_replay_data, _f, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save replay JSON next to %s: %s", output_path, exc)

    return {
        "video_path": video_path,
        "replay_path": replay_path,
        "winner": winner,
        "end_reason": info.get("end_reason"),
        "total_reward": total_reward,
        "steps": steps,
        "episode_stats": episode_stats,
        "step_stats": step_stats,
    }


def record_replay_to_video(
    replay_data: dict[str, Any],
    output_path: str = "replay.mp4",
    fps: int = 4,
    scale: int | None = None,
    use_pixel_art: bool | None = False,
    show_hud: bool = True,
    progress_callback: Any = None,
) -> str:
    """
    Record a saved replay file to MP4 video using headless rendering.

    This replays the action history from a saved replay JSON file,
    capturing a frame after each action.

    Args:
        replay_data: Replay dict (from FileIO.load_replay()) with keys
                    'actions', 'game_info'
        output_path: Path for the output video file
        fps: Frames per second for the output video
        scale: Integer upscale factor (nearest-neighbour). ``None``
                    (default) auto-picks the smallest factor that makes
                    the short side at least ~480 px.
        use_pixel_art: Render with the bundled pixel-art sprites instead
                    of the fallback (coloured rects + unit letters).
                    ``None`` follows ``settings.json``, matching the
                    interactive renderer.
        show_hud: Stamp a compact turn/player badge onto each frame.
        progress_callback: Optional ``callback(done, total)`` invoked
                    after each encoded action so long exports can report
                    progress (e.g. the in-game replay viewer).

    Returns:
        Path to the saved video file
    """
    import pandas as pd

    _ensure_headless_pygame()
    from reinforcetactics.core.game_state import GameState
    from reinforcetactics.ui.renderer import Renderer

    actions = replay_data.get("actions", [])
    game_info = replay_data.get("game_info", {})
    initial_map = game_info.get("initial_map")

    if initial_map is None:
        raise ValueError("Replay data missing 'initial_map' in game_info")

    map_df = pd.DataFrame(initial_map)

    # Add an ocean border for framing. Unlike the interactive ReplayPlayer,
    # no MIN_MAP_SIZE padding is applied: that padding exists to leave room
    # for on-screen controls, and in a video it only produces a huge dead
    # ocean margin around a small board.
    border = 2
    h2, w2 = map_df.shape
    bordered = pd.DataFrame(np.full((h2 + 2 * border, w2 + 2 * border), "o", dtype=object))
    bordered.iloc[border : border + h2, border : border + w2] = map_df.values
    offset_x, offset_y = border, border

    # Create game state and headless renderer
    game_state = GameState(bordered, num_players=game_info.get("num_players", 2))
    renderer = Renderer(game_state, replay_mode=True, headless=True, pixel_art=use_pixel_art)

    # Helper to translate coordinates
    def _translate(x, y):
        return x + offset_x, y + offset_y

    def _capture_frame():
        renderer.render()
        if show_hud:
            _draw_video_hud(renderer.screen, game_state)
        return renderer.get_rgb_array()

    with _VideoWriter(output_path, fps, scale=scale) as writer:
        # Capture initial state
        last_frame = _capture_frame()
        writer.write(last_frame)

        # Execute each action and capture frame. Stop the moment the game
        # ends so we don't replay post-victory "cosmetic" actions that some
        # older replays accumulated (bots that didn't break out of their
        # per-unit loop on game_over -- now fixed, but old saved replays
        # still carry the trailing actions). This also keeps end_turn()
        # from advancing current_player/turn_number past the actual end,
        # which would otherwise make the closing-frame overlay misreport
        # the final turn number.
        schema_version = get_schema_version(game_info)
        total = len(actions)
        for i, action in enumerate(actions):
            _execute_replay_action(game_state, action, _translate, schema_version)
            last_frame = _capture_frame()
            writer.write(last_frame)
            # Hold one extra frame at turn boundaries so the handover
            # between players reads as a beat instead of flashing past.
            if action.get("type") == "end_turn":
                writer.write(last_frame)
            if progress_callback is not None:
                progress_callback(i + 1, total)
            if game_state.game_over:
                break

        # Hold last frame with a "Game Over -- Winner: X" overlay so viewers
        # can see how the game ended. Tournament replays don't record an
        # explicit end_reason, so we synthesise one from game_info + final
        # state. Recaptured without the turn badge: the post-game turn
        # counter would contradict the overlay's authoritative turn count.
        renderer.render()
        final_frame = _overlay_game_over(renderer.get_rgb_array(), game_info, game_state)
        for _ in range(3 * fps):
            writer.write(final_frame)

    return output_path


def _draw_video_hud(screen, game_state) -> None:
    """Stamp a compact status badge (turn + player to move) onto a frame.

    Headless renders skip the interactive HUD entirely, so without this a
    video gives no indication of the turn number or whose move it is. The
    badge sits in the top-left ocean border and scales with frame size.
    """
    import pygame

    from reinforcetactics.constants import PLAYER_COLORS
    from reinforcetactics.utils.fonts import get_font

    turn_text = f"Turn {game_state.turn_number + 1}"
    if game_state.max_turns:
        turn_text += f"/{game_state.max_turns}"
    player = game_state.current_player
    label_text = f"{turn_text}  •  P{player}"

    width = screen.get_width()
    font = get_font(max(12, min(22, width // 32)))
    label = font.render(label_text, True, (255, 255, 255))

    pad = 6
    swatch = max(8, label.get_height() - 8)
    badge_w = pad + swatch + pad + label.get_width() + pad
    badge_h = label.get_height() + 2 * (pad - 2)

    badge = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
    badge.fill((10, 10, 20, 170))
    swatch_rect = pygame.Rect(pad, (badge_h - swatch) // 2, swatch, swatch)
    pygame.draw.rect(badge, PLAYER_COLORS.get(player, (255, 255, 255)), swatch_rect)
    pygame.draw.rect(badge, (255, 255, 255), swatch_rect, 1)
    badge.blit(label, (pad + swatch + pad, (badge_h - label.get_height()) // 2))
    screen.blit(badge, (8, 8))


def _overlay_game_over(frame, game_info: dict[str, Any], game_state) -> np.ndarray:
    """Stamp a "Game Over -- Winner: X" caption onto the final frame.

    Used to give the held closing frame a visible end-state caption -- the
    bare renderer has no game-over UI of its own, so without this the video
    just freezes on an ordinary mid-board frame and looks truncated.
    """
    import pygame

    winner = game_info.get("winner")
    if winner is None and getattr(game_state, "game_over", False):
        winner = game_state.winner
    # Any positive player number counts as a win: FFA and team games have
    # winners 3 and 4, which the old ``winner in (1, 2)`` check misreported
    # as a draw. 0/None remain draws.
    has_winner = isinstance(winner, int) and winner >= 1
    winner_name = game_info.get("winner_name")
    if not winner_name:
        if has_winner:
            winner_name = f"Player {winner}"
        else:
            winner_name = "Draw"

    turns = game_info.get("total_turns") or game_info.get("turns") or game_state.turn_number
    max_turns = game_info.get("max_turns") or game_state.max_turns
    end_reason = game_info.get("end_reason") or getattr(game_state, "end_reason", None)

    reason_label = {
        "hq_capture": "HQ Captured",
        "elimination": "Elimination",
        "resign": "Resignation",
        "max_turns_draw": "Turn Limit",
        "max_steps_truncate": "Step Limit",
    }.get(end_reason or "", None)

    if has_winner:
        headline = "Game Over"
        subline = f"Winner: {winner_name}  (Turn {turns})"
        if reason_label:
            subline += f"  --  {reason_label}"
    else:
        # Distinguish max-turns draw from any other draw. Prefer engine-
        # reported end_reason; fall back to a turn-count heuristic for old
        # replays that don't carry end_reason.
        if end_reason == "max_turns_draw" or (not end_reason and max_turns and turns and turns >= max_turns):
            headline = "Draw -- Turn Limit"
        else:
            headline = "Draw"
        subline = f"Turn {turns}" + (f" / {max_turns}" if max_turns else "")

    h, w = frame.shape[:2]
    surface = pygame.image.frombuffer(np.ascontiguousarray(frame).tobytes(), (w, h), "RGB").convert()

    # Translucent backdrop so text reads on any underlying tile.
    band_h = max(48, int(h * 0.22))
    band_y = (h - band_h) // 2
    band = pygame.Surface((w, band_h), pygame.SRCALPHA)
    band.fill((0, 0, 0, 180))
    surface.blit(band, (0, band_y))

    from reinforcetactics.utils.fonts import get_font

    head_size = max(18, min(48, w // 12))
    sub_size = max(12, min(28, w // 20))
    head_font = get_font(head_size)
    sub_font = get_font(sub_size)

    head_surf = head_font.render(headline, True, (255, 235, 120))
    sub_surf = sub_font.render(subline, True, (235, 235, 235))

    spacing = 6
    total_h = head_surf.get_height() + spacing + sub_surf.get_height()
    top = band_y + (band_h - total_h) // 2
    surface.blit(head_surf, head_surf.get_rect(midtop=(w // 2, top)))
    surface.blit(sub_surf, sub_surf.get_rect(midtop=(w // 2, top + head_surf.get_height() + spacing)))

    return np.frombuffer(pygame.image.tobytes(surface, "RGB"), dtype=np.uint8).reshape((h, w, 3))


class _VideoWriter:
    """Streaming MP4 writer: encodes one frame at a time so the full video
    is never held in memory at once. Dozens of long games can be rendered
    back-to-back without OOMing.

    ``scale`` upscales each frame with nearest-neighbour interpolation
    before encoding so small grids (a 6x6 map renders at 192x192 px)
    produce crisp, readable video. ``scale=None`` auto-picks the smallest
    factor that brings the short side to at least ``AUTO_SCALE_TARGET``
    pixels (capped at ``AUTO_SCALE_MAX``), so callers get readable output
    for tiny boards without upscaling already-large ones.

    Prefers ``imageio`` with the bundled ffmpeg backend (H.264 MP4s that
    play in QuickTime / browsers / VLC). Falls back to ``cv2.VideoWriter``
    with the legacy ``mp4v`` codec when the ffmpeg backend is unavailable.
    Backend is chosen on the first ``write`` call so the source frame size
    is known and we can correct odd dimensions (yuv420p needs even width
    and height).
    """

    AUTO_SCALE_TARGET = 480
    AUTO_SCALE_MAX = 4

    def __init__(self, output_path: str, fps: int, scale: int | None = None, quality: int = 8):
        self.output_path = str(output_path)
        self.fps = fps
        self.scale = None if scale is None else max(1, int(scale))
        # imageio/ffmpeg variable-bitrate quality, 0 (worst) .. 10 (best).
        # The imageio default of 5 visibly smears pixel-art edges.
        self.quality = quality
        self._writer: Any = None
        self._cv2_writer: Any = None
        self._target_wh: tuple | None = None
        self._resolved_scale = 1
        self._need_resize = False
        self._n_frames = 0
        self._backend = "unknown"
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def write(self, frame) -> None:
        if self._closed:
            raise RuntimeError("VideoWriter already closed")
        if self._writer is None and self._cv2_writer is None:
            self._lazy_init(frame)
        conformed = self._conform(frame)
        if self._writer is not None:
            self._writer.append_data(conformed)
        else:
            import cv2

            self._cv2_writer.write(cv2.cvtColor(conformed, cv2.COLOR_RGB2BGR))
        self._n_frames += 1

    def _lazy_init(self, first_frame) -> None:
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        src_h, src_w = first_frame.shape[:2]
        scale = self.scale
        if scale is None:
            short_side = min(src_w, src_h)
            scale = max(1, min(self.AUTO_SCALE_MAX, -(-self.AUTO_SCALE_TARGET // short_side)))
        self._resolved_scale = scale
        width = src_w * scale
        height = src_h * scale
        # yuv420p requires even width/height
        width += width % 2
        height += height % 2
        self._target_wh = (width, height)
        self._need_resize = (width, height) != (src_w, src_h)

        try:
            import imageio
            import imageio_ffmpeg  # noqa: F401

            self._writer = imageio.get_writer(
                self.output_path,
                fps=self.fps,
                codec="libx264",
                quality=self.quality,
                pixelformat="yuv420p",
                macro_block_size=1,
                # +faststart moves the moov atom to the front so embedded
                # players (notebooks, browsers) start before the download
                # finishes; animation tuning suits flat-colour board art.
                output_params=["-movflags", "+faststart", "-tune", "animation"],
            )
            self._backend = "h264"
            return
        except (ImportError, Exception) as exc:  # noqa: BLE001
            logger.warning(
                "imageio/ffmpeg unavailable (%s); falling back to cv2 mp4v (may not play in QuickTime/Preview)",
                exc,
            )

        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self._cv2_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        if not self._cv2_writer.isOpened():
            raise RuntimeError(
                f"cv2.VideoWriter failed to open {self.output_path}; install imageio-ffmpeg for reliable H.264 output"
            )
        self._backend = "mp4v"

    def _conform(self, frame):
        # Pure-numpy resize so the imageio/H.264 path has no OpenCV
        # dependency: integer nearest-neighbour upscale plus edge-padding
        # to even dimensions.
        if not self._need_resize:
            return frame
        if self._resolved_scale > 1:
            frame = frame.repeat(self._resolved_scale, axis=0).repeat(self._resolved_scale, axis=1)
        h, w = frame.shape[:2]
        target_w, target_h = self._target_wh
        if (w, h) != (target_w, target_h):
            frame = np.pad(frame, ((0, target_h - h), (0, target_w - w), (0, 0)), mode="edge")
        return frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            self._writer.close()
        if self._cv2_writer is not None:
            self._cv2_writer.release()
        if self._n_frames > 0:
            logger.info(
                "Video saved to %s (%d frames, %.1fs at %d fps, %s)",
                self.output_path,
                self._n_frames,
                self._n_frames / self.fps,
                self.fps,
                self._backend,
            )


def _write_frames_to_video(frames, output_path: str, fps: int, scale: int | None = None) -> str:
    """Stream an iterable of RGB ndarrays to an MP4. Memory stays bounded
    at ~one frame regardless of total length.
    """
    n = 0
    with _VideoWriter(output_path, fps, scale=scale) as writer:
        for frame in frames:
            writer.write(frame)
            n += 1
    if n == 0:
        raise ValueError("No frames to write")
    return output_path


def display_video_in_notebook(video_path: str):
    """Display an MP4 video inline in a Jupyter notebook."""
    try:
        from IPython.display import Video, display

        display(Video(video_path, embed=True, mimetype="video/mp4"))
    except ImportError:
        print(f"Video saved to: {video_path}")
        print("Install IPython to display videos inline.")
