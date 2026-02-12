"""
Headless video recording utilities for Jupyter notebooks and CI.

Provides functions to record game replays and agent evaluations to MP4 video
without requiring a display server. Works on Google Colab and headless Linux.
"""
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_headless_pygame():
    """Ensure pygame is initialised with a dummy video driver for headless use."""
    if not os.environ.get('SDL_VIDEODRIVER'):
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    import pygame
    if not pygame.get_init():
        pygame.init()
    return pygame


def record_game_to_video(
    game_states: List[Any],
    output_path: str = "game_replay.mp4",
    fps: int = 4,
    map_file: Optional[str] = None,
) -> str:
    """
    Record a sequence of game state snapshots to an MP4 video.

    Args:
        game_states: List of game state snapshot dicts (from GameState.to_dict())
        output_path: Path for the output video file
        fps: Frames per second for the output video
        map_file: Path to the map CSV file (needed to reconstruct game states)

    Returns:
        Path to the saved video file
    """
    import cv2
    _ensure_headless_pygame()
    from reinforcetactics.ui.renderer import Renderer
    from reinforcetactics.core.game_state import GameState

    if not game_states:
        raise ValueError("No game states to record")

    frames = []
    for state_dict in game_states:
        gs = GameState.from_dict(state_dict, state_dict.get('map_data'))
        renderer = Renderer(gs, replay_mode=True, headless=True)
        renderer.render()
        frames.append(renderer.get_rgb_array())

    return _write_frames_to_video(frames, output_path, fps)


def record_evaluation_to_video(
    env,
    model,
    output_path: str = "agent_replay.mp4",
    fps: int = 4,
    max_steps: int = 500,
    deterministic: bool = True,
) -> Dict[str, Any]:
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

    Returns:
        Dict with keys:
            - video_path: Path to the saved video
            - winner: Winner of the game (1, 2, or None)
            - total_reward: Total episode reward
            - steps: Number of steps taken
    """
    import cv2
    _ensure_headless_pygame()
    from reinforcetactics.ui.renderer import Renderer

    # Determine the unwrapped game_state accessor
    _inner = env.unwrapped if hasattr(env, 'unwrapped') else env
    _get_gs = lambda: _inner.game_state

    # Check whether the env supports action masking (ActionMaskedEnv)
    _has_masks = hasattr(env, 'action_masks') and callable(env.action_masks)

    obs, info = env.reset()
    # Create headless renderer after reset so game_state is fresh
    renderer = Renderer(_get_gs(), replay_mode=True, headless=True)

    frames = []

    # Capture initial state
    renderer.render()
    frames.append(renderer.get_rgb_array())

    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:
        predict_kwargs = dict(deterministic=deterministic)
        if _has_masks:
            predict_kwargs['action_masks'] = env.action_masks()
        action, _ = model.predict(obs, **predict_kwargs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        # Capture frame after each action
        renderer.game_state = _get_gs()
        renderer.render()
        frames.append(renderer.get_rgb_array())

    # Hold last frame for 1 second so viewer can see final state
    for _ in range(fps):
        frames.append(frames[-1])

    video_path = _write_frames_to_video(frames, output_path, fps)

    gs = _get_gs()
    winner = info.get('winner') or (gs.winner if gs.game_over else None)
    episode_stats = info.get('episode_stats', {})

    return {
        'video_path': video_path,
        'winner': winner,
        'total_reward': total_reward,
        'steps': steps,
        'episode_stats': episode_stats,
    }


def record_replay_to_video(
    replay_data: Dict[str, Any],
    output_path: str = "replay.mp4",
    fps: int = 4,
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

    Returns:
        Path to the saved video file
    """
    import cv2
    import pandas as pd
    _ensure_headless_pygame()
    from reinforcetactics.core.game_state import GameState
    from reinforcetactics.ui.renderer import Renderer
    from reinforcetactics.constants import MIN_MAP_SIZE

    actions = replay_data.get('actions', [])
    game_info = replay_data.get('game_info', {})
    initial_map = game_info.get('initial_map')

    if initial_map is None:
        raise ValueError("Replay data missing 'initial_map' in game_info")

    # Convert to DataFrame
    map_df = pd.DataFrame(initial_map)

    # Pad small maps (same logic as ReplayPlayer)
    height, width = map_df.shape
    offset_x, offset_y = 0, 0

    if height < MIN_MAP_SIZE or width < MIN_MAP_SIZE:
        min_h = max(height, MIN_MAP_SIZE)
        min_w = max(width, MIN_MAP_SIZE)
        pad_w = max(0, min_w - width)
        pad_h = max(0, min_h - height)
        if pad_w > 0 or pad_h > 0:
            padded = pd.DataFrame(np.full((min_h, min_w), 'o', dtype=object))
            sy, sx = pad_h // 2, pad_w // 2
            padded.iloc[sy:sy + height, sx:sx + width] = map_df.values
            map_df = padded
            offset_x, offset_y = sx, sy

    # Add ocean border
    border = 2
    h2, w2 = map_df.shape
    bordered = pd.DataFrame(np.full((h2 + 2 * border, w2 + 2 * border), 'o', dtype=object))
    bordered.iloc[border:border + h2, border:border + w2] = map_df.values
    offset_x += border
    offset_y += border

    # Create game state and headless renderer
    game_state = GameState(bordered, num_players=game_info.get('num_players', 2))
    renderer = Renderer(game_state, replay_mode=True, headless=True)

    frames = []

    # Capture initial state
    renderer.render()
    frames.append(renderer.get_rgb_array())

    # Helper to translate coordinates
    def _translate(x, y):
        return x + offset_x, y + offset_y

    # Execute each action and capture frame
    for action in actions:
        _execute_replay_action(game_state, action, _translate)
        renderer.render()
        frames.append(renderer.get_rgb_array())

    # Hold last frame
    for _ in range(fps):
        frames.append(frames[-1])

    return _write_frames_to_video(frames, output_path, fps)


def _execute_replay_action(game_state, action, translate_fn):
    """Execute a single replay action on the game state."""
    action_type = action.get('type')
    try:
        if action_type == 'create_unit':
            px, py = translate_fn(action['x'], action['y'])
            game_state.create_unit(action['unit_type'], px, py, action['player'])
        elif action_type == 'move':
            fx, fy = translate_fn(action['from_x'], action['from_y'])
            tx, ty = translate_fn(action['to_x'], action['to_y'])
            unit = game_state.get_unit_at_position(fx, fy)
            if unit and unit.player == action['player']:
                game_state.move_unit(unit, tx, ty)
        elif action_type == 'attack':
            ap = translate_fn(*action['attacker_pos'])
            tp = translate_fn(*action['target_pos'])
            attacker = game_state.get_unit_at_position(*ap)
            target = game_state.get_unit_at_position(*tp)
            if attacker and target:
                game_state.attack(attacker, target)
        elif action_type == 'seize':
            pos = translate_fn(*action['position'])
            unit = game_state.get_unit_at_position(*pos)
            if unit:
                game_state.seize(unit)
        elif action_type == 'paralyze':
            pp = translate_fn(*action['paralyzer_pos'])
            tp = translate_fn(*action['target_pos'])
            paralyzer = game_state.get_unit_at_position(*pp)
            target = game_state.get_unit_at_position(*tp)
            if paralyzer and target:
                game_state.paralyze(paralyzer, target)
        elif action_type == 'heal':
            hp = translate_fn(*action['healer_pos'])
            tp = translate_fn(*action['target_pos'])
            healer = game_state.get_unit_at_position(*hp)
            target = game_state.get_unit_at_position(*tp)
            if healer and target:
                game_state.heal(healer, target)
        elif action_type == 'cure':
            cp = translate_fn(*action['curer_pos'])
            tp = translate_fn(*action['target_pos'])
            curer = game_state.get_unit_at_position(*cp)
            target = game_state.get_unit_at_position(*tp)
            if curer and target:
                game_state.cure(curer, target)
        elif action_type == 'haste':
            sp = translate_fn(*action['sorcerer_pos'])
            tp = translate_fn(*action['target_pos'])
            sorcerer = game_state.get_unit_at_position(*sp)
            target = game_state.get_unit_at_position(*tp)
            if sorcerer and target:
                game_state.haste(sorcerer, target)
        elif action_type == 'defence_buff':
            sp = translate_fn(*action['sorcerer_pos'])
            tp = translate_fn(*action['target_pos'])
            sorcerer = game_state.get_unit_at_position(*sp)
            target = game_state.get_unit_at_position(*tp)
            if sorcerer and target:
                game_state.defence_buff(sorcerer, target)
        elif action_type == 'attack_buff':
            sp = translate_fn(*action['sorcerer_pos'])
            tp = translate_fn(*action['target_pos'])
            sorcerer = game_state.get_unit_at_position(*sp)
            target = game_state.get_unit_at_position(*tp)
            if sorcerer and target:
                game_state.attack_buff(sorcerer, target)
        elif action_type == 'resign':
            game_state.resign(action['player'])
        elif action_type == 'end_turn':
            old_history = game_state.action_history
            game_state.action_history = []
            game_state.end_turn()
            game_state.action_history = old_history
    except Exception as e:
        logger.warning("Error executing replay action %s: %s", action_type, e)


def _write_frames_to_video(frames: list, output_path: str, fps: int) -> str:
    """Write a list of RGB numpy arrays to an MP4 video file."""
    import cv2

    if not frames:
        raise ValueError("No frames to write")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    logger.info("Video saved to %s (%d frames, %.1fs at %d fps)",
                output_path, len(frames), len(frames) / fps, fps)
    return output_path


def display_video_in_notebook(video_path: str):
    """Display an MP4 video inline in a Jupyter notebook."""
    try:
        from IPython.display import Video, display
        display(Video(video_path, embed=True, mimetype='video/mp4'))
    except ImportError:
        print(f"Video saved to: {video_path}")
        print("Install IPython to display videos inline.")
