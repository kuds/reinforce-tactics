"""
File I/O utilities for maps, saves, and replays.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from reinforcetactics.constants import MIN_MAP_SIZE, MIN_STRIP_SIZE


class FileIO:
    """Handles all file I/O operations for the game."""

    @staticmethod
    def load_map(filepath, for_ui=False, border_size=2):
        """
        Load a map from a CSV file.

        Args:
            filepath: Path to the CSV map file
            for_ui: If True, adds water borders for UI display (default: False)
            border_size: Number of ocean border layers to add when for_ui=True (default: 2)

        Returns:
            pandas DataFrame containing the map data
        """
        result = FileIO.load_map_with_metadata(filepath, for_ui, border_size)
        if result is None:
            return None
        return result['map_data']

    @staticmethod
    def load_map_with_metadata(filepath, for_ui=False, border_size=2):
        """
        Load a map from a CSV file with metadata about padding.

        Args:
            filepath: Path to the CSV map file
            for_ui: If True, adds minimum size padding and water borders for UI display (default: False)
            border_size: Number of ocean border layers to add when for_ui=True (default: 2)

        Returns:
            dict with keys:
                - map_data: pandas DataFrame containing the map data (padded if for_ui=True)
                - original_map_data: 2D list of the unpadded map data
                - original_width: width before padding
                - original_height: height before padding
                - padding_offset_x: x offset due to padding (0 if no padding or for_ui=False)
                - padding_offset_y: y offset due to padding (0 if no padding or for_ui=False)
                - map_file: filepath that was loaded
        """
        try:
            # Load CSV file - force all data to be strings and strip whitespace
            map_data = pd.read_csv(
                filepath,
                header=None,
                dtype=str,  # Force everything to be strings
                skipinitialspace=True  # Skip whitespace after delimiter
            )

            # Strip whitespace from all cells and drop empty rows/columns
            map_data = map_data.map(lambda x: str(x).strip() if pd.notna(x) else 'p')

            # Drop rows that are all 'p' (likely empty rows)
            # But keep at least some content
            map_data = map_data.replace('nan', 'o')  # Replace any 'nan' strings with grass

            # Remove completely empty rows and columns
            map_data = map_data.dropna(axis=0, how='all')  # Drop empty rows
            map_data = map_data.dropna(axis=1, how='all')  # Drop empty columns

            # Reset index after dropping
            map_data = map_data.reset_index(drop=True)

            # Store original dimensions and data
            original_height, original_width = map_data.shape
            original_map_data = map_data.values.tolist()  # Store unpadded map as 2D list
            padding_offset_x = 0
            padding_offset_y = 0

            # Only apply minimum size padding if loading for UI
            if for_ui:
                height, width = map_data.shape
                if height < MIN_MAP_SIZE or width < MIN_MAP_SIZE:
                    print(
                        f"⚠️  Map size ({width}x{height}) is smaller than "
                        f"minimum ({MIN_MAP_SIZE}x{MIN_MAP_SIZE})"
                    )
                    print("   Padding map to minimum size for UI...")
                    map_data, padding_offset_x, padding_offset_y = FileIO._pad_map(
                        map_data, MIN_MAP_SIZE, MIN_MAP_SIZE
                    )

            # Add water borders if loading for UI (after ensuring minimum size)
            if for_ui:
                map_data = FileIO.add_water_border(map_data, border_size)

            height, width = map_data.shape
            print(f"✅ Map loaded: {width}x{height}")

            return {
                'map_data': map_data,
                'original_map_data': original_map_data,
                'original_width': original_width,
                'original_height': original_height,
                'padding_offset_x': padding_offset_x,
                'padding_offset_y': padding_offset_y,
                'map_file': filepath
            }

        except FileNotFoundError:
            print(f"❌ Map file not found: {filepath}")
            return None
        except Exception as e:
            print(f"❌ Error loading map: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _pad_map(map_data, min_width, min_height):
        """
        Pad a map to minimum dimensions with ocean tiles.

        Args:
            map_data: pandas DataFrame with map data
            min_width: Minimum width
            min_height: Minimum height

        Returns:
            tuple: (padded_map, offset_x, offset_y) where offsets are the padding on left/top
        """
        current_height, current_width = map_data.shape

        # Calculate padding needed
        pad_width = max(0, min_width - current_width)
        pad_height = max(0, min_height - current_height)

        if pad_width > 0 or pad_height > 0:
            # Pad with ocean tiles ('o')
            padded = pd.DataFrame(
                np.full((min_height, min_width), 'o', dtype=object)
            )

            # Copy original data into center
            start_y = pad_height // 2
            start_x = pad_width // 2
            end_y = start_y + current_height
            end_x = start_x + current_width
            padded.iloc[start_y:end_y, start_x:end_x] = map_data.values

            return padded, start_x, start_y

        return map_data, 0, 0

    @staticmethod
    def strip_water_border(map_data, min_size=MIN_STRIP_SIZE):
        """
        Strip ocean borders from a map by removing full-ocean rows/columns
        from each side independently until a non-ocean row/column is found
        or minimum size is reached.

        This handles off-center maps by stripping each side based on how many
        consecutive ocean rows/columns exist on that side, rather than requiring
        all four sides to have ocean borders before stripping.

        Args:
            map_data: pandas DataFrame with map data
            min_size: Minimum map size to preserve (default: MIN_STRIP_SIZE)

        Returns:
            pandas DataFrame with ocean borders stripped
        """
        if map_data is None or map_data.empty:
            return map_data

        result = map_data.copy()
        height, width = result.shape

        # Count consecutive ocean rows from top
        top_strip = 0
        for i in range(height):
            if all(tile == 'o' for tile in result.iloc[i, :].values):
                top_strip += 1
            else:
                break

        # Count consecutive ocean rows from bottom
        bottom_strip = 0
        for i in range(height - 1, -1, -1):
            if all(tile == 'o' for tile in result.iloc[i, :].values):
                bottom_strip += 1
            else:
                break

        # Count consecutive ocean columns from left
        left_strip = 0
        for j in range(width):
            if all(tile == 'o' for tile in result.iloc[:, j].values):
                left_strip += 1
            else:
                break

        # Count consecutive ocean columns from right
        right_strip = 0
        for j in range(width - 1, -1, -1):
            if all(tile == 'o' for tile in result.iloc[:, j].values):
                right_strip += 1
            else:
                break

        # Calculate how much we can actually strip while respecting min_size
        max_vertical_strip = max(0, height - min_size)
        max_horizontal_strip = max(0, width - min_size)

        # Limit strips to respect minimum size
        total_vertical = top_strip + bottom_strip
        total_horizontal = left_strip + right_strip

        if total_vertical > max_vertical_strip:
            # Scale down proportionally if we would go below min_size
            if total_vertical > 0:
                scale = max_vertical_strip / total_vertical
                # Use round() to minimize leftover ocean borders
                top_strip = round(top_strip * scale)
                bottom_strip = round(bottom_strip * scale)
                # Ensure we don't exceed max by adjusting the larger value
                while top_strip + bottom_strip > max_vertical_strip:
                    if top_strip >= bottom_strip:
                        top_strip -= 1
                    else:
                        bottom_strip -= 1

        if total_horizontal > max_horizontal_strip:
            # Scale down proportionally if we would go below min_size
            if total_horizontal > 0:
                scale = max_horizontal_strip / total_horizontal
                # Use round() to minimize leftover ocean borders
                left_strip = round(left_strip * scale)
                right_strip = round(right_strip * scale)
                # Ensure we don't exceed max by adjusting the larger value
                while left_strip + right_strip > max_horizontal_strip:
                    if left_strip >= right_strip:
                        left_strip -= 1
                    else:
                        right_strip -= 1

        # Apply the strips
        if top_strip > 0 or bottom_strip > 0 or left_strip > 0 or right_strip > 0:
            row_end = height - bottom_strip if bottom_strip > 0 else height
            col_end = width - right_strip if right_strip > 0 else width
            result = result.iloc[top_strip:row_end, left_strip:col_end]
            result = result.reset_index(drop=True)
            result.columns = range(result.shape[1])

        return result

    @staticmethod
    def add_water_border(map_data, border_size=2):
        """
        Add ocean borders around a map for UI display.

        Args:
            map_data: pandas DataFrame with map data
            border_size: Number of ocean border layers to add (default: 2)

        Returns:
            pandas DataFrame with ocean borders added
        """
        if map_data is None or map_data.empty or border_size <= 0:
            return map_data

        height, width = map_data.shape
        new_height = height + 2 * border_size
        new_width = width + 2 * border_size

        # Create new map filled with ocean
        bordered = pd.DataFrame(
            np.full((new_height, new_width), 'o', dtype=object)
        )

        # Copy original data into center
        bordered.iloc[border_size:border_size + height,
                      border_size:border_size + width] = map_data.values

        return bordered

    @staticmethod
    def generate_random_map(width, height, num_players=2):
        """
        Generate a random map.

        Args:
            width: Map width
            height: Map height
            num_players: Number of players (2-4)

        Returns:
            pandas DataFrame containing the generated map
        """
        # Ensure minimum size
        width = max(width, MIN_MAP_SIZE)
        height = max(height, MIN_MAP_SIZE)

        # Create base map with grass
        map_data = np.full((height, width), 'o', dtype=object)

        # Add some variety - forests, mountains, water
        num_tiles = width * height

        # Add forests (10% of tiles)
        for _ in range(num_tiles // 10):
            x, y = np.random.randint(0, width), np.random.randint(0, height)
            map_data[y, x] = 'f'

        # Add mountains (5% of tiles)
        for _ in range(num_tiles // 20):
            x, y = np.random.randint(0, width), np.random.randint(0, height)
            map_data[y, x] = 'm'

        # Add water (3% of tiles)
        for _ in range(num_tiles // 33):
            x, y = np.random.randint(0, width), np.random.randint(0, height)
            map_data[y, x] = 'w'

        # Place headquarters for each player in corners
        if num_players >= 1:
            map_data[1, 1] = 'h_1'  # Player 1 HQ (top-left)
            map_data[1, 2] = 'b_1'  # Player 1 Building
            map_data[2, 1] = 'b_1'  # Player 1 Building

        if num_players >= 2:
            map_data[height-2, width-2] = 'h_2'  # Player 2 HQ (bottom-right)
            map_data[height-2, width-3] = 'b_2'  # Player 2 Building
            map_data[height-3, width-2] = 'b_2'  # Player 2 Building

        if num_players >= 3:
            map_data[1, width-2] = 'h_3'  # Player 3 HQ (top-right)
            map_data[1, width-3] = 'b_3'
            map_data[2, width-2] = 'b_3'

        if num_players >= 4:
            map_data[height-2, 1] = 'h_4'  # Player 4 HQ (bottom-left)
            map_data[height-2, 2] = 'b_4'
            map_data[height-3, 1] = 'b_4'

        # Add some neutral towers in the center area
        center_x, center_y = width // 2, height // 2
        for dx, dy in [(0, 0), (3, 0), (0, 3), (3, 3)]:
            x, y = center_x + dx - 2, center_y + dy - 2
            if 0 <= x < width and 0 <= y < height:
                if map_data[y, x] == 'p':  # Only place on grass
                    map_data[y, x] = 't'

        return pd.DataFrame(map_data)

    @staticmethod
    def save_game(game_state, filepath=None):
        """
        Save a game state to a JSON file.

        Args:
            game_state: GameState instance to save
            filepath: Path to save file (auto-generated if None)

        Returns:
            Path to the saved file
        """
        if filepath is None:
            # Auto-generate filename
            saves_dir = Path("saves")
            saves_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = saves_dir / f"save_{timestamp}.json"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Convert game state to dictionary
            save_data = game_state.to_dict()

            # Save to JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)

            print(f"✅ Game saved: {filepath}")
            return str(filepath)

        except Exception as e:
            print(f"❌ Error saving game: {e}")
            return None

    @staticmethod
    def load_game(filepath):
        """
        Load a game state from a JSON file.

        Args:
            filepath: Path to the save file

        Returns:
            Dictionary with game state data
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                save_data = json.load(f)

            print(f"✅ Game loaded: {filepath}")
            return save_data

        except FileNotFoundError:
            print(f"❌ Save file not found: {filepath}")
            return None
        except Exception as e:
            print(f"❌ Error loading game: {e}")
            return None

    @staticmethod
    def list_saves():
        """
        List all available save files.

        Returns:
            List of save file paths
        """
        saves_dir = Path("saves")
        if not saves_dir.exists():
            return []

        saves = list(saves_dir.glob("*.json"))
        saves.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Most recent first
        return saves

    @staticmethod
    def save_replay(action_history, game_info, filepath=None):
        """
        Save a replay to a JSON file.

        Args:
            action_history: List of actions taken during the game
            game_info: Dictionary with game metadata (winner, turns, etc.)
            filepath: Path to save file (auto-generated if None)

        Returns:
            Path to the saved replay file
        """
        if filepath is None:
            # Auto-generate filename
            replays_dir = Path("replays")
            replays_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = replays_dir / f"replay_{timestamp}.json"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            replay_data = {
                'timestamp': datetime.now().isoformat(),
                'game_info': game_info,
                'actions': action_history
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(replay_data, f, indent=2)

            print(f"✅ Replay saved: {filepath}")
            return str(filepath)

        except Exception as e:
            print(f"❌ Error saving replay: {e}")
            return None

    @staticmethod
    def load_replay(filepath):
        """
        Load a replay from a JSON file.

        Args:
            filepath: Path to the replay file

        Returns:
            Dictionary with replay data
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                replay_data = json.load(f)

            print(f"✅ Replay loaded: {filepath}")
            return replay_data

        except FileNotFoundError:
            print(f"❌ Replay file not found: {filepath}")
            return None
        except Exception as e:
            print(f"❌ Error loading replay: {e}")
            return None

    @staticmethod
    def list_replays():
        """
        List all available replay files.

        Returns:
            List of replay file paths
        """
        replays_dir = Path("replays")
        if not replays_dir.exists():
            return []

        replays = list(replays_dir.glob("*.json"))
        replays.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Most recent first
        return replays

    @staticmethod
    def list_maps(map_type="1v1"):
        """
        List all available map files.

        Args:
            map_type: Type of maps to list (1v1, 2v2, etc.)

        Returns:
            List of map file paths
        """
        maps_dir = Path(f"maps/{map_type}")
        if not maps_dir.exists():
            return []

        maps = list(maps_dir.glob("*.csv"))
        maps.sort()
        return maps

    @staticmethod
    def save_map(map_data, filepath):
        """
        Save a map to a CSV file.

        Ocean borders are automatically stripped before saving to reduce file size.

        Args:
            map_data: pandas DataFrame with map data
            filepath: Path to save the map

        Returns:
            Path to the saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Strip ocean borders before saving
            stripped_map = FileIO.strip_water_border(map_data)

            stripped_map.to_csv(filepath, header=False, index=False)
            print(f"✅ Map saved: {filepath}")
            return str(filepath)

        except Exception as e:
            print(f"❌ Error saving map: {e}")
            return None

    @staticmethod
    def export_replay_video(output_filepath=None):
        """
        Export a replay to a video file (requires opencv-python).

        Note: This is a placeholder function for future implementation.

        Args:
            output_filepath: Path to save the video (auto-generated if None)

        Returns:
            Path to the saved video file
        """
        try:
            import cv2  # pylint: disable=unused-import,import-outside-toplevel
        except ImportError:
            print("❌ opencv-python not installed. Install with: pip install opencv-python")
            return None

        if output_filepath is None:
            videos_dir = Path("videos")
            videos_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filepath = videos_dir / f"replay_{timestamp}.mp4"

        # This would require more implementation to actually render frames
        # For now, just a placeholder
        print("⚠️  Video export not yet fully implemented")
        print(f"   Would export to: {output_filepath}")
        return str(output_filepath)

    @staticmethod
    def ensure_directories():
        """
        Ensure all necessary directories exist.
        """
        directories = [
            "maps/1v1",
            "maps/2v2",
            "saves",
            "replays",
            "models",
            "checkpoints",
            "tensorboard",
            "logs",
            "videos"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

        print("✅ All directories created")

    @staticmethod
    def get_map_info(filepath):
        """
        Get information about a map file.

        Args:
            filepath: Path to the map file

        Returns:
            Dictionary with map information
        """
        try:
            map_data = pd.read_csv(filepath, header=None)
            height, width = map_data.shape

            # Count different tile types
            tiles = map_data.values.flatten()
            unique_tiles = {}
            for tile in tiles:
                tile_type = str(tile).split('_', maxsplit=1)[0]
                unique_tiles[tile_type] = unique_tiles.get(tile_type, 0) + 1

            # Count players
            num_players = 0
            for tile in tiles:
                if '_' in str(tile):
                    parts = str(tile).split('_')
                    if len(parts) >= 2 and parts[1].isdigit():
                        num_players = max(num_players, int(parts[1]))

            return {
                'width': width,
                'height': height,
                'num_players': num_players,
                'tile_counts': unique_tiles,
                'total_tiles': width * height
            }

        except Exception as e:
            print(f"❌ Error reading map info: {e}")
            return None
