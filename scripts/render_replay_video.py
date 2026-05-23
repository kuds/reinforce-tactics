"""Render a single replay JSON to an MP4.

Designed to be invoked as a one-shot subprocess so the host process
doesn't accumulate pygame surfaces and frame buffers across many
renders (which OOMs in Colab when rendering dozens of tournament games).
"""

import argparse
import sys

from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.utils.video import record_replay_to_video


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_path", help="Path to replay JSON")
    parser.add_argument("output_path", help="Path to write MP4")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument(
        "--pixel-art",
        action="store_true",
        help="Render with pixel-art sprites instead of flat tiles",
    )
    args = parser.parse_args()

    replay_data = FileIO.load_replay(args.replay_path)
    if replay_data is None:
        print(f"Failed to load replay {args.replay_path}", file=sys.stderr)
        return 1

    game_info = replay_data.setdefault("game_info", {})
    if "initial_map" not in game_info and game_info.get("map"):
        meta = FileIO.load_map_with_metadata(game_info["map"])
        game_info["initial_map"] = meta["original_map_data"]

    record_replay_to_video(
        replay_data,
        output_path=args.output_path,
        fps=args.fps,
        scale=args.scale,
        use_pixel_art=args.pixel_art,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
