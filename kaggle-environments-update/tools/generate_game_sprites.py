#!/usr/bin/env python3
"""Regenerate the visualizer's game-art sprites from the main repo assets.

Writes 32x32 RGBA PNGs into
``kaggle-environments-update/kaggle_environments/envs/reinforce_tactics/
visualizer/default/src/assets/sprites/game/``:

- Units: the first idle frame of each animation sheet — frame (row 0,
  col 0) of the 64x64 grid, centre-cropped to 32x32. This matches
  ``ANIMATION_CONFIG`` in ``reinforcetactics/constants.py`` (frame size
  64, crop 32, ``frame_map["idle"][0] == (0, 0)``).
- Terrain and structures: byte-identical copies of
  ``assets/sprites/tiles/*.png``. The tower sprite is the game's
  ``city.png`` (``TILE_IMAGES["TOWER"] = "city.png"``).

Run from the reinforce-tactics repository root:

    python3 kaggle-environments-update/tools/generate_game_sprites.py

Requires Pillow.
"""

import shutil
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_UNITS = REPO_ROOT / "assets" / "sprites" / "units"
SRC_TILES = REPO_ROOT / "assets" / "sprites" / "tiles"
DST = (
    REPO_ROOT
    / "kaggle-environments-update"
    / "kaggle_environments"
    / "envs"
    / "reinforce_tactics"
    / "visualizer"
    / "default"
    / "src"
    / "assets"
    / "sprites"
    / "game"
)

FRAME_SIZE = 64
CROP_SIZE = 32

UNITS = ["warrior", "mage", "cleric", "archer", "knight", "rogue", "sorcerer", "barbarian"]

# Visualizer filename -> game tile filename. Terrain keys match the
# engine's TileType codes; the tower art is the game's city.png.
TILES = {
    "grass": "grass",
    "forest": "forest",
    "mountain": "mountain",
    "water": "water",
    "ocean": "ocean",
    "road": "road",
    "building": "building",
    "headquarters": "headquarters",
    "tower": "city",
}


def extract_idle_frame(sheet_path: Path) -> Image.Image:
    """Return the first idle frame, centre-cropped, as a fresh image.

    Rebuilding the image from raw pixel data drops any ancillary PNG
    chunks (sRGB etc.) so browsers report the exact authored RGB values
    in canvas getImageData — required for the exact-match palette swap.
    """
    sheet = Image.open(sheet_path).convert("RGBA")
    # Idle frame_map[0] is (row 0, col 0) of the 64x64 frame grid.
    frame = sheet.crop((0, 0, FRAME_SIZE, FRAME_SIZE))
    margin = (FRAME_SIZE - CROP_SIZE) // 2
    frame = frame.crop((margin, margin, margin + CROP_SIZE, margin + CROP_SIZE))
    clean = Image.new("RGBA", (CROP_SIZE, CROP_SIZE))
    clean.putdata(list(frame.getdata()))
    return clean


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)

    for unit in UNITS:
        out = DST / f"{unit}.png"
        extract_idle_frame(SRC_UNITS / f"{unit}_sheet.png").save(out)
        print(f"wrote {out.relative_to(REPO_ROOT)}")

    for dst_name, src_name in TILES.items():
        out = DST / f"{dst_name}.png"
        shutil.copyfile(SRC_TILES / f"{src_name}.png", out)
        print(f"wrote {out.relative_to(REPO_ROOT)} (from tiles/{src_name}.png)")


if __name__ == "__main__":
    main()
