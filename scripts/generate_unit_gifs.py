#!/usr/bin/env python3
"""
Generate animated GIF idle sprites from unit sprite sheets.

Extracts the idle animation frames from each unit's sprite sheet and
creates animated GIFs suitable for the documentation website.

Usage:
    python scripts/generate_unit_gifs.py <sprites_dir> [--output <output_dir>] [--scale <factor>]

Example:
    python scripts/generate_unit_gifs.py ./my_sprites --output docs-site/static/img/units --scale 3

Sprite sheet format (6 columns x 5 rows of 64x64 frames, centre-cropped to 32x32):
    Idle frames are the first 4 cells: [0,0] [0,1] [0,2] [0,3]
"""
import argparse
import os
import sys

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

# Unit names matching the animation_path values in constants.py
UNIT_NAMES = [
    'warrior', 'mage', 'cleric', 'archer',
    'knight', 'rogue', 'sorcerer', 'barbarian',
]

# Idle frame coordinates (row, col) from ANIMATION_CONFIG
IDLE_FRAMES = [(0, 0), (0, 1), (0, 2), (0, 3)]

# Source frame size on the sprite sheet
FRAME_WIDTH = 64
FRAME_HEIGHT = 64

# Centre-crop to this size before scaling
CROP_WIDTH = 32
CROP_HEIGHT = 32


def find_sprite_sheet(sprites_dir, unit_name):
    """Find the sprite sheet file for a unit, trying common naming patterns."""
    candidates = [
        f"{unit_name}_sheet.png",
        f"{unit_name}_spritesheet.png",
        f"{unit_name}.png",
    ]
    for name in candidates:
        path = os.path.join(sprites_dir, name)
        if os.path.exists(path):
            return path
    return None


def extract_idle_frames(sheet_path, scale=1):
    """
    Extract idle animation frames from a sprite sheet.

    Args:
        sheet_path: Path to the sprite sheet PNG
        scale: Integer scale factor (e.g. 3 = 96x96 output)

    Returns:
        List of PIL Image frames
    """
    sheet = Image.open(sheet_path).convert('RGBA')
    frames = []

    for row, col in IDLE_FRAMES:
        x = col * FRAME_WIDTH
        y = row * FRAME_HEIGHT
        box = (x, y, x + FRAME_WIDTH, y + FRAME_HEIGHT)

        if x + FRAME_WIDTH > sheet.width or y + FRAME_HEIGHT > sheet.height:
            print(f"  Warning: frame ({row},{col}) out of bounds, skipping")
            continue

        frame = sheet.crop(box)

        # Centre-crop from source size to crop size
        if CROP_WIDTH < FRAME_WIDTH or CROP_HEIGHT < FRAME_HEIGHT:
            cx = (FRAME_WIDTH - CROP_WIDTH) // 2
            cy = (FRAME_HEIGHT - CROP_HEIGHT) // 2
            frame = frame.crop((cx, cy, cx + CROP_WIDTH, cy + CROP_HEIGHT))

        if scale > 1:
            new_size = (CROP_WIDTH * scale, CROP_HEIGHT * scale)
            frame = frame.resize(new_size, Image.NEAREST)

        frames.append(frame)

    return frames


def create_gif(frames, output_path, frame_duration_ms=200):
    """
    Create an animated GIF from a list of PIL Image frames.

    Args:
        frames: List of PIL Image objects
        output_path: Where to save the GIF
        frame_duration_ms: Milliseconds per frame
    """
    if not frames:
        return False

    # Convert RGBA frames to a format suitable for GIF
    # GIF doesn't support full alpha; use a transparency index
    gif_frames = []
    for frame in frames:
        # Create a palette-based image with transparency
        # Use a magenta background as the transparent colour
        bg = Image.new('RGBA', frame.size, (0, 0, 0, 0))
        bg.paste(frame, (0, 0), frame)

        # Convert to palette mode, preserving transparency
        quantized = bg.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        gif_frames.append(quantized)

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=frame_duration_ms,
        loop=0,  # Loop forever
        transparency=0,
        disposal=2,  # Clear frame before drawing next
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate animated GIF idle sprites from unit sprite sheets.'
    )
    parser.add_argument(
        'sprites_dir',
        help='Directory containing unit sprite sheet PNG files',
    )
    parser.add_argument(
        '--output', '-o',
        default='docs-site/static/img/units',
        help='Output directory for generated GIFs (default: docs-site/static/img/units)',
    )
    parser.add_argument(
        '--scale', '-s',
        type=int, default=3,
        help='Scale factor for output GIFs (default: 3, producing 96x96 GIFs)',
    )
    parser.add_argument(
        '--duration', '-d',
        type=int, default=200,
        help='Frame duration in milliseconds (default: 200)',
    )
    args = parser.parse_args()

    if not os.path.isdir(args.sprites_dir):
        print(f"Error: sprites directory not found: {args.sprites_dir}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    generated = 0
    skipped = 0

    for unit_name in UNIT_NAMES:
        sheet_path = find_sprite_sheet(args.sprites_dir, unit_name)
        if not sheet_path:
            print(f"  [{unit_name}] sprite sheet not found, skipping")
            skipped += 1
            continue

        print(f"  [{unit_name}] extracting idle frames from {os.path.basename(sheet_path)}")
        frames = extract_idle_frames(sheet_path, scale=args.scale)

        if not frames:
            print(f"  [{unit_name}] no frames extracted, skipping")
            skipped += 1
            continue

        output_path = os.path.join(args.output, f"{unit_name}_idle.gif")
        if create_gif(frames, output_path, frame_duration_ms=args.duration):
            print(f"  [{unit_name}] -> {output_path} ({len(frames)} frames, {args.scale}x scale)")
            generated += 1
        else:
            print(f"  [{unit_name}] failed to create GIF")
            skipped += 1

    print(f"\nDone: {generated} GIFs generated, {skipped} skipped")


if __name__ == '__main__':
    main()
