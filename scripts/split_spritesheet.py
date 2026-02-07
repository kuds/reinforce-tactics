#!/usr/bin/env python3
"""
Split a PNG image into a grid of 32x32 tiles.

Usage:
    python scripts/split_spritesheet.py <input.png> [--output <dir>] [--size <px>]

Examples:
    python scripts/split_spritesheet.py knight.png
    python scripts/split_spritesheet.py tileset.png --output tiles/ --size 32

Output files are named {stem}_{row}_{col}.png (0-indexed).
"""
import argparse
import os
import sys

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


def split_image(input_path, output_dir, tile_size=32):
    img = Image.open(input_path)
    width, height = img.size

    if width % tile_size != 0:
        print(
            f"Error: image width {width}px is not divisible by {tile_size}. "
            f"Remainder: {width % tile_size}px",
            file=sys.stderr,
        )
        sys.exit(1)

    if height % tile_size != 0:
        print(
            f"Error: image height {height}px is not divisible by {tile_size}. "
            f"Remainder: {height % tile_size}px",
            file=sys.stderr,
        )
        sys.exit(1)

    cols = width // tile_size
    rows = height // tile_size
    stem = os.path.splitext(os.path.basename(input_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for row in range(rows):
        for col in range(cols):
            x = col * tile_size
            y = row * tile_size
            tile = img.crop((x, y, x + tile_size, y + tile_size))
            out_path = os.path.join(output_dir, f"{stem}_{row}_{col}.png")
            tile.save(out_path)
            count += 1

    print(f"Split {width}x{height} into {rows} rows x {cols} cols = {count} tiles ({tile_size}x{tile_size} each)")
    print(f"Output: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Split a PNG image into a grid of 32x32 tiles."
    )
    parser.add_argument("input", help="Path to input PNG file")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: {input_stem}_tiles/)",
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=32,
        help="Tile size in pixels (default: 32)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output
    if output_dir is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        output_dir = f"{stem}_tiles"

    split_image(args.input, output_dir, tile_size=args.size)


if __name__ == "__main__":
    main()
