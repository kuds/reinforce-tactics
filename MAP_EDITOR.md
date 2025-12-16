# Map Editor

The Map Editor allows you to create and modify game maps for Reinforce Tactics.

## Features

- **Create New Maps**: Design maps from scratch with custom dimensions (minimum 20×20)
- **Edit Existing Maps**: Load and modify existing maps from the maps directory
- **Tile Palette**: Select from 6 terrain types and 3 structure types
- **Player Ownership**: Assign buildings, towers, and headquarters to specific players (1-4)
- **Visual Editing**: Click to paint, right-click to erase, with grid overlay
- **Map Validation**: Ensures each player has exactly one headquarters before saving
- **Multi-language Support**: Available in English, French, and Korean

## Access

From the main menu, select **"Map Editor"** to access:
- **New Map**: Create a new map with custom dimensions
- **Edit Existing Map**: Load and modify an existing map

## Controls

### Mouse
- **Left Click**: Paint selected tile
- **Right Click**: Erase (replace with grass)
- **Mouse Wheel**: Scroll/pan the map
- **Hover**: View tile coordinates

### Keyboard Shortcuts
- **Ctrl+S**: Save map
- **G**: Toggle grid display
- **+/-**: Zoom in/out
- **1-9**: Quick select tile types
  - 1: Grass
  - 2: Ocean
  - 3: Water
  - 4: Mountain
  - 5: Forest
  - 6: Road
  - 7: Tower
  - 8: Building
  - 9: Headquarters
- **Esc**: Exit editor

## Tile Types

### Terrain
- **Grass** (p): Walkable terrain
- **Ocean** (o): Not walkable
- **Water** (w): Not walkable
- **Mountain** (m): Not walkable
- **Forest** (f): Walkable
- **Road** (r): Walkable

### Structures
- **Tower** (t): Neutral or player-owned, generates $50/turn
- **Building** (b): Neutral or player-owned, generates $100/turn
- **Headquarters** (h): Neutral or player-owned, generates $150/turn (required: 1 per player)

## Map Format

Maps are saved as CSV files with tile codes:
- Single-letter codes for terrain: `p`, `o`, `w`, `m`, `f`, `r`
- Neutral structures (no owner): `t`, `b`, `h`
- Owner-specific codes for structures: `h_1`, `b_2`, `t_3`

Example:
```
o,o,o,o,o
o,p,p,p,o
o,p,h_1,p,o
o,b,t,b,o
o,o,o,o,o
```

In this example, `h_1` is Player 1's headquarters, while `b` and `t` are neutral structures that can be captured by any player during gameplay.

## Map Requirements

For a map to be valid:
1. Minimum size: 20×20 tiles
2. Each player must have exactly one headquarters (`h_1`, `h_2`, etc.)
3. Maps are automatically saved to `maps/1v1/` (2 players) or `maps/2v2/` (4 players)

## Tips

- Start with a base terrain (ocean or grass) and build up from there
- Place headquarters in corners for balanced gameplay
- Add neutral towers in strategic central locations
- Use terrain variety (mountains, forests) to create interesting tactical choices
- Test your maps by playing them!
