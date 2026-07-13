# Kaggle Environments Update — Game Sprite Art

Staged update for the `reinforce_tactics` environment in
[Kaggle/kaggle-environments](https://github.com/Kaggle/kaggle-environments),
ready to copy into a checkout of that repository.

**Base:** upstream `master` at commit
[`e9c121a`](https://github.com/Kaggle/kaggle-environments/commit/e9c121ac432dd8fe92594c1b9f433a582c0cbfe4)
("reinforce_tactics: use seed to pick a built-in map (#1237)", 2026-06-11).
Everything in here is that snapshot plus the visualizer sprite changes
listed below. The Python interpreter, engine, agents, spec, and test
files are unchanged from upstream.

## How to Apply

Copy the two trees over a kaggle-environments checkout (this `README.md`
and `tools/` stay behind — don't copy them):

```bash
cp -r kaggle-environments-update/kaggle_environments /path/to/kaggle-environments/
cp -r kaggle-environments-update/tests /path/to/kaggle-environments/
```

The update is purely additive plus in-place file modifications — nothing
needs to be deleted when applying over upstream `master`.

> **Note (fork only):** the `kuds/kaggle-environments` fork carries an
> earlier partial sprite update (`f2add72`) that replaced the root
> placeholder tiles and added an unused
> `visualizer/default/src/assets/sprites/city.png`. Applying this update
> over the fork restores the placeholder art at the root paths (it now
> lives alongside the game art, see below) — delete the leftover
> `city.png` by hand, it was never referenced.

## What Changed vs Upstream

All changes are inside
`kaggle_environments/envs/reinforce_tactics/visualizer/default/`:

| File | Change |
| --- | --- |
| `src/assets/sprites/game/` (new) | Sprite art from the main Reinforce Tactics repository: 6 terrain tiles, 3 structures, 8 unit idle frames (all 32×32 RGBA). |
| `src/sprites.ts` | Loads both art sets, owner-aware sprite lookups, runtime palette swap for team colours, sprite-theme toggle state. |
| `src/renderer.ts` | Theme-aware rendering (full-tile team-coloured structures, player-colour unit borders, nearest-neighbour scaling for pixel art), per-theme player accent colours, art toggle button in the status bar. |
| `src/style.css` | Styling for the art toggle button. |
| `../README.md` (env README) | Documents the two art sets and the toggle. |

The original placeholder art is untouched at
`src/assets/sprites/*.png` and remains available in the visualizer via
the **Art** toggle button ("game" ⇄ "classic"). The choice persists in
`localStorage` when available.

### Team colour handling (matches the main repository)

The game art is authored in blue. Exactly like
`reinforcetactics/constants.py` + the pygame renderer in the main repo,
the visualizer recolours sprites at load time with an exact-match
palette swap of the nine `BASE_SPRITE_COLORS` blue tones:

- **Player 1** → red (`TEAM_PALETTES[1]`), accent `#ff3232` (255, 50, 50)
- **Player 2** → base blue kept (`TEAM_PALETTES[2] = None`), accent `#4d79ff` (77, 121, 255)
- **Neutral structures** (owner 0) → gray (`NEUTRAL_STRUCTURE_PALETTE`)

This also corrects the player accent colours for the game art set: the
placeholder art shipped with Player 1 blue / Player 2 red, while the
main game is Player 1 red / Player 2 blue. The classic art set keeps its
original colours.

Sprite movement/animation is intentionally out of scope: each unit is
represented by the first idle frame of its animation sheet.

## Sprite Provenance

Generated from the main repository's assets by
`tools/generate_game_sprites.py` (run it from the repo root to
regenerate):

- **Units** (`warrior.png` … `barbarian.png`): first idle frame of
  `assets/sprites/units/<unit>_sheet.png` — frame (row 0, col 0) of the
  64×64 grid, centre-cropped to 32×32, per `ANIMATION_CONFIG` in
  `reinforcetactics/constants.py`.
- **Terrain + structures**: byte-identical copies of
  `assets/sprites/tiles/*.png`. The tower sprite is the game's
  `city.png`, mirroring `TILE_IMAGES["TOWER"] = "city.png"`.

## Verification

Run inside a kaggle-environments checkout with this update applied:

- `pnpm --filter @kaggle-environments/reinforce_tactics-visualizer build` — passes (tsc + vite single-file bundle).
- `pnpm exec playwright test --project reinforce_tactics` — all 3 e2e tests pass.
- `pnpm exec prettier --check` / `pnpm exec eslint` on the changed TypeScript — clean.
- Both art modes rendered and screenshot-checked against the test replay, including the toggle round-trip.
