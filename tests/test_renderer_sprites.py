"""The game draws the bundled pixel art by default, with feet-anchored unit sprites.

A fresh install (no ``settings.json``) used to load no sprites: the sprite
paths defaulted to empty, so ``Renderer(game)`` drew coloured tiles and unit
letters. With no path configured the renderer now falls back to the bundled
``assets/sprites/`` and draws tile sprites; ``graphics.pixel_art`` off keeps
the letter mode.

Each 64x64 sheet frame used to be centre-cropped to 32x32, which cut off every
unit's feet and drop shadow (they end at y=51) and some weapons and hats. The
crop is now the 48x48 at (8, 4), drawn with its bottom edge on the tile's
bottom edge so it overflows the tile upward and sideways, and units are drawn
top to bottom so overlaps stack correctly.
"""

import json

import numpy as np
import pygame
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.ui import theme
from reinforcetactics.ui.assets import ANIMATION_CONFIG, TILE_SIZE, UNIT_ASSETS
from reinforcetactics.ui.renderer import Renderer, _resolve_bundled_sprites_path
from reinforcetactics.ui.sprite_animator import scale_unit_sprite
from reinforcetactics.utils import settings as settings_module


def _map():
    grid = np.array([["p"] * 10 for _ in range(10)], dtype=object)
    grid[0][0] = "h_1"
    grid[9][9] = "h_2"
    return grid


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """A fresh settings file (no settings.json yet), with a dummy display."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.chdir(tmp_path)
    fresh = settings_module.Settings(str(tmp_path / "settings.json"))
    monkeypatch.setattr(settings_module, "_settings_instance", fresh)
    pygame.init()
    yield fresh
    pygame.quit()


@pytest.fixture
def game():
    return GameState(_map(), num_players=2)


def _sheet_frames(unit_type):
    """(state, source 64x64 frame) for every frame of a unit's bundled sheet."""
    name = UNIT_ASSETS[unit_type]["animation_path"]
    sheet = pygame.image.load(f"{_resolve_bundled_sprites_path()}/units/{name}_sheet.png")
    fw, fh = ANIMATION_CONFIG["frame_width"], ANIMATION_CONFIG["frame_height"]
    for state, coords in ANIMATION_CONFIG["frame_map"].items():
        for row, col in coords:
            yield state, sheet.subsurface(pygame.Rect(col * fw, row * fh, fw, fh))


class TestBundledArtByDefault:
    def test_a_fresh_install_draws_tile_and_unit_sprites(self, settings, game):
        renderer = Renderer(game, headless=True)

        assert set(renderer.tile_variants) >= {"GRASS", "HEADQUARTERS", "OCEAN"}
        assert renderer.animator is not None
        assert all(renderer.animator.has_animations(t) for t in UNIT_ASSETS)

    def test_the_default_game_screen_shows_no_unit_letters(self, settings, game, monkeypatch):
        game.place_unit("W", 4, 4, 1)
        renderer = Renderer(game, headless=True)
        letters = []
        monkeypatch.setattr(renderer, "_draw_unit_letter", letters.append)

        renderer.render()

        assert letters == []

    def test_pixel_art_off_in_settings_draws_letters(self, settings, game, monkeypatch):
        settings.settings["graphics"]["pixel_art"] = False
        game.place_unit("W", 4, 4, 1)
        renderer = Renderer(game, headless=True)
        letters = []
        monkeypatch.setattr(renderer, "_draw_unit_letter", letters.append)

        renderer.render()

        assert renderer.tile_variants == {}
        assert renderer.animator is None
        assert len(letters) == 1

    def test_a_configured_sprites_path_wins_over_the_bundled_art(self, settings, game, tmp_path):
        custom = tmp_path / "my_sprites"
        (custom / "units").mkdir(parents=True)
        (custom / "tiles").mkdir()
        settings.settings["graphics"]["sprites_path"] = str(custom)

        renderer = Renderer(game, headless=True)

        assert renderer._resolve_sprites_path("units") == str(custom / "units")
        assert renderer.tile_variants == {}
        assert not renderer.animator.has_animations("W")

    def test_custom_static_unit_sprites_are_not_hidden_by_the_bundled_sheets(self, settings, game, tmp_path):
        custom = tmp_path / "static_units"
        custom.mkdir()
        pygame.image.save(pygame.Surface((TILE_SIZE, TILE_SIZE)), str(custom / UNIT_ASSETS["W"]["static_path"]))
        settings.settings["graphics"]["unit_sprites_path"] = str(custom)

        renderer = Renderer(game, headless=True)

        assert renderer.unit_images["W"] is not None
        assert renderer.animator is None
        assert "GRASS" in renderer.tile_variants

    def test_tile_sprites_off_keeps_the_bundled_units(self, settings, game):
        settings.settings["graphics"]["use_tile_sprites"] = False

        renderer = Renderer(game, headless=True)

        assert renderer.tile_variants == {}
        assert renderer.animator.has_animations("W")


class TestSettingsDefaults:
    def test_pixel_art_and_tile_sprites_default_on(self, tmp_path):
        s = settings_module.Settings(str(tmp_path / "settings.json"))

        assert s.get("graphics.pixel_art") is True
        assert s.get("graphics.use_tile_sprites") is True

    @staticmethod
    def _load(tmp_path, graphics):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"graphics": graphics}))
        return settings_module.Settings(str(path))

    def test_the_old_saved_default_no_longer_turns_tile_sprites_off(self, tmp_path):
        # Every settings.json saved before this change carries the old default,
        # which had no visible effect without a sprites path.
        s = self._load(tmp_path, {"sprites_path": "", "tile_sprites_path": "", "use_tile_sprites": False})

        assert s.get("graphics.use_tile_sprites") is True

    def test_tile_sprites_off_with_a_custom_path_is_kept(self, tmp_path):
        s = self._load(tmp_path, {"sprites_path": "/my/sprites", "use_tile_sprites": False})

        assert s.get("graphics.use_tile_sprites") is False

    def test_tile_sprites_turned_off_since_this_change_are_kept(self, tmp_path):
        s = self._load(tmp_path, {"pixel_art": True, "sprites_path": "", "use_tile_sprites": False})

        assert s.get("graphics.use_tile_sprites") is False


class TestFeetAnchoredCrop:
    @pytest.mark.parametrize("unit_type", sorted(UNIT_ASSETS))
    def test_no_frame_loses_a_pixel(self, settings, game, unit_type):
        crop = pygame.Rect(ANIMATION_CONFIG["crop"])
        for state, source in _sheet_frames(unit_type):
            assert crop.contains(source.get_bounding_rect()), (unit_type, state)

    def test_frames_keep_the_pixel_scale_of_the_tiles(self, settings, game):
        renderer = Renderer(game, headless=True)
        crop = pygame.Rect(ANIMATION_CONFIG["crop"])

        for unit_type in UNIT_ASSETS:
            frames = renderer.animator.sprite_sheets[unit_type]
            assert {f.get_size() for state in frames.values() for f in state} == {crop.size}

            source = next(_sheet_frames(unit_type))[1]
            expected = source.subsurface(crop)
            got = frames["idle"][0]
            assert pygame.image.tobytes(got, "RGBA") == pygame.image.tobytes(expected, "RGBA"), unit_type

    def test_sheet_frames_are_never_smoothscaled(self):
        # A two-colour checkerboard: nearest-neighbour keeps exactly those two
        # colours at any ratio, smoothscale would blend them.
        src = pygame.Surface((48, 48), pygame.SRCALPHA)
        for y in range(48):
            for x in range(48):
                src.set_at((x, y), (255, 0, 0, 255) if (x + y) % 2 else (0, 0, 255, 255))

        assert scale_unit_sprite(src, 48, nearest=True) is src
        scaled = scale_unit_sprite(src, (72, 72), nearest=True)
        colours = {tuple(scaled.get_at((x, y))) for y in range(72) for x in range(72)}
        assert colours == {(255, 0, 0, 255), (0, 0, 255, 255)}


class TestUnitPlacementAndOrder:
    def test_the_sprite_stands_on_the_tile_bottom_and_overflows_up_and_sideways(self, settings, game):
        unit = game.place_unit("W", 4, 4, 1)
        renderer = Renderer(game, headless=True)
        colour = (1, 254, 3, 255)
        sprite = pygame.Surface((48, 48), pygame.SRCALPHA)
        sprite.fill(colour)
        renderer.screen.fill((0, 0, 0))

        renderer._draw_unit_sprite(unit, sprite)

        left, top = unit.x * TILE_SIZE, unit.y * TILE_SIZE
        bottom, centre = top + TILE_SIZE, left + TILE_SIZE // 2
        at = renderer.screen.get_at
        assert tuple(at((centre, bottom - 1))) == colour
        assert tuple(at((centre, bottom))) != colour
        assert tuple(at((centre, bottom - 48))) == colour
        assert tuple(at((centre, bottom - 49))) != colour
        assert tuple(at((centre - 24, bottom - 10))) == colour
        assert tuple(at((centre + 23, bottom - 10))) == colour

    def test_units_are_drawn_top_to_bottom_with_status_after_every_sprite(self, settings, game, monkeypatch):
        game.place_unit("W", 5, 6, 1)
        game.place_unit("M", 3, 2, 2)
        game.place_unit("C", 1, 6, 1)
        game.place_unit("K", 3, 3, 2)
        renderer = Renderer(game, headless=True)
        calls = []
        monkeypatch.setattr(renderer, "_draw_unit", lambda u: calls.append(("body", u.x, u.y)))
        monkeypatch.setattr(renderer, "_draw_unit_status", lambda u: calls.append(("status", u.x, u.y)))

        renderer._draw_units()

        order = [(3, 2), (3, 3), (1, 6), (5, 6)]
        assert calls == [("body", *p) for p in order] + [("status", *p) for p in order]

    def test_a_health_bar_is_not_covered_by_the_unit_below(self, settings, game):
        upper = game.place_unit("W", 4, 4, 1)
        game.place_unit("B", 4, 5, 2)
        renderer = Renderer(game, headless=True)
        # The lower unit is a solid 48x48 sprite, so its top 16 rows cover
        # the bottom half of the tile above, where the upper unit's bar sits.
        renderer.animator = None
        tall = pygame.Surface((48, 48), pygame.SRCALPHA)
        tall.fill((1, 254, 3, 255))
        renderer.unit_images["B"] = tall

        renderer.render()

        bar_y = upper.y * TILE_SIZE + TILE_SIZE - theme.HEALTH_BAR_UNIT_HEIGHT - theme.HEALTH_BAR_MARGIN
        mid = (upper.x * TILE_SIZE + TILE_SIZE // 2, bar_y + theme.HEALTH_BAR_UNIT_HEIGHT // 2)
        assert tuple(renderer.screen.get_at(mid))[:3] == tuple(theme.HEALTH_GOOD)
