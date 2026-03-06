"""Tests for Settings manager."""

import json
import os
import tempfile

from reinforcetactics.utils.settings import Settings


class TestSettingsBasics:
    """Tests for basic Settings functionality."""

    def test_default_settings_created(self):
        """Settings creates defaults when no file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            assert s.settings is not None
            assert s.get("language") == "english"
            # File should be created
            assert os.path.exists(path)

    def test_load_existing_settings(self):
        """Settings loads from existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            data = {"language": "korean", "paths": {"maps": "custom_maps"}}
            with open(path, "w") as f:
                json.dump(data, f)
            s = Settings(settings_file=path)
            assert s.get("language") == "korean"
            assert s.get("paths.maps") == "custom_maps"

    def test_load_corrupt_file_falls_back_to_defaults(self):
        """Settings uses defaults when file is corrupt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            with open(path, "w") as f:
                f.write("{invalid json")
            s = Settings(settings_file=path)
            assert s.get("language") == "english"

    def test_merge_with_defaults(self):
        """Loaded settings are merged with defaults to fill missing keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            data = {"language": "spanish"}
            with open(path, "w") as f:
                json.dump(data, f)
            s = Settings(settings_file=path)
            # Custom value preserved
            assert s.get("language") == "spanish"
            # Default values filled in
            assert s.get("video.fullscreen") is False
            assert s.get("audio.enabled") is True


class TestSettingsGetSet:
    """Tests for get/set operations."""

    def test_get_nested_key(self):
        """get() supports dotted key paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            assert s.get("video.fps") == 60
            assert s.get("audio.music_volume") == 0.7

    def test_get_nonexistent_key_returns_default(self):
        """get() returns default for missing keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            assert s.get("nonexistent.key") is None
            assert s.get("nonexistent.key", "fallback") == "fallback"

    def test_set_creates_nested_keys(self):
        """set() creates nested dictionaries as needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            s.set("custom.nested.key", "value")
            assert s.get("custom.nested.key") == "value"

    def test_set_persists_to_file(self):
        """set() saves changes to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            s.set("language", "french")
            # Re-load from file
            with open(path) as f:
                data = json.load(f)
            assert data["language"] == "french"


class TestSettingsLanguage:
    """Tests for language get/set."""

    def test_get_language(self):
        """get_language returns current language."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            assert s.get_language() == "english"

    def test_set_language(self):
        """set_language normalizes to lowercase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            s.set_language("Korean")
            assert s.get_language() == "korean"


class TestSettingsPaths:
    """Tests for path management."""

    def test_get_path(self):
        """get_path returns configured path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            # get_path returns whatever is in the paths dict
            assert isinstance(s.get_path("maps"), str)

    def test_set_path(self):
        """set_path updates a path setting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            s.set_path("maps", "/custom/maps")
            assert s.get_path("maps") == "/custom/maps"

    def test_set_path_creates_paths_dict(self):
        """set_path creates paths dict if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            del s.settings["paths"]
            s.set_path("maps", "new_maps")
            assert s.get_path("maps") == "new_maps"

    def test_ensure_directories(self):
        """ensure_directories creates all path directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            # Point paths to subdirectories of tmpdir
            s.settings["paths"] = {
                "maps": os.path.join(tmpdir, "maps"),
                "saves": os.path.join(tmpdir, "saves"),
            }
            s.ensure_directories()
            assert os.path.isdir(os.path.join(tmpdir, "maps"))
            assert os.path.isdir(os.path.join(tmpdir, "saves"))


class TestSettingsAPIKeys:
    """Tests for API key management."""

    def test_get_api_key_default(self):
        """get_api_key returns empty string by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            assert s.get_api_key("openai") == ""

    def test_set_and_get_api_key(self):
        """set_api_key stores and retrieves API key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            s.set_api_key("openai", "sk-test-123")
            assert s.get_api_key("openai") == "sk-test-123"

    def test_get_api_key_creates_dict_if_missing(self):
        """get_api_key creates llm_api_keys dict if not present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            del s.settings["llm_api_keys"]
            assert s.get_api_key("anthropic") == ""
            assert "llm_api_keys" in s.settings

    def test_set_api_key_creates_dict_if_missing(self):
        """set_api_key creates llm_api_keys dict if not present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            del s.settings["llm_api_keys"]
            s.set_api_key("google", "goog-key")
            assert s.get_api_key("google") == "goog-key"


class TestSettingsUnits:
    """Tests for unit enable/disable management."""

    def test_get_enabled_units_default(self):
        """All units enabled by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            units = s.get_enabled_units()
            assert set(units) == {"W", "M", "C", "A", "K", "R", "S", "B"}

    def test_set_enabled_units(self):
        """set_enabled_units updates the list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            s.set_enabled_units(["W", "M"])
            assert s.get_enabled_units() == ["W", "M"]

    def test_is_unit_enabled(self):
        """is_unit_enabled checks unit status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            assert s.is_unit_enabled("W") is True
            s.set_enabled_units(["W"])
            assert s.is_unit_enabled("M") is False

    def test_toggle_unit_off(self):
        """toggle_unit disables an enabled unit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            result = s.toggle_unit("W")
            assert result is False
            assert "W" not in s.get_enabled_units()

    def test_toggle_unit_on(self):
        """toggle_unit enables a disabled unit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            s.set_enabled_units(["M"])
            result = s.toggle_unit("W")
            assert result is True
            assert "W" in s.get_enabled_units()

    def test_get_enabled_units_creates_game_dict(self):
        """get_enabled_units creates game dict if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            del s.settings["game"]
            units = s.get_enabled_units()
            assert len(units) == 8

    def test_set_enabled_units_creates_game_dict(self):
        """set_enabled_units creates game dict if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            del s.settings["game"]
            s.set_enabled_units(["W"])
            assert s.get_enabled_units() == ["W"]


class TestSettingsSpritePaths:
    """Tests for sprite path resolution."""

    def test_get_sprites_path_empty_default(self):
        """get_sprites_path returns empty string when not configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            assert s.get_sprites_path("units") == ""

    def test_get_sprites_path_override(self):
        """get_sprites_path returns per-category override when set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            s.settings["graphics"]["unit_sprites_path"] = "/custom/units"
            assert s.get_sprites_path("units") == "/custom/units"
            # Clean up: restore default to avoid mutating shared DEFAULT_SETTINGS
            s.settings["graphics"]["unit_sprites_path"] = ""

    def test_get_sprites_path_from_base(self):
        """get_sprites_path discovers subdirectory from base path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            # Ensure override is empty
            s.settings["graphics"]["unit_sprites_path"] = ""
            # Create base/units/ directory
            base = os.path.join(tmpdir, "sprites")
            units_dir = os.path.join(base, "units")
            os.makedirs(units_dir)
            s.settings["graphics"]["sprites_path"] = base
            assert s.get_sprites_path("units") == units_dir
            # Clean up
            s.settings["graphics"]["sprites_path"] = ""

    def test_get_sprites_path_animation_fallback_to_base(self):
        """get_sprites_path falls back to base for animation when no subdir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            s.settings["graphics"]["animation_sprites_path"] = ""
            s.settings["graphics"]["sprites_path"] = ""
            base = os.path.join(tmpdir, "sprites")
            os.makedirs(base)
            s.settings["graphics"]["sprites_path"] = base
            assert s.get_sprites_path("animation") == base
            s.settings["graphics"]["sprites_path"] = ""


class TestSettingsReset:
    """Tests for reset functionality."""

    def test_reset_to_defaults(self):
        """reset_to_defaults restores default settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            s.set("language", "korean")
            s.reset_to_defaults()
            assert s.get("language") == "english"

    def test_save_returns_true_on_success(self):
        """save() returns True on successful write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "settings.json")
            s = Settings(settings_file=path)
            assert s.save() is True

    def test_save_returns_false_on_error(self):
        """save() returns False when write fails."""
        s = Settings(settings_file="/nonexistent/dir/settings.json")
        # Override settings to avoid load error
        s.settings = Settings.DEFAULT_SETTINGS.copy()
        assert s.save() is False
