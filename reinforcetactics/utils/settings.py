"""
Settings manager for game configuration
"""
import json
import os
from pathlib import Path


class Settings:
    """Manages game settings with persistence."""

    DEFAULT_SETTINGS = {
        'language': 'english',
        'paths': {
            'maps': 'maps',
            'videos': 'videos',
            'replays': 'replays',
            'saves': 'saves',
            'models': 'models'
        },
        'video': {
            'fullscreen': False,
            'resolution': [900, 700],
            'fps': 60
        },
        'audio': {
            'music_volume': 0.7,
            'sfx_volume': 0.8,
            'enabled': True
        },
        'graphics': {
            'unit_sprites_path': '',
            'tile_sprites_path': '',
            'use_unit_sprites': False,
            'use_tile_sprites': False
        },
        'llm_api_keys': {
            'openai': '',
            'anthropic': '',
            'google': ''
        }
    }

    def __init__(self, settings_file='settings.json'):
        """Initialize settings manager."""
        self.settings_file = settings_file
        self.settings = self.load()

    def load(self):
        """Load settings from file or create defaults."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return self._merge_with_defaults(loaded_settings)
            except Exception as e:
                print(f"⚠️  Error loading settings: {e}")
                print("Using default settings")
                return self.DEFAULT_SETTINGS.copy()
        else:
            print("No settings file found, creating defaults...")
            settings = self.DEFAULT_SETTINGS.copy()
            self.save(settings)
            return settings

    def _merge_with_defaults(self, loaded):
        """Merge loaded settings with defaults to ensure all keys exist."""
        result = self.DEFAULT_SETTINGS.copy()

        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict):
                # Merge nested dicts
                result[key].update(value)
            else:
                result[key] = value

        return result

    def save(self, settings=None):
        """Save settings to file."""
        if settings is not None:
            self.settings = settings

        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2)
            print(f"✅ Settings saved to {self.settings_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving settings: {e}")
            return False

    def get(self, key, default=None):
        """Get a setting value."""
        keys = key.split('.')
        value = self.settings

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key, value):
        """Set a setting value."""
        keys = key.split('.')
        settings = self.settings

        for k in keys[:-1]:
            if k not in settings:
                settings[k] = {}
            settings = settings[k]

        settings[keys[-1]] = value
        self.save()

    def get_language(self):
        """Get current language."""
        return self.settings.get('language', 'english')

    def set_language(self, language):
        """Set language."""
        self.settings['language'] = language.lower()
        self.save()

    def get_path(self, path_type):
        """Get a configured path."""
        return self.settings['paths'].get(path_type, path_type)

    def set_path(self, path_type, path):
        """Set a configured path."""
        if 'paths' not in self.settings:
            self.settings['paths'] = {}
        self.settings['paths'][path_type] = path
        self.save()

    def ensure_directories(self):
        """Create all configured directories if they don't exist."""
        for _path_type, path in self.settings['paths'].items():
            Path(path).mkdir(parents=True, exist_ok=True)
        print("✅ All configured directories created")

    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.save()
        print("✅ Settings reset to defaults")

    def get_api_key(self, provider):
        """
        Get API key for a specific LLM provider.

        Args:
            provider: Provider name ('openai', 'anthropic', 'google')

        Returns:
            API key string or empty string if not set
        """
        if 'llm_api_keys' not in self.settings:
            self.settings['llm_api_keys'] = {}
        return self.settings['llm_api_keys'].get(provider, '')

    def set_api_key(self, provider, api_key):
        """
        Set API key for a specific LLM provider.

        Args:
            provider: Provider name ('openai', 'anthropic', 'google')
            api_key: API key string
        """
        if 'llm_api_keys' not in self.settings:
            self.settings['llm_api_keys'] = {}
        self.settings['llm_api_keys'][provider] = api_key
        self.save()


# Global settings instance
_settings_instance = None  # pylint: disable=invalid-name

def get_settings():
    """Get global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
