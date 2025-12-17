"""Tests for the map preview generator."""
import os
import unittest
import pygame

from reinforcetactics.ui.components.map_preview import MapPreviewGenerator


class TestMapPreviewGenerator(unittest.TestCase):
    """Test map preview generation functionality."""

    @classmethod
    def setUpClass(cls):
        """Initialize pygame before tests."""
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        """Clean up pygame after tests."""
        pygame.quit()

    def setUp(self):
        """Set up test fixtures."""
        self.generator = MapPreviewGenerator()
        self.test_map_path = "maps/1v1/beginner.csv"

    def test_format_display_name(self):
        """Test display name formatting."""
        test_cases = [
            ("beginner.csv", "Beginner"),
            ("center_mountains.csv", "Center Mountains"),
            ("corner_points.csv", "Corner Points"),
            ("difficult_terrain.csv", "Difficult Terrain"),
            ("funnel_point.csv", "Funnel Point"),
        ]

        for filename, expected in test_cases:
            result = self.generator._format_display_name(filename)
            self.assertEqual(result, expected, f"Failed for {filename}")

    def test_generate_preview_random(self):
        """Test preview generation for random map."""
        preview, metadata = self.generator.generate_preview("random", 150, 150)

        self.assertIsNotNone(preview)
        self.assertIsInstance(preview, pygame.Surface)
        self.assertEqual(preview.get_width(), 150)
        self.assertEqual(preview.get_height(), 150)
        self.assertEqual(metadata['name'], "Random Map")

    def test_generate_preview_real_map(self):
        """Test preview generation for real map file."""
        if not os.path.exists(self.test_map_path):
            self.skipTest(f"Test map not found: {self.test_map_path}")

        preview, metadata = self.generator.generate_preview(self.test_map_path, 150, 150)

        self.assertIsNotNone(preview)
        self.assertIsInstance(preview, pygame.Surface)
        self.assertEqual(preview.get_width(), 150)
        self.assertEqual(preview.get_height(), 150)

        # Check metadata
        self.assertIn('name', metadata)
        self.assertIn('width', metadata)
        self.assertIn('height', metadata)
        self.assertIn('player_count', metadata)
        self.assertIn('terrain_breakdown', metadata)
        self.assertIn('difficulty', metadata)

        # For beginner map
        self.assertEqual(metadata['name'], "Beginner")
        self.assertGreater(metadata['width'], 0)
        self.assertGreater(metadata['height'], 0)
        self.assertGreater(metadata['player_count'], 0)

    def test_cache_functionality(self):
        """Test that preview caching works."""
        if not os.path.exists(self.test_map_path):
            self.skipTest(f"Test map not found: {self.test_map_path}")

        # Generate preview twice
        preview1, metadata1 = self.generator.generate_preview(self.test_map_path, 100, 100)
        preview2, metadata2 = self.generator.generate_preview(self.test_map_path, 100, 100)

        # Should return same objects from cache
        self.assertIs(preview1, preview2)
        self.assertIs(metadata1, metadata2)

    def test_clear_cache(self):
        """Test cache clearing."""
        if not os.path.exists(self.test_map_path):
            self.skipTest(f"Test map not found: {self.test_map_path}")

        # Generate and cache preview
        self.generator.generate_preview(self.test_map_path, 100, 100)
        self.assertGreater(len(self.generator._cache), 0)

        # Clear cache
        self.generator.clear_cache()
        self.assertEqual(len(self.generator._cache), 0)

    def test_difficulty_calculation(self):
        """Test difficulty calculation based on map size."""
        test_cases = [
            (6, 6, "Beginner"),     # 36 tiles
            (10, 10, "Easy"),       # 100 tiles
            (14, 14, "Medium"),     # 196 tiles
            (24, 24, "Medium"),     # 576 tiles
            (32, 32, "Expert"),     # 1024 tiles
        ]

        for width, height, expected_difficulty in test_cases:
            difficulty = self.generator._calculate_difficulty(width, height, {})
            self.assertEqual(difficulty, expected_difficulty,
                           f"Failed for {width}×{height}")


if __name__ == '__main__':
    unittest.main()
