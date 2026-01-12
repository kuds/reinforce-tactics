"""Pytest configuration and shared fixtures for testing."""
import os
import pytest
import numpy as np
from reinforcetactics.core.unit import Unit
from reinforcetactics.core.tile import Tile
from reinforcetactics.core.grid import TileGrid


# ==============================================================================
# PYTEST HOOKS - Marker registration and configuration
# ==============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Fast unit tests (< 100ms)")
    config.addinivalue_line("markers", "integration: Integration tests (100ms - 2s)")
    config.addinivalue_line("markers", "slow: Long-running tests (> 2s)")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU acceleration")
    config.addinivalue_line("markers", "external: Tests requiring external services")
    config.addinivalue_line("markers", "ui: Tests requiring pygame display")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on naming conventions and skip slow tests if requested."""
    skip_slow = config.getoption("--skip-slow", default=False) if hasattr(config.option, 'skip_slow') else False
    run_external = os.environ.get("RUN_EXTERNAL_TESTS", "").lower() == "true"

    for item in items:
        # Skip slow tests if --skip-slow is passed
        if skip_slow and "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="Skipping slow test (--skip-slow)"))

        # Skip external tests unless explicitly enabled
        if "external" in item.keywords and not run_external:
            item.add_marker(pytest.mark.skip(reason="External tests disabled (set RUN_EXTERNAL_TESTS=true)"))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip tests marked as slow"
    )
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="Run tests that require external services"
    )


@pytest.fixture
def warrior():
    """Create a test warrior unit."""
    return Unit('W', 5, 5, 1)


@pytest.fixture
def mage():
    """Create a test mage unit."""
    return Unit('M', 3, 3, 1)


@pytest.fixture
def cleric():
    """Create a test cleric unit."""
    return Unit('C', 4, 4, 1)


@pytest.fixture
def archer():
    """Create a test archer unit."""
    return Unit('A', 5, 5, 1)


@pytest.fixture
def enemy_warrior():
    """Create an enemy warrior unit."""
    return Unit('W', 6, 5, 2)


@pytest.fixture
def grass_tile():
    """Create a grass tile."""
    return Tile('p', 0, 0)


@pytest.fixture
def water_tile():
    """Create a water tile."""
    return Tile('w', 1, 1)


@pytest.fixture
def tower_tile():
    """Create a tower tile owned by player 1."""
    return Tile('t_1', 2, 2)


@pytest.fixture
def hq_tile():
    """Create an HQ tile owned by player 1."""
    return Tile('h_1', 3, 3)


@pytest.fixture
def building_tile():
    """Create a building tile owned by player 2."""
    return Tile('b_2', 4, 4)


@pytest.fixture
def mock_grid():
    """Create a simple mock grid for testing."""
    # Create a 10x10 grid with grass tiles
    map_data = np.array([['p' for _ in range(10)] for _ in range(10)], dtype=object)
    # Add some water tiles
    map_data[0][0] = 'w'
    map_data[1][1] = 'w'
    return TileGrid(map_data)
