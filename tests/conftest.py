"""Pytest configuration and shared fixtures for testing."""
import pytest
import numpy as np
from reinforcetactics.core.unit import Unit
from reinforcetactics.core.tile import Tile
from reinforcetactics.core.grid import TileGrid


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
