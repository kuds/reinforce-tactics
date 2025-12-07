"""Tests for the Unit class."""
from reinforcetactics.core.unit import Unit


class TestUnitInitialization:
    """Test unit initialization with different types."""

    def test_warrior_initialization(self):
        """Test warrior unit initialization."""
        warrior = Unit('W', 5, 5, 1)
        assert warrior.type == 'W'
        assert warrior.x == 5
        assert warrior.y == 5
        assert warrior.player == 1
        assert warrior.health == 15  # From UNIT_DATA
        assert warrior.max_health == 15
        assert warrior.movement_range == 3
        assert warrior.paralyzed_turns == 0

    def test_mage_initialization(self):
        """Test mage unit initialization."""
        mage = Unit('M', 3, 3, 2)
        assert mage.type == 'M'
        assert mage.x == 3
        assert mage.y == 3
        assert mage.player == 2
        assert mage.health == 10
        assert mage.max_health == 10
        assert mage.movement_range == 2

    def test_cleric_initialization(self):
        """Test cleric unit initialization."""
        cleric = Unit('C', 4, 4, 1)
        assert cleric.type == 'C'
        assert cleric.x == 4
        assert cleric.y == 4
        assert cleric.player == 1
        assert cleric.health == 8
        assert cleric.max_health == 8
        assert cleric.movement_range == 2


class TestAttackDamage:
    """Test attack damage calculations."""

    def test_warrior_melee_damage(self, warrior):
        """Test warrior deals damage only at adjacent range."""
        # Adjacent target (distance = 1)
        damage = warrior.get_attack_damage(6, 5)
        assert damage == 10

        # Non-adjacent target (distance = 2)
        damage = warrior.get_attack_damage(7, 5)
        assert damage == 0

    def test_mage_adjacent_damage(self, mage):
        """Test mage deals adjacent damage at distance 1."""
        # Adjacent target (distance = 1)
        damage = mage.get_attack_damage(4, 3)
        assert damage == 8

    def test_mage_range_damage(self, mage):
        """Test mage deals range damage at distance 2."""
        # Range target (distance = 2)
        damage = mage.get_attack_damage(5, 3)
        assert damage == 12

    def test_mage_no_damage_at_distance_3(self, mage):
        """Test mage deals no damage at distance 3."""
        damage = mage.get_attack_damage(6, 3)
        assert damage == 0

    def test_cleric_adjacent_damage(self, cleric):
        """Test cleric deals damage only at adjacent range."""
        # Adjacent target (distance = 1)
        damage = cleric.get_attack_damage(5, 4)
        assert damage == 2

        # Non-adjacent target (distance = 2)
        damage = cleric.get_attack_damage(6, 4)
        assert damage == 0


class TestTakeDamage:
    """Test unit taking damage."""

    def test_unit_survives_damage(self, warrior):
        """Test unit survives when health remains above 0."""
        alive = warrior.take_damage(5)
        assert alive is True
        assert warrior.health == 10

    def test_unit_dies(self, warrior):
        """Test unit dies when health drops to 0 or below."""
        alive = warrior.take_damage(15)
        assert alive is False
        assert warrior.health == 0

    def test_unit_dies_from_overkill(self, mage):
        """Test unit dies when taking more damage than health."""
        alive = mage.take_damage(20)
        assert alive is False
        assert mage.health == 0


class TestParalysis:
    """Test paralysis status."""

    def test_unit_is_paralyzed(self, warrior):
        """Test unit is paralyzed when paralyzed_turns > 0."""
        warrior.paralyzed_turns = 2
        assert warrior.is_paralyzed() is True

    def test_unit_not_paralyzed(self, warrior):
        """Test unit is not paralyzed when paralyzed_turns == 0."""
        warrior.paralyzed_turns = 0
        assert warrior.is_paralyzed() is False


class TestMovement:
    """Test unit movement functionality."""

    def test_get_reachable_positions(self, warrior, mock_grid):
        """Test getting reachable positions within movement range."""
        def can_move_func(x, y):
            if not (0 <= x < mock_grid.width and 0 <= y < mock_grid.height):
                return False
            tile = mock_grid.get_tile(x, y)
            return tile.is_walkable()

        reachable = warrior.get_reachable_positions(
            mock_grid.width,
            mock_grid.height,
            can_move_func
        )

        # Warrior at (5,5) with movement 3 should have several reachable positions
        assert len(reachable) > 0
        # Check that current position is not in reachable (BFS starts with distance > 0)
        assert (5, 5) not in reachable
        # Check some expected positions are reachable
        assert (6, 5) in reachable  # Distance 1
        assert (5, 6) in reachable  # Distance 1

    def test_move_to_updates_position(self, warrior):
        """Test move_to updates unit position and flags."""
        original_x = warrior.x
        original_y = warrior.y

        warrior.move_to(7, 6)

        assert warrior.x == 7
        assert warrior.y == 6
        assert warrior.has_moved is True
        assert warrior.selected is False
        # Original position should remain unchanged
        assert warrior.original_x == original_x
        assert warrior.original_y == original_y

    def test_cancel_move_restores_position(self, warrior):
        """Test cancel_move restores original position."""
        warrior.move_to(7, 6)
        assert warrior.has_moved is True

        result = warrior.cancel_move()

        assert result is True
        assert warrior.x == 5
        assert warrior.y == 5
        assert warrior.has_moved is False

    def test_cancel_move_when_not_moved(self, warrior):
        """Test cancel_move returns False when unit hasn't moved."""
        result = warrior.cancel_move()
        assert result is False


class TestTurnManagement:
    """Test turn management."""

    def test_end_unit_turn_resets_flags(self, warrior):
        """Test end_unit_turn resets all flags."""
        warrior.can_move = True
        warrior.can_attack = True
        warrior.selected = True
        warrior.has_moved = True
        warrior.move_to(7, 6)

        warrior.end_unit_turn()

        assert warrior.can_move is False
        assert warrior.can_attack is False
        assert warrior.selected is False
        assert warrior.has_moved is False
        assert warrior.original_x == 7
        assert warrior.original_y == 6


class TestSerialization:
    """Test unit serialization and deserialization."""

    def test_to_dict(self, warrior):
        """Test unit serialization to dictionary."""
        warrior.health = 12
        warrior.paralyzed_turns = 1
        warrior.can_move = True
        warrior.can_attack = False

        data = warrior.to_dict()

        assert data['type'] == 'W'
        assert data['x'] == 5
        assert data['y'] == 5
        assert data['player'] == 1
        assert data['health'] == 12
        assert data['paralyzed_turns'] == 1
        assert data['can_move'] is True
        assert data['can_attack'] is False

    def test_from_dict(self):
        """Test unit deserialization from dictionary."""
        data = {
            'type': 'M',
            'x': 3,
            'y': 4,
            'player': 2,
            'health': 8,
            'paralyzed_turns': 2,
            'can_move': False,
            'can_attack': True
        }

        unit = Unit.from_dict(data)

        assert unit.type == 'M'
        assert unit.x == 3
        assert unit.y == 4
        assert unit.player == 2
        assert unit.health == 8
        assert unit.paralyzed_turns == 2
        assert unit.can_move is False
        assert unit.can_attack is True
