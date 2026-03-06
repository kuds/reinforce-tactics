"""Tests for LLM prompt system."""

import pytest

from reinforcetactics.game.llm_prompts import (
    DEFAULT_PROMPT,
    PROMPT_BASIC,
    PROMPT_REGISTRY,
    PROMPT_STRATEGIC,
    PROMPT_TWO_PHASE_EXECUTE,
    PROMPT_TWO_PHASE_PLAN,
    UNIT_DESCRIPTIONS,
    UNIT_DESCRIPTIONS_SHORT,
    UNIT_STRATEGY_TIPS,
    get_available_actions_section,
    get_dynamic_prompt,
    get_prompt,
    get_strategy_tips,
    get_unit_types_section,
    get_unit_types_section_short,
    list_prompts,
    register_prompt,
)


class TestPromptRegistry:
    """Tests for prompt registry functions."""

    def test_get_prompt_basic(self):
        """get_prompt('basic') returns PROMPT_BASIC."""
        assert get_prompt("basic") is PROMPT_BASIC

    def test_get_prompt_strategic(self):
        """get_prompt('strategic') returns PROMPT_STRATEGIC."""
        assert get_prompt("strategic") is PROMPT_STRATEGIC

    def test_get_prompt_two_phase_plan(self):
        """get_prompt returns two-phase plan prompt."""
        assert get_prompt("two_phase_plan") is PROMPT_TWO_PHASE_PLAN

    def test_get_prompt_two_phase_execute(self):
        """get_prompt returns two-phase execute prompt."""
        assert get_prompt("two_phase_execute") is PROMPT_TWO_PHASE_EXECUTE

    def test_get_prompt_unknown_raises(self):
        """get_prompt raises ValueError for unknown prompt name."""
        with pytest.raises(ValueError, match="Unknown prompt"):
            get_prompt("nonexistent")

    def test_list_prompts(self):
        """list_prompts returns all registered prompt names."""
        names = list_prompts()
        assert "basic" in names
        assert "strategic" in names
        assert "two_phase_plan" in names
        assert "two_phase_execute" in names

    def test_register_prompt(self):
        """register_prompt adds a custom prompt."""
        register_prompt("test_custom", "Custom prompt text")
        assert get_prompt("test_custom") == "Custom prompt text"
        # Cleanup
        del PROMPT_REGISTRY["test_custom"]

    def test_default_prompt_is_basic(self):
        """DEFAULT_PROMPT should be PROMPT_BASIC."""
        assert DEFAULT_PROMPT is PROMPT_BASIC


class TestPromptContent:
    """Tests for prompt content correctness."""

    def test_basic_prompt_contains_game_rules(self):
        """Basic prompt should contain key game information."""
        assert "GAME OBJECTIVE" in PROMPT_BASIC
        assert "UNIT TYPES" in PROMPT_BASIC
        assert "AVAILABLE ACTIONS" in PROMPT_BASIC
        assert "COMBAT RULES" in PROMPT_BASIC

    def test_strategic_prompt_contains_tactical_planning(self):
        """Strategic prompt should contain multi-step tactical planning."""
        assert "MULTI-STEP TACTICAL PLANNING" in PROMPT_STRATEGIC
        assert "ELIMINATION" in PROMPT_STRATEGIC
        assert "ACTION ORDERING" in PROMPT_STRATEGIC

    def test_two_phase_plan_contains_planning_sections(self):
        """Two-phase plan prompt should contain planning-specific sections."""
        assert "STRATEGIC PLANNING" in PROMPT_TWO_PHASE_PLAN
        assert "IMMEDIATE OPPORTUNITIES" in PROMPT_TWO_PHASE_PLAN
        assert "situation_assessment" in PROMPT_TWO_PHASE_PLAN

    def test_two_phase_execute_contains_plan_placeholder(self):
        """Two-phase execute prompt should contain {plan} placeholder."""
        assert "{plan}" in PROMPT_TWO_PHASE_EXECUTE


class TestUnitDescriptions:
    """Tests for unit description dictionaries."""

    def test_all_unit_types_have_descriptions(self):
        """All 8 unit types should have descriptions."""
        expected_types = {"W", "M", "C", "A", "K", "R", "S", "B"}
        assert set(UNIT_DESCRIPTIONS.keys()) == expected_types

    def test_all_unit_types_have_short_descriptions(self):
        """All 8 unit types should have short descriptions."""
        expected_types = {"W", "M", "C", "A", "K", "R", "S", "B"}
        assert set(UNIT_DESCRIPTIONS_SHORT.keys()) == expected_types

    def test_strategy_tips_for_special_units(self):
        """Strategy tips exist for units with special abilities."""
        assert "M" in UNIT_STRATEGY_TIPS  # Mage
        assert "C" in UNIT_STRATEGY_TIPS  # Cleric
        assert "A" in UNIT_STRATEGY_TIPS  # Archer
        assert "K" in UNIT_STRATEGY_TIPS  # Knight
        assert "R" in UNIT_STRATEGY_TIPS  # Rogue
        assert "S" in UNIT_STRATEGY_TIPS  # Sorcerer
        assert "B" in UNIT_STRATEGY_TIPS  # Barbarian


class TestDynamicPromptGeneration:
    """Tests for dynamic prompt generation functions."""

    def test_get_unit_types_section_all_units(self):
        """Full unit types section includes all units."""
        section = get_unit_types_section(["W", "M", "C", "A", "K", "R", "S", "B"])
        assert "UNIT TYPES:" in section
        assert "Warrior" in section
        assert "Mage" in section
        assert "Barbarian" in section

    def test_get_unit_types_section_subset(self):
        """Unit types section with subset only includes selected units."""
        section = get_unit_types_section(["W", "A"])
        assert "Warrior" in section
        assert "Archer" in section
        # Cleric should not appear as a numbered entry
        assert "Cleric (C):" not in section

    def test_get_unit_types_section_empty_defaults_to_all(self):
        """Empty enabled_units list defaults to all units."""
        section = get_unit_types_section([])
        assert "Warrior" in section
        assert "Barbarian" in section

    def test_get_unit_types_section_short(self):
        """Short unit types section works with subset."""
        section = get_unit_types_section_short(["W", "M"])
        assert "UNIT TYPES (for reference):" in section
        assert "Warrior" in section
        assert "Mage" in section
        assert "Archer" not in section

    def test_get_unit_types_section_short_empty_defaults_to_all(self):
        """Short section with empty list defaults to all units."""
        section = get_unit_types_section_short([])
        assert "Warrior" in section
        assert "Barbarian" in section

    def test_get_strategy_tips_all(self):
        """Strategy tips includes all relevant tips."""
        tips = get_strategy_tips(["W", "M", "C", "A", "K", "R", "S", "B"])
        assert "STRATEGY TIPS:" in tips
        assert "paralyze" in tips  # Mage tip
        assert "Position units" in tips  # Always-present tip

    def test_get_strategy_tips_subset(self):
        """Strategy tips only includes tips for enabled units."""
        tips = get_strategy_tips(["W"])
        assert "STRATEGY TIPS:" in tips
        assert "paralyze" not in tips  # No mage tip
        assert "Position units" in tips

    def test_get_strategy_tips_empty_defaults(self):
        """Empty list defaults to all strategy tips."""
        tips = get_strategy_tips([])
        assert "paralyze" in tips

    def test_get_available_actions_section_all(self):
        """Available actions include all unit-specific actions when all enabled."""
        actions = get_available_actions_section(["W", "M", "C", "A", "K", "R", "S", "B"])
        assert "PARALYZE" in actions
        assert "HEAL" in actions
        assert "CURE" in actions
        assert "HASTE" in actions
        assert "DEFENCE_BUFF" in actions
        assert "ATTACK_BUFF" in actions
        assert "SEIZE" in actions
        assert "END_TURN" in actions

    def test_get_available_actions_section_warriors_only(self):
        """Available actions with only warriors should not include special abilities."""
        actions = get_available_actions_section(["W"])
        assert "CREATE_UNIT" in actions
        assert "MOVE" in actions
        assert "ATTACK" in actions
        assert "SEIZE" in actions
        assert "PARALYZE" not in actions
        assert "HEAL" not in actions
        assert "HASTE" not in actions

    def test_get_available_actions_section_with_mage(self):
        """Actions with mage enabled include PARALYZE."""
        actions = get_available_actions_section(["W", "M"])
        assert "PARALYZE" in actions
        assert "HEAL" not in actions

    def test_get_available_actions_section_with_cleric(self):
        """Actions with cleric enabled include HEAL and CURE."""
        actions = get_available_actions_section(["W", "C"])
        assert "HEAL" in actions
        assert "CURE" in actions
        assert "PARALYZE" not in actions

    def test_get_available_actions_section_with_sorcerer(self):
        """Actions with sorcerer enabled include buff actions."""
        actions = get_available_actions_section(["W", "S"])
        assert "HASTE" in actions
        assert "DEFENCE_BUFF" in actions
        assert "ATTACK_BUFF" in actions

    def test_get_available_actions_section_empty_defaults(self):
        """Empty list defaults to all actions."""
        actions = get_available_actions_section([])
        assert "PARALYZE" in actions
        assert "HEAL" in actions
        assert "HASTE" in actions

    def test_get_dynamic_prompt_all_units_returns_original(self):
        """Dynamic prompt with all units returns unmodified original."""
        all_units = ["W", "M", "C", "A", "K", "R", "S", "B"]
        result = get_dynamic_prompt("basic", all_units)
        assert result is PROMPT_BASIC

    def test_get_dynamic_prompt_subset_adds_disabled_note(self):
        """Dynamic prompt with subset adds disabled units note."""
        result = get_dynamic_prompt("basic", ["W", "M", "C"])
        # Should contain a note about disabled units
        assert "DISABLED" in result or "disabled" in result.lower()

    def test_get_dynamic_prompt_unknown_base_raises(self):
        """Dynamic prompt with unknown base prompt raises ValueError."""
        with pytest.raises(ValueError):
            get_dynamic_prompt("nonexistent", ["W"])
