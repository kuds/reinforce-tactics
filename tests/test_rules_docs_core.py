"""The rules as documented must match the engine's constants (review core-23).

The README said Cleric move 2 / HP 8 / range 1-2, Sorcerer cost 400 / HP 10 /
buffs +35%, Barbarian HP 24 and starting gold $200; the docs site said heals
restore 5 HP at range 1-2; the LLM system prompts said buffs were 35% with
3-turn cooldowns and the HQ earned 100 gold. None matched ``constants.py``,
so players, prompt authors and balance analysts worked from wrong numbers.
These tests parse every place the numbers are written down and fail when one
drifts from ``UNIT_DATA`` and the rule constants.
"""

import re
from pathlib import Path

import pytest

from reinforcetactics import rules as C
from reinforcetactics.core.unit import Unit
from reinforcetactics.game import llm_prompts
from reinforcetactics.rules import UNIT_DATA

README = Path("README.md").read_text(encoding="utf-8")
MECHANICS = Path("docs-site/docs/game-mechanics.md").read_text(encoding="utf-8")
NAME_TO_CODE = {data["name"]: code for code, data in UNIT_DATA.items()}


def _pct(fraction: float) -> str:
    return f"{round(fraction * 100)}%"


def _table(text: str, header: str) -> list[list[str]]:
    """Cells of the markdown table whose header row starts with ``header``."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip().strip("*").strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _attack_text(code: str) -> str:
    attack = UNIT_DATA[code]["attack"]
    if isinstance(attack, dict):
        return f"{attack['adjacent']} (adjacent) / {attack['range']} (range)"
    return str(attack)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def test_readme_unit_table_matches_unit_data():
    rows = _table(README, "| Unit | Cost | Move | HP |")
    assert sorted(row[0] for row in rows) == sorted(NAME_TO_CODE)
    for name, cost, move, hp, _special in rows:
        stats = UNIT_DATA[NAME_TO_CODE[name]]
        assert (int(cost), int(move), int(hp)) == (stats["cost"], stats["movement"], stats["health"]), name


def test_readme_abilities_and_economy_match_the_constants():
    special = {row[0]: row[4] for row in _table(README, "| Unit | Cost | Move | HP |")}
    assert f"paralyze ({C.PARALYZE_DURATION} turns, {C.PARALYZE_COOLDOWN}-turn cooldown)" in special["Mage"]
    assert f"+{C.HEAL_AMOUNT} HP" in special["Cleric"] and f"range 1-{C.CLERIC_HEAL_RANGE}" in special["Cleric"]
    assert f"+{_pct(C.CHARGE_BONUS)} dmg if moved {C.CHARGE_MIN_DISTANCE}+ tiles" in special["Knight"]
    assert f"Flank (+{_pct(C.FLANK_BONUS)} dmg)" in special["Rogue"]
    evade = f"Evade ({_pct(C.ROGUE_EVADE_CHANCE)} dodge, {_pct(C.ROGUE_EVADE_CHANCE + C.ROGUE_FOREST_EVADE_BONUS)} in forest)"
    assert evade in special["Rogue"]
    assert C.SORCERER_ATTACK_BUFF_AMOUNT == C.SORCERER_DEFENCE_BUFF_AMOUNT  # the README states one number
    buff = (
        f"+{_pct(C.SORCERER_ATTACK_BUFF_AMOUNT)}, {C.SORCERER_BUFF_DURATION} turns, {C.SORCERER_BUFF_COOLDOWN}-turn cooldown"
    )
    assert buff in special["Sorcerer"]
    assert C.HASTE_COOLDOWN == C.SORCERER_BUFF_COOLDOWN  # "2-turn cooldown" covers haste too

    assert f"Starting gold ${C.STARTING_GOLD}." in README
    income = f"(HQ: ${C.HEADQUARTERS_INCOME}, Building: ${C.BUILDING_INCOME}, Tower: ${C.TOWER_INCOME})"
    assert income in README


def test_readme_documents_every_optional_terrain_rule_with_its_default():
    from reinforcetactics.core.terrain_rules import TERRAIN_RULE_KEYS, TerrainRules

    rows = {row[0].strip("`"): row[1] for row in _table(README, "| Key | Default | Effect when set |")}
    assert set(rows) == set(TERRAIN_RULE_KEYS)
    defaults = TerrainRules()
    assert rows["charge_distance"] == f'`"{defaults.charge_distance}"`'
    assert rows["forest_concealment"] == str(defaults.forest_concealment).lower().join("``")
    assert rows["hq_always_visible"] == str(defaults.hq_always_visible).lower().join("``")
    assert defaults.move_costs == {} and "costs 1" in rows["terrain_move_cost"]


# ---------------------------------------------------------------------------
# docs site
# ---------------------------------------------------------------------------


def test_mechanics_page_unit_table_matches_unit_data():
    rows = _table(MECHANICS, "| Unit | Code | Cost | Health | Movement | Attack | Defence |")
    assert sorted(row[1] for row in rows) == sorted(UNIT_DATA)
    for name, code, cost, health, movement, attack, defence, _special in rows:
        stats = UNIT_DATA[code]
        assert name == stats["name"]
        assert (int(cost), int(health), int(movement), int(defence)) == (
            stats["cost"],
            stats["health"],
            stats["movement"],
            stats["defence"],
        ), name
        assert attack == _attack_text(code), name


@pytest.mark.parametrize("code", sorted(UNIT_DATA))
def test_mechanics_page_unit_details_match_unit_data(code):
    stats = UNIT_DATA[code]
    section = MECHANICS.split(f"#### {stats['name']} ({code})\n", 1)[1].split("\n#### ", 1)[0]
    assert f"- **Cost**: ${stats['cost']}\n" in section
    attack = stats["attack"]
    attack_text = f"{attack['adjacent']}/{attack['range']}" if isinstance(attack, dict) else str(attack)
    expected = (
        f"- **Stats**: {stats['health']} HP, {stats['movement']} Movement, {attack_text} Attack, {stats['defence']} Defence"
    )
    assert expected in section


def test_mechanics_page_structure_table_matches_constants():
    rows = {row[1]: row for row in _table(MECHANICS, "| Structure | Code | Max Health | Income/Turn |")}
    expected = {
        "h": (C.HEADQUARTERS_MAX_HEALTH, C.HEADQUARTERS_INCOME),
        "b": (C.BUILDING_MAX_HEALTH, C.BUILDING_INCOME),
        "t": (C.TOWER_MAX_HEALTH, C.TOWER_INCOME),
    }
    assert set(rows) == set(expected)
    for code, (hp, income) in expected.items():
        assert rows[code][2] == f"{hp} HP" and rows[code][3] == f"${income}", code


def test_constants_quoted_on_the_mechanics_page_are_current():
    """Every ``NAME = value`` the page quotes must be the constant's value."""
    quoted = re.findall(r"`([A-Z][A-Z_]+) = ([0-9.]+)`", MECHANICS)
    assert len(quoted) >= 10
    for name, value in quoted:
        assert float(value) == getattr(C, name), name
    assert f"restores **{C.HEAL_AMOUNT} HP**" in MECHANICS
    assert f"HEAL allies (+{C.HEAL_AMOUNT} HP) at range 1-{C.CLERIC_HEAL_RANGE}" in MECHANICS


def test_mechanics_page_documents_every_optional_terrain_rule():
    from reinforcetactics.core.terrain_rules import TERRAIN_RULE_KEYS

    rows = _table(MECHANICS, "| Key | Default | Effect when set |")
    assert {row[0].strip("`") for row in rows} == set(TERRAIN_RULE_KEYS)


# ---------------------------------------------------------------------------
# LLM system prompts (sent to the LLM bots)
# ---------------------------------------------------------------------------


def _prompt_texts() -> dict[str, str]:
    everything = sorted(UNIT_DATA)
    texts = {name: text for name, text in vars(llm_prompts).items() if name.startswith("PROMPT_") and isinstance(text, str)}
    texts["UNIT_DESCRIPTIONS"] = "\n".join(llm_prompts.UNIT_DESCRIPTIONS.values())
    texts["UNIT_DESCRIPTIONS_SHORT"] = "\n".join(llm_prompts.UNIT_DESCRIPTIONS_SHORT.values())
    texts["actions_section"] = llm_prompts.get_available_actions_section(everything)
    return texts


@pytest.mark.parametrize("name", sorted(_prompt_texts()))
def test_llm_prompt_numbers_match_the_constants(name):
    text = _prompt_texts()[name]
    for unit, code, cost, hp, attack, defence, movement in re.findall(
        r"(\w+) \((\w)\): Cost (\d+) gold, HP (\d+), Attack (.+?), Defense (\d+), Movement (\d+)", text
    ):
        stats = UNIT_DATA[code]
        assert unit == stats["name"]
        assert (int(cost), int(hp), int(defence), int(movement)) == (
            stats["cost"],
            stats["health"],
            stats["defence"],
            stats["movement"],
        ), (name, code)
        raw = stats["attack"]
        assert attack == (f"{raw['adjacent']} (adjacent) or {raw['range']} (range)" if isinstance(raw, dict) else str(raw))
    for unit, code, hp, movement in re.findall(r"(\w+) \((\w)\): [^\n]*?, (\d+) HP, (\d+) movement", text):
        assert (int(hp), int(movement)) == (UNIT_DATA[code]["health"], UNIT_DATA[code]["movement"]), (name, code)

    for cooldown in re.findall(r"HASTE: [^\n]*\((\d+)-turn cooldown\)", text):
        assert int(cooldown) == C.HASTE_COOLDOWN, name
    for pct, turns, cooldown in re.findall(
        r"BUFF: Give ally [-+](\d+)% damage \w+ for (\d+) turns \((\d+)-turn cooldown\)", text
    ):
        assert (f"{pct}%", int(turns), int(cooldown)) == (
            _pct(C.SORCERER_ATTACK_BUFF_AMOUNT),
            C.SORCERER_BUFF_DURATION,
            C.SORCERER_BUFF_COOLDOWN,
        ), name
    for pct, turns in re.findall(r"Give an ally (\d+)% damage (?:reduction|boost) for (\d+) turns", text):
        assert (f"{pct}%", int(turns)) == (_pct(C.SORCERER_ATTACK_BUFF_AMOUNT), C.SORCERER_BUFF_DURATION), name
    assert "35%" not in text, name
    for heal_range in re.findall(r"(?:HEAL|CURE)[^\n]*range 1-(\d+)", text):
        assert int(heal_range) == C.CLERIC_HEAL_RANGE, name
    assert "adjacent ally" not in text, name  # heal/cure reach CLERIC_HEAL_RANGE, not 1
    archer = Unit("A", 0, 0, 1)
    for low, high in re.findall(r"Archers?(?: attack)? at range (\d+)-(\d+)", text):
        assert (int(low), int(high)) == archer.get_attack_range(), name
    assert "Mages/Archers" not in text, name  # they have different ranges
    for pct, forest_pct in re.findall(r"EVADE[^\n]*?(\d+)% (?:chance to )?dodge[^\n]*?(\d+)% in forest", text):
        assert (f"{pct}%", f"{forest_pct}%") == (
            _pct(C.ROGUE_EVADE_CHANCE),
            _pct(C.ROGUE_EVADE_CHANCE + C.ROGUE_FOREST_EVADE_BONUS),
        ), name
    incomes = {"HQ": C.HEADQUARTERS_INCOME, "Building": C.BUILDING_INCOME, "Tower": C.TOWER_INCOME}
    for structure, gold in re.findall(r"- (HQ|Building|Tower) \(\w\): (?:Generates )?(\d+) gold/turn", text):
        assert int(gold) == incomes[structure], (name, structure)


# ---------------------------------------------------------------------------
# tile.py reports bad map data through logging, not print
# ---------------------------------------------------------------------------


def test_invalid_tile_type_is_logged_not_printed(caplog, capsys):
    from reinforcetactics.core.tile import Tile

    with caplog.at_level("WARNING", logger="reinforcetactics.core.tile"):
        tile = Tile("zz", 3, 4)
    assert tile.type == "o"  # ocean: impassable, not open ground
    assert "Invalid tile type 'zz' at (3, 4)" in caplog.text
    assert capsys.readouterr().out == ""
