"""Engine rules and UI art live in separate modules (review core-24).

``constants.py`` mixed the engine's rules with colours, sprite paths and the
animation layout, and repeated data (``UNIT_COLORS`` copied every unit's
``color``; ``TILE_COLORS`` and ``TILE_TYPES`` listed each tile code twice).
Rules now live in ``reinforcetactics.rules`` and art in
``reinforcetactics.ui.assets``; ``constants`` re-exports both so old imports
keep working.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

from reinforcetactics import constants, rules
from reinforcetactics.ui import assets

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PACKAGES = ("core", "game", "rl")

# Every public name ``reinforcetactics/constants.py`` defined before the split
# (commit 60cdb92), minus the ``Enum`` / ``Self`` it imported for ``TileType``.
PRE_SPLIT_CONSTANTS_NAMES = [
    "ALL_UNIT_TYPES",
    "ANIMATION_CONFIG",
    "BASE_SPRITE_COLORS",
    "BUILDING_INCOME",
    "BUILDING_MAX_HEALTH",
    "CHARGE_BONUS",
    "CHARGE_MIN_DISTANCE",
    "CLERIC_HEAL_RANGE",
    "COUNTER_ATTACK_MULTIPLIER",
    "DEFENCE_REDUCTION_PER_POINT",
    "FLANK_BONUS",
    "FPS",
    "HASTE_COOLDOWN",
    "HEADQUARTERS_INCOME",
    "HEADQUARTERS_MAX_HEALTH",
    "HEAL_AMOUNT",
    "MAX_UNITS_PER_PLAYER",
    "MIN_MAP_SIZE",
    "MIN_STRIP_SIZE",
    "NEUTRAL_STRUCTURE_PALETTE",
    "PARALYZE_COOLDOWN",
    "PARALYZE_DURATION",
    "PLAYER_COLORS",
    "ROGUE_EVADE_CHANCE",
    "ROGUE_FOREST_EVADE_BONUS",
    "SORCERER_ATTACK_BUFF_AMOUNT",
    "SORCERER_BUFF_COOLDOWN",
    "SORCERER_BUFF_DURATION",
    "SORCERER_DEFENCE_BUFF_AMOUNT",
    "STARTING_GOLD",
    "STRUCTURE_REGEN_RATE",
    "STRUCTURE_TILE_TYPES",
    "TEAM_PALETTES",
    "TILE_COLORS",
    "TILE_IMAGES",
    "TILE_SIZE",
    "TILE_TYPES",
    "TOWER_INCOME",
    "TOWER_MAX_HEALTH",
    "TileType",
    "UNIT_COLORS",
    "UNIT_DATA",
    "UNIT_TYPE_TO_IDX",
]

# Imports every module of the engine packages, then reports which UI-side
# modules ended up loaded. Run in a fresh interpreter: the test session has
# long since imported pygame and the UI.
_IMPORT_ENGINE = """
import importlib, json, pkgutil, sys
for package in sys.argv[1:]:
    root = importlib.import_module(package)
    for info in pkgutil.walk_packages(root.__path__, package + "."):
        importlib.import_module(info.name)
print(json.dumps(sorted(
    name for name in sys.modules
    if name == "pygame" or name.startswith(("pygame.", "reinforcetactics.ui", "reinforcetactics.constants"))
)))
"""

_IMPORT_FACADE = """
import json, sys
import reinforcetactics.constants
print(json.dumps(sorted(name for name in sys.modules if name == "pygame" or name.startswith("pygame."))))
"""


def _run(code: str, *args: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", code, *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_engine_packages_load_no_ui_module_or_pygame():
    packages = [f"reinforcetactics.{name}" for name in ENGINE_PACKAGES]
    assert _run(_IMPORT_ENGINE, *packages) == []


def test_engine_source_never_imports_ui_art_or_the_facade():
    # Also covers imports inside functions, which the import check above
    # never runs. (gym_env lazily imports the Renderer to render; that is
    # the UI drawing the game, not the engine reading art.)
    banned = ("reinforcetactics.constants", "reinforcetactics.ui.assets", "pygame")
    offenders = []
    for package in ENGINE_PACKAGES:
        for path in sorted((REPO_ROOT / "reinforcetactics" / package).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
                else:
                    continue
                if any(name == b or name.startswith(b + ".") for name in names for b in banned):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == []


def test_constants_facade_imports_without_pygame():
    assert _run(_IMPORT_FACADE) == []


def test_constants_facade_still_exports_every_pre_split_name():
    missing = [name for name in PRE_SPLIT_CONSTANTS_NAMES if not hasattr(constants, name)]
    assert missing == []
    assert sorted(constants.__all__) == sorted(PRE_SPLIT_CONSTANTS_NAMES)  # what ``import *`` gives


def test_constants_facade_shares_the_owning_modules_objects():
    # Same objects, so a notebook that edits constants.UNIT_DATA in place
    # (balance_analysis.ipynb does) still changes what the engine plays.
    for name in constants.__all__:
        if name == "UNIT_COLORS":
            continue
        owner = rules if hasattr(rules, name) else assets
        assert getattr(constants, name) is getattr(owner, name), name
    assert constants.UNIT_COLORS == {code: art["color"] for code, art in assets.UNIT_ASSETS.items()}


def test_unit_rules_and_unit_art_are_keyed_alike():
    assert list(assets.UNIT_ASSETS) == list(rules.UNIT_DATA)
    assert set(rules.UNIT_DATA) == set(rules.ALL_UNIT_TYPES)
    for code, stats in rules.UNIT_DATA.items():
        assert set(stats) == {"name", "cost", "movement", "health", "attack", "defence"}, code
        assert set(assets.UNIT_ASSETS[code]) == {"static_path", "animation_path", "color"}, code


def test_tile_tables_list_each_tile_code_once():
    codes = [tile_type.value for tile_type in rules.TileType]
    assert list(rules.TILE_TYPES) == codes
    assert list(assets.TILE_COLORS) == codes
    assert rules.TILE_TYPES == {tile_type.value: tile_type.name for tile_type in rules.TileType}
    assert set(assets.TILE_IMAGES) == set(rules.TILE_TYPES.values())
    assert assets.STRUCTURE_TILE_TYPES == {t.name for t in rules.TileType if t.is_capturable()}
