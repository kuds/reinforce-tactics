"""Every command-line opponent list comes from the bot registry (review rltrain-22 / rulebots-7).

The CLIs used to hand-write their ``--opponent`` choices (``["bot",
"random"]``, ``["bot", "random", "noop", "self"]``, ...), which drifted from
the registry: ``master`` and ``balanced_random`` could not be chosen, and
``self`` was offered where nothing wraps the env for self-play, so it meant
no opponent at all. This scans the scripts' argparse calls: every
opponent-like option must take its choices from
``bot_registry.accepted_names()`` (or ``gym_env.accepted_opponents()``
where self-play is real).
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(
    [
        *(REPO_ROOT / "scripts").rglob("*.py"),
        *(REPO_ROOT / "reinforcetactics" / "cli").rglob("*.py"),
        *(REPO_ROOT / "examples").rglob("*.py"),
    ]
)
REGISTRY_CALLS = {"accepted_names", "accepted_opponents"}


def _opponent_options(path: Path) -> list[tuple[str, ast.Call]]:
    options = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if any(f.startswith("--") and f.endswith("opponent") for f in flags):
            options.append((flags[0], node))
    return options


def test_scan_finds_the_opponent_options():
    found = {(p.relative_to(REPO_ROOT).as_posix(), flag) for p in SOURCES for flag, _ in _opponent_options(p)}
    assert {
        ("reinforcetactics/cli/main.py", "--opponent"),
        ("scripts/eval_agent.py", "--opponent"),
        ("scripts/train/train_feudal_rl.py", "--opponent"),
        ("scripts/train/train_feudal_rl.py", "--eval-opponent"),
        ("scripts/train/train_self_play.py", "--bot-opponent"),
        ("scripts/train/train_self_play.py", "--eval-opponent"),
    } <= found


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_opponent_choices_come_from_the_registry(path):
    for flag, call in _opponent_options(path):
        choices = next((kw.value for kw in call.keywords if kw.arg == "choices"), None)
        assert choices is not None, f"{flag} in {path.name} takes any string; use choices=accepted_names()"
        assert isinstance(choices, ast.Call) and getattr(choices.func, "id", None) in REGISTRY_CALLS, (
            f"{flag} in {path.name} hand-writes its choices; derive them from the bot registry"
        )


def test_self_is_offered_only_where_self_play_is_wired():
    """'self' without a self-play wrapper is an env with no opponent."""
    for path in SOURCES:
        for flag, call in _opponent_options(path):
            choices = next(kw.value for kw in call.keywords if kw.arg == "choices")
            if getattr(choices.func, "id", None) == "accepted_opponents":
                assert (path.name, flag) == ("train_feudal_rl.py", "--opponent")
