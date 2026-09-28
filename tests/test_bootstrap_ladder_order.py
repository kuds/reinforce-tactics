"""configs/ppo/bootstrap.yaml climbs the scripted-bot ladder: no stage is a step down.

``scripts/eval/bot_ladder.py`` flags a curriculum stage whose opponent rates
weaker than the previous stage's opponent on the same map, or scores
significantly below half against it head to head (95% Wilson interval). A
full ladder takes ~40 minutes on 4 CPUs, so the default test re-runs the
tool's check (its own ``check_curriculum``) on a cached copy of one run's
games: ``tests/data/bootstrap_ladder.json`` stores every game's result, per
board and pairing, as one character per game (``W`` / ``D`` / ``L`` from the
pairing's first bot's side), seed-major and seat 1 before seat 2, exactly the
order the tool plays and sorts them.

The fast tests fail when the curriculum changes in a way the cache does not
cover (a new opponent, map, game length or engine setting), or when the
cached games flag a stage. After changing the curriculum's opponents, a bot
or the engine, re-run the ladder and refresh the cache::

    python scripts/eval/bot_ladder.py --config configs/ppo/bootstrap.yaml \\
        --seeds 25 --seed-base 7000 --workers 4 --fail-on-flag --out-dir DIR
    python tests/test_bootstrap_ladder_order.py DIR/ladder.json

The slow test replays a sample of the cached games (the first seeds of every
pair of consecutive stage opponents, both seats) to check the cache still
matches the engine and the bots.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "eval" / "bot_ladder.py"
CONFIG = REPO_ROOT / "configs" / "ppo" / "bootstrap.yaml"
CACHE = Path(__file__).resolve().parent / "data" / "bootstrap_ladder.json"
_CHAR_TO_WINNER = {"W": "a", "D": None, "L": "b"}
_WINNER_TO_CHAR = {winner: char for char, winner in _CHAR_TO_WINNER.items()}


def _load_ladder_module():
    spec = importlib.util.spec_from_file_location("bot_ladder_order", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: dataclasses look their module up in sys.modules.
    sys.modules["bot_ladder_order"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ladder():
    try:
        yield _load_ladder_module()
    finally:
        sys.modules.pop("bot_ladder_order", None)


def _cache() -> dict[str, Any]:
    return json.loads(CACHE.read_text(encoding="utf-8"))


def _expand(cache: Mapping[str, Any]) -> dict[str, Any]:
    """The cache in the format ``bot_ladder.load_json`` reads (games without end reason or length)."""
    seed_base = int(cache["meta"]["seed_base"])
    boards = []
    for board in cache["boards"]:
        pairings = []
        for a_index, b_index, outcomes in board["pairings"]:
            games = [
                [seed_base + i // 2, 1 + i % 2, _CHAR_TO_WINNER[char], "cached", 0, 0, None] for i, char in enumerate(outcomes)
            ]
            pairings.append({"a_index": a_index, "b_index": b_index, "games": games})
        boards.append({**{k: v for k, v in board.items() if k != "pairings"}, "pairings": pairings})
    return {"meta": dict(cache["meta"]), "boards": boards}


def _cached_result(ladder):
    cache = _cache()
    return ladder.load_json(_expand(cache)), cache


def _curriculum(ladder):
    """The config's boards, default opponent list and stages, exactly as the tool resolves them."""
    args = ladder.build_parser().parse_args(["--config", str(CONFIG)])
    return ladder.resolve_inputs(args)


def test_cache_covers_the_shipped_curriculum(ladder):
    """The cached run played the boards and bots the tool would play for today's config."""
    result, cache = _cached_result(ladder)
    boards, opponents, stages = _curriculum(ladder)
    assert [br.board for br in result.boards] == boards, (
        "bootstrap.yaml's maps, game lengths or engine settings changed: refresh tests/data/bootstrap_ladder.json"
    )
    n_games = 2 * int(cache["meta"]["n_seeds"])
    assert n_games >= 50, "the cache should hold >= 25 seeds x both seats per pairing"
    for br in result.boards:
        assert [o.key for o in br.opponents] == [o.key for o in opponents], (
            f"{br.board.label}: bootstrap.yaml's opponents changed: refresh tests/data/bootstrap_ladder.json"
        )
        # Every pair of bots met in both seats on every seed.
        assert len(br.pairings) == len(opponents) * (len(opponents) - 1) // 2
        for pairing in br.pairings.values():
            assert pairing.seat(1).n == pairing.seat(2).n == n_games // 2
    assert {s.map_file for s in stages} == {b.map_file for b in boards}


def test_no_stage_is_easier_than_the_one_before(ladder):
    """The tool's curriculum check on the cached games: every stage checked, none flagged."""
    result, _ = _cached_result(ladder)
    _, _, stages = _curriculum(ladder)
    checks = ladder.check_curriculum(stages, result)
    assert [c.stage.name for c in checks] == [s.name for s in stages]
    unchecked = [c.stage.name for c in checks if c.verdict in ("not played", "same opponent")]
    assert not unchecked
    flagged = [f"{c.stage.name}: {ladder._verdict_text(c)}" for c in checks if c.flagged]
    assert not flagged, "stages easier than the one before them:\n" + "\n".join(flagged)
    # Each map block starts once, and every later stage on it is compared with its predecessor.
    firsts = [c.stage.name for c in checks if c.previous is None]
    assert len(firsts) == len({s.map_file for s in stages})


def test_the_check_has_teeth(ladder):
    """Reversing a map block on the cached games flags its stages: the pass above is not vacuous."""
    result, _ = _cached_result(ladder)
    _, _, stages = _curriculum(ladder)
    beginner = [s for s in stages if s.map_file.endswith("beginner.csv")]
    others = [s for s in stages if not s.map_file.endswith("beginner.csv")]
    checks = ladder.check_curriculum(others + beginner[::-1], result)
    flagged = {c.stage.name for c in checks if c.flagged}
    assert {"beginner_mixed_med_adv_50", "beginner_mixed_50"} <= flagged


REPLAY_SEEDS = 3


@pytest.mark.slow
def test_cached_games_still_replay(ladder):
    """The engine and bots still play the cached games: the first seeds of each stage step, both seats."""
    result, cache = _cached_result(ladder)
    _, _, stages = _curriculum(ladder)
    seeds = range(int(cache["meta"]["seed_base"]), int(cache["meta"]["seed_base"]) + REPLAY_SEEDS)
    replayed = 0
    for check in ladder.check_curriculum(stages, result):
        if check.previous is None:
            continue
        br = ladder.board_of(result, check.board.map_file, check.board.max_turns)
        pairing, _ = br.pairing(check.stage.opponent, check.previous.opponent)
        for record in (r for r in pairing.records if r.seed in seeds):
            game = ladder.play_game(br.board, pairing.a, pairing.b, record.seed, record.a_seat)
            assert game.error is None, game.error
            assert game.winner == record.winner, (
                f"{br.board.label} {pairing.a.label} vs {pairing.b.label}, seed {record.seed}, a in seat "
                f"{record.a_seat}: the engine or a bot changed since the cache was made; refresh "
                "tests/data/bootstrap_ladder.json"
            )
            replayed += 1
    assert replayed == 2 * REPLAY_SEEDS * (len(stages) - len({s.map_file for s in stages}))


def write_cache(ladder_json: Path, out: Path = CACHE) -> None:
    """Compact a ``bot_ladder.py --out-dir`` run's ladder.json into the cache this test reads."""
    data = json.loads(ladder_json.read_text(encoding="utf-8"))
    meta = data["meta"]
    if data.get("errors"):
        raise SystemExit(f"{ladder_json}: the run has errored games; not caching it")
    seed_base, n_seeds = int(meta["seed_base"]), int(meta["n_seeds"])
    order = [(seed_base + i // 2, 1 + i % 2) for i in range(2 * n_seeds)]
    boards = []
    for board in data["boards"]:
        pairings = []
        for pairing in board["pairings"]:
            games = sorted(pairing["games"], key=lambda g: (g[0], g[1]))
            if [(g[0], g[1]) for g in games] != order:
                raise SystemExit(f"{board['map_file']} {pairing['a']} vs {pairing['b']}: not every seed in both seats")
            outcomes = "".join(_WINNER_TO_CHAR[g[2]] for g in games)
            pairings.append([pairing["a_index"], pairing["b_index"], outcomes])
        boards.append(
            {
                "map_file": board["map_file"],
                "max_turns": board["max_turns"],
                "fog_of_war": board["fog_of_war"],
                "enabled_units": board["enabled_units"],
                "engine_overrides": board["engine_overrides"],
                "opponents": [{"name": o["name"], "kwargs": o["kwargs"]} for o in board["opponents"]],
                "pairings": pairings,
            }
        )
    # Output paths are machine-local: kept out of the cache.
    argv = list(meta.get("argv") or [])
    for i, arg in enumerate(argv[:-1]):
        if arg == "--out-dir":
            argv[i + 1] = "DIR"
    compact = {
        "meta": {
            "source": "scripts/eval/bot_ladder.py; refresh with `python tests/test_bootstrap_ladder_order.py DIR/ladder.json`",
            "config": meta.get("config"),
            "argv": argv,
            "git_commit": meta.get("git_commit"),
            "git_dirty": meta.get("git_dirty"),
            "n_seeds": n_seeds,
            "seed_base": seed_base,
            "games": "per pairing [a_index, b_index, outcomes]: one char per game (W/D/L for a), "
            "seed-major from seed_base, a in seat 1 then seat 2",
        },
        "boards": boards,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dump(compact), encoding="utf-8")


def _dump(compact: Mapping[str, Any]) -> str:
    """JSON with one opponent and one pairing per line, so a refresh's diff stays readable."""

    def block(items: list[Any], indent: str) -> list[str]:
        return [f"{indent}{json.dumps(item)}{',' if i < len(items) - 1 else ''}" for i, item in enumerate(items)]

    lines = ["{", f' "meta": {json.dumps(compact["meta"], indent=1)},'.replace("\n", "\n "), ' "boards": [']
    for bi, board in enumerate(compact["boards"]):
        lines.append("  {")
        for key, value in board.items():
            if key in ("opponents", "pairings"):
                continue
            lines.append(f"   {json.dumps(key)}: {json.dumps(value)},")
        lines.append('   "opponents": [')
        lines += block(board["opponents"], "    ")
        lines.append("   ],")
        lines.append('   "pairings": [')
        lines += block(board["pairings"], "    ")
        lines.append("   ]")
        lines.append("  }" + ("," if bi < len(compact["boards"]) - 1 else ""))
    lines += [" ]", "}"]
    text = "\n".join(lines) + "\n"
    assert json.loads(text) == json.loads(json.dumps(compact))
    return text


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tests/test_bootstrap_ladder_order.py DIR/ladder.json")
    write_cache(Path(sys.argv[1]))
    print(f"wrote {CACHE}")
