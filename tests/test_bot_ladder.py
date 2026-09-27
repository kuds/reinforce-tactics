"""scripts/eval/bot_ladder.py: seeded scripted-bot ladder and its curriculum check.

The default run keeps to tiny games (a handful of seeds, short max_turns,
the in-process path). The process-pool path is exercised by the slow test
at the bottom (``pytest -m slow --no-cov``).
"""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "eval" / "bot_ladder.py"
STARTER = "maps/1v1/starter.csv"
BEGINNER = "maps/1v1/beginner.csv"


@pytest.fixture(scope="module")
def ladder():
    spec = importlib.util.spec_from_file_location("bot_ladder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: dataclasses look their module up in sys.modules.
    sys.modules["bot_ladder"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("bot_ladder", None)


def _records(ladder, wins, draws, losses):
    """Fabricated games for a pairing: ``a``'s wins, draws and losses, alternating seats."""
    outcomes = ["a"] * wins + [None] * draws + ["b"] * losses
    return [
        ladder.GameRecord(
            seed=i // 2, a_seat=1 + i % 2, winner=w, end_reason="elimination" if w else "max_turns_draw", turns=10
        )
        for i, w in enumerate(outcomes)
    ]


def _board_result(ladder, board, results):
    """A rated BoardResult from ``{(a, b): (w, d, l)}`` over OpponentSpecs."""
    opponents = ladder.dedupe(spec for pair in results for spec in pair)
    br = ladder.BoardResult(board, opponents)
    for (a, b), (w, d, lost) in results.items():
        br.pairings[(a.key, b.key)] = ladder.Pairing(a, b, _records(ladder, w, d, lost))
    ladder.rate_board(br)
    return br


# --------------------------------------------------------------------------
# Opponents
# --------------------------------------------------------------------------


def test_opponent_specs_are_canonical(ladder):
    spec = ladder.OpponentSpec
    # Defaults are dropped, aliases resolved: one bot, one key.
    assert spec.parse("random") == spec.parse("random:max_actions=20") == spec.create("RandomBot", {"max_actions": 20})
    assert spec.parse("bot") == spec.create("simple")
    assert spec.parse("random").label == "random_20"
    assert spec.parse("random:max_actions=10").label == "random_10"
    mixed = spec.parse("mixed:easy=medium,hard=advanced,p_hard=0.5")
    assert mixed == spec.parse('mixed:{"easy": "medium", "hard": "advanced", "p_hard": 0.5}')
    assert mixed.label == "mix(medium/advanced,0.5)"
    assert spec.parse("mixed").label == "mix(simple/medium,0.5)"
    nested = spec.create("mixed", {"easy": "random", "hard": "simple", "easy_kwargs": {"max_actions": 10}})
    assert nested.label == "mix(random_10/simple,0.5)"
    assert len(ladder.dedupe([spec.parse("random"), spec.parse("random:max_actions=20"), spec.parse("noop")])) == 2


@pytest.mark.parametrize("text", ["medium:max_actions=3", "random:max_actions=0", "nosuchbot", "random:max_actions"])
def test_bad_opponent_specs_are_rejected(ladder, text):
    with pytest.raises((ValueError, KeyError)):
        ladder.OpponentSpec.parse(text)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_bradley_terry_orders_a_transitive_ladder_and_stays_finite(ladder):
    # 0 beats 1 beats 2, every game: the maximum-likelihood strengths are
    # infinite; the prior keeps them finite and in order.
    games = {(0, 1): (20.0, 20), (1, 2): (20.0, 20), (0, 2): (20.0, 20)}
    elo, se = ladder.bradley_terry(3, games)
    assert elo[0] > elo[1] > elo[2]
    assert all(math.isfinite(x) for x in elo + se)
    assert sum(elo) == pytest.approx(0.0, abs=1e-6)
    # Even results rate everyone the same; a draw counts half each way.
    even, _ = ladder.bradley_terry(3, {(0, 1): (10.0, 20), (1, 2): (10.0, 20), (0, 2): (10.0, 20)})
    assert even == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_wilson_interval(ladder):
    low, high = ladder.wilson(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-3)
    low, high = ladder.wilson(5, 10)
    assert low == pytest.approx(1 - high)
    assert ladder.wilson(0, 0) == (0.0, 1.0)


def test_ratings_record_mean_score_and_beaten(ladder):
    spec = ladder.OpponentSpec.create
    simple, medium, noop = spec("simple"), spec("medium"), spec("noop")
    board = ladder.Board(BEGINNER, 75)
    br = _board_result(ladder, board, {(medium, simple): (40, 0, 10), (simple, noop): (10, 40, 0), (medium, noop): (50, 0, 0)})
    assert [r.spec for r in br.ratings] == [noop, simple, medium]
    by_bot = {r.spec: r for r in br.ratings}
    assert by_bot[medium].mean_score == pytest.approx((0.8 + 1.0) / 2)
    # noop never wins, but simple beats it only 1 game in 5: it is hard to *beat*.
    assert by_bot[noop].beaten == pytest.approx((0.2 + 1.0) / 2)
    assert str(by_bot[simple].record) == "20-40-40"
    assert br.head_to_head(simple, medium).as_dict() == {"w": 10, "d": 0, "l": 40}
    assert str(br.seat1_score) == "55-40-55"
    # By seat: the first bot named sits in seat 1.
    assert str(br.seat_record(medium, simple)) == "20-0-5" and str(br.seat_record(simple, medium)) == "5-0-20"
    assert br.seat_record(medium, medium) is None


# --------------------------------------------------------------------------
# Curriculum check
# --------------------------------------------------------------------------


def test_curriculum_check_flags_a_stage_weaker_than_the_one_before(ladder):
    spec = ladder.OpponentSpec.create
    simple, medium, advanced = spec("simple"), spec("medium"), spec("advanced")
    board = ladder.Board(BEGINNER, 75)
    br = _board_result(
        ladder, board, {(medium, advanced): (45, 0, 5), (simple, medium): (1, 0, 49), (simple, advanced): (2, 0, 48)}
    )
    result = ladder.LadderResult([br], n_seeds=25, seed_base=0)
    stage = ladder.StageRef
    stages = [
        stage("b_simple", BEGINNER, 75, simple),
        stage("b_medium", BEGINNER, 75, medium),
        stage("b_medium_again", BEGINNER, 75, medium),
        stage("b_advanced", BEGINNER, 75, advanced),
        stage("unplayed", "maps/1v1/skirmish.csv", 120, simple),
    ]
    checks = {c.stage.name: c for c in ladder.check_curriculum(stages, result)}
    assert set(checks) == {"b_simple", "b_medium", "b_medium_again", "b_advanced"}
    assert checks["b_simple"].verdict == "first on map" and not checks["b_simple"].flagged
    assert checks["b_medium"].verdict == "harder" and checks["b_medium"].significance == "significant"
    assert checks["b_medium_again"].verdict == "same opponent"
    flagged = checks["b_advanced"]
    assert flagged.flagged and flagged.previous.name == "b_medium_again"
    assert str(flagged.head_to_head) == "5-0-45" and flagged.significance == "significant"
    assert flagged.ci[1] < 0.5

    order = [s.name for s, _ in ladder.rating_sorted_order(stages, result)]
    assert order == ["b_simple", "b_advanced", "b_medium", "b_medium_again", "unplayed"]

    markdown = ladder.render_markdown(result, list(checks.values()), ladder.rating_sorted_order(stages, result))
    assert "**FLAG: weaker** (significant)" in markdown
    assert "| 5 | unplayed | simple | n/a |" in markdown


def test_bots_that_only_draw_each_other_are_tied_not_flagged(ladder):
    spec = ladder.OpponentSpec.create
    r10, r15, medium = spec("random", {"max_actions": 10}), spec("random", {"max_actions": 15}), spec("medium")
    board = ladder.Board(BEGINNER, 75)
    br = _board_result(ladder, board, {(r10, r15): (0, 50, 0), (medium, r10): (50, 0, 0), (medium, r15): (50, 0, 0)})
    result = ladder.LadderResult([br], n_seeds=25, seed_base=0)
    stages = [ladder.StageRef("r10", BEGINNER, 75, r10), ladder.StageRef("r15", BEGINNER, 75, r15)]
    (_, second) = ladder.check_curriculum(stages, result)
    assert second.verdict == "tied" and not second.flagged and second.significance == "within noise"


# --------------------------------------------------------------------------
# Playing
# --------------------------------------------------------------------------


def test_games_are_seeded_and_seat_paired(ladder):
    spec = ladder.OpponentSpec.parse
    board = ladder.Board(STARTER, 12)
    mixed, rand = spec("mixed:easy=random,hard=simple,p_hard=0.5"), spec("random:max_actions=5")
    first = ladder.play_game(board, mixed, rand, seed=3, a_seat=1)
    assert first == ladder.play_game(board, mixed, rand, seed=3, a_seat=1)
    # A bot draws the same stream in both seats of a pair, a different one per opponent.
    assert ladder.bot_rng(3, mixed, rand).random() == ladder.bot_rng(3, mixed, rand).random()
    assert ladder.bot_rng(3, mixed, rand).random() != ladder.bot_rng(3, mixed, spec("simple")).random()
    assert first.error is None and first.end_reason in {"hq_capture", "elimination", "max_turns_draw"}


def test_tiny_ladder_smoke(ladder, tmp_path):
    """Tiny N end to end: every pairing, both seats, outputs, and a JSON round trip."""
    spec = ladder.OpponentSpec.parse
    opponents = [spec("simple"), spec("noop"), spec("random:max_actions=5")]
    boards = ladder.boards_for([(STARTER, 10)])
    result = ladder.run_ladder(boards, opponents, n_seeds=2, workers=1)
    (br,) = result.boards
    assert len(br.pairings) == 3 and not result.errors
    for pairing in br.pairings.values():
        assert pairing.total.n == 4
        assert pairing.seat(1).n == pairing.seat(2).n == 2
    noop = next(r for r in br.ratings if r.spec.name == "noop")
    assert noop.record.w == 0

    markdown = ladder.render_markdown(result)
    assert "## maps/1v1/starter.csv, max_turns 10" in markdown and "| A | B | A total | A as P1 | A as P2 |" in markdown
    assert "By seat: the row bot's W-D-L in seat 1" in markdown and markdown.count("| # | Bot | 1 | 2 | 3 |") == 2
    ladder.write_csv(result, tmp_path / "ladder.csv")
    rows = (tmp_path / "ladder.csv").read_text().splitlines()
    assert rows[0].startswith("map_file,max_turns,a,b,games,a_w,a_d,a_l,a_p1_w") and len(rows) == 4

    data = json.loads(json.dumps(ladder.to_json(result)))
    again = ladder.load_json(data)
    (br2,) = again.boards
    assert [str(p.total) for p in br2.pairings.values()] == [str(p.total) for p in br.pairings.values()]
    assert [round(r.elo, 6) for r in br2.ratings] == [round(r.elo, 6) for r in br.ratings]


def test_cli_checks_a_curriculum_and_fails_on_flag(ladder, tmp_path, capsys):
    config = tmp_path / "curriculum.yaml"
    starter = str(REPO_ROOT / STARTER)
    config.write_text(
        "env:\n  max_turns: 20\ncurriculum:\n  stages:\n"
        f"    - {{name: s_medium, map_file: {starter}, opponent: medium}}\n"
        f"    - {{name: s_noop, map_file: {starter}, opponent: noop}}\n"
    )
    out = tmp_path / "out"
    argv = ["--config", str(config), "--opponents", "medium", "noop", "--seeds", "1", "--workers", "1", "--quiet"]
    assert ladder.main([*argv, "--out-dir", str(out)]) == 0
    report = capsys.readouterr().out
    assert "| s_noop | starter@20 | noop |" in report and "**FLAG: weaker**" in report
    data = json.loads((out / "ladder.json").read_text())
    assert [c["flagged"] for c in data["curriculum_check"]] == [False, True]
    assert ladder.main([*argv, "--fail-on-flag"]) == ladder.EXIT_FLAGGED
    # Re-rendered from the JSON: same check, no games played.
    assert ladder.main(["--from-json", str(out / "ladder.json"), "--config", str(config), "--quiet", "--fail-on-flag"]) == 3


def test_default_inputs_cover_the_shipped_curriculum(ladder):
    args = ladder.build_parser().parse_args(["--config", str(REPO_ROOT / "configs/ppo/bootstrap.yaml")])
    boards, opponents, stages = ladder.resolve_inputs(args)
    assert [b.label for b in boards] == ["starter@20", "beginner@75", "intermediate@60", "skirmish@120", "corner_points@200"]
    labels = {o.label for o in opponents}
    assert {s.opponent.label for s in stages} <= labels
    assert {"simple", "medium", "advanced", "master", "random_20", "balanced_random", "noop"} <= labels
    assert {"random_10", "random_15", "mix(medium/advanced,0.5)", "mix(balanced_random/simple,0.25)"} <= labels
    assert len(labels) == len(opponents)

    # --maps under another spelling of a stage's map takes the stage's spelling
    # (so the curriculum check finds the board) and its max_turns.
    config = str(REPO_ROOT / "configs/ppo/bootstrap.yaml")
    args = ladder.build_parser().parse_args(["--config", config, "--maps", str(REPO_ROOT / BEGINNER)])
    boards, _, _ = ladder.resolve_inputs(args)
    assert [(b.map_file, b.max_turns) for b in boards] == [(BEGINNER, 75)]
    args = ladder.build_parser().parse_args(["--config", config, "--maps", BEGINNER, "--max-turns", "30"])
    boards, _, stages = ladder.resolve_inputs(args)
    assert [b.label for b in boards] == ["beginner@30"] and {s.max_turns for s in stages} == {30}


# --------------------------------------------------------------------------
# Slow: the process pool
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_worker_pool_matches_in_process_run(ladder, tmp_path):
    """The CLI with a process pool plays the same games as the in-process path."""
    opponents = ["simple", "medium", "random:max_actions=10", "mixed:easy=random,hard=simple,p_hard=0.5"]
    common = ["--maps", STARTER, BEGINNER, "--max-turns", "20", "--seeds", "3", "--quiet", "--opponents", *opponents]
    pooled = tmp_path / "pooled"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *common, "--workers", "2", "--chunk", "1", "--out-dir", str(pooled)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    serial = tmp_path / "serial"
    assert ladder.main([*common, "--workers", "1", "--out-dir", str(serial)]) == 0

    def games(path):
        data = json.loads((path / "ladder.json").read_text())
        return [(b["map_file"], p["a"], p["b"], p["games"]) for b in data["boards"] for p in b["pairings"]]

    assert games(pooled) == games(serial)
