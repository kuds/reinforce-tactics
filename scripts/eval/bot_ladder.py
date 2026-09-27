#!/usr/bin/env python3
"""Scripted-bot ladder: seeded cross-tier matches on a curriculum's maps.

A curriculum stage is only a step up if its opponent is harder than the one
before it. That was assumed, never measured, and the rule-bot legality fixes
broke it: on the small curriculum maps MediumBot now beats AdvancedBot and
MasterBot. This script measures the ladder directly. For every map a
curriculum config uses (or ``--maps``) it plays every pair of scripted
opponents against each other on seeded games, in both seats, and reports:

* W/D/L per pairing per map, overall and per seat (markdown, CSV and JSON);
* a per-map strength ordering: a Bradley-Terry rating fitted to the results
  (a draw counts half a win for each side; a weak prior of one virtual win
  and one virtual loss against an average opponent keeps the ratings finite
  when a bot wins or loses every game) and each bot's mean score;
* a curriculum check: each stage's opponent against the opponent of the
  previous stage on the same map, flagged when it rates weaker, with the
  head-to-head score of the two and its 95% Wilson interval;
* the curriculum's stages re-sorted by opponent rating inside each map block
  (a mechanical order to read next to the flags, not a curriculum edit).

Opponents
    ``--config`` gives the default list: every (opponent, opponent_kwargs)
    pair the curriculum's stages use, plus the plain tiers ``simple medium
    advanced master random balanced_random noop``, deduplicated after
    dropping kwargs equal to the constructor defaults (so ``random`` with
    ``max_actions: 20`` and plain ``random`` are one bot, shown as
    ``random_20``). ``--opponents`` replaces the list and ``--add-opponents``
    extends it; each entry is ``name`` or ``name:key=value,key=value``
    (values parsed as YAML) or ``name:{json}``, e.g. ``random:max_actions=10``
    or ``mixed:easy=simple,hard=medium,p_hard=0.5``.

Maps and game length
    A board is one (map, max_turns) pair. With ``--config`` each map is played
    at the ``max_turns`` of the stages that use it (a stage's own
    ``max_turns``, else ``env.max_turns``; a map whose stages disagree is
    played once per value), under the config's ``env.fog_of_war``,
    ``env.enabled_units`` and ``env.engine_overrides``. ``--maps`` picks the
    maps; ``--max-turns`` replaces every board's game length.

Seeding
    Seed ``k`` (``--seed-base`` .. ``--seed-base + --seeds - 1``) plays two
    games per pairing, one per seat assignment, both on ``GameState(seed=k)``.
    Each bot gets ``rng=random.Random("ladder:<k>:<bot>:vs:<other bot>")``
    (a string seed, which Python hashes with SHA-512, so it is stable across
    processes), the same in both games of the pair, so the only difference
    between them is the seat (a MixedBot plays the same inner bot in both);
    in a mirror match the seat number is appended so the two copies draw
    different streams. The other bot is part of the seed so a MixedBot's
    coin flips are independent across its pairings: seeded on ``k`` alone,
    it would play the same inner bot against every opponent at seed ``k``
    and its realised mix would rest on ``--seeds`` flips in all. A game's
    result depends only on the map, max_turns, the two bots, the seat
    assignment and ``k``: not on the opponent list, its order or
    ``--workers``.

    A bot that fails to end its turn has it ended for it (counted as
    ``forced_end_turns``). A bot that raises loses nothing: the game is
    recorded as an error, left out of W/D/L, and the script exits 1.

Examples::

    python scripts/eval/bot_ladder.py --config configs/ppo/bootstrap.yaml \\
        --seeds 25 --workers 4 --out-dir /tmp/ladder
    python scripts/eval/bot_ladder.py --maps maps/1v1/beginner.csv --max-turns 75 \\
        --opponents simple medium advanced master --seeds 10

Exit codes: 0 done; 1 a game raised or a bad argument value; 2 usage error
(argparse); 3 a stage was flagged and ``--fail-on-flag`` was given.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# Import this checkout's package, not whatever copy is installed.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from reinforcetactics.core.game_state import GameState  # noqa: E402
from reinforcetactics.game.bot_registry import (  # noqa: E402
    build_scripted,
    canonical_name,
    resolve_scripted,
    validate_scripted_kwargs,
)
from reinforcetactics.utils.file_io import FileIO  # noqa: E402

# The plain tiers every default ladder includes, weakest-intended first.
BASE_TIERS: tuple[str, ...] = ("noop", "balanced_random", "random", "simple", "medium", "advanced", "master")
# Bots whose games run long (random play rarely ends a game before
# max_turns); scheduled first so the pool's tail is short.
_SLOW_BOTS = frozenset({"random", "balanced_random", "mixed", "noop"})
DEFAULT_SEEDS = 25
DEFAULT_CHUNK = 5
Z_95 = 1.959963984540054
_KNOWN_END_REASONS = frozenset({"hq_capture", "elimination", "max_turns_draw"})
# Rating gap (Elo points) below which two stage opponents count as tied.
TIE_ELO = 1.0
EXIT_ERRORS = 1
EXIT_FLAGGED = 3


# --------------------------------------------------------------------------
# Opponents
# --------------------------------------------------------------------------


def _constructor_defaults(name: str) -> dict[str, Any]:
    """Keyword defaults of a scripted bot's constructor (``game_state``/``player``/``rng`` excluded)."""
    params = inspect.signature(resolve_scripted(name)).parameters.values()
    return {
        p.name: p.default
        for p in params
        if p.default is not inspect.Parameter.empty and p.name not in ("game_state", "player", "rng")
    }


@dataclass(frozen=True)
class OpponentSpec:
    """A scripted bot and its constructor kwargs, in canonical form.

    ``kwargs`` is stored as sorted JSON so the spec is hashable and
    picklable; kwargs equal to the constructor's defaults are dropped, so
    two spellings of the same bot compare equal.
    """

    name: str
    kwargs_json: str = "{}"

    @classmethod
    def create(cls, name: str, kwargs: Mapping[str, Any] | None = None) -> OpponentSpec:
        canon = canonical_name(name)
        kw = dict(kwargs or {})
        validate_scripted_kwargs(canon, kw)
        defaults = _constructor_defaults(canon)
        kw = {k: v for k, v in kw.items() if not (k in defaults and defaults[k] == v)}
        return cls(canon, json.dumps(kw, sort_keys=True))

    @classmethod
    def parse(cls, text: str) -> OpponentSpec:
        """``name``, ``name:key=value,key=value`` (values as YAML) or ``name:{json}``."""
        name, sep, rest = text.strip().partition(":")
        if not sep or not rest.strip():
            return cls.create(name)
        rest = rest.strip()
        if rest.startswith("{"):
            kwargs = json.loads(rest)
        else:
            kwargs = {}
            for item in rest.split(","):
                key, eq, value = item.partition("=")
                if not eq or not key.strip():
                    raise ValueError(f"bad opponent kwarg {item!r} in {text!r}; expected key=value")
                kwargs[key.strip()] = yaml.safe_load(value.strip())
        if not isinstance(kwargs, dict):
            raise ValueError(f"opponent kwargs in {text!r} must be a mapping")
        return cls.create(name, kwargs)

    @property
    def kwargs(self) -> dict[str, Any]:
        return json.loads(self.kwargs_json)

    @property
    def key(self) -> str:
        return f"{self.name}{self.kwargs_json}"

    @property
    def label(self) -> str:
        """Short display name: ``random_10``, ``mix(simple/medium,0.5)``, ``medium``."""
        return _label(self.name, self.kwargs)

    @property
    def slow(self) -> bool:
        inner = {self.name}
        if self.name == "mixed":
            kw = self.kwargs
            inner |= {kw.get("easy", "simple"), kw.get("hard", "medium")}
        return bool(inner & _SLOW_BOTS)

    def build(self, game: GameState, player: int, rng: random.Random) -> Any:
        return build_scripted(self.name, game, player=player, rng=rng, **self.kwargs)


def _label(name: str, kwargs: Mapping[str, Any]) -> str:
    kwargs = dict(kwargs)
    if name == "random":
        max_actions = kwargs.pop("max_actions", _constructor_defaults("random").get("max_actions"))
        base = f"random_{max_actions}"
    elif name == "mixed":
        defaults = _constructor_defaults("mixed")
        easy = _label(canonical_name(kwargs.pop("easy", defaults["easy"])), kwargs.pop("easy_kwargs", None) or {})
        hard = _label(canonical_name(kwargs.pop("hard", defaults["hard"])), kwargs.pop("hard_kwargs", None) or {})
        p_hard = kwargs.pop("p_hard", defaults["p_hard"])
        base = f"mix({easy}/{hard},{p_hard:g})"
    else:
        base = name
    if kwargs:
        base += "(" + ",".join(f"{k}={kwargs[k]}" for k in sorted(kwargs)) + ")"
    return base


def dedupe(specs: Iterable[OpponentSpec]) -> list[OpponentSpec]:
    seen: dict[str, OpponentSpec] = {}
    for spec in specs:
        seen.setdefault(spec.key, spec)
    return list(seen.values())


# --------------------------------------------------------------------------
# Boards and games
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Board:
    """One map at one game length, with the env settings the games run under."""

    map_file: str
    max_turns: int
    fog_of_war: bool = False
    enabled_units: tuple[str, ...] | None = None
    engine_overrides_json: str | None = None

    @property
    def label(self) -> str:
        return f"{Path(self.map_file).stem}@{self.max_turns}"

    @property
    def engine_overrides(self) -> dict[str, Any] | None:
        return json.loads(self.engine_overrides_json) if self.engine_overrides_json else None


def resolve_map_path(map_file: str) -> Path:
    """``map_file`` as given if it exists, else relative to the repository root."""
    path = Path(map_file)
    if path.is_file():
        return path
    candidate = REPO_ROOT / map_file
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"map file not found: {map_file}")


@lru_cache(maxsize=32)
def _map_data(map_file: str) -> Any:
    # GameState copies the tile codes into its own grid, so one DataFrame per
    # map serves every game in the worker (the gym env reuses it the same way).
    return FileIO.load_map(str(resolve_map_path(map_file)))


def bot_rng(seed: int, spec: OpponentSpec, other: OpponentSpec, mirror_seat: int | None = None) -> random.Random:
    """The rng ``spec`` plays with against ``other`` at ``seed`` (see Seeding in the module docstring)."""
    suffix = "" if mirror_seat is None else f":seat{mirror_seat}"
    return random.Random(f"ladder:{seed}:{spec.key}:vs:{other.key}{suffix}")


@dataclass
class GameRecord:
    """One game, from the point of view of the pairing's first bot (``a``)."""

    seed: int
    a_seat: int  # 1 or 2
    winner: str | None  # "a", "b" or None (draw); None too when error is set
    end_reason: str
    turns: int
    forced_end_turns: int = 0
    error: str | None = None

    def as_list(self) -> list[Any]:
        return [self.seed, self.a_seat, self.winner, self.end_reason, self.turns, self.forced_end_turns, self.error]


def play_game(board: Board, a: OpponentSpec, b: OpponentSpec, seed: int, a_seat: int) -> GameRecord:
    """Play one seeded game with ``a`` in seat ``a_seat`` and ``b`` in the other."""
    game = GameState(
        _map_data(board.map_file),
        num_players=2,
        max_turns=board.max_turns,
        enabled_units=list(board.enabled_units) if board.enabled_units is not None else None,
        fog_of_war=board.fog_of_war,
        engine_overrides=board.engine_overrides,
        seed=seed,
    )
    mirror = a == b
    seats = {a_seat: a, 3 - a_seat: b}
    bots = {
        seat: spec.build(game, seat, bot_rng(seed, spec, seats[3 - seat], mirror_seat=seat if mirror else None))
        for seat, spec in seats.items()
    }
    forced = 0
    try:
        # max_turns ends the game at the end of a full round; the bound is a
        # guard against an engine change that stops doing so.
        for _ in range(2 * board.max_turns + 2):
            if game.game_over:
                break
            player = game.current_player
            bots[player].take_turn()
            if not game.game_over and game.current_player == player:
                forced += 1
                game.end_turn()
    except Exception as exc:  # noqa: BLE001 -- recorded and reported, never counted as a result
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return GameRecord(seed, a_seat, None, "error", game.turn_number, forced, error=detail)
    if not game.game_over:
        return GameRecord(seed, a_seat, None, "unfinished", game.turn_number, forced)
    winner = None
    if game.winner is not None:
        winner = "a" if game.winner == a_seat else "b"
    return GameRecord(seed, a_seat, winner, game.end_reason or "unknown", game.turn_number, forced)


@dataclass(frozen=True)
class Job:
    board: Board
    a: OpponentSpec
    b: OpponentSpec
    seeds: tuple[int, ...]

    @property
    def cost(self) -> float:
        slow = int(self.a.slow) + int(self.b.slow)
        return self.board.max_turns * len(self.seeds) * (1 + 4 * slow)


def run_job(job: Job) -> tuple[Job, list[GameRecord]]:
    records = []
    for seed in job.seeds:
        for a_seat in (1, 2):
            records.append(play_game(job.board, job.a, job.b, seed, a_seat))
    return job, records


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class WDL:
    w: int = 0
    d: int = 0
    l: int = 0  # noqa: E741

    @property
    def n(self) -> int:
        return self.w + self.d + self.l

    @property
    def score(self) -> float:
        return self.w + 0.5 * self.d

    @property
    def score_rate(self) -> float:
        return self.score / self.n if self.n else float("nan")

    def add(self, winner: str | None, me: str) -> None:
        if winner is None:
            self.d += 1
        elif winner == me:
            self.w += 1
        else:
            self.l += 1

    def flipped(self) -> WDL:
        return WDL(self.l, self.d, self.w)

    def __str__(self) -> str:
        return f"{self.w}-{self.d}-{self.l}"

    def as_dict(self) -> dict[str, int]:
        return {"w": self.w, "d": self.d, "l": self.l}


@dataclass
class Pairing:
    """Every game between ``a`` and ``b`` on one board, W/D/L from ``a``'s side."""

    a: OpponentSpec
    b: OpponentSpec
    records: list[GameRecord] = field(default_factory=list)

    def _games(self) -> list[GameRecord]:
        return [r for r in self.records if r.error is None]

    def seat(self, a_seat: int) -> WDL:
        out = WDL()
        for r in self._games():
            if r.a_seat == a_seat:
                out.add(r.winner, "a")
        return out

    @property
    def total(self) -> WDL:
        out = WDL()
        for r in self._games():
            out.add(r.winner, "a")
        return out

    @property
    def end_reasons(self) -> Counter:
        return Counter(r.end_reason for r in self.records)

    @property
    def errors(self) -> list[GameRecord]:
        return [r for r in self.records if r.error is not None]

    @property
    def mean_turns(self) -> float:
        games = self._games()
        return sum(r.turns for r in games) / len(games) if games else float("nan")

    @property
    def forced_end_turns(self) -> int:
        return sum(r.forced_end_turns for r in self.records)

    @property
    def mirror(self) -> bool:
        return self.a == self.b


@dataclass
class Rating:
    """One bot's standing on a board.

    ``mean_score``: its score against each other bot (win 1, draw 0.5),
    averaged over the bots. ``beaten``: how often each other bot beat it
    (wins only; a draw does not count), averaged over the bots -- the
    number closest to a curriculum gate, which counts only wins, so a bot
    that holds games to a draw is hard to promote against even if it rarely
    wins itself.
    """

    spec: OpponentSpec
    elo: float
    elo_se: float
    mean_score: float
    beaten: float
    record: WDL


@dataclass
class BoardResult:
    board: Board
    opponents: list[OpponentSpec]
    pairings: dict[tuple[str, str], Pairing] = field(default_factory=dict)
    ratings: list[Rating] = field(default_factory=list)

    def pairing(self, x: OpponentSpec, y: OpponentSpec) -> tuple[Pairing | None, bool]:
        """The pairing of ``x`` and ``y`` and whether ``x`` is its ``b`` side."""
        if (x.key, y.key) in self.pairings:
            return self.pairings[(x.key, y.key)], False
        if (y.key, x.key) in self.pairings:
            return self.pairings[(y.key, x.key)], True
        return None, False

    def head_to_head(self, x: OpponentSpec, y: OpponentSpec) -> WDL | None:
        """``x``'s W/D/L against ``y`` over both seats (None if they did not meet)."""
        pairing, flipped = self.pairing(x, y)
        if pairing is None:
            return None
        return pairing.total.flipped() if flipped else pairing.total

    def seat_record(self, x: OpponentSpec, y: OpponentSpec) -> WDL | None:
        """``x``'s W/D/L in seat 1 against ``y`` in seat 2 (``x == y``: the mirror match's seat 1)."""
        pairing, flipped = self.pairing(x, y)
        if pairing is None:
            return None
        return pairing.seat(2).flipped() if flipped else pairing.seat(1)

    def rating_of(self, spec: OpponentSpec) -> Rating | None:
        return next((r for r in self.ratings if r.spec == spec), None)

    @property
    def seat1_score(self) -> WDL:
        """Seat 1's W/D/L over every game on the board, mirrors included (the first-move edge)."""
        out = WDL()
        for pairing in self.pairings.values():
            for r in pairing._games():
                out.add(None if r.winner is None else ("p1" if (r.winner == "a") == (r.a_seat == 1) else "p2"), "p1")
        return out


def wilson(successes: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for ``successes / n`` (fractional successes allowed: a draw is half)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bradley_terry(
    n_players: int,
    games: Mapping[tuple[int, int], tuple[float, int]],
    prior: float = 1.0,
    iterations: int = 10_000,
    tol: float = 1e-10,
) -> tuple[list[float], list[float]]:
    """Fit Bradley-Terry strengths by minorization-maximization (Hunter 2004).

    ``games[(i, j)] = (score of i against j, games played)`` for ``i != j``
    (only one orientation needed). Each player also plays ``2 * prior``
    virtual games against a fixed strength-1 opponent and scores half of
    them, which keeps every strength finite and positive when a player won
    or lost everything. Returns ``(elo, elo_se)``: ``400 * log10(strength)``
    centred on the mean, and an approximate standard error from the
    diagonal of the Fisher information.
    """
    n_ij = [[0.0] * n_players for _ in range(n_players)]
    wins = [prior] * n_players
    for (i, j), (score, n) in games.items():
        if i == j or n <= 0:
            continue
        n_ij[i][j] += n
        n_ij[j][i] += n
        wins[i] += score
        wins[j] += n - score
    strength = [1.0] * n_players
    for _ in range(iterations):
        new = []
        for i in range(n_players):
            denom = 2 * prior / (strength[i] + 1.0)
            denom += sum(n_ij[i][j] / (strength[i] + strength[j]) for j in range(n_players) if n_ij[i][j])
            new.append(wins[i] / denom)
        delta = max(abs(math.log(x) - math.log(y)) for x, y in zip(new, strength, strict=True)) if new else 0.0
        strength = new
        if delta < tol:
            break
    scale = 400 / math.log(10)
    elo = [scale * math.log(s) for s in strength]
    mean = sum(elo) / len(elo) if elo else 0.0
    se = []
    for i in range(n_players):
        info = 2 * prior * strength[i] / (strength[i] + 1.0) ** 2
        info += sum(
            n_ij[i][j] * strength[i] * strength[j] / (strength[i] + strength[j]) ** 2 for j in range(n_players) if n_ij[i][j]
        )
        se.append(scale / math.sqrt(info) if info > 0 else float("inf"))
    return [e - mean for e in elo], se


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def rate_board(result: BoardResult, prior: float = 1.0) -> None:
    """Fill ``result.ratings`` (weakest first) from its non-mirror pairings."""
    index = {spec.key: i for i, spec in enumerate(result.opponents)}
    games: dict[tuple[int, int], tuple[float, int]] = {}
    records = [WDL() for _ in result.opponents]
    per_opponent: list[list[float]] = [[] for _ in result.opponents]
    beaten: list[list[float]] = [[] for _ in result.opponents]
    for pairing in result.pairings.values():
        if pairing.mirror:
            continue
        total = pairing.total
        if not total.n:
            continue
        i, j = index[pairing.a.key], index[pairing.b.key]
        games[(i, j)] = (total.score, total.n)
        for side, wdl in ((i, total), (j, total.flipped())):
            records[side].w += wdl.w
            records[side].d += wdl.d
            records[side].l += wdl.l
            per_opponent[side].append(wdl.score_rate)
            beaten[side].append(wdl.l / wdl.n)
    elo, se = bradley_terry(len(result.opponents), games, prior=prior)
    ratings = [
        Rating(spec, elo[i], se[i], _mean(per_opponent[i]), _mean(beaten[i]), records[i])
        for i, spec in enumerate(result.opponents)
    ]
    result.ratings = sorted(ratings, key=lambda r: (r.elo, r.mean_score))


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


@dataclass
class LadderResult:
    boards: list[BoardResult]
    n_seeds: int
    seed_base: int
    elapsed_s: float = 0.0

    @property
    def errors(self) -> list[tuple[Board, Pairing, GameRecord]]:
        return [(br.board, p, r) for br in self.boards for p in br.pairings.values() for r in p.errors]


def make_jobs(
    boards: Sequence[Board],
    opponents: Sequence[OpponentSpec],
    seeds: Sequence[int],
    *,
    mirrors: bool = False,
    chunk: int = DEFAULT_CHUNK,
) -> list[Job]:
    jobs = []
    chunk = max(1, chunk)
    for board in boards:
        for i, a in enumerate(opponents):
            for b in opponents[i if mirrors else i + 1 :]:
                for start in range(0, len(seeds), chunk):
                    jobs.append(Job(board, a, b, tuple(seeds[start : start + chunk])))
    # Longest first: the pool then finishes on short jobs.
    jobs.sort(key=lambda job: -job.cost)
    return jobs


def run_ladder(
    boards: Sequence[Board],
    opponents: Sequence[OpponentSpec],
    *,
    n_seeds: int = DEFAULT_SEEDS,
    seed_base: int = 0,
    workers: int = 1,
    mirrors: bool = False,
    chunk: int = DEFAULT_CHUNK,
    prior: float = 1.0,
    progress: Callable[[int, int, float], None] | None = None,
) -> LadderResult:
    """Play every pairing of ``opponents`` on every board and rate the results.

    ``workers > 1`` plays the games in a process pool; the results are the
    same for any ``workers`` (each game is seeded on its own).
    """
    opponents = dedupe(opponents)
    if len(opponents) < 2 and not mirrors:
        raise ValueError("need at least two distinct opponents")
    seeds = list(range(seed_base, seed_base + n_seeds))
    jobs = make_jobs(boards, opponents, seeds, mirrors=mirrors, chunk=chunk)
    results = {board: BoardResult(board, list(opponents)) for board in boards}
    started = time.monotonic()

    def collect(job: Job, records: list[GameRecord]) -> None:
        br = results[job.board]
        pairing = br.pairings.setdefault((job.a.key, job.b.key), Pairing(job.a, job.b))
        pairing.records.extend(records)

    done = 0
    if workers <= 1:
        for job in jobs:
            collect(*run_job(job))
            done += 1
            if progress:
                progress(done, len(jobs), time.monotonic() - started)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                collect(*future.result())
                done += 1
                if progress:
                    progress(done, len(jobs), time.monotonic() - started)

    ordered = []
    for board in boards:
        br = results[board]
        # Deterministic order (the pool completes jobs in any order).
        br.pairings = {
            key: Pairing(p.a, p.b, sorted(p.records, key=lambda r: (r.seed, r.a_seat)))
            for key, p in sorted(br.pairings.items(), key=lambda kv: _pair_order(br.opponents, kv[1]))
        }
        rate_board(br, prior=prior)
        ordered.append(br)
    return LadderResult(ordered, n_seeds, seed_base, time.monotonic() - started)


def _pair_order(opponents: Sequence[OpponentSpec], pairing: Pairing) -> tuple[int, int]:
    keys = [o.key for o in opponents]
    return keys.index(pairing.a.key), keys.index(pairing.b.key)


# --------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageRef:
    """What the ladder needs from a curriculum stage."""

    name: str
    map_file: str
    max_turns: int
    opponent: OpponentSpec


@dataclass
class StageCheck:
    stage: StageRef
    board: Board
    previous: StageRef | None
    rating: float | None = None
    previous_rating: float | None = None
    beaten: float | None = None  # Rating.beaten of this stage's opponent
    previous_beaten: float | None = None
    head_to_head: WDL | None = None  # this stage's opponent vs the previous stage's
    ci: tuple[float, float] | None = None
    verdict: str = "first on map"  # first on map | same opponent | harder | tied | weaker

    @property
    def flagged(self) -> bool:
        return self.verdict == "weaker"

    @property
    def significance(self) -> str:
        """Whether the head-to-head backs the verdict at 95%: ``significant`` or ``within noise``."""
        if self.ci is None:
            return ""
        low, high = self.ci
        return "significant" if high < 0.5 or low > 0.5 else "within noise"


def stages_from_config(cfg: Any, max_turns_override: int | None = None, default_max_turns: int = 100) -> list[StageRef]:
    """The curriculum's stages as :class:`StageRef` (game length as the env would run it)."""
    refs = []
    for stage in cfg.curriculum.stages:
        max_turns = max_turns_override or stage.resolve_max_turns(cfg.env) or default_max_turns
        refs.append(
            StageRef(stage.name, stage.map_file, int(max_turns), OpponentSpec.create(stage.opponent, stage.opponent_kwargs))
        )
    return refs


def boards_for(
    map_turns: Iterable[tuple[str, int]],
    *,
    fog_of_war: bool = False,
    enabled_units: Sequence[str] | None = None,
    engine_overrides: Mapping[str, Any] | None = None,
) -> list[Board]:
    boards: dict[tuple[str, int], Board] = {}
    for map_file, max_turns in map_turns:
        boards.setdefault(
            (map_file, max_turns),
            Board(
                map_file,
                int(max_turns),
                bool(fog_of_war),
                tuple(enabled_units) if enabled_units is not None else None,
                json.dumps(engine_overrides, sort_keys=True) if engine_overrides else None,
            ),
        )
    return list(boards.values())


def board_of(result: LadderResult, map_file: str, max_turns: int) -> BoardResult | None:
    return next((br for br in result.boards if br.board.map_file == map_file and br.board.max_turns == max_turns), None)


def check_curriculum(stages: Sequence[StageRef], result: LadderResult) -> list[StageCheck]:
    """Compare each stage's opponent with the previous stage's on the same map.

    Both are rated on the current stage's board (the ladder plays every
    opponent on every board). A stage is flagged ``weaker`` when its
    opponent's rating is below the previous one's by at least ``TIE_ELO``
    (``tied`` within it: two bots that only ever draw each other rate the
    same up to rounding); ``significance`` says whether the direct
    head-to-head between the two confirms the order. Stages on a map the
    ladder did not play are skipped.
    """
    checks = []
    last_on_map: dict[str, StageRef] = {}
    for stage in stages:
        br = board_of(result, stage.map_file, stage.max_turns)
        previous = last_on_map.get(stage.map_file)
        last_on_map[stage.map_file] = stage
        if br is None:
            continue
        check = StageCheck(stage, br.board, previous)
        mine = br.rating_of(stage.opponent)
        if mine is not None:
            check.rating, check.beaten = mine.elo, mine.beaten
        if previous is None:
            checks.append(check)
            continue
        theirs = br.rating_of(previous.opponent)
        if theirs is not None:
            check.previous_rating, check.previous_beaten = theirs.elo, theirs.beaten
        if previous.opponent == stage.opponent:
            check.verdict = "same opponent"
        elif check.rating is not None and check.previous_rating is not None:
            diff = check.rating - check.previous_rating
            check.verdict = "tied" if abs(diff) < TIE_ELO else ("weaker" if diff < 0 else "harder")
            h2h = br.head_to_head(stage.opponent, previous.opponent)
            if h2h is not None and h2h.n:
                check.head_to_head = h2h
                check.ci = wilson(h2h.score, h2h.n)
        checks.append(check)
    return checks


def rating_sorted_order(stages: Sequence[StageRef], result: LadderResult) -> list[tuple[StageRef, float | None]]:
    """The stages re-sorted by opponent rating inside each map block (maps keep their first-use order).

    A stable sort, so stages with the same opponent keep their relative
    order. Mechanical: it ignores everything a stage is for besides its
    opponent's strength (bridge stages, entropy schedules, thresholds).
    """
    blocks: dict[str, list[tuple[int, StageRef, float | None]]] = defaultdict(list)
    for position, stage in enumerate(stages):
        br = board_of(result, stage.map_file, stage.max_turns)
        rating = br.rating_of(stage.opponent) if br else None
        blocks[stage.map_file].append((position, stage, rating.elo if rating else None))
    order: list[tuple[StageRef, float | None]] = []
    for block in blocks.values():
        block.sort(key=lambda item: (item[2] if item[2] is not None else math.inf, item[0]))
        order.extend((stage, elo) for _, stage, elo in block)
    return order


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _fmt(x: float | None, spec: str = "+.0f") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return format(x, spec)


def _board_markdown(br: BoardResult, h: str) -> list[str]:
    board = br.board
    lines: list[str] = []
    out = lines.append
    extras = []
    if board.fog_of_war:
        extras.append("fog of war")
    if board.enabled_units is not None:
        extras.append("units " + ",".join(board.enabled_units))
    if board.engine_overrides_json:
        extras.append("engine overrides")
    out(f"{h} {board.map_file}, max_turns {board.max_turns}" + (f" ({'; '.join(extras)})" if extras else ""))
    out("")
    seat1 = br.seat1_score
    if seat1.n:
        out(
            f"Seat 1 (the player who moves first) scores {seat1.score_rate:.1%} over every game on this board ({seat1} W-D-L)."
        )
        out("")
    out("Strength ordering, weakest first. Rating: Bradley-Terry, Elo scale, centred on 0 (± one standard error).")
    out("Mean score: the bot's score against each other bot (win 1, draw 0.5), averaged.")
    out("Beaten: how often each other bot beat it (wins only), averaged; what a win-rate gate sees.")
    out("")
    out("| # | Bot | Rating | Mean score | Beaten | W-D-L |")
    out("|---|---|---|---|---|---|")
    for rank, rating in enumerate(br.ratings, 1):
        out(
            f"| {rank} | {rating.spec.label} | {_fmt(rating.elo)} ± {_fmt(rating.elo_se, '.0f')} | "
            f"{_fmt(rating.mean_score, '.2f')} | {_fmt(rating.beaten, '.2f')} | {rating.record} |"
        )
    out("")
    ranked = [r.spec for r in br.ratings]
    out("Crosstable: the row bot's W-D-L against the column bot over both seats (columns numbered as the rows).")
    out("")
    lines += _crosstable(ranked, lambda row, col: None if row == col else br.head_to_head(row, col))
    out("")
    out(
        "By seat: the row bot's W-D-L in seat 1 (moving first) against the column bot in seat 2; "
        "the column bot's own seat-1 games are in its row. The diagonal is the mirror match, when played."
    )
    out("")
    lines += _crosstable(ranked, br.seat_record)
    out("")
    out("Per pairing and seat: W-D-L for A. End reasons over all games: hq = HQ capture, elim = elimination.")
    out("")
    out("| A | B | A total | A as P1 | A as P2 | A score | hq | elim | draw | mean turns |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    for pairing in br.pairings.values():
        reasons = pairing.end_reasons
        out(
            f"| {pairing.a.label} | {pairing.b.label} | {pairing.total} | {pairing.seat(1)} | {pairing.seat(2)} | "
            f"{_fmt(pairing.total.score_rate, '.2f')} | {reasons['hq_capture']} | {reasons['elimination']} | "
            f"{reasons['max_turns_draw']} | {_fmt(pairing.mean_turns, '.1f')} |"
        )
    other = sorted({reason for p in br.pairings.values() for reason in p.end_reasons} - _KNOWN_END_REASONS)
    if other:
        out("")
        out("Other end reasons: " + ", ".join(other) + ".")
    forced = sum(p.forced_end_turns for p in br.pairings.values())
    if forced:
        out("")
        out(f"Turns ended for a bot that did not end them: {forced}.")
    out("")
    return lines


def _crosstable(ranked: Sequence[OpponentSpec], cell: Callable[[OpponentSpec, OpponentSpec], WDL | None]) -> list[str]:
    lines = ["| # | Bot | " + " | ".join(str(i) for i in range(1, len(ranked) + 1)) + " |", "|---|---|" + "---|" * len(ranked)]
    for i, row in enumerate(ranked, 1):
        cells = []
        for col in ranked:
            wdl = cell(row, col)
            cells.append(str(wdl) if wdl is not None else ("·" if row == col else ""))
        lines.append(f"| {i} | {row.label} | " + " | ".join(cells) + " |")
    return lines


def _verdict_text(check: StageCheck) -> str:
    if check.flagged:
        return f"**FLAG: weaker** ({check.significance})" if check.significance else "**FLAG: weaker**"
    if check.verdict in ("harder", "tied") and check.significance:
        return f"{check.verdict} ({check.significance})"
    return check.verdict


def _checks_markdown(checks: Sequence[StageCheck], h: str) -> list[str]:
    lines = [
        f"{h} Curriculum check",
        "",
        "Each stage's opponent against the previous stage's on the same map, both rated on this stage's board. "
        "Beaten as in the ordering tables. H2H: this stage's opponent's W-D-L against the previous one and its "
        "score with the 95% Wilson interval; *significant* when the interval excludes 0.5.",
        "",
        "| Stage | Board | Opponent | Rating | Beaten | Previous stage (opponent) | Rating | Beaten | H2H | Score [95% CI] | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for check in checks:
        prev = f"{check.previous.name} ({check.previous.opponent.label})" if check.previous else "-"
        h2h = str(check.head_to_head) if check.head_to_head else "-"
        score = "-"
        if check.head_to_head and check.ci:
            score = f"{check.head_to_head.score_rate:.2f} [{check.ci[0]:.2f}, {check.ci[1]:.2f}]"
        lines.append(
            f"| {check.stage.name} | {check.board.label} | {check.stage.opponent.label} | {_fmt(check.rating)} | "
            f"{_fmt(check.beaten, '.2f')} | {prev} | {_fmt(check.previous_rating)} | {_fmt(check.previous_beaten, '.2f')} | "
            f"{h2h} | {score} | {_verdict_text(check)} |"
        )
    lines.append("")
    return lines


def render_markdown(
    result: LadderResult,
    checks: Sequence[StageCheck] = (),
    order: Sequence[tuple[StageRef, float | None]] = (),
    level: int = 2,
) -> str:
    """The report as markdown; ``level`` is the heading level of its sections."""
    h = "#" * level
    lines = [
        f"Seeds {result.seed_base}..{result.seed_base + result.n_seeds - 1}, both seats: "
        f"{2 * result.n_seeds} games per pairing.",
        "",
    ]
    for br in result.boards:
        lines += _board_markdown(br, h)
    if checks:
        lines += _checks_markdown(checks, h)
    if order:
        lines += [
            f"{h} Stages sorted by opponent rating within each map",
            "",
            "Mechanical (opponent strength only); read with the flags above, not as a curriculum edit.",
            "",
            "| # | Stage | Opponent | Rating |",
            "|---|---|---|---|",
        ]
        lines += [f"| {i} | {stage.name} | {stage.opponent.label} | {_fmt(elo)} |" for i, (stage, elo) in enumerate(order, 1)]
        lines.append("")
    if result.errors:
        lines += [f"{h} Errors", ""]
        lines += [
            f"- {board.label} {pairing.a.label} vs {pairing.b.label} seed {record.seed} (A in seat {record.a_seat}): {record.error}"
            for board, pairing, record in result.errors[:50]
        ]
        lines.append("")
    return "\n".join(lines)


CSV_FIELDS = (
    "map_file",
    "max_turns",
    "a",
    "b",
    "games",
    "a_w",
    "a_d",
    "a_l",
    "a_p1_w",
    "a_p1_d",
    "a_p1_l",
    "a_p2_w",
    "a_p2_d",
    "a_p2_l",
    "a_score",
    "hq_capture",
    "elimination",
    "max_turns_draw",
    "other_end",
    "errors",
    "mean_turns",
    "forced_end_turns",
)


def write_csv(result: LadderResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for br in result.boards:
            for pairing in br.pairings.values():
                total, p1, p2 = pairing.total, pairing.seat(1), pairing.seat(2)
                reasons = pairing.end_reasons
                known = sum(reasons[r] for r in _KNOWN_END_REASONS) + reasons["error"]
                writer.writerow(
                    {
                        "map_file": br.board.map_file,
                        "max_turns": br.board.max_turns,
                        "a": pairing.a.label,
                        "b": pairing.b.label,
                        "games": total.n,
                        "a_w": total.w,
                        "a_d": total.d,
                        "a_l": total.l,
                        "a_p1_w": p1.w,
                        "a_p1_d": p1.d,
                        "a_p1_l": p1.l,
                        "a_p2_w": p2.w,
                        "a_p2_d": p2.d,
                        "a_p2_l": p2.l,
                        "a_score": round(total.score_rate, 4) if total.n else "",
                        "hq_capture": reasons["hq_capture"],
                        "elimination": reasons["elimination"],
                        "max_turns_draw": reasons["max_turns_draw"],
                        "other_end": sum(reasons.values()) - known,
                        "errors": len(pairing.errors),
                        "mean_turns": round(pairing.mean_turns, 2) if total.n else "",
                        "forced_end_turns": pairing.forced_end_turns,
                    }
                )


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _spec_json(spec: OpponentSpec) -> dict[str, Any]:
    return {"label": spec.label, "name": spec.name, "kwargs": spec.kwargs}


def _stage_json(stage: StageRef | None) -> dict[str, Any] | None:
    if stage is None:
        return None
    return {"name": stage.name, "map_file": stage.map_file, "max_turns": stage.max_turns, "opponent": stage.opponent.label}


def to_json(
    result: LadderResult,
    checks: Sequence[StageCheck] = (),
    order: Sequence[tuple[StageRef, float | None]] = (),
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def num(x: float | None) -> float | None:
        return None if x is None or math.isnan(x) or math.isinf(x) else round(x, 3)

    return {
        "meta": {
            **dict(meta or {}),
            "n_seeds": result.n_seeds,
            "seed_base": result.seed_base,
            "games_per_pairing": 2 * result.n_seeds,
            "elapsed_s": round(result.elapsed_s, 1),
            "git_commit": _git_commit(),
            "game_record_fields": ["seed", "a_seat", "winner", "end_reason", "turns", "forced_end_turns", "error"],
        },
        "boards": [
            {
                "map_file": br.board.map_file,
                "max_turns": br.board.max_turns,
                "fog_of_war": br.board.fog_of_war,
                "enabled_units": list(br.board.enabled_units) if br.board.enabled_units is not None else None,
                "engine_overrides": br.board.engine_overrides,
                "opponents": [_spec_json(s) for s in br.opponents],
                "seat1": br.seat1_score.as_dict(),
                "ratings": [
                    {
                        "opponent": r.spec.label,
                        "elo": num(r.elo),
                        "elo_se": num(r.elo_se),
                        "mean_score": num(r.mean_score),
                        "beaten": num(r.beaten),
                        "record": r.record.as_dict(),
                    }
                    for r in br.ratings
                ],
                "pairings": [
                    {
                        "a": p.a.label,
                        "b": p.b.label,
                        # Indices into this board's "opponents" (load_json reads these).
                        "a_index": br.opponents.index(p.a),
                        "b_index": br.opponents.index(p.b),
                        "total": p.total.as_dict(),
                        "a_as_p1": p.seat(1).as_dict(),
                        "a_as_p2": p.seat(2).as_dict(),
                        "end_reasons": dict(p.end_reasons),
                        "mean_turns": num(p.mean_turns),
                        "forced_end_turns": p.forced_end_turns,
                        "games": [r.as_list() for r in p.records],
                    }
                    for p in br.pairings.values()
                ],
            }
            for br in result.boards
        ],
        "curriculum_check": [
            {
                "stage": _stage_json(c.stage),
                "previous": _stage_json(c.previous),
                "board": c.board.label,
                "rating": num(c.rating),
                "previous_rating": num(c.previous_rating),
                "beaten": num(c.beaten),
                "previous_beaten": num(c.previous_beaten),
                "head_to_head": c.head_to_head.as_dict() if c.head_to_head else None,
                "ci95": [round(x, 4) for x in c.ci] if c.ci else None,
                "verdict": c.verdict,
                "flagged": c.flagged,
                "significance": c.significance,
            }
            for c in checks
        ],
        "rating_sorted_order": [{"stage": s.name, "opponent": s.opponent.label, "rating": num(elo)} for s, elo in order],
        "errors": [
            {"board": b.label, "a": p.a.label, "b": p.b.label, "seed": r.seed, "a_seat": r.a_seat, "error": r.error}
            for b, p, r in result.errors
        ],
    }


def load_json(data: Mapping[str, Any], prior: float = 1.0) -> LadderResult:
    """Rebuild a :class:`LadderResult` from :func:`to_json` output (the games, re-rated with ``prior``)."""
    boards = []
    for raw in data["boards"]:
        overrides = raw.get("engine_overrides")
        units = raw.get("enabled_units")
        board = Board(
            raw["map_file"],
            int(raw["max_turns"]),
            bool(raw.get("fog_of_war", False)),
            tuple(units) if units is not None else None,
            json.dumps(overrides, sort_keys=True) if overrides else None,
        )
        opponents = [OpponentSpec.create(o["name"], o["kwargs"]) for o in raw["opponents"]]
        br = BoardResult(board, opponents)
        for pairing in raw["pairings"]:
            a, b = opponents[pairing["a_index"]], opponents[pairing["b_index"]]
            records = [GameRecord(*game) for game in pairing["games"]]
            br.pairings[(a.key, b.key)] = Pairing(a, b, records)
        rate_board(br, prior=prior)
        boards.append(br)
    meta = data.get("meta", {})
    return LadderResult(boards, int(meta.get("n_seeds", 0)), int(meta.get("seed_base", 0)), float(meta.get("elapsed_s", 0.0)))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Seeded cross-tier matches between scripted bots on a curriculum's maps (both seats).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="Curriculum config (e.g. configs/ppo/bootstrap.yaml): default maps, opponents, max_turns")
    p.add_argument("--maps", nargs="+", help="Map files to play (default: every map the config's stages use)")
    p.add_argument(
        "--opponents",
        nargs="+",
        help="Replace the opponent list; each 'name', 'name:key=value,...' or 'name:{json}' "
        f"(default: the config's stage opponents plus {' '.join(BASE_TIERS)})",
    )
    p.add_argument("--add-opponents", nargs="+", default=[], help="Extra opponents on top of the default list")
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Seeds per pairing; each seed plays both seats")
    p.add_argument("--seed-base", type=int, default=0, help="First seed")
    p.add_argument("--max-turns", type=int, help="Game length for every board (default: each stage's max_turns)")
    p.add_argument(
        "--default-max-turns",
        type=int,
        default=100,
        help="Game length for a map with no max_turns from the config (and no --max-turns)",
    )
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Worker processes (1: in-process)")
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="Seeds per worker job")
    p.add_argument("--mirrors", action="store_true", help="Also play each bot against itself (left out of the ratings)")
    p.add_argument("--prior", type=float, default=1.0, help="Bradley-Terry prior: virtual wins (and losses) per bot")
    p.add_argument("--out-dir", help="Write ladder.md, ladder.csv and ladder.json here")
    p.add_argument("--fail-on-flag", action="store_true", help="Exit 3 when a curriculum stage is flagged")
    p.add_argument(
        "--from-json",
        help="Re-rate and re-render a previous run's ladder.json instead of playing (--config adds the curriculum check)",
    )
    p.add_argument("--quiet", action="store_true", help="No progress lines on stderr")
    return p


def _load_config(path: str) -> Any:
    # Deferred: the rl package imports torch and SB3 (seconds), which a
    # --maps-only run does not need.
    from reinforcetactics.rl.config import load_config

    return load_config(path)


def resolve_inputs(args: argparse.Namespace) -> tuple[list[Board], list[OpponentSpec], list[StageRef]]:
    """Boards, opponents and curriculum stages from the parsed arguments."""
    cfg = _load_config(args.config) if args.config else None
    stages = stages_from_config(cfg, args.max_turns, args.default_max_turns) if cfg is not None else []

    if args.maps:
        # A --maps entry naming a stage's map under another spelling
        # (./maps/..., an absolute path) takes the stage's spelling, so the
        # curriculum check finds its board.
        spelling = {resolve_map_path(s.map_file).resolve(): s.map_file for s in stages}
        map_turns: list[tuple[str, int]] = []
        for given in args.maps:
            map_file = spelling.get(resolve_map_path(given).resolve(), given)
            turns = sorted({s.max_turns for s in stages if s.map_file == map_file})
            if args.max_turns:
                turns = [args.max_turns]
            elif not turns:
                env_turns = cfg.env.max_turns if cfg is not None else None
                turns = [env_turns or args.default_max_turns]
            map_turns.extend((map_file, t) for t in turns)
    elif stages:
        map_turns = [(s.map_file, s.max_turns) for s in stages]
    else:
        raise ValueError("give --config or --maps")
    for map_file, _ in map_turns:
        resolve_map_path(map_file)

    env = cfg.env if cfg is not None else None
    boards = boards_for(
        map_turns,
        fog_of_war=bool(env.fog_of_war) if env is not None else False,
        enabled_units=env.enabled_units if env is not None else None,
        engine_overrides=env.engine_overrides if env is not None else None,
    )

    if args.opponents:
        opponents = [OpponentSpec.parse(text) for text in args.opponents]
    else:
        opponents = [s.opponent for s in stages] + [OpponentSpec.create(name) for name in BASE_TIERS]
    opponents += [OpponentSpec.parse(text) for text in args.add_opponents]
    opponents = dedupe(opponents)
    return boards, opponents, stages


def _progress(done: int, total: int, elapsed: float) -> None:
    step = max(1, total // 50)
    if done == total or done % step == 0:
        eta = elapsed / done * (total - done) if done else 0.0
        print(f"[bot_ladder] {done}/{total} jobs, {elapsed:.0f}s elapsed, ~{eta:.0f}s left", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    if args.max_turns is not None and args.max_turns < 1:
        parser.error("--max-turns must be >= 1")
    try:
        if args.from_json:
            result = load_json(json.loads(Path(args.from_json).read_text(encoding="utf-8")), prior=args.prior)
            cfg = _load_config(args.config) if args.config else None
            stages = stages_from_config(cfg, args.max_turns, args.default_max_turns) if cfg is not None else []
        else:
            boards, opponents, stages = resolve_inputs(args)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        print(f"bot_ladder: {exc}", file=sys.stderr)
        return EXIT_ERRORS

    if not args.from_json:
        if not args.quiet:
            print(
                f"[bot_ladder] {len(boards)} board(s) x {len(opponents)} bots x {args.seeds} seeds x 2 seats, "
                f"{args.workers} worker(s): {', '.join(b.label for b in boards)}",
                file=sys.stderr,
                flush=True,
            )
        result = run_ladder(
            boards,
            opponents,
            n_seeds=args.seeds,
            seed_base=args.seed_base,
            workers=args.workers,
            mirrors=args.mirrors,
            chunk=args.chunk,
            prior=args.prior,
            progress=None if args.quiet else _progress,
        )
    # Stages on a map the ladder did not play (left out by --maps) are not checked.
    played = {br.board.map_file for br in result.boards}
    stages = [s for s in stages if s.map_file in played]
    checks = check_curriculum(stages, result)
    order = rating_sorted_order(stages, result)
    markdown = render_markdown(result, checks, order)
    print(markdown)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ladder.md").write_text(markdown + "\n", encoding="utf-8")
        write_csv(result, out / "ladder.csv")
        meta = {"config": args.config, "argv": list(argv) if argv is not None else sys.argv[1:], "workers": args.workers}
        # Compact: the per-game records run to tens of thousands of rows.
        payload = json.dumps(to_json(result, checks, order, meta), separators=(",", ":"))
        (out / "ladder.json").write_text(payload + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"[bot_ladder] wrote {out}/ladder.md, ladder.csv, ladder.json", file=sys.stderr)
    if result.errors:
        print(f"bot_ladder: {len(result.errors)} game(s) raised; see the Errors section", file=sys.stderr)
        return EXIT_ERRORS
    if args.fail_on_flag and any(c.flagged for c in checks):
        return EXIT_FLAGGED
    return 0


if __name__ == "__main__":
    sys.exit(main())
