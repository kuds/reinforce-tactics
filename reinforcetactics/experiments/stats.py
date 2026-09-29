"""Small statistics helpers for the seed-replication tooling (stdlib only).

``reinforcetactics.rl`` pulls in torch and SB3 at import, so the aggregator
(``run_summary``) cannot borrow ``rl.evaluation.wilson_lower_bound``; the
formula is repeated here and ``tests/test_run_summary.py`` checks that both
give the same bound. There is no scipy dependency, so the Student-t quantiles
of the across-seed confidence intervals come from a table.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from statistics import NormalDist
from typing import Any

# Two-sided 95% Student-t quantiles t(0.975, df) for df = 1..30. Above 30 the
# normal quantile is used (1.96; t(0.975, 30) is 2.042, so the error there is
# under 4% of the interval's half-width).
T_975: tuple[float, ...] = (
    12.7062,
    4.3027,
    3.1824,
    2.7764,
    2.5706,
    2.4469,
    2.3646,
    2.3060,
    2.2622,
    2.2281,
    2.2010,
    2.1788,
    2.1604,
    2.1448,
    2.1314,
    2.1199,
    2.1098,
    2.1009,
    2.0930,
    2.0860,
    2.0796,
    2.0739,
    2.0687,
    2.0639,
    2.0595,
    2.0555,
    2.0518,
    2.0484,
    2.0452,
    2.0423,
)


def t_quantile_975(df: int) -> float:
    """t(0.975, df): the multiplier of a two-sided 95% interval with ``df`` degrees of freedom."""
    if df < 1:
        raise ValueError(f"degrees of freedom must be >= 1, got {df}")
    if df <= len(T_975):
        return T_975[df - 1]
    return float(NormalDist().inv_cdf(0.975))


def z_two_sided(confidence: float) -> float:
    """The normal quantile of a two-sided interval at ``confidence`` (0.95 -> 1.960)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return float(NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0))


def wilson_interval(successes: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score interval for ``successes`` out of ``n`` at ``confidence``.

    The same formula as ``reinforcetactics.rl.evaluation.wilson_lower_bound``
    (whose ``z`` here is the two-sided quantile). ``n == 0`` gives (0, 1):
    nothing measured.
    """
    if n <= 0:
        return 0.0, 1.0
    z = z_two_sided(confidence)
    p = min(max(float(successes) / n, 0.0), 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def finite(values: Iterable[Any]) -> list[float]:
    """The values that are real, finite numbers (bools and None dropped), as floats."""
    out: list[float] = []
    for v in values:
        if isinstance(v, bool) or v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def describe(values: Sequence[Any]) -> dict[str, Any]:
    """n, mean, sd (sample), min, max, median and a two-sided 95% t-interval of the finite ``values``.

    The interval (``ci95_lo`` / ``ci95_hi``) needs n >= 2; with fewer values
    it and ``sd`` are None. Everything is None for an empty input.
    """
    xs = finite(values)
    n = len(xs)
    out: dict[str, Any] = {
        "n": n,
        "mean": None,
        "sd": None,
        "min": None,
        "max": None,
        "median": None,
        "ci95_lo": None,
        "ci95_hi": None,
    }
    if n == 0:
        return out
    mean = statistics.fmean(xs)
    out.update(mean=mean, min=min(xs), max=max(xs), median=statistics.median(xs))
    if n >= 2:
        sd = statistics.stdev(xs)
        half = t_quantile_975(n - 1) * sd / math.sqrt(n)
        out.update(sd=sd, ci95_lo=mean - half, ci95_hi=mean + half)
    return out


def median(values: Sequence[Any]) -> float | None:
    xs = finite(values)
    return statistics.median(xs) if xs else None
