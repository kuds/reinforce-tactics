"""Reinforce Tactics - Turn-Based Strategy Game"""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _read_pyproject_version() -> str:
    """Read ``version = "X.Y.Z"`` from the repo's pyproject.toml.

    importlib.metadata only sees the version if the package was installed
    (``pip install -e .`` etc.), so source-from-clone runs -- notebooks,
    training scripts, the tournament runner invoked from a fresh checkout
    -- otherwise fall through to ``0.0.0+unknown`` and lose version
    provenance in saved replays. Parsing the project file directly keeps
    those runs honest without adding a hard ``tomllib``/``tomli`` dependency
    (we just need one line out of the file).
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        return "0.0.0+unknown"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return "0.0.0+unknown"
    # Match the first ``version = "..."`` under [project]. Anchored so a
    # tool's own ``version = "..."`` in a later table can't shadow it.
    match = re.search(r'(?ms)^\[project\][^\[]*?^version\s*=\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    return "0.0.0+unknown"


try:
    __version__ = version("reinforcetactics")
except PackageNotFoundError:
    __version__ = _read_pyproject_version()
