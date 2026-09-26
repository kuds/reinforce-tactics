"""Every third-party import in shipped code must be a declared dependency.

``tqdm`` and ``rich`` were used but declared nowhere: SB3's
``model.learn(progress_bar=True)`` imports both, so the Docker image's
default ``CMD`` (``main.py --mode train``), the Vertex submit script's default
job and the training scripts built the env and the model and then died with
``ImportError`` at the start of training. ``scripts/eval_agent.py`` imported
``tqdm`` directly and failed on ``--help``. Nothing exercised those entry
points, and ``test_packaging.py`` only diffs requirements.txt against
pyproject, so CI stayed green.

This module scans the source with ``ast`` (nothing is imported, so the
result does not depend on what happens to be installed) and checks each
third-party import against ``pyproject.toml``.
"""

import ast
import functools
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

# Code that ships in the package, the Docker images or the documented entry
# points. tests/ is not scanned (test-only packages belong in the [dev]
# extra), nor are notebooks (they pip-install what they need in a cell).
SCANNED_ROOTS = ("reinforcetactics", "scripts", "examples", "docker", "main.py")

FIRST_PARTY = frozenset({"reinforcetactics"})

# Import name -> distribution name where the two differ. Keys are dotted
# module prefixes and the longest matching prefix wins, so the ``google``
# namespace maps per sub-package: a future ``from google.cloud import
# bigquery`` must not pass because google-cloud-storage is declared. An import
# not listed here maps to its own top-level name after PEP 503 normalisation,
# which is already right for numpy, torch, gymnasium, pandas, ...
IMPORT_TO_DISTRIBUTION = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "pygame": "pygame-ce",
    "google.cloud.storage": "google-cloud-storage",
    "google.genai": "google-genai",
    "sb3_contrib": "sb3-contrib",
    "stable_baselines3": "stable-baselines3",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "IPython": "ipython",
}

# Exception names whose handler can swallow a failed import. ``Exception``
# and a bare ``except:`` count: ``run_config.py`` probes psutil that way.
_IMPORT_ERROR_NAMES = frozenset({"ImportError", "ModuleNotFoundError", "Exception", "BaseException"})


@dataclass(frozen=True)
class ImportSite:
    """One imported module path, where it was imported, and whether it is optional."""

    module: str
    path: Path
    lineno: int
    optional: bool

    def __str__(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}:{self.lineno} imports {self.module}"


def _normalize(name: str) -> str:
    """PEP 503 normalisation, so Pillow == pillow and sb3_contrib == sb3-contrib."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names(lines: list[str]) -> set[str]:
    """Normalised distribution names from requirement strings (versions, extras, markers stripped)."""
    names = set()
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[\[<>=!~;\s]", line, maxsplit=1)[0]
        if name:
            names.add(_normalize(name))
    return names


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:  # bare ``except:``
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    for exc in types:
        name = exc.id if isinstance(exc, ast.Name) else exc.attr if isinstance(exc, ast.Attribute) else None
        if name in _IMPORT_ERROR_NAMES:
            return True
    return False


def _has_fallback(try_node: ast.Try | ast.TryStar) -> bool:
    """True when some handler catches a failed import and carries on.

    A handler that re-raises (``except ImportError: raise ImportError("pip
    install ...")``) is a nicer error message, not a fallback: the code path
    still needs the package, so its import stays required.
    """
    return any(
        _catches_import_error(handler) and not any(isinstance(stmt, ast.Raise) for stmt in handler.body)
        for handler in try_node.handlers
    )


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _collect(node: ast.AST, path: Path, optional: bool, sites: list[ImportSite]) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
        # A body defined inside a ``try`` runs when it is called, outside that
        # try, so an enclosing handler guards none of its imports.
        optional = False
    if isinstance(node, ast.If) and _is_type_checking(node.test):
        # ``if TYPE_CHECKING:`` bodies never run; only the else branch does.
        for stmt in node.orelse:
            _collect(stmt, path, optional, sites)
        return
    if isinstance(node, (ast.Try, ast.TryStar)):
        guarded = optional or _has_fallback(node)
        for stmt in node.body:
            _collect(stmt, path, guarded, sites)
        for other in (*node.handlers, *node.orelse, *node.finalbody):
            _collect(other, path, optional, sites)
        return
    if isinstance(node, ast.Import):
        sites.extend(ImportSite(alias.name, path, node.lineno, optional) for alias in node.names)
        return
    if isinstance(node, ast.ImportFrom):
        if node.level or not node.module:  # relative imports are first-party by definition
            return
        # ``from google import genai`` names the module ``google.genai``; keep
        # the imported name so namespace packages resolve to the right distribution.
        modules = [f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*"] or [node.module]
        sites.extend(ImportSite(module, path, node.lineno, optional) for module in modules)
        return
    for child in ast.iter_child_nodes(node):
        _collect(child, path, optional, sites)


def scan_source(source: str, path: Path) -> list[ImportSite]:
    """All absolute imports in ``source``, flagged optional when a handler falls back."""
    sites: list[ImportSite] = []
    _collect(ast.parse(source, filename=str(path)), path, False, sites)
    return sites


@functools.cache
def _local_modules(directory: Path) -> frozenset[str]:
    """Modules a script can import from its own directory (``sys.path[0]`` when run directly)."""
    modules = {p.stem for p in directory.glob("*.py")}
    modules |= {p.name for p in directory.iterdir() if (p / "__init__.py").is_file()}
    return frozenset(modules)


def is_third_party(site: ImportSite) -> bool:
    top = site.module.split(".", 1)[0]
    if top in sys.stdlib_module_names or top in FIRST_PARTY:
        return False
    # Only a script run directly can import its neighbours by bare name.
    # Inside the package, ``import foo`` is absolute and never means a sibling
    # foo.py, so a same-named module there must not hide a missing dependency.
    if site.path.is_relative_to(REPO_ROOT / "reinforcetactics"):
        return True
    return top not in _local_modules(site.path.parent)


def distribution_for(module: str) -> str:
    """Normalised distribution name providing ``module`` (longest mapped prefix wins)."""
    parts = module.split(".")
    for end in range(len(parts), 0, -1):
        prefix = ".".join(parts[:end])
        if prefix in IMPORT_TO_DISTRIBUTION:
            return _normalize(IMPORT_TO_DISTRIBUTION[prefix])
    return _normalize(parts[0])


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for root in SCANNED_ROOTS:
        path = REPO_ROOT / root
        files.extend([path] if path.is_file() else sorted(path.rglob("*.py")))
    return files


def _progress_bar_calls(source: str, path: Path) -> list[str]:
    """``file:line`` of every call passing a literal ``progress_bar=True``."""
    return [
        f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
        for node in ast.walk(ast.parse(source, filename=str(path)))
        if isinstance(node, ast.Call)
        and any(
            kw.arg == "progress_bar" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in node.keywords
        )
    ]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


@pytest.fixture(scope="module")
def core_dependencies(pyproject) -> set[str]:
    return _requirement_names(pyproject["project"]["dependencies"])


@pytest.fixture(scope="module")
def declared_distributions(pyproject, core_dependencies) -> set[str]:
    """[project].dependencies plus every extra (``all`` only re-exports the others)."""
    declared = set(core_dependencies)
    for extra, requirements in pyproject["project"].get("optional-dependencies", {}).items():
        if extra != "all":
            declared |= _requirement_names(requirements)
    return declared


@pytest.fixture(scope="module")
def sources() -> dict[Path, str]:
    return {path: path.read_text(encoding="utf-8") for path in _scanned_files()}


def test_scan_covers_the_entry_points(sources):
    """Guard against the scan silently finding nothing (a renamed root, a bad glob)."""
    for expected in (
        "main.py",
        "scripts/eval_agent.py",
        "reinforcetactics/cli/commands.py",
        "docker/tournament/run_tournament.py",
    ):
        assert REPO_ROOT / expected in sources, f"{expected} is not scanned"


def test_every_third_party_import_is_declared(sources, declared_distributions):
    undeclared = []
    for path, source in sources.items():
        for site in scan_source(source, path):
            if site.optional or not is_third_party(site):
                continue
            distribution = distribution_for(site.module)
            if distribution not in declared_distributions:
                undeclared.append(f"{site} -> {distribution}")
    assert not undeclared, (
        "third-party imports with no pyproject dependency or extra:\n  "
        + "\n  ".join(undeclared)
        + "\nDeclare the distribution in [project].dependencies (and requirements.txt) or the "
        "matching extra. If the import name differs from the distribution name, add it to "
        "IMPORT_TO_DISTRIBUTION; if the import is genuinely optional, guard it with "
        "try/except ImportError and a fallback."
    )


def test_progress_bar_dependencies_are_core(sources, core_dependencies):
    """SB3 needs tqdm and rich for progress_bar=True; they must ship with the base install.

    Core rather than an extra: ``reinforcetactics/cli/commands.py`` (the Docker
    image's default command) passes ``progress_bar=True``, and the image
    installs only requirements.txt plus ``[cloud]``.
    """
    calls = [call for path, source in sources.items() for call in _progress_bar_calls(source, path)]
    if not calls:
        pytest.skip("no source passes progress_bar=True")
    required = _requirement_names(REQUIREMENTS.read_text(encoding="utf-8").splitlines())
    for distribution in ("tqdm", "rich"):
        assert distribution in core_dependencies, (
            f"{distribution} is not in [project].dependencies, but SB3's progress bar needs it at: {calls}"
        )
        assert distribution in required, f"{distribution} is not in requirements.txt (CI and the Docker images install it)"


# ---------------------------------------------------------------------------
# The scanner itself: pin the rules the checks above rely on.
# ---------------------------------------------------------------------------


def _scan(source: str) -> list[ImportSite]:
    return scan_source(source, REPO_ROOT / "reinforcetactics" / "synthetic.py")


def test_scanner_treats_import_with_fallback_as_optional():
    sites = _scan("try:\n    import wandb\nexcept ImportError:\n    wandb = None\n")
    assert [(s.module, s.optional) for s in sites] == [("wandb", True)]


def test_scanner_keeps_reraised_import_required():
    source = (
        "try:\n    import sb3_contrib\nexcept ImportError as exc:\n    raise ImportError('pip install sb3-contrib') from exc\n"
    )
    assert [(s.module, s.optional) for s in _scan(source)] == [("sb3_contrib", False)]


def test_scanner_does_not_guard_functions_defined_inside_try():
    source = "try:\n    def load():\n        import wandb\nexcept ImportError:\n    load = None\n"
    assert [(s.module, s.optional) for s in _scan(source)] == [("wandb", False)]


def test_scanner_skips_type_checking_and_relative_imports():
    source = (
        "from typing import TYPE_CHECKING\n"
        "from . import sibling\n"
        "if TYPE_CHECKING:\n    import pandas\nelse:\n    import numpy\n"
    )
    assert [s.module for s in _scan(source)] == ["typing.TYPE_CHECKING", "numpy"]


def test_scanner_finds_function_level_imports():
    """Lazy imports inside functions still crash at call time when missing."""
    assert [s.module for s in _scan("def train():\n    from tqdm import tqdm\n")] == ["tqdm.tqdm"]


@pytest.mark.parametrize(
    ("module", "distribution"),
    [
        ("google.cloud.storage", "google-cloud-storage"),
        ("google.genai.types", "google-genai"),
        ("google.cloud.bigquery", "google"),
        ("cv2", "opencv-python"),
        ("PIL.Image", "pillow"),
        ("yaml", "pyyaml"),
        ("pygame.font", "pygame-ce"),
        ("stable_baselines3.common.callbacks", "stable-baselines3"),
        ("numpy.random", "numpy"),
    ],
)
def test_distribution_for(module, distribution):
    assert distribution_for(module) == distribution


def test_stdlib_and_first_party_are_not_third_party():
    path = REPO_ROOT / "reinforcetactics" / "synthetic.py"
    assert not is_third_party(ImportSite("os.path", path, 1, False))
    assert not is_third_party(ImportSite("reinforcetactics.rl.bootstrap", path, 1, False))
    assert is_third_party(ImportSite("tqdm", path, 1, False))


def test_progress_bar_detector():
    path = REPO_ROOT / "reinforcetactics" / "synthetic.py"
    source = "model.learn(10, progress_bar=True)\nmodel.learn(10, progress_bar=False)\nmodel.learn(10, progress_bar=flag)\n"
    assert _progress_bar_calls(source, path) == ["reinforcetactics/synthetic.py:1"]
