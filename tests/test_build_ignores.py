"""Secrets and bulky local outputs stay out of Docker images, Cloud Build uploads and git.

``.dockerignore`` used to let ``docker/tournament/.env`` into both images:
each Dockerfile does ``COPY . .`` with the repo root as its build context,
and ``docker/tournament/README.md`` tells users to put their
OpenAI/Anthropic/Google API keys in that file and then push the image.

The checks below evaluate the ignore files with the same matching rules as
Docker and git, so a reordered or overridden pattern fails here even if the
line itself is still present. Neither tool is run: the rules are small enough
to reimplement, and the tests then do not depend on a daemon or checkout.
"""

import posixpath
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Must never reach an image, a Cloud Build upload or a commit.
SECRET_PATHS = [
    ".env",
    "docker/tournament/.env",
    "docker/tournament/.env.local",
    "scripts/cloud/.env.production",
    "gcp-credentials.json",
    "configs/application_default_credentials.json",
    "docker/tournament/my-project-service-account.json",
]

# Tournament results and LLM conversation logs: runtime output, mounted as a volume.
TOURNAMENT_OUTPUT = "docker/tournament/output/conversations/game_001.json"

# train_bootstrap.py's default run directory (critic-gaps-6).
BOOTSTRAP_RUN_FILE = "benchmarks/bootstrap/20260926_120000/checkpoints/stage_1.zip"

# What the Dockerfiles need from the build context. The template for the
# secrets file is harmless and documents the variables, so it stays.
SHIPPED_PATHS = [
    "Dockerfile",
    ".dockerignore",
    "docker/tournament/Dockerfile",
    "docker/tournament/run_tournament.py",
    "docker/tournament/config.json",
    "docker/tournament/.env.example",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "main.py",
    "reinforcetactics/__init__.py",
    "reinforcetactics/cloud/storage.py",
    "scripts/cloud/vertex_train.py",
    "scripts/train/train_bootstrap.py",
    "configs/ppo/bootstrap.yaml",
    "maps/1v1/beginner.csv",
]


# ---------------------------------------------------------------------------
# .dockerignore semantics (moby/patternmatcher, as used by `docker build`)
# ---------------------------------------------------------------------------


def _docker_regex(pattern: str) -> re.Pattern[str]:
    """Translate one cleaned .dockerignore pattern the way moby's patternmatcher does."""
    out = ["^"]
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            if pattern[i + 1 : i + 2] == "*":
                i += 1
                if pattern[i + 1 : i + 2] == "/":  # "**/" is treated as "**"
                    i += 1
                # "**" at the end matches everything; elsewhere any number of
                # directories, including none (so "**/.env" matches ".env").
                out.append(".*" if i + 1 >= len(pattern) else "(.*/)?")
            else:
                out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch in "[\\":
            raise NotImplementedError(f"extend the test matcher before using {ch!r} in .dockerignore: {pattern!r}")
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("".join(out) + "$")


def parse_dockerignore(text: str) -> list[tuple[bool, re.Pattern[str]]]:
    rules = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:].strip()
        # Docker runs filepath.Clean (which drops a trailing "/") and strips a
        # leading "/": every pattern is relative to the context root.
        line = posixpath.normpath(line)
        if len(line) > 1 and line.startswith("/"):
            line = line[1:]
        rules.append((negate, _docker_regex(line)))
    return rules


def docker_excludes(path: str, rules: list[tuple[bool, re.Pattern[str]]]) -> bool:
    """Last matching rule wins; a rule matching a parent directory matches the path."""
    parts = path.split("/")
    candidates = [path] + ["/".join(parts[:i]) for i in range(1, len(parts))]
    excluded = False
    for negate, regex in rules:
        if any(regex.match(candidate) for candidate in candidates):
            excluded = not negate
    return excluded


# ---------------------------------------------------------------------------
# .gitignore semantics (also what gcloud applies to .gcloudignore)
# ---------------------------------------------------------------------------


def _git_glob(pattern: str) -> str:
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
            continue
        if pattern.startswith("/**", i) and i + 3 == len(pattern):
            out.append("/.*")
            i += 3
            continue
        ch = pattern[i]
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            end = pattern.index("]", i + 1)
            body = pattern[i + 1 : end]
            out.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
            i = end
        elif ch == "\\":
            raise NotImplementedError(f"extend the test matcher before escaping in an ignore file: {pattern!r}")
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)


def parse_gitignore(text: str) -> list[tuple[bool, bool, re.Pattern[str]]]:
    """``(negate, directory_only, regex)`` per rule of a root-level ignore file."""
    rules = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        directory_only = line.endswith("/")
        line = line.rstrip("/")
        if "/" in line:  # a slash anywhere but the end anchors to the root
            regex = "^" + _git_glob(line.lstrip("/")) + "$"
        else:  # otherwise the name matches at any depth
            regex = "^(?:.*/)?" + _git_glob(line) + "$"
        rules.append((negate, directory_only, re.compile(regex)))
    return rules


def _git_last_match(path: str, is_dir: bool, rules) -> bool:
    excluded = False
    for negate, directory_only, regex in rules:
        if directory_only and not is_dir:
            continue
        if regex.match(path):
            excluded = not negate
    return excluded


def git_excludes(path: str, rules) -> bool:
    """Whether ``path`` (a file) is ignored.

    Directories are checked shallowest first: git does not descend into an
    ignored directory, so nothing beneath one can be re-included.
    """
    parts = path.split("/")
    if any(_git_last_match("/".join(parts[:i]), True, rules) for i in range(1, len(parts))):
        return True
    return _git_last_match(path, False, rules)


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def _dockerfiles() -> list[Path]:
    found = [REPO_ROOT / "Dockerfile", *(REPO_ROOT / "docker").rglob("Dockerfile*")]
    return sorted(p for p in found if p.is_file() and not p.name.endswith(".dockerignore"))


def _effective_dockerignore(dockerfile: Path) -> Path:
    """BuildKit prefers ``<Dockerfile>.dockerignore`` next to the Dockerfile.

    Otherwise it reads ``.dockerignore`` at the context root, which is the
    repo root for both images (``docker build .`` in the docs, and
    ``context: ../..`` in docker/tournament/docker-compose.yml).
    """
    specific = dockerfile.with_name(dockerfile.name + ".dockerignore")
    return specific if specific.is_file() else REPO_ROOT / ".dockerignore"


def test_both_images_are_covered():
    assert {p.relative_to(REPO_ROOT).as_posix() for p in _dockerfiles()} >= {"Dockerfile", "docker/tournament/Dockerfile"}


@pytest.mark.parametrize("dockerfile", _dockerfiles(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_docker_build_context_excludes_secrets_and_tournament_output(dockerfile):
    ignore_file = _effective_dockerignore(dockerfile)
    rules = parse_dockerignore(ignore_file.read_text(encoding="utf-8"))
    leaked = [p for p in [*SECRET_PATHS, TOURNAMENT_OUTPUT] if not docker_excludes(p, rules)]
    assert not leaked, f"{ignore_file.name} lets these into the {dockerfile.relative_to(REPO_ROOT)} image: {leaked}"
    dropped = [p for p in SHIPPED_PATHS if docker_excludes(p, rules)]
    assert not dropped, f"{ignore_file.name} excludes files the image needs: {dropped}"


def test_gitignore_excludes_secrets_and_bootstrap_runs():
    rules = parse_gitignore((REPO_ROOT / ".gitignore").read_text(encoding="utf-8"))
    not_ignored = [p for p in [*SECRET_PATHS, BOOTSTRAP_RUN_FILE] if not git_excludes(p, rules)]
    assert not not_ignored, f"the root .gitignore does not ignore: {not_ignored}"
    tracked = [
        "docker/tournament/.env.example",
        "benchmarks/ppo_vs_simplebot/RESULTS.md",
        "saves/cavalry_charge_scenario.json",
        *SHIPPED_PATHS,
    ]
    wrongly_ignored = [p for p in tracked if git_excludes(p, rules)]
    assert not wrongly_ignored, f"the root .gitignore ignores tracked files: {wrongly_ignored}"


@pytest.fixture(scope="module")
def gcloudignore_rules():
    path = REPO_ROOT / ".gcloudignore"
    assert path.is_file(), "without .gcloudignore, `gcloud builds submit` uploads every local benchmarks/ run"
    return parse_gitignore(path.read_text(encoding="utf-8"))


def test_gcloudignore_excludes_secrets_and_local_runs(gcloudignore_rules):
    uploaded = [p for p in [*SECRET_PATHS, TOURNAMENT_OUTPUT, BOOTSTRAP_RUN_FILE] if not git_excludes(p, gcloudignore_rules)]
    assert not uploaded, f".gcloudignore uploads these to Cloud Build: {uploaded}"
    dropped = [p for p in SHIPPED_PATHS if git_excludes(p, gcloudignore_rules)]
    assert not dropped, f".gcloudignore drops files the Cloud Build docker build needs: {dropped}"


def test_gcloudignore_keeps_everything_the_image_keeps(gcloudignore_rules):
    """A Cloud Build image must contain what a local `docker build .` would.

    Walks the working tree (hidden tool directories aside, and pruning what
    .dockerignore excludes) and checks that .gcloudignore uploads every file
    .dockerignore would let into the image.
    """
    docker_rules = parse_dockerignore((REPO_ROOT / ".dockerignore").read_text(encoding="utf-8"))
    missing = []
    stack = [p for p in REPO_ROOT.iterdir() if not p.name.startswith(".") or p.name == ".dockerignore"]
    while stack:
        path = stack.pop()
        rel = path.relative_to(REPO_ROOT).as_posix()
        if docker_excludes(rel, docker_rules):
            continue
        if path.is_dir():
            stack.extend(path.iterdir())
        elif git_excludes(rel, gcloudignore_rules):
            missing.append(rel)
    assert not missing, f".gcloudignore drops files a local docker build would include: {sorted(missing)[:20]}"


# ---------------------------------------------------------------------------
# The matchers themselves, on cases whose Docker/git behaviour is documented.
# ---------------------------------------------------------------------------


def test_docker_matcher_semantics():
    rules = parse_dockerignore("models/\n**/.env\n**/.env.*\n!**/.env.example\n*.pyc\n")
    assert docker_excludes("models/ppo_final.zip", rules)  # a directory pattern covers its contents
    assert not docker_excludes("reinforcetactics/models/x.py", rules)  # patterns are root-anchored
    assert docker_excludes(".env", rules)  # "**/" matches zero directories
    assert docker_excludes("a/b/.env.local", rules)
    assert not docker_excludes("a/.env.example", rules)  # the last matching rule wins
    assert not docker_excludes("pkg/mod.pyc", rules)  # "*" never crosses "/"


def test_git_matcher_semantics():
    rules = parse_gitignore("/logs\n.env\n!.env.example\n/saves/*\n!/saves/*_scenario.json\nbuild/\n!build/keep.txt\n")
    assert git_excludes("logs/run.txt", rules)
    assert not git_excludes("pkg/logs/run.txt", rules)  # a leading "/" anchors
    assert git_excludes("deep/dir/.env", rules)  # no slash: any depth
    assert not git_excludes("deep/.env.example", rules)
    assert git_excludes("saves/game.json", rules)
    assert not git_excludes("saves/intro_scenario.json", rules)
    assert git_excludes("build/keep.txt", rules)  # cannot re-include inside an ignored directory
    assert not git_excludes("build", rules)  # "build/" only matches a directory
