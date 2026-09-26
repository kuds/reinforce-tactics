"""Secrets and bulky local outputs stay out of Docker images, Cloud Build uploads and git.

``.dockerignore`` used to let ``docker/tournament/.env`` into both images:
each Dockerfile does ``COPY . .`` with the repo root as its build context,
and ``docker/tournament/README.md`` tells users to put their
OpenAI/Anthropic/Google API keys in that file and then push the image.

The checks below evaluate the ignore files with the same matching rules as
Docker, git and gcloud, so a reordered or overridden pattern fails here even
if the line itself is still present. None of the tools is run: the rules are
small enough to reimplement, and the tests then do not depend on a daemon or
an SDK install.
"""

import fnmatch
import posixpath
import re
import subprocess
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
    # The Cloud Console's name for a downloaded service-account key.
    "my-project-0123456789ab.json",
    "configs/my-project-0123456789ab.json",
    # Settings() writes the settings menu's LLM API keys to settings.json in
    # whatever directory the game or a tournament was started from.
    "settings.json",
    "scripts/settings.json",
    "docker/tournament/settings.json",
]

# Underscore-prefixed agent scratch files at the root (Drive crawls, decoded
# payloads, working notes); the root .gitignore ignores them.
SCRATCH_PATHS = ["_drive_crawl.json", "_notes.md", "_probe.py", "_dump.txt"]

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
        elif ch == "[":
            # Character classes pass through to the regexp unchanged, ranges
            # and "^" negation included.
            end = pattern.index("]", i + 1)
            out.append(pattern[i : end + 1])
            i = end
        elif ch == "\\":
            raise NotImplementedError(f"extend the test matcher before escaping in .dockerignore: {pattern!r}")
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
# .gitignore semantics
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
# .gcloudignore semantics (googlecloudsdk command_lib/util/gcloudignore.py and
# glob.py). Close to git's, with two differences that matter here: a
# "#!include:FILE" comment splices in FILE's rules, and only a leading "/"
# anchors a pattern (gcloud matches pattern components right to left, so
# "a/b" also matches "x/a/b").
# ---------------------------------------------------------------------------

GcloudRule = tuple[bool, bool, str]  # (negate, directory_only, pattern)


def parse_gcloudignore(path: Path, recurse: int = 1) -> list[GcloudRule]:
    """Rules of ``path``, with ``#!include:`` directives expanded as gcloud does.

    gcloud honours the directives in .gcloudignore itself (``recurse=1``) but
    not inside the files they include, and only includes files from the same
    directory.
    """
    rules: list[GcloudRule] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            directive = line[1:].lstrip()
            if directive.startswith("!include:"):
                included = directive[len("!include:") :]
                assert "/" not in included, f"gcloud refuses to include a file from another directory: {line!r}"
                if recurse:
                    rules.extend(parse_gcloudignore(path.parent / included, recurse - 1))
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        directory_only = line.endswith("/")
        line = line.removesuffix("/").rstrip(" ")
        if not line:
            continue
        if "\\" in line:
            raise NotImplementedError(f"extend the test matcher before escaping in .gcloudignore: {line!r}")
        rules.append((negate, directory_only, line))
    return rules


def _gcloud_path_prefixes(path: str) -> list[str]:
    """``"a/b/c"`` -> ``["", "a", "a/b", "a/b/c"]`` (gcloud's GetPathPrefixes)."""
    prefixes = [path]
    tail = path
    while path and tail:
        path, tail = posixpath.split(path)
        prefixes.insert(0, path)
    return prefixes


def _gcloud_glob_matches(parts: list[str], path: str | None) -> bool:
    """gcloud's Glob._MatchesHelper: match pattern components right to left.

    ``path is None`` means "above the root"; a leading ``""`` component (from a
    leading "/") only matches the root itself, which is what anchors a pattern.
    """
    if not parts:
        return True
    if path is None:
        return False
    *rest, part = parts
    if path:
        path = posixpath.normpath(path)
    remaining: str | None
    remaining, name = posixpath.split(path)
    if not name:
        remaining = None
    if part == "**":
        # Whatever precedes "**" must match some prefix of the path, anchored.
        if not (rest and rest[0] == ""):
            rest = ["", *rest]
        return any(_gcloud_glob_matches(rest, prefix) for prefix in _gcloud_path_prefixes(path))
    if part == "*" and not rest and remaining and len(remaining) > 1:
        return False
    if not fnmatch.fnmatchcase(name, part):
        return False
    return _gcloud_glob_matches(rest, remaining)


def gcloud_excludes(path: str, rules: list[GcloudRule]) -> bool:
    """FileChooser.IsIncluded, negated: shallowest ignored prefix wins, last rule per prefix."""
    for prefix in _gcloud_path_prefixes(path)[1:]:
        ignored = None
        for negate, directory_only, pattern in rules:
            if directory_only and prefix == path:  # ``path`` is a file
                continue
            if _gcloud_glob_matches(pattern.split("/"), prefix):
                ignored = not negate
        if ignored:
            return True
    return False


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


@pytest.fixture(scope="module")
def tracked_files() -> list[str]:
    """Every file git tracks: what a fresh checkout hands to docker build and gcloud."""
    try:
        listing = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("not a git checkout")
    return [name for name in listing.decode("utf-8").split("\0") if name]


def test_both_images_are_covered():
    assert {p.relative_to(REPO_ROOT).as_posix() for p in _dockerfiles()} >= {"Dockerfile", "docker/tournament/Dockerfile"}


@pytest.mark.parametrize("dockerfile", _dockerfiles(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_docker_build_context_excludes_secrets_and_tournament_output(dockerfile):
    ignore_file = _effective_dockerignore(dockerfile)
    rules = parse_dockerignore(ignore_file.read_text(encoding="utf-8"))
    leaked = [p for p in [*SECRET_PATHS, *SCRATCH_PATHS, TOURNAMENT_OUTPUT] if not docker_excludes(p, rules)]
    assert not leaked, f"{ignore_file.name} lets these into the {dockerfile.relative_to(REPO_ROOT)} image: {leaked}"
    dropped = [p for p in SHIPPED_PATHS if docker_excludes(p, rules)]
    assert not dropped, f"{ignore_file.name} excludes files the image needs: {dropped}"


def test_gitignore_excludes_secrets_and_bootstrap_runs():
    rules = parse_gitignore((REPO_ROOT / ".gitignore").read_text(encoding="utf-8"))
    not_ignored = [p for p in [*SECRET_PATHS, *SCRATCH_PATHS, BOOTSTRAP_RUN_FILE] if not git_excludes(p, rules)]
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
def gcloudignore_rules() -> list[GcloudRule]:
    path = REPO_ROOT / ".gcloudignore"
    assert path.is_file(), "without .gcloudignore, `gcloud builds submit` uploads every local benchmarks/ run"
    return parse_gcloudignore(path)


def test_gcloudignore_excludes_secrets_and_local_runs(gcloudignore_rules):
    uploaded = [
        p
        for p in [*SECRET_PATHS, *SCRATCH_PATHS, TOURNAMENT_OUTPUT, BOOTSTRAP_RUN_FILE]
        if not gcloud_excludes(p, gcloudignore_rules)
    ]
    assert not uploaded, f".gcloudignore uploads these to Cloud Build: {uploaded}"
    dropped = [p for p in SHIPPED_PATHS if gcloud_excludes(p, gcloudignore_rules)]
    assert not dropped, f".gcloudignore drops files the Cloud Build docker build needs: {dropped}"


def test_gcloudignore_still_excludes_what_the_gitignore_fallback_did(gcloudignore_rules):
    """Adding .gcloudignore must not upload anything gcloud's .gitignore fallback kept back.

    With no .gcloudignore, gcloud excludes what the root .gitignore ignores.
    Once the file exists it replaces that fallback, so without the
    ``#!include:.gitignore`` directive settings.json below the root and the
    root scratch files went from excluded to uploaded, and from there into
    the image, since .dockerignore did not exclude them either.
    """
    git_rules = parse_gitignore((REPO_ROOT / ".gitignore").read_text(encoding="utf-8"))
    local_state = [
        *SECRET_PATHS,
        *SCRATCH_PATHS,
        BOOTSTRAP_RUN_FILE,
        "uv.lock",
        "models/ppo_final.zip",
        "saves/game_20260926.json",
        "replays/game_1.json",
        "videos/stage_1.mp4",
        "tournament_results/latest/results.csv",
        "reinforcetactics/core/__pycache__/game_state.cpython-312.pyc",
        "reinforcetactics.egg-info/PKG-INFO",
        "build/lib/reinforcetactics/__init__.py",
        "htmlcov/index.html",
        "test_map_editor.py",
    ]
    ignored_by_git = [p for p in local_state if git_excludes(p, git_rules)]
    # Guard against the list drifting away from what .gitignore covers.
    assert len(ignored_by_git) == len(local_state), sorted(set(local_state) - set(ignored_by_git))
    uploaded = [p for p in ignored_by_git if not gcloud_excludes(p, gcloudignore_rules)]
    assert not uploaded, f".gcloudignore uploads files the .gitignore fallback excluded: {uploaded}"


def test_cloud_build_uploads_every_tracked_file_the_image_gets(gcloudignore_rules, tracked_files):
    """A Cloud Build image must contain what a local ``docker build .`` of a checkout would.

    Checks the tracked files rather than walking the working tree, so the
    result does not depend on what else happens to sit in a developer's
    checkout (a local venv, a symlink loop).
    """
    docker_rules = parse_dockerignore((REPO_ROOT / ".dockerignore").read_text(encoding="utf-8"))
    missing = [
        path for path in tracked_files if not docker_excludes(path, docker_rules) and gcloud_excludes(path, gcloudignore_rules)
    ]
    assert not missing, f".gcloudignore drops tracked files a local docker build would include: {sorted(missing)[:20]}"


# ---------------------------------------------------------------------------
# The matchers themselves, on cases whose Docker/git/gcloud behaviour is documented.
# ---------------------------------------------------------------------------


def test_docker_matcher_semantics():
    rules = parse_dockerignore("models/\n**/.env\n**/.env.*\n!**/.env.example\n*.pyc\n**/*-[0-9a-f][0-9a-f].json\n")
    assert docker_excludes("models/ppo_final.zip", rules)  # a directory pattern covers its contents
    assert not docker_excludes("reinforcetactics/models/x.py", rules)  # patterns are root-anchored
    assert docker_excludes(".env", rules)  # "**/" matches zero directories
    assert docker_excludes("a/b/.env.local", rules)
    assert not docker_excludes("a/.env.example", rules)  # the last matching rule wins
    assert not docker_excludes("pkg/mod.pyc", rules)  # "*" never crosses "/"
    assert docker_excludes("keys/proj-0f.json", rules)  # character classes
    assert not docker_excludes("keys/proj-0g.json", rules)


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


def test_gcloud_matcher_semantics(tmp_path):
    (tmp_path / ".gitignore").write_text("settings.json\n/_*.json\n", encoding="utf-8")
    (tmp_path / "other").write_text("#!include:nested\n", encoding="utf-8")
    (tmp_path / "nested").write_text("nested.txt\n", encoding="utf-8")
    (tmp_path / ".gcloudignore").write_text(
        "#!include:.gitignore\n#!include:other\n/docs/\ndocker/output/\n**/cache/\nbuild/\n!build/keep.txt\n",
        encoding="utf-8",
    )
    rules = parse_gcloudignore(tmp_path / ".gcloudignore")
    assert gcloud_excludes("scripts/settings.json", rules)  # included from .gitignore, any depth
    assert gcloud_excludes("_crawl.json", rules)
    assert not gcloud_excludes("pkg/_crawl.json", rules)  # a leading "/" anchors
    assert not gcloud_excludes("nested.txt", rules)  # includes inside included files are ignored
    assert gcloud_excludes("docs/index.md", rules)
    assert not gcloud_excludes("pkg/docs/index.md", rules)
    assert gcloud_excludes("docker/output/log.json", rules)
    assert gcloud_excludes("x/docker/output/log.json", rules)  # unlike git, an inner "/" does not anchor
    assert gcloud_excludes("cache/a", rules)  # "**/" matches zero directories
    assert gcloud_excludes("a/b/cache/c", rules)
    assert gcloud_excludes("build/keep.txt", rules)  # cannot re-include inside an ignored directory
    assert not gcloud_excludes("build", rules)  # "build/" only matches a directory
