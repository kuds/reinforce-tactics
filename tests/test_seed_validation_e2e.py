"""End to end: run_seeds.py launches two real seeds of a tiny curriculum in parallel, summarize_seeds.py reports them.

The §2.1 pipeline in miniature: the launcher's group manifest, one
train_bootstrap.py per seed (``--resume-if-exists``), per-session logs, a
re-launch that finds both seeds done, and the report over the group.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_curriculum_recovery import _tiny_config

REPO_ROOT = Path(__file__).resolve().parents[1]
GROUP = "20260928_120000_e2e"


@pytest.mark.slow
def test_two_seeds_launched_in_parallel_then_summarized(tmp_path):
    config = _tiny_config(tmp_path, 256)
    root = tmp_path / "root"
    env = {k: v for k, v in os.environ.items() if k not in ("GCS_OUTPUT_URI", "AIP_MODEL_DIR", "GCS_WRAPPER_SYNC")}
    env.update(PYTHONPATH=str(REPO_ROOT), SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", MPLBACKEND="Agg")
    launch = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train" / "run_seeds.py"),
        "--config",
        str(config),
        "--seeds",
        "0,1000",
        "--group",
        GROUP,
        "--root",
        str(root),
        "--parallel",
        "2",
        "--device",
        "cpu",
        "--",
        "--strict",
        "--skip-videos",
        "--skip-plots",
        "--sanity-episodes",
        "0",
        "--no-gcs",
    ]
    first = subprocess.run(launch, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900, check=False)
    assert first.returncode == 0, first.stdout[-4000:] + first.stderr[-4000:]
    manifest_path = root / "_groups" / GROUP / "seed_group.json"
    manifest = json.loads(manifest_path.read_text())
    for seed in ("0", "1000"):
        run = manifest["runs"][seed]
        assert run["last_state"] == "completed" and [s["exit_code"] for s in run["sessions"]] == [0]
        run_dir = root / run["run_dir"]
        assert (run_dir / "final_model.zip").is_file()
        (log,) = (run_dir / "logs").glob("train.*.log")
        assert "Bootstrap run" in log.read_text()
        assert json.loads((run_dir / "s1" / "config.json").read_text())["seed"] == int(seed)

    # The identical command again: both seeds are done, nothing is relaunched.
    again = subprocess.run(launch, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300, check=False)
    assert again.returncode == 0, again.stdout[-4000:] + again.stderr[-4000:]
    assert [len(r["sessions"]) for r in json.loads(manifest_path.read_text())["runs"].values()] == [1, 1]

    summarize = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval" / "summarize_seeds.py"),
        "--group-manifest",
        str(manifest_path),
        "--quiet",
    ]
    report = subprocess.run(summarize, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300, check=False)
    assert report.returncode == 0, report.stdout[-4000:] + report.stderr[-4000:]
    summary = json.loads((manifest_path.parent / "report" / "summary.json").read_text())
    assert [r["seed"] for r in summary["runs"]] == [0, 1000]
    assert [[s["stage"] for s in r["stages"]] for r in summary["runs"]] == [["s1", "s2"], ["s1", "s2"]]
    assert all(r["status"] == "completed" and r["stages_cleared"] == 2 for r in summary["runs"])
    assert summary["group"]["replicate_check"]["passed"] is True
    assert summary["group"]["config_digest"] == manifest["config_digest"]
    text = (manifest_path.parent / "report" / "report.md").read_text()
    assert (
        f"# Seed validation report: {GROUP}" in text
        and "| s0 | completed | 2/2 |" in text
        and "| s1000 | completed | 2/2 |" in text
    )
