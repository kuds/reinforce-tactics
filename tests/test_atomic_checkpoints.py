"""Checkpoints are replaced atomically, and uploads skip files still being written.

SB3's ``model.save`` writes its zip in place. Interrupted mid-save (by the
SIGTERM scripts/train/train_bootstrap.py turns into ``SystemExit`` when Vertex
cancels or preempts a job, or by an OOM kill), it left a zip missing entries
where the last good ``best_model.zip`` had been. The final upload and the
entrypoint's sync then stored that over the good copy an earlier periodic
sync had put in GCS.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reinforcetactics.cloud.storage import upload_tree
from reinforcetactics.rl.callbacks import PeriodicEvalCallback, save_model_atomically

REPO_ROOT = Path(__file__).resolve().parent.parent


class _Logger:
    def record(self, key, value):
        pass


class _Model:
    """Writes a checkpoint the way SB3 does: to the path it is given, in place."""

    def __init__(self, interrupt: bool = False):
        self.interrupt = interrupt
        self.num_timesteps = 0
        self.logger = _Logger()
        self.saved_to: list[str] = []

    def save(self, path):
        self.saved_to.append(str(path))
        Path(path).write_bytes(b"PK half a zip" if self.interrupt else b"new checkpoint")
        if self.interrupt:
            raise SystemExit(143)


def _eval_metrics(win_rate: float) -> dict:
    return {
        "win_rate": win_rate,
        "avg_reward": win_rate,
        "avg_length": 1.0,
        "avg_turns": 1.0,
        "std_reward": 0.0,
        "wins": 1,
        "losses": 0,
        "draws": 0,
        "seize_available_rate": 0.0,
        "max_legal_actions": 0,
    }


def test_interrupted_best_model_save_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    best = tmp_path / "best_model.zip"
    best.write_bytes(b"good checkpoint")
    monkeypatch.setattr("reinforcetactics.rl.callbacks.evaluate_model", lambda model, env, **kwargs: _eval_metrics(0.9))
    callback = PeriodicEvalCallback(eval_env=object(), eval_freq=100, n_eval_episodes=1, save_dir=tmp_path, verbose=0)
    callback.model = _Model(interrupt=True)
    callback.num_timesteps = callback.model.num_timesteps = 100
    callback._last_eval_block = 1

    with pytest.raises(SystemExit):
        callback._do_eval()

    assert best.read_bytes() == b"good checkpoint"
    assert [p.name for p in tmp_path.iterdir()] == ["best_model.zip"]  # no .partial left behind


def test_completed_save_replaces_the_checkpoint(tmp_path):
    target = tmp_path / "stage_1" / "best_model.zip"
    target.parent.mkdir()
    target.write_bytes(b"old checkpoint")
    model = _Model()
    save_model_atomically(model, target)
    assert target.read_bytes() == b"new checkpoint"
    assert model.saved_to == [str(target) + ".partial"]
    assert [p.name for p in target.parent.iterdir()] == ["best_model.zip"]


def test_suffixless_path_gets_zip_like_sb3(tmp_path):
    save_model_atomically(_Model(), tmp_path / "final_model")
    assert [p.name for p in tmp_path.iterdir()] == ["final_model.zip"]


def test_uploads_skip_files_still_being_written(tmp_path):
    (tmp_path / "best_model.zip").write_bytes(b"x")
    (tmp_path / "best_model.zip.partial").write_bytes(b"half")
    uploaded: list[str] = []
    client = SimpleNamespace(
        bucket=lambda name: SimpleNamespace(
            blob=lambda blob: SimpleNamespace(upload_from_filename=lambda path: uploaded.append(blob))
        )
    )
    assert upload_tree(str(tmp_path), "gs://bucket/run", client=client) == 1
    assert uploaded == ["run/best_model.zip"]


@pytest.fixture
def train_bootstrap(monkeypatch):
    monkeypatch.setattr(sys, "path", [*sys.path])
    monkeypatch.setenv("SDL_VIDEODRIVER", os.environ.get("SDL_VIDEODRIVER", "dummy"))
    monkeypatch.setenv("MPLBACKEND", os.environ.get("MPLBACKEND", "Agg"))
    spec = importlib.util.spec_from_file_location("train_bootstrap_atomic", REPO_ROOT / "scripts/train/train_bootstrap.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interrupted_checkpoint_snapshot_leaves_no_truncated_copy(train_bootstrap, tmp_path, monkeypatch):
    """train_bootstrap flattens each stage's checkpoints into checkpoints/ after training."""
    import shutil

    stage_dir = tmp_path / "stage_1"
    stage_dir.mkdir()
    (stage_dir / "stage_final.zip").write_bytes(b"stage checkpoint")

    def interrupted_copy(src, dst, **kwargs):
        Path(dst).write_bytes(b"PK half")
        raise SystemExit(143)

    monkeypatch.setattr(shutil, "copy2", interrupted_copy)
    result = {"history": [{"stage": "stage_1", "promoted": True, "best_win_rate": 0.9}]}
    cfg = SimpleNamespace(curriculum=SimpleNamespace(stages=[SimpleNamespace(name="stage_1", map_file="m", opponent="o")]))
    with pytest.raises(SystemExit):
        train_bootstrap._snapshot_stage_checkpoints(result, cfg, tmp_path)
    assert list((tmp_path / "checkpoints").iterdir()) == []


class _VecModel(_Model):
    """A _Model that the checkpoint callbacks can query for a VecNormalize env."""

    def get_vec_normalize_env(self):
        return None


class TestAtomicSB3Callbacks:
    """The CheckpointCallback / EvalCallback replacements used by train_self_play.py and the CLI trainer."""

    def test_checkpoint_callback_keeps_the_previous_checkpoint_on_an_interrupted_save(self, tmp_path):
        from reinforcetactics.rl.callbacks import AtomicCheckpointCallback

        callback = AtomicCheckpointCallback(save_freq=1, save_path=str(tmp_path), name_prefix="sp")
        callback.model = _VecModel()
        callback.n_calls, callback.num_timesteps = 1, 100
        assert callback._on_step() is True
        good = tmp_path / "sp_100_steps.zip"
        assert good.read_bytes() == b"new checkpoint"

        # Same timestep again, interrupted mid-save: the good zip survives
        # and no half-written file is left for a sync to pick up.
        callback.model = _VecModel(interrupt=True)
        with pytest.raises(SystemExit):
            callback._on_step()
        assert good.read_bytes() == b"new checkpoint"
        assert not list(tmp_path.glob("*.partial"))

    def test_checkpoint_callback_saves_only_on_its_frequency(self, tmp_path):
        from reinforcetactics.rl.callbacks import AtomicCheckpointCallback

        callback = AtomicCheckpointCallback(save_freq=3, save_path=str(tmp_path))
        callback.model = _VecModel()
        callback.n_calls, callback.num_timesteps = 2, 64
        callback._on_step()
        assert callback.model.saved_to == []

    def test_new_best_callback_writes_atomically(self, tmp_path):
        from reinforcetactics.rl.callbacks import SaveModelAtomicallyCallback

        path = tmp_path / "best_model" / "best_model.zip"
        path.parent.mkdir()
        path.write_bytes(b"old best")
        callback = SaveModelAtomicallyCallback(path)
        callback.model = _Model(interrupt=True)
        with pytest.raises(SystemExit):
            callback._on_step()
        assert path.read_bytes() == b"old best"
        assert not list(path.parent.glob("*.partial"))

        callback.model = _Model()
        callback._on_step()
        assert path.read_bytes() == b"new checkpoint"
