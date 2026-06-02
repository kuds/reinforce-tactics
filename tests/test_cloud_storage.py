"""Tests for reinforcetactics.cloud.storage."""

import pytest

from reinforcetactics.cloud.storage import (
    GCSUploader,
    is_available,
    parse_gcs_uri,
    resolve_output_base,
    sync_directories,
    upload_tree,
)

# ---------------------------------------------------------------------------
# Fake GCS client (records uploads instead of touching the network)
# ---------------------------------------------------------------------------


class _FakeBlob:
    def __init__(self, name, store):
        self.name = name
        self._store = store

    def upload_from_filename(self, path):
        self._store.append((self.name, path))


class _FakeBucket:
    def __init__(self, name, store):
        self.name = name
        self._store = store

    def blob(self, name):
        return _FakeBlob(name, self._store)


class _FakeClient:
    def __init__(self):
        self.uploads = []

    def bucket(self, name):
        return _FakeBucket(name, self.uploads)


# ---------------------------------------------------------------------------
# parse_gcs_uri
# ---------------------------------------------------------------------------


class TestParseGcsUri:
    def test_bucket_and_prefix(self):
        assert parse_gcs_uri("gs://my-bucket/jobs/run1") == ("my-bucket", "jobs/run1")

    def test_bucket_only(self):
        assert parse_gcs_uri("gs://my-bucket") == ("my-bucket", "")

    def test_trailing_slash_stripped(self):
        assert parse_gcs_uri("gs://my-bucket/prefix/") == ("my-bucket", "prefix")

    @pytest.mark.parametrize("bad", ["", "my-bucket/x", "s3://b/p", "gs://"])
    def test_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            parse_gcs_uri(bad)


# ---------------------------------------------------------------------------
# resolve_output_base
# ---------------------------------------------------------------------------


class TestResolveOutputBase:
    def test_explicit_uri_wins(self):
        env = {"GCS_OUTPUT_URI": "gs://b/run", "AIP_MODEL_DIR": "gs://other/model"}
        assert resolve_output_base(env) == "gs://b/run"

    def test_explicit_uri_trailing_slash_stripped(self):
        assert resolve_output_base({"GCS_OUTPUT_URI": "gs://b/run/"}) == "gs://b/run"

    def test_falls_back_to_vertex_model_dir(self):
        # Vertex sets AIP_MODEL_DIR = <base>/model
        env = {"AIP_MODEL_DIR": "gs://b/jobs/run1/model"}
        assert resolve_output_base(env) == "gs://b/jobs/run1"

    def test_model_dir_at_bucket_root(self):
        # baseOutputDirectory = bucket root -> AIP_MODEL_DIR = gs://b/model
        assert resolve_output_base({"AIP_MODEL_DIR": "gs://b/model"}) == "gs://b"

    def test_model_dir_trailing_slash_not_corrupted(self):
        # A stray trailing slash must not strip part of the gs:// scheme
        # (the old rsplit approach turned 'gs://b/' into 'gs:/').
        assert resolve_output_base({"AIP_MODEL_DIR": "gs://b/"}) == "gs://b"
        assert resolve_output_base({"AIP_MODEL_DIR": "gs://b/jobs/run1/model/"}) == "gs://b/jobs/run1"

    def test_none_when_unset(self):
        assert resolve_output_base({}) is None


# ---------------------------------------------------------------------------
# GCSUploader (with an injected fake client)
# ---------------------------------------------------------------------------


class TestGCSUploader:
    def test_upload_file_applies_prefix(self):
        client = _FakeClient()
        uploader = GCSUploader("bucket", prefix="jobs/run1", client=client)
        uri = uploader.upload_file("/tmp/model.zip", "models/model.zip")

        assert uri == "gs://bucket/jobs/run1/models/model.zip"
        assert client.uploads == [("jobs/run1/models/model.zip", "/tmp/model.zip")]

    def test_upload_file_no_prefix_uses_basename(self):
        client = _FakeClient()
        uploader = GCSUploader("bucket", client=client)
        uploader.upload_file("/tmp/a/model.zip")

        assert client.uploads == [("model.zip", "/tmp/a/model.zip")]

    def test_upload_directory_recurses(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "sub" / "b.txt").write_text("b")

        client = _FakeClient()
        uploader = GCSUploader("bucket", prefix="p", client=client)
        count = uploader.upload_directory(str(tmp_path), remote_prefix="models")

        assert count == 2
        remote_names = sorted(name for name, _ in client.uploads)
        assert remote_names == ["p/models/a.txt", "p/models/sub/b.txt"]

    def test_upload_directory_missing_dir_is_noop(self, tmp_path):
        client = _FakeClient()
        uploader = GCSUploader("bucket", client=client)
        assert uploader.upload_directory(str(tmp_path / "nope")) == 0
        assert client.uploads == []


# ---------------------------------------------------------------------------
# sync_directories
# ---------------------------------------------------------------------------


class TestSyncDirectories:
    def test_no_base_uri_is_noop(self, tmp_path):
        assert sync_directories(None, root=str(tmp_path), client=_FakeClient()) == {}

    def test_syncs_existing_dirs_only(self, tmp_path):
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "final.zip").write_text("x")
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "ckpt_100.zip").write_text("y")
        # tensorboard/ and logs/ intentionally absent

        client = _FakeClient()
        result = sync_directories("gs://bucket/jobs/run1", root=str(tmp_path), client=client)

        assert result == {"models": 1, "checkpoints": 1}
        uploaded = sorted(name for name, _ in client.uploads)
        assert uploaded == ["jobs/run1/checkpoints/ckpt_100.zip", "jobs/run1/models/final.zip"]

    def test_invalid_base_uri_is_noop(self, tmp_path):
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "final.zip").write_text("x")
        assert sync_directories("not-a-gcs-uri", root=str(tmp_path), client=_FakeClient()) == {}

    def test_empty_dirs_excluded_from_result(self, tmp_path):
        (tmp_path / "models").mkdir()  # exists but empty
        client = _FakeClient()
        assert sync_directories("gs://bucket/run", root=str(tmp_path), client=client) == {}


class TestUploadTree:
    def test_uploads_whole_tree_under_dest_prefix(self, tmp_path):
        run_dir = tmp_path / "20260601_120000"
        (run_dir / "charts").mkdir(parents=True)
        (run_dir / "videos").mkdir()
        (run_dir / "charts" / "summary.png").write_text("png")
        (run_dir / "videos" / "stage1.mp4").write_text("vid")
        (run_dir / "bootstrap_results.csv").write_text("csv")

        client = _FakeClient()
        count = upload_tree(str(run_dir), "gs://bucket/jobs/job1/20260601_120000", client=client)

        assert count == 3
        uploaded = sorted(name for name, _ in client.uploads)
        assert uploaded == [
            "jobs/job1/20260601_120000/bootstrap_results.csv",
            "jobs/job1/20260601_120000/charts/summary.png",
            "jobs/job1/20260601_120000/videos/stage1.mp4",
        ]

    def test_no_dest_is_noop(self, tmp_path):
        (tmp_path / "f.txt").write_text("x")
        assert upload_tree(str(tmp_path), None, client=_FakeClient()) == 0

    def test_invalid_dest_is_noop(self, tmp_path):
        (tmp_path / "f.txt").write_text("x")
        assert upload_tree(str(tmp_path), "not-gcs", client=_FakeClient()) == 0


class TestManifestSkip:
    def test_unchanged_files_skipped_across_syncs(self, tmp_path):
        (tmp_path / "models").mkdir()
        f = tmp_path / "models" / "a.zip"
        f.write_text("v1")
        manifest = {}

        first = sync_directories("gs://b/run", root=str(tmp_path), client=_FakeClient(), manifest=manifest)
        assert first == {"models": 1}

        # Nothing changed → the second pass uploads nothing.
        client2 = _FakeClient()
        second = sync_directories("gs://b/run", root=str(tmp_path), client=client2, manifest=manifest)
        assert second == {}
        assert client2.uploads == []

        # Changing the file (different size) → re-uploaded.
        f.write_text("v2-much-larger")
        client3 = _FakeClient()
        third = sync_directories("gs://b/run", root=str(tmp_path), client=client3, manifest=manifest)
        assert third == {"models": 1}
        assert len(client3.uploads) == 1


class TestIsAvailable:
    def test_returns_bool(self):
        # True or False depending on whether google-cloud-storage is installed;
        # the contract is simply that it never raises and returns a bool.
        assert isinstance(is_available(), bool)
