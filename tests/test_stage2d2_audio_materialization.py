import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

from swara.data.spicor_audio import SpicorAudioError, SpicorAudioResolver


def test_resolver_priority(tmp_path):
    prepared = tmp_path / "prepared.wav"; prepared.write_bytes(b"prepared")
    cache = tmp_path / "cache" / "archive"; cache.mkdir(parents=True)
    (cache / "u2.wav").write_bytes(b"cached")
    inventory = {
        "u1": {"prepared_audio_path": str(prepared), "source_wav_member": "root/u1.wav"},
        "u2": {"source_wav_member": "root/u2.wav"},
        "u3": {"source_wav_member": "root/u3.wav"},
    }
    archive = tmp_path / "corpus.tar.gz"; archive.write_bytes(b"placeholder")
    resolver = SpicorAudioResolver(inventory, repo_root=tmp_path, archive_path=archive, selected_cache_root=tmp_path / "cache")
    assert resolver.resolve("u1").source_type == "PREPARED_LOCAL"
    assert resolver.resolve("u2").source_type == "ARCHIVE_EXTRACTED"
    assert resolver.resolve("u3").status == "ARCHIVE_MEMBER_AVAILABLE"


def test_safe_target_rejects_path_traversal(tmp_path):
    with pytest.raises(SpicorAudioError):
        SpicorAudioResolver._safe_target(tmp_path, "../escape.wav")


def test_materialization_deduplicates_ids_and_extracts_only_selected_members(tmp_path):
    archive_path = tmp_path / "corpus.tar.gz"
    payloads = {"root/u1.wav": b"one", "root/u2.wav": b"two", "root/other.wav": b"other"}
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    inventory = {
        key: {"source_wav_member": f"root/{key}.wav", "source_size_bytes": len(payloads[f"root/{key}.wav"])}
        for key in ("u1", "u2")
    }
    resolver = SpicorAudioResolver(inventory, repo_root=tmp_path, archive_path=archive_path, selected_cache_root=tmp_path / "cache")
    resolved = resolver.materialize(["u1", "u1", "u2"])
    assert resolved["u1"].selected_audio_path.read_bytes() == b"one"
    assert resolved["u2"].selected_audio_path.read_bytes() == b"two"
    assert not (tmp_path / "cache" / "archive" / "other.wav").exists()


def test_materialization_rejects_missing_archive_member(tmp_path):
    inventory = {"u1": {"source_wav_member": "root/u1.wav", "source_size_bytes": 1}}
    archive_path = tmp_path / "corpus.tar.gz"
    with tarfile.open(archive_path, mode="w:gz"):
        pass
    resolver = SpicorAudioResolver(inventory, repo_root=tmp_path, archive_path=archive_path, selected_cache_root=tmp_path / "cache")
    with pytest.raises(SpicorAudioError):
        resolver.materialize(["u1"])
