"""Archive-aware, non-destructive access to the SPICOR audio inventory."""

from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


class SpicorAudioError(ValueError):
    """Raised when a SPICOR audio resolution or extraction contract fails."""


@dataclass(frozen=True)
class AudioResolution:
    utterance_id: str
    status: str
    source_type: str | None
    original_inventory_path: str | None
    archive_member: str | None
    selected_audio_path: Path | None


class SpicorAudioResolver:
    """Resolve prepared files, selected cache files, or verified archive members."""

    def __init__(self, inventory: Mapping[str, Mapping[str, Any]], *, repo_root: str | Path, archive_path: str | Path, selected_cache_root: str | Path):
        self.inventory = {str(key): value for key, value in inventory.items()}
        self.repo_root = Path(repo_root).resolve()
        self.archive_path = Path(archive_path).resolve()
        self.selected_cache_root = Path(selected_cache_root).resolve()

    def _prepared(self, row: Mapping[str, Any]) -> tuple[str | None, Path | None]:
        value = row.get("prepared_audio_path")
        if not isinstance(value, str) or not value:
            return None, None
        path = Path(value)
        resolved = path if path.is_absolute() else self.repo_root / path
        return value, resolved

    def _cache_path(self, row: Mapping[str, Any]) -> Path | None:
        member = row.get("source_wav_member")
        if not isinstance(member, str) or not member:
            return None
        return self.selected_cache_root / "archive" / Path(member).name

    def resolve(self, utterance_id: str) -> AudioResolution:
        row = self.inventory.get(str(utterance_id))
        if row is None:
            return AudioResolution(str(utterance_id), "MISSING", None, None, None, None)
        inventory_path, prepared = self._prepared(row)
        if prepared is not None and prepared.is_file():
            return AudioResolution(str(utterance_id), "RESOLVES", "PREPARED_LOCAL", inventory_path, row.get("source_wav_member"), prepared)
        cached = self._cache_path(row)
        if cached is not None and cached.is_file():
            return AudioResolution(str(utterance_id), "RESOLVES", "ARCHIVE_EXTRACTED", inventory_path, row.get("source_wav_member"), cached)
        member = row.get("source_wav_member")
        if self.archive_path.is_file() and isinstance(member, str) and member:
            return AudioResolution(str(utterance_id), "ARCHIVE_MEMBER_AVAILABLE", None, inventory_path, member, None)
        return AudioResolution(str(utterance_id), "MISSING", None, inventory_path, member if isinstance(member, str) else None, None)

    def estimate_archive_bytes(self, utterance_ids: Iterable[str]) -> int:
        total = 0
        for utterance_id in utterance_ids:
            row = self.inventory.get(str(utterance_id))
            if row is None:
                raise SpicorAudioError(f"unknown utterance ID: {utterance_id}")
            if self.resolve(str(utterance_id)).source_type == "PREPARED_LOCAL":
                continue
            size = row.get("source_size_bytes")
            if not isinstance(size, int) or size < 0:
                raise SpicorAudioError(f"missing source_size_bytes for {utterance_id}")
            total += size
        return total

    @staticmethod
    def _safe_target(root: Path, filename: str) -> Path:
        if not filename or Path(filename).name != filename:
            raise SpicorAudioError(f"archive member is not a plain expected filename: {filename}")
        target = (root / filename).resolve()
        if root.resolve() not in target.parents:
            raise SpicorAudioError(f"archive path escapes extraction root: {filename}")
        return target

    def materialize(self, utterance_ids: Iterable[str], *, minimum_headroom_bytes: int = 512 * 1024 * 1024) -> dict[str, AudioResolution]:
        ids = sorted(set(map(str, utterance_ids)))
        resolutions = {utterance_id: self.resolve(utterance_id) for utterance_id in ids}
        archive_ids = [utterance_id for utterance_id, resolution in resolutions.items() if resolution.status == "ARCHIVE_MEMBER_AVAILABLE"]
        missing = [utterance_id for utterance_id, resolution in resolutions.items() if resolution.status == "MISSING"]
        if missing:
            raise SpicorAudioError(f"missing SPICOR audio: {missing[:5]}")
        if not archive_ids:
            return {utterance_id: self.resolve(utterance_id) for utterance_id in ids}
        if not self.archive_path.is_file():
            raise SpicorAudioError(f"archive does not exist: {self.archive_path}")
        estimated = self.estimate_archive_bytes(archive_ids)
        self.selected_cache_root.mkdir(parents=True, exist_ok=True)
        archive_root = self.selected_cache_root / "archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(self.selected_cache_root).free
        if free_bytes < estimated + minimum_headroom_bytes:
            raise SpicorAudioError(f"insufficient disk headroom: free={free_bytes}, estimated={estimated}, required_headroom={minimum_headroom_bytes}")
        member_to_id: dict[str, str] = {}
        for utterance_id in archive_ids:
            member = resolutions[utterance_id].archive_member
            if not member:
                raise SpicorAudioError(f"archive member missing for {utterance_id}")
            if member in member_to_id:
                raise SpicorAudioError(f"duplicate archive member: {member}")
            member_to_id[member] = utterance_id
        found: set[str] = set()
        with tarfile.open(self.archive_path, mode="r|gz") as archive:
            for member in archive:
                utterance_id = member_to_id.get(member.name)
                if utterance_id is None:
                    continue
                if not member.isfile():
                    raise SpicorAudioError(f"selected archive member is not a regular file: {member.name}")
                target = self._safe_target(archive_root, Path(member.name).name)
                if target.exists():
                    if target.stat().st_size != member.size:
                        raise SpicorAudioError(f"existing selected file has wrong size: {target}")
                else:
                    source = archive.extractfile(member)
                    if source is None:
                        raise SpicorAudioError(f"cannot read archive member: {member.name}")
                    with target.open("xb") as handle:
                        shutil.copyfileobj(source, handle, length=1024 * 1024)
                    if target.stat().st_size != member.size:
                        raise SpicorAudioError(f"extracted size mismatch: {target}")
                found.add(utterance_id)
        not_found = sorted(set(archive_ids) - found)
        if not_found:
            raise SpicorAudioError(f"selected archive members not found: {not_found[:5]}")
        return {utterance_id: self.resolve(utterance_id) for utterance_id in ids}
