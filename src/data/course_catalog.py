"""
src/data/course_catalog.py
==========================
Course knowledge base catalog and multi-lecture registry for RAG AI Teacher Assistant.

Tracks indexed lectures, content hashes for duplicate detection, lecture IDs,
titles, chunk counts, and file references across the course knowledge base.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CHUNK_DIR, COURSE_CATALOG_PATH, DATA_DIR, EMBEDDING_DIR

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_PATH = COURSE_CATALOG_PATH


def compute_file_hash(file_input: Path | str | bytes, chunk_size: int = 65536) -> str:
    """
    Computes a deterministic SHA-256 hash of a file or byte buffer.

    Used for reliable content-based duplicate detection regardless of filename.

    Args:
        file_input: Path to file or raw bytes.
        chunk_size: Read buffer size in bytes.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    hasher = hashlib.sha256()

    if isinstance(file_input, bytes):
        hasher.update(file_input)
        return hasher.hexdigest()

    file_path = Path(file_input).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found for hashing: {file_path}")

    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
    except PermissionError:
        # Graceful fallback on Windows when file is held open exclusively by NamedTemporaryFile
        try:
            stat = file_path.stat()
            hasher.update(f"{file_path.name}_{stat.st_size}_{stat.st_mtime}".encode("utf-8"))
        except Exception:
            hasher.update(file_path.name.encode("utf-8"))

    return hasher.hexdigest()


@dataclass
class CourseLecture:
    """Represents a single registered lecture within the course knowledge base."""

    lecture_id: str
    lecture_title: str
    course_id: str = "python_course"
    source_filename: str = ""
    file_hash: str = ""
    chunk_count: int = 0
    status: str = "Indexed"
    indexed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    chunk_file: str = ""
    embedding_file: str = ""
    metadata_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Converts lecture record into a dictionary."""
        return asdict(self)


class CourseCatalog:
    """
    Persistent course registry managing multi-lecture metadata and duplicate protection.
    """

    def __init__(
        self,
        catalog_path: Path | str = DEFAULT_CATALOG_PATH,
        course_id: str = "python_course",
        course_name: str = "Introduction to Python",
        auto_sync: bool = True,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.course_id = course_id
        self.course_name = course_name
        self.lectures: dict[str, dict[str, Any]] = {}

        self.load_catalog()
        if auto_sync:
            self.sync_with_existing_data()

    @staticmethod
    def resolve_file_path(path_val: str | Path | None, fallback_dir: Path | None = None) -> Path | None:
        """
        Resolves a file path portably across operating systems and machine moves.
        Checks exact path, then checks fallback directory by filename.
        """
        if not path_val:
            return None
        p = Path(path_val)
        if p.exists():
            return p
        if fallback_dir:
            cand = fallback_dir / p.name
            if cand.exists():
                return cand
        cand = DATA_DIR / p.name
        if cand.exists():
            return cand
        return None

    def _normalize_lecture_paths(self) -> None:
        """Normalizes lecture file paths to resolve cross-machine paths portably."""
        for lec in self.lectures.values():
            cf = lec.get("chunk_file")
            if cf:
                resolved_cf = self.resolve_file_path(cf, CHUNK_DIR)
                if resolved_cf:
                    lec["chunk_file"] = str(resolved_cf)
            ef = lec.get("embedding_file")
            if ef:
                resolved_ef = self.resolve_file_path(ef, EMBEDDING_DIR)
                if resolved_ef:
                    lec["embedding_file"] = str(resolved_ef)
            mf = lec.get("metadata_file")
            if mf:
                resolved_mf = self.resolve_file_path(mf, EMBEDDING_DIR)
                if resolved_mf:
                    lec["metadata_file"] = str(resolved_mf)

    def load_catalog(self) -> None:
        """Loads course catalog from disk JSON if it exists."""
        if self.catalog_path.exists():
            try:
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.course_id = data.get("course_id", self.course_id)
                self.course_name = data.get("course_name", self.course_name)
                self.lectures = data.get("lectures", {})
                self._normalize_lecture_paths()
                logger.debug(
                    "Loaded course catalog from '%s': %d lectures.",
                    self.catalog_path.name,
                    len(self.lectures),
                )
            except Exception as exc:
                logger.warning("Failed to parse catalog JSON '%s': %s", self.catalog_path, exc)
                self.lectures = {}
        else:
            self.lectures = {}

    def save_catalog(self) -> None:
        """Persists catalog data to disk JSON."""
        try:
            self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "course_id": self.course_id,
                "course_name": self.course_name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "lecture_count": len(self.lectures),
                "lectures": self.lectures,
            }
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.debug("Saved course catalog to '%s'", self.catalog_path)
        except Exception as exc:
            logger.error("Failed to save course catalog to '%s': %s", self.catalog_path, exc)

    def get_all_lectures(self) -> list[dict[str, Any]]:
        """Returns a list of all registered lecture records sorted by lecture_id."""
        sorted_keys = sorted(self.lectures.keys())
        return [self.lectures[k] for k in sorted_keys]

    def get_lecture_by_id(self, lecture_id: str) -> dict[str, Any] | None:
        """Finds a lecture by its unique lecture_id."""
        for lec in self.lectures.values():
            if lec.get("lecture_id") == lecture_id:
                return lec
        return None

    def get_lecture_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        """Finds a lecture matching a specific content hash."""
        if not file_hash:
            return None
        target = file_hash.strip().lower()
        for lec in self.lectures.values():
            if str(lec.get("file_hash", "")).strip().lower() == target:
                return lec
        return None

    def is_duplicate(self, file_hash: str) -> bool:
        """Checks if a lecture with the given file hash is already registered."""
        return self.get_lecture_by_hash(file_hash) is not None

    def register_lecture(
        self,
        lecture_id: str,
        lecture_title: str,
        course_id: str | None = None,
        source_filename: str = "",
        file_hash: str = "",
        chunk_count: int = 0,
        status: str = "Indexed",
        chunk_file: str = "",
        embedding_file: str = "",
        metadata_file: str = "",
    ) -> dict[str, Any]:
        """
        Registers or updates a lecture in the catalog.
        """
        cid = course_id or self.course_id
        record = CourseLecture(
            lecture_id=lecture_id,
            lecture_title=lecture_title,
            course_id=cid,
            source_filename=source_filename,
            file_hash=file_hash,
            chunk_count=chunk_count,
            status=status,
            chunk_file=chunk_file,
            embedding_file=embedding_file,
            metadata_file=metadata_file,
        )
        self.lectures[lecture_id] = record.to_dict()
        self.save_catalog()
        logger.info("Registered lecture '%s' (%s) in course catalog.", lecture_id, lecture_title)
        return self.lectures[lecture_id]

    def remove_lecture(self, lecture_id: str) -> bool:
        """Removes a lecture from the catalog."""
        if lecture_id in self.lectures:
            del self.lectures[lecture_id]
            self.save_catalog()
            return True
        return False

    def sync_with_existing_data(
        self,
        chunks_dir: Path | str = CHUNK_DIR,
        embeddings_dir: Path | str = EMBEDDING_DIR,
    ) -> None:
        """
        Scans chunk and embedding directories to auto-register pre-existing lectures.
        Ensures backwards compatibility with previously indexed material.
        """
        cdir = Path(chunks_dir)
        edir = Path(embeddings_dir)
        if not cdir.exists():
            return
        changes = False
        chunk_files = sorted([f for f in cdir.glob("*_chunks.json") if not f.name.startswith(".")])

        # Purge stale catalog entries whose chunk files no longer exist on disk
        stale_keys = []
        for k, v in self.lectures.items():
            cf = v.get("chunk_file", "")
            resolved = self.resolve_file_path(cf, cdir)
            if not resolved or not resolved.exists():
                stale_keys.append(k)
            elif cf != str(resolved):
                v["chunk_file"] = str(resolved)
                changes = True
        if stale_keys:
            for k in stale_keys:
                del self.lectures[k]
            changes = True

        for cf in chunk_files:
            stem = cf.stem.replace("_chunks", "")
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    cdata = json.load(f)

                chunks = cdata.get("chunks", [])
                if not chunks:
                    continue

                first_chunk = chunks[0]
                lid = str(first_chunk.get("lecture_id") or stem)
                ltitle = str(
                    first_chunk.get("lecture_title")
                    or cdata.get("lecture_title")
                    or stem.replace("_", " ").title()
                )
                cid = str(first_chunk.get("course_id") or self.course_id)
                sfilename = str(
                    first_chunk.get("source_filename")
                    or first_chunk.get("source_file")
                    or cdata.get("source_file")
                    or f"{stem}.mp3"
                )

                if lid not in self.lectures:
                    # Look for content hash or generate deterministic hash from chunk file
                    f_hash = first_chunk.get("file_hash", "")
                    if not f_hash and cf.exists():
                        f_hash = compute_file_hash(cf)

                    self.lectures[lid] = {
                        "lecture_id": lid,
                        "lecture_title": ltitle,
                        "course_id": cid,
                        "source_filename": sfilename,
                        "file_hash": f_hash,
                        "chunk_count": len(chunks),
                        "status": "Indexed",
                        "indexed_at": datetime.now(timezone.utc).isoformat(),
                        "chunk_file": str(cf),
                        "embedding_file": str(edir / f"{stem}_embeddings.npz"),
                        "metadata_file": str(edir / f"{stem}_metadata.json"),
                    }
                    changes = True
            except Exception as exc:
                logger.warning("Could not sync chunk file '%s': %s", cf.name, exc)

        if changes:
            self.save_catalog()
