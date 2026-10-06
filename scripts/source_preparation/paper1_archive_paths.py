"""Resolve relocated Paper 1 archives without changing recorded identities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

V4_RELATIVE = Path("output/paper1_r2_v4/2026-09-12")
CHECKPOINT_RELATIVE = Path("output/paper1_r2_rebuild/2026-09-07")
REGIONS = ("HITRANS", "Nestrans", "SESTRAN", "SPT", "SWESTRANS", "Tactran", "ZetTrans")


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Return the digest of an existing file; missing files remain failures."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def require_writable_copy(run_dir: Path, original: Path) -> Path:
    """Reject the retained archive as a verification write destination."""
    run = run_dir.resolve()
    retained = original.resolve()
    if run == retained or retained in run.parents:
        raise ValueError(f"Use a separate writable archive copy, outside {retained}")
    if not run.is_dir():
        raise FileNotFoundError(f"Writable archive copy does not exist: {run}")
    return run


class ArchivePaths:
    """Resolve explicit relocations and check the original recorded hashes.

    Relative keys and absolute keys under ``root`` identify the same source.
    Other historical absolute paths require an explicit mapping. No file is
    accepted by name alone, and no recorded digest is modified.
    """

    def __init__(self, root: Path, *, run_dir: Path | None = None,
                 checkpoint_dir: Path | None = None,
                 path_map: Mapping[str, str] | None = None) -> None:
        self.root = root.resolve()
        self.run_dir = run_dir
        self.checkpoint_dir = checkpoint_dir
        self.path_map: dict[str, Path] = {}
        self.checked: list[dict[str, str]] = []
        for original, replacement in (path_map or {}).items():
            if not isinstance(original, str) or not isinstance(replacement, str):
                raise ValueError("Path map must contain original-path: replacement-path strings")
            key = self.key(original)
            target = Path(replacement)
            target = target if target.is_absolute() else self.root / target
            if key in self.path_map and self.path_map[key] != target:
                raise ValueError(f"Conflicting source relocations: {original}")
            self.path_map[key] = target

    def key(self, name: str | Path) -> str:
        """Normalise an absolute path under this root to its relative key."""
        path = Path(name)
        return str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)

    def resolve(self, name: str | Path) -> Path:
        """Resolve explicit source maps before archive-prefix relocations."""
        key = self.key(name)
        if key in self.path_map:
            return self.path_map[key]
        logical = Path(key)
        for prefix, target in ((V4_RELATIVE, self.run_dir),
                               (CHECKPOINT_RELATIVE, self.checkpoint_dir)):
            if target is not None and logical.is_relative_to(prefix):
                return target / logical.relative_to(prefix)
        return logical if logical.is_absolute() else self.root / logical

    def verify(self, name: str | Path, expected: str,
               algorithm: str = "sha256") -> Path:
        """Verify original expected bytes and log the actual source used."""
        path = self.resolve(name)
        actual = file_digest(path, algorithm)
        if actual != expected:
            raise ValueError(f"Source hash mismatch: {name} -> {path}; "
                             f"expected {expected}, actual {actual}")
        self.checked.append({"recorded_path": str(name), "resolved_path": str(path),
                             "algorithm": algorithm, "expected": expected, "actual": actual})
        return path
