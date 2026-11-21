"""The data store: one pair of files per week, never overwritten.

Layout under the store root::

    weeks/2026-11-16/r1/matches.jsonl.gz    one match per line
    weeks/2026-11-16/r1/players.jsonl.gz    ten player rows per match

A new pull of a week whose content changed writes a new revision (``r2``, ``r3``). The
manifest, a JSON file kept in the repository, says which revision is current and holds the
SHA-256 of every file, so any reader can check that it has exactly the data a result was
computed from. Files are written with a fixed gzip header, which makes the same content
produce the same bytes and therefore the same hash.

Reading goes through ``Store.read_week``, which removes the match result from every row.
The result is available only through ``vexil.labels``.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from vexil.hashing import sha256_file
from vexil.matchdata import is_usable

FILES = ("matches", "players")


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    """Write rows as gzip JSON lines with a fixed header (mtime 0), so bytes are reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=6) as handle:
        handle.write(text.encode())
    path.write_bytes(buffer.getvalue())


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


@dataclass
class Store:
    """Weekly files on disk plus the manifest that describes them."""

    root: Path
    manifest_path: Path

    def __post_init__(self) -> None:
        self.root, self.manifest_path = Path(self.root), Path(self.manifest_path)
        self.manifest: dict = {"weeks": {}}
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())

    # ------------------------------------------------------------------ paths
    def file_path(self, week: str, revision: int, name: str) -> Path:
        return self.root / "weeks" / week / f"r{revision}" / f"{name}.jsonl.gz"

    def asset_name(self, week: str, revision: int, name: str) -> str:
        """Name of the file when stored as a release asset (flat namespace)."""
        return f"week-{week}.r{revision}.{name}.jsonl.gz"

    def weeks(self) -> list[str]:
        return sorted(self.manifest["weeks"])

    # ------------------------------------------------------------------ writing
    def write_week(self, week: str, matches: list[dict], players: list[dict], source: str) -> bool:
        """Store a week. Returns False, and writes nothing, if the content is unchanged."""
        matches = sorted(matches, key=lambda m: m["match_id"])
        players = sorted(players, key=lambda p: (p["match_id"], p["player_slot"]))
        entry = self.manifest["weeks"].get(week)
        revision = entry["revision"] + 1 if entry else 1
        paths = {name: self.file_path(week, revision, name) for name in FILES}
        write_jsonl_gz(paths["matches"], matches)
        write_jsonl_gz(paths["players"], players)
        hashes = {name: sha256_file(path) for name, path in paths.items()}
        if entry and entry["sha256"] == hashes:
            for path in paths.values():
                path.unlink()
            paths["matches"].parent.rmdir()
            return False
        self.manifest["weeks"][week] = {
            "revision": revision,
            "sha256": hashes,
            "bytes": {name: path.stat().st_size for name, path in paths.items()},
            "matches": len(matches),
            "usable": sum(is_usable(m) for m in matches),
            "source": source,
        }
        self.save_manifest()
        return True

    def save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest, indent=1, sort_keys=True) + "\n")

    # ------------------------------------------------------------------ reading
    def verify_week(self, week: str) -> None:
        """Raise if a file of the current revision is missing or differs from the manifest."""
        entry = self.manifest["weeks"][week]
        for name in FILES:
            path = self.file_path(week, entry["revision"], name)
            if not path.exists():
                raise FileNotFoundError(f"week {week}: {path} is not in the local store")
            if sha256_file(path) != entry["sha256"][name]:
                raise ValueError(f"week {week}: {name} file does not match the manifest hash")

    def _read_raw(self, week: str) -> tuple[list[dict], list[dict]]:
        self.verify_week(week)
        revision = self.manifest["weeks"][week]["revision"]
        return (
            read_jsonl_gz(self.file_path(week, revision, "matches")),
            read_jsonl_gz(self.file_path(week, revision, "players")),
        )

    def read_week(self, week: str) -> Iterator[tuple[dict, list[dict]]]:
        """Yield (match, players) in match_id order. The match result is removed."""
        matches, players = self._read_raw(week)
        by_match: dict[int, list[dict]] = {}
        for row in players:
            by_match.setdefault(row["match_id"], []).append(row)
        for match in matches:
            match.pop("radiant_win", None)
            yield match, by_match.get(match["match_id"], [])
