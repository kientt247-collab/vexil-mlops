"""The only way to read a match result.

Every read is written to a log with its purpose. Results of matches that start at or
after the phase 2 cutoff (2026-11-20 00:00 UTC) are withheld until the go-live lock file
exists, because the alert rules and the promotion gate must be frozen before the first
prospective result is seen.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from vexil.data.store import Store
from vexil.weeks import CUTOFF


@dataclass
class LabelLoader:
    store: Store
    log_path: Path
    lock_path: Path

    def _log(self, purpose: str, kind: str, n: int, n_after_cutoff: int, withheld: int) -> None:
        self.log_path = Path(self.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "purpose": purpose,
            "kind": kind,
            "matches": n,
            "after_cutoff": n_after_cutoff,
            "withheld": withheld,
        }
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def week_labels(self, week: str, purpose: str) -> dict[int, int]:
        """Results (1 if Radiant won, else 0) of the matches of one week.

        Without the lock file, results of matches that start at or after the cutoff are
        withheld: they are not returned, and the log records how many were held back.
        """
        matches, _ = self.store._read_raw(week)
        known = [m for m in matches if m.get("radiant_win") is not None]
        locked = Path(self.lock_path).exists()
        allowed = [m for m in known if locked or m["start_time"] < CUTOFF]
        after = sum(m["start_time"] >= CUTOFF for m in allowed)
        self._log(purpose, "labels", len(allowed), after, withheld=len(known) - len(allowed))
        return {m["match_id"]: int(bool(m["radiant_win"])) for m in allowed}
