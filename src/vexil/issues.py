"""Alarms as GitHub issues: one issue per alarm, comments while it lasts, closed when quiet.

The weekly job talks to GitHub through the ``gh`` command-line tool. The replay uses
``LogChannel`` instead, which writes the same messages to a file, so that replaying a year
does not open issues.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Protocol

import polars as pl

QUIET_WEEKS_TO_CLOSE = 4
LABEL = "alert"


def issue_title(family: str, subject: str) -> str:
    return f"[alert] {family} / {subject}"


class Channel(Protocol):
    def open_titles(self) -> set[str]: ...
    def open(self, title: str, body: str) -> None: ...
    def comment(self, title: str, body: str) -> None: ...
    def close(self, title: str, body: str) -> None: ...


class LogChannel:
    """Keeps issues in memory and appends every action to a JSON-lines file."""

    def __init__(self, path: Path) -> None:
        self.path, self._open = Path(path), set()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, action: str, title: str, body: str) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps({"action": action, "title": title, "body": body}) + "\n")

    def open_titles(self) -> set[str]:
        return set(self._open)

    def open(self, title: str, body: str) -> None:
        self._open.add(title)
        self._write("open", title, body)

    def comment(self, title: str, body: str) -> None:
        self._write("comment", title, body)

    def close(self, title: str, body: str) -> None:
        self._open.discard(title)
        self._write("close", title, body)


class GitHubChannel:
    """Issues of one repository, through the gh command-line tool."""

    def __init__(self, repository: str) -> None:
        self.repository = repository

    def _gh(self, *args: str) -> str:
        command = ["gh", *args, "--repo", self.repository]
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout

    def _numbers(self) -> dict[str, int]:
        listed = self._gh(
            "issue",
            "list",
            "--state",
            "open",
            "--label",
            LABEL,
            "--json",
            "number,title",
            "--limit",
            "100",
        )
        return {item["title"]: item["number"] for item in json.loads(listed)}

    def open_titles(self) -> set[str]:
        return set(self._numbers())

    def open(self, title: str, body: str) -> None:
        subprocess.run(
            ["gh", "label", "create", LABEL, "--repo", self.repository, "--color", "B60205", "--force"],  # noqa: E501
            check=False, capture_output=True,
        )  # fmt: skip
        self._gh("issue", "create", "--title", title, "--body", body, "--label", LABEL)

    def comment(self, title: str, body: str) -> None:
        self._gh("issue", "comment", str(self._numbers()[title]), "--body", body)

    def close(self, title: str, body: str) -> None:
        self._gh("issue", "close", str(self._numbers()[title]), "--comment", body)


def sync(
    states: pl.DataFrame, week: str, reports: dict[tuple[str, str], str], channel: Channel
) -> list[str]:
    """Bring the issues in line with the states of one newly final week. Returns what was done.

    A rule that starts firing opens an issue, or comments if one is already open. An open
    issue whose rule has been quiet for 4 final weeks is closed.
    """
    weeks = sorted(w for w in states["week"].unique() if w <= week)
    recent = weeks[-QUIET_WEEKS_TO_CLOSE:]
    open_now = channel.open_titles()
    done = []
    subjects = states.select("family", "subject").unique(maintain_order=True)
    for family, subject in subjects.iter_rows():
        title = issue_title(family, subject)
        mine = states.filter((pl.col("family") == family) & (pl.col("subject") == subject))
        this_week = mine.filter(pl.col("week") == week)
        if this_week.height and this_week["alarm"].any():
            body = reports.get((family, subject), f"Rule fired in week {week}.")
            if title in open_now:
                channel.comment(title, f"Fired again in week {week}.\n\n{body}")
                done.append(f"comment: {title}")
            else:
                channel.open(title, body)
                done.append(f"open: {title}")
        elif title in open_now and len(recent) == QUIET_WEEKS_TO_CLOSE:
            quiet = not mine.filter(pl.col("week").is_in(recent))["firing"].any()
            if quiet:
                channel.close(
                    title,
                    f"Quiet for {QUIET_WEEKS_TO_CLOSE} final weeks up to {week}. Closed by the pipeline.",  # noqa: E501
                )
                done.append(f"close: {title}")
    return done
