"""Weekly files as assets of a GitHub release.

Downloads use the public asset URL and need no credentials. Uploads call the ``gh``
command-line tool, which a GitHub Actions run has by default, with the token of that run.
One release per year holds the weeks of that year (a release takes up to 1,000 assets).
"""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from vexil.data.store import FILES, Store
from vexil.hashing import sha256_file
from vexil.weeks import week_bounds


def release_tag(week: str) -> str:
    """Release that holds a week: named after the ISO year of the week."""
    start, _ = week_bounds(week)
    iso_year = dt.datetime.fromtimestamp(start, dt.UTC).isocalendar().year
    return f"data-{iso_year}"


def download_week(store: Store, week: str, repository: str) -> None:
    """Fetch the current revision of a week into the local store, unless it is there."""
    entry = store.manifest["weeks"][week]
    for name in FILES:
        path = store.file_path(week, entry["revision"], name)
        if path.exists() and sha256_file(path) == entry["sha256"][name]:
            continue
        asset = store.asset_name(week, entry["revision"], name)
        url = f"https://github.com/{repository}/releases/download/{release_tag(week)}/{asset}"
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as response:
            path.write_bytes(response.read())
    store.verify_week(week)


def upload_week(store: Store, week: str, repository: str) -> None:
    """Attach the current revision of a week to its release (creating the release once)."""
    entry, tag = store.manifest["weeks"][week], release_tag(week)
    exists = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repository], capture_output=True, check=False
    )
    if exists.returncode != 0:
        notes = (
            "Weekly match files from OpenDota (https://www.opendota.com). See data/manifest.json."
        )
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                "--repo",
                repository,
                "--title",
                tag,
                "--notes",
                notes,
            ],
            check=True,
        )
    # gh names an asset after the file it is given (text after "#" is only a display
    # label), so each file is copied under its asset name first
    with tempfile.TemporaryDirectory() as folder:
        for name in FILES:
            source = store.file_path(week, entry["revision"], name)
            named = Path(folder) / store.asset_name(week, entry["revision"], name)
            shutil.copy(source, named)
            subprocess.run(
                ["gh", "release", "upload", tag, str(named), "--repo", repository, "--clobber"],
                check=True,
            )
