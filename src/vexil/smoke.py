"""Walking skeleton for node M0: prove that the platform can do each step the plan needs.

Run as ``python -m vexil.smoke --out site``. It makes one OpenDota explorer call that
returns a match count (no match result is requested), then writes a JSON record and a
one-line HTML page. The workflow stores the record as a release asset, publishes the
page and opens a test issue.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import urllib.parse
import urllib.request
from pathlib import Path

OPENDOTA = "https://api.opendota.com/api"
COUNT_SQL = (
    "SELECT count(*) AS n FROM matches "
    "WHERE start_time >= extract(epoch from now())::int - 7 * 86400"
)


def explorer_url(sql: str) -> str:
    """Return the OpenDota explorer URL for one SQL statement."""
    return f"{OPENDOTA}/explorer?sql={urllib.parse.quote(sql)}"


def fetch_json(url: str, timeout: int = 120) -> dict:
    """GET a URL and decode its JSON body (gzip or plain)."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "vexil-mlops (smoke test)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    return json.loads(body)


def parse_count(payload: dict) -> int:
    """Read the single count out of an explorer response."""
    if payload.get("err"):
        raise RuntimeError(f"explorer error: {payload['err']}")
    rows = payload["rows"]
    if len(rows) != 1:
        raise ValueError(f"expected one row, got {len(rows)}")
    return int(rows[0]["n"])


def render_page(record: dict) -> str:
    """One-line page that shows the smoke record."""
    return (
        "<!doctype html><meta charset='utf-8'><title>vexil-mlops smoke</title>"
        f"<p>vexil-mlops walking skeleton. Run at {record['run_at_utc']} UTC. "
        f"OpenDota matches in the last 7 days: {record['matches_last_7_days']}.</p>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args(argv)

    count = parse_count(fetch_json(explorer_url(COUNT_SQL)))
    record = {
        "run_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "matches_last_7_days": count,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "smoke.json").write_text(json.dumps(record, indent=2) + "\n")
    (args.out / "index.html").write_text(render_page(record) + "\n")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
