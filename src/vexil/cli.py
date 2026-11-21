"""Command line of the pipeline: ``python -m vexil <command>``.

Each command is one step of the weekly job, so a failed step can be run again by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from vexil.data import remote
from vexil.data.opendota import OpenDotaClient
from vexil.data.pull import pull_weeks
from vexil.data.store import Store
from vexil.weeks import open_weeks

DEFAULT_REPOSITORY = "kientt247-collab/vexil-mlops"


def _store(args: argparse.Namespace) -> Store:
    return Store(args.store, args.manifest)


def cmd_pull(args: argparse.Namespace) -> int:
    """Pull every open week from OpenDota; upload the weeks whose content changed."""
    store = _store(args)
    weeks = open_weeks(args.now)
    for week in weeks:
        if week in store.manifest["weeks"]:
            remote.download_week(store, week, args.repository)
    client = OpenDotaClient()
    report = pull_weeks(store, client.explorer, weeks)
    for week, entry in report.items():
        if entry["changed"] and args.upload:
            remote.upload_week(store, week, args.repository)
    summary = {
        week: {k: v for k, v in entry.items() if k != "reasons"} for week, entry in report.items()
    }
    print(json.dumps({"weeks": summary, "explorer calls": client.calls}, indent=1))
    return 0


def cmd_verify_remote(args: argparse.Namespace) -> int:
    """Download every week of the manifest from the release and check its hashes."""
    store = _store(args)
    for week in store.weeks():
        remote.download_week(store, week, args.repository)
    print(json.dumps({"weeks verified": len(store.weeks())}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vexil", description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("_store"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument(
        "--repository", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY)
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pull = commands.add_parser("pull", help=cmd_pull.__doc__)
    pull.add_argument("--now", type=int, default=int(time.time()), help="run time, Unix seconds")
    pull.add_argument("--no-upload", dest="upload", action="store_false")
    pull.set_defaults(run=cmd_pull)

    verify = commands.add_parser("verify-remote", help=cmd_verify_remote.__doc__)
    verify.set_defaults(run=cmd_verify_remote)

    args = parser.parse_args(argv)
    return args.run(args)
