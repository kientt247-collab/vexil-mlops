"""Command line of the pipeline: ``python -m vexil <command>``.

Each command is one step of the weekly job, so a failed step can be run again by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
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


def cmd_run(args: argparse.Namespace) -> int:
    """The weekly job: pull, score, evaluate, alarms, challengers."""
    from vexil.issues import GitHubChannel, LogChannel
    from vexil.job import Layout, run_job

    layout = Layout(args.root, args.store)
    if args.dry_run:  # nothing leaves this machine: no issue is opened, no file is uploaded
        channel = LogChannel(Path(tempfile.mkdtemp()) / "issues.jsonl")
    else:
        channel = GitHubChannel(args.repository)
    summary = run_job(
        layout, args.now, channel, OpenDotaClient().explorer,
        repository=args.repository, upload=not args.dry_run,
    )  # fmt: skip
    print(json.dumps(summary, indent=1))
    return 0


def _scratch_root(source: Path, scratch: Path) -> Path:
    """Copy what a replay reads (models, manifest, rules, gate) into a folder of its own,
    so that the tables committed in the repository are left alone."""
    for folder in ("monitoring", "data"):
        (scratch / folder).mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "models", scratch / "models", dirs_exist_ok=True)
    shutil.copy(source / "data" / "manifest.json", scratch / "data")
    for name in ("rules.yaml", "gate.yaml"):
        shutil.copy(source / "monitoring" / name, scratch / "monitoring")
    return scratch


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay past weeks from the local store as if the job had run every Monday."""
    from vexil.issues import LogChannel
    from vexil.job import Layout, replay

    root = args.root
    if args.scratch:
        root = _scratch_root(args.root, args.scratch)
    store = _store(args)
    for week in store.weeks():
        if week <= args.last_week:
            remote.download_week(store, week, args.repository)
    layout = Layout(root, args.store)
    log = layout.monitoring / "replay_issues.jsonl"
    log.unlink(missing_ok=True)
    summaries = replay(layout, args.first_week, args.last_week, LogChannel(log))
    actions = [a for s in summaries for a in s.get("actions", [])]
    print(json.dumps({"runs": len(summaries), "actions": actions}, indent=1))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Build the static dashboard from the monitoring tables."""
    from vexil.dashboard import build
    from vexil.job import Layout

    print(build(Layout(args.root, args.store), args.out))
    return 0


def cmd_lock(args: argparse.Namespace) -> int:
    """Write lock.json: hashes of the rules, the gate, the starting models and the code."""
    import datetime as dt

    from vexil.job import Layout, lock_contents

    layout = Layout(args.root, args.store)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    contents = {
        **lock_contents(layout, args.version, stamp),
        "approved_by": args.approved_by,
        "reason": args.reason,
    }
    layout.lock.write_text(json.dumps(contents, indent=1) + "\n")
    print(json.dumps(contents, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vexil", description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
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

    run = commands.add_parser("run", help=cmd_run.__doc__)
    run.add_argument("--now", type=int, default=int(time.time()), help="run time, Unix seconds")
    run.add_argument("--dry-run", action="store_true", help="open no issue and upload no file")
    run.set_defaults(run=cmd_run)

    replay = commands.add_parser("replay", help=cmd_replay.__doc__)
    replay.add_argument(
        "--first-week", required=True, help="Monday of the first week, e.g. 2025-12-29"
    )
    replay.add_argument("--last-week", required=True, help="Monday of the last week")
    replay.add_argument(
        "--scratch", type=Path, help="write into this folder instead of the repository"
    )
    replay.set_defaults(run=cmd_replay)

    dashboard = commands.add_parser("dashboard", help=cmd_dashboard.__doc__)
    dashboard.add_argument("--out", type=Path, default=Path("site"))
    dashboard.set_defaults(run=cmd_dashboard)

    lock = commands.add_parser("lock", help=cmd_lock.__doc__)
    lock.add_argument("--version", type=int, required=True)
    lock.add_argument("--approved-by", required=True)
    lock.add_argument("--reason", required=True)
    lock.set_defaults(run=cmd_lock)

    args = parser.parse_args(argv)
    return args.run(args)
