"""The weekly job, one function: pull, score, evaluate, alarm, challengers.

``run_job`` is what the schedule runs every Monday, and what the replay runs for every
past Monday. It only ever reads weeks that ended before its run time `now`.

Order of work:

1. pull the open weeks (the three most recent ended weeks) and store what changed;
2. load the rating state as it stood before the first open week, score the open weeks;
3. apply the frozen rules to all final weeks;
4. for the week that became final in this run: open, update or close issues;
5. challengers: apply the gate to those whose shadow period is over, create new ones
   when an alarm or a new patch asks for it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml

from vexil import retrain
from vexil.candidates import ResultHistory
from vexil.data import remote
from vexil.data.opendota import Explorer
from vexil.data.pull import pull_weeks
from vexil.data.store import Store
from vexil.evaluate import add_groups
from vexil.hashing import sha256_file
from vexil.investigate import recommend, report
from vexil.issues import Channel, sync
from vexil.labels import LabelLoader
from vexil.monitor import check_coverage, evaluate_rules
from vexil.pipeline import process_week
from vexil.ratings import MatchMeta, RatingState
from vexil.registry import append_entry, load_registry, set_role
from vexil.run import (
    effective_scores,
    read_matches,
    training_rows,
    write_challenger,
    write_tables,
)
from vexil.weeks import WEEK_SECONDS, is_final, open_weeks, week_bounds, week_label

SNAPSHOTS_KEPT = 4


@dataclass
class Layout:
    """Where everything lives, relative to the repository root."""

    root: Path
    store_dir: Path

    def __post_init__(self) -> None:
        self.root, self.store_dir = Path(self.root), Path(self.store_dir)
        self.monitoring = self.root / "monitoring"
        self.models = self.root / "models"
        self.state = self.root / "state"
        self.manifest = self.root / "data" / "manifest.json"
        self.registry = self.models / "registry.yaml"
        self.rules = self.monitoring / "rules.yaml"
        self.gate = self.monitoring / "gate.yaml"
        self.lock = self.root / "lock.json"
        self.label_log = self.monitoring / "label_reads.log"
        self.challengers = self.monitoring / "challengers.json"

    def snapshot(self, week: str) -> Path:
        return self.state / f"ratings-before-{week}.json.gz"

    def history_snapshot(self, week: str) -> Path:
        return self.state / f"history-before-{week}.json.gz"


LOCKED_FILES = (
    "monitoring/rules.yaml", "monitoring/gate.yaml",
    "models/candidate.json", "models/rstar.json", "models/anchor.json",
    # lock version 3: the four candidates of the vexil-auto loop, their item constants, and
    # the resolved dependencies (the numbers depend on the library versions)
    "models/constants/items.json",
    "models/candidates/A1/train.py", "models/candidates/A1/model.json",
    "models/candidates/A3/train.py", "models/candidates/A3/model.json",
    "models/candidates/A4/train.py", "models/candidates/A4/model.json",
    "models/candidates/A5/train.py", "models/candidates/A5/model.json",
    # lock version 4: cand-R, its train.py, its frozen table of initial ratings and its fitted file
    "models/candidates/R/train.py", "models/candidates/R/model.json",
    "models/candidates/R/initial_ratings_2026-01-01.csv",
    "uv.lock",
)  # fmt: skip


class ClockError(RuntimeError):
    """A live run was started with a clock earlier than the last recorded run."""


class LockError(RuntimeError):
    """A locked file changed. The job refuses to run until a new lock version is approved."""


def source_tree_sha256(root: Path) -> str:
    """One hash over every file of the package, so that a code change is visible in the lock."""
    import hashlib

    digest = hashlib.sha256()
    package = Path(root) / "src" / "vexil"
    files = sorted(p for p in package.rglob("*") if p.suffix in (".py", ".html"))
    for path in files:
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def lock_contents(layout: Layout, version: int, locked_at: str) -> dict:
    return {
        "version": version,
        "locked_at": locked_at,
        "files": {name: sha256_file(layout.root / name) for name in LOCKED_FILES},
        "source_tree_sha256": source_tree_sha256(layout.root),
    }


def verify_lock(layout: Layout) -> None:
    """If a lock exists, every locked file and the code must match it."""
    if not layout.lock.exists():
        return
    lock = json.loads(layout.lock.read_text())
    for name, digest in lock["files"].items():
        if sha256_file(layout.root / name) != digest:
            raise LockError(f"{name} differs from lock version {lock['version']}")
    if source_tree_sha256(layout.root) != lock["source_tree_sha256"]:
        raise LockError(f"the code under src/vexil differs from lock version {lock['version']}")


def _history(
    layout: Layout, store: Store, loader: LabelLoader, week: str, repository: str | None = None
) -> ResultHistory:
    """Facts and results of every match before `week`: from the snapshot, else replayed."""
    path = layout.history_snapshot(week)
    if path.exists():
        return ResultHistory.load(path)
    history = ResultHistory.empty()
    for earlier in (w for w in store.weeks() if w < week):
        if repository:
            remote.download_week(store, earlier, repository)
        results = loader.week_labels(earlier, purpose="rebuild of the result history")
        history.add_week(list(store.read_week(earlier)), results)
    return history


def _rating_state(
    layout: Layout, store: Store, loader: LabelLoader, week: str, repository: str | None = None
) -> RatingState:
    """Ratings as they stood before `week`: from the snapshot, else replayed from the start."""
    path = layout.snapshot(week)
    if path.exists():
        return RatingState.load(path)
    state = RatingState()
    for earlier in (w for w in store.weeks() if w < week):
        if repository:
            remote.download_week(store, earlier, repository)
        results = loader.week_labels(earlier, purpose="rebuild of the rating state")
        metas = [MatchMeta.from_rows(m, p) for m, p in store.read_week(earlier)]
        state.process(metas, results)
    return state


def _next_week(week: str) -> str:
    return week_label(week_bounds(week)[0] + WEEK_SECONDS)


def effective_predictions(layout: Layout, weeks: list[str], model_id: str) -> pl.DataFrame:
    """Minute rows of one model for the given weeks: out-of-fold rows for development
    matches (where phase 2 saved them), the registered model's rows elsewhere."""
    matches = read_matches(layout.monitoring).filter(pl.col("week").is_in(weeks))
    development = set(matches.filter(pl.col("region") == "development")["match_id"])
    frames = []
    for week in weeks:
        path = layout.monitoring / "predictions" / f"{week}.parquet"
        if path.exists():
            part = pl.read_parquet(path).filter(pl.col("model_id") == model_id)
            frames.append(part.filter(~pl.col("match_id").is_in(development)))
    oof_path = layout.monitoring / "oof_phase2.parquet"
    if oof_path.exists() and development:
        oof = pl.read_parquet(oof_path).filter(
            (pl.col("model_id") == model_id) & pl.col("match_id").is_in(development)
        )
        frames.append(oof)
    columns = ["week", "match_id", "t", "model_id", "p"]
    return pl.concat([f.select(columns) for f in frames]) if frames else pl.DataFrame()


def _read_challengers(layout: Layout) -> list[dict]:
    return json.loads(layout.challengers.read_text()) if layout.challengers.exists() else []


def _apply_gate(layout: Layout, record: dict, champion_id: str, gate: dict, final: list[str]):
    """Gate result of one challenger once every week of its shadow period is final, else None."""
    first = week_bounds(record["shadow_from"])[0]
    shadow = [week_label(first + k * WEEK_SECONDS) for k in range(gate["shadow_weeks"])]
    if not set(shadow) <= set(final):
        return None
    scores = pl.read_parquet(layout.monitoring / "match_scores.parquet").filter(
        pl.col("week").is_in(shadow)
    )
    matches = add_groups(read_matches(layout.monitoring).filter(pl.col("week").is_in(shadow)))
    # The champion is judged on predictions that were out-of-sample for it (out-of-fold rows
    # in the development region). Otherwise a challenger would be compared with a model
    # that was fitted on the very matches of the shadow period.
    fair = effective_scores(layout.monitoring).filter(pl.col("week").is_in(shadow))
    champion = fair.filter(pl.col("model_id") == champion_id).select(
        "match_id", "n", pl.col("loss").alias("champion")
    )
    challenger = scores.filter(pl.col("model_id") == record["id"]).select(
        "match_id", "y", pl.col("loss").alias("challenger")
    )
    both = champion.join(challenger, on="match_id").join(
        matches.select("match_id", "group"), on="match_id"
    )
    rows = pl.concat(
        [pl.read_parquet(layout.monitoring / "predictions" / f"{w}.parquet") for w in shadow]
    ).filter((pl.col("model_id") == record["id"]) & (pl.col("t") < 30))
    rows = rows.join(both.select("match_id", "y"), on="match_id")
    return retrain.gate(
        both["champion"].to_numpy(), both["challenger"].to_numpy(), both["n"].to_numpy().astype(float),  # noqa: E501
        both["group"].to_numpy(), rows["p"].to_numpy().astype(float), rows["y"].to_numpy().astype(float),  # noqa: E501
        settings={k: gate[k] for k in retrain.GATE_DEFAULTS},
    )  # fmt: skip


def _create_challenger(
    layout: Layout, store: Store, loader: LabelLoader, kind: str, trigger: str, now: int,
    champion, gate: dict, final: list[str], repository: str | None = None,
) -> dict:  # fmt: skip
    """Fit a challenger, save its file, add it to the registry, return its record."""
    current = week_label(now)
    model_id = f"{kind}-{current}"
    if kind == "refit":
        weeks = [w for w in store.weeks() if w < current]
        if repository:  # a refit needs every stored week, not only the open ones
            for week in weeks:
                remote.download_week(store, week, repository)
        table, y = training_rows(store, loader, weeks, now - 86400, f"refit challenger {model_id}")
        spec = retrain.refit_spec(table, y, gate["refit_recipe"])
    else:
        weeks = final[-gate["recalibration_weeks"] :]
        scores = pl.read_parquet(layout.monitoring / "match_scores.parquet").filter(
            pl.col("week").is_in(weeks)
        )
        oof = layout.monitoring / "match_scores_oof.parquet"
        if oof.exists():
            scores = pl.concat(
                [scores.select("match_id", "y"), pl.read_parquet(oof).select("match_id", "y")]
            )
        results = dict(scores.select("match_id", "y").unique().iter_rows())
        predictions = effective_predictions(layout, weeks, champion.id)
        base = os.path.relpath(champion.file, layout.models / "challengers")
        spec = retrain.recalibration_spec(predictions, results, base)
    path, digest = write_challenger(layout.models / "challengers", model_id, spec)
    append_entry(
        layout.registry,
        {
            "id": model_id,
            "role": "challenger",
            "kind": spec["kind"],
            "file": str(path.relative_to(layout.root)),
            "sha256": digest,
            "description": f"{kind} challenger created in week {current}; trigger: {trigger}",
        },  # fmt: skip
    )
    return {
        "id": model_id,
        "kind": kind,
        "trigger": trigger,
        "shadow_from": current,
        "status": "shadow",
    }


def run_job(
    layout: Layout,
    now: int,
    channel: Channel,
    explorer: Explorer | None = None,
    repository: str | None = None,
    upload: bool = False,
    until_week: str | None = None,
) -> dict:
    """Run the weekly job as of time `now` (Unix seconds). Returns a summary.

    `until_week` is used by the replay only: weeks after it are ignored, so that a replay of
    2026 stops at the phase 2 cutoff even when later weeks are in the store.
    """
    verify_lock(layout)
    store = Store(layout.store_dir, layout.manifest)
    weeks = open_weeks(now)
    summary: dict = {"now": now, "open_weeks": weeks}
    if explorer is not None:
        last_run = layout.monitoring / "last_run.json"
        # a run that uploads is a run that will be stored; a dry run may use any clock
        if upload and last_run.exists() and now < json.loads(last_run.read_text())["now"]:
            raise ClockError(
                "this run is earlier than the last recorded run (monitoring/last_run.json); "
                "running it would move the stored state back in time"
            )
        for week in weeks:
            if week in store.manifest["weeks"] and repository:
                remote.download_week(store, week, repository)
        pulled = pull_weeks(store, explorer, weeks)
        summary["pull"] = {
            w: {k: v for k, v in r.items() if k != "reasons"} for w, r in pulled.items()
        }
        if upload and repository:
            for week, entry in pulled.items():
                if entry["changed"]:
                    remote.upload_week(store, week, repository)
    weeks = [
        w for w in weeks if w in store.manifest["weeks"] and (not until_week or w <= until_week)
    ]
    if not weeks:
        return summary

    registry = load_registry(layout.registry, layout.root)
    rules = yaml.safe_load(layout.rules.read_text())
    gate = yaml.safe_load(layout.gate.read_text())
    check_coverage(rules, [registry.one("champion").id])
    loader = LabelLoader(store, layout.label_log, layout.lock)

    # 2. ratings and scores of the open weeks
    live = repository if explorer is not None else None
    state = _rating_state(layout, store, loader, weeks[0], live)
    history = (
        _history(layout, store, loader, weeks[0], live)
        if any(e.kind == "frozen_train" for e in registry.active())
        else None
    )
    outputs, results = {}, {}
    for index, week in enumerate(weeks):
        week_results = loader.week_labels(week, purpose="weekly job")
        outputs[week] = process_week(store, week, registry, state, week_results, history)
        results.update(week_results)
        if index == 0:
            state.save(layout.snapshot(_next_week(week)))  # the start point of the next run
            if history is not None:
                history.save(layout.history_snapshot(_next_week(week)))
    write_tables(layout.monitoring, outputs, results)
    for pattern in ("ratings-before-*.json.gz", "history-before-*.json.gz"):
        for old in sorted(layout.state.glob(pattern))[:-SNAPSHOTS_KEPT]:
            old.unlink()

    # 3. rules on final weeks
    counts = pl.read_parquet(layout.monitoring / "weekly_counts.parquet")
    final = [w for w in sorted(counts["week"]) if is_final(w, now)]
    summary["final_weeks"] = len(final)
    if not final:
        return summary
    scores = effective_scores(layout.monitoring)
    matches = read_matches(layout.monitoring)
    result = evaluate_rules(scores, matches, counts, rules, final)
    result.states.write_parquet(layout.monitoring / "alert_states.parquet")
    result.events.write_parquet(layout.monitoring / "patch_events.parquet")

    # 4. issues for the week that became final in this run
    champion = registry.one("champion")
    actions: list[str] = []
    decisions: list[tuple[str, str]] = []  # (action code, trigger text) of this run's alarms
    seen_path = layout.monitoring / "final_weeks.json"
    seen = set(json.loads(seen_path.read_text())) if seen_path.exists() else set()
    seen_path.write_text(json.dumps(final) + "\n")
    for week in (w for w in final if w not in seen):  # weeks that became final in this run
        reports = {}
        new_alarms = result.states.filter((pl.col("week") == week) & pl.col("alarm"))
        for alarm in new_alarms.iter_rows(named=True):
            action = recommend(result.states, alarm, champion.id, result.events)
            in_window = scores.filter(pl.col("week").is_in(final))
            reports[(alarm["family"], alarm["subject"])] = report(
                alarm, action, result.states, in_window, matches, champion.id, rules["anchor"]
            )
            decisions.append(
                (action[0], f"{alarm['family']} alarm on {alarm['subject']}, week {week}")
            )
        actions += sync(result.states, week, reports, channel)
    summary["issues"] = actions

    # 5. challengers
    records = _read_challengers(layout)
    for record in (r for r in records if r["status"] == "shadow"):
        outcome = _apply_gate(layout, record, champion.id, gate, final)
        if outcome is None:
            continue
        record.update(
            status="proposed" if outcome.passed else "rejected",
            gain=outcome.gain, interval=list(outcome.interval), slope=outcome.slope,
            matches=outcome.matches, reasons=outcome.reasons,
        )  # fmt: skip
        title = f"[challenger] {record['id']}"
        detail = (
            f"Shadow period from week {record['shadow_from']}, {outcome.matches} judged matches. "
            f"Gain over the champion {outcome.gain:+.4f} (95% interval {outcome.interval[0]:+.4f} "
            f"to {outcome.interval[1]:+.4f}), slope {outcome.slope:.3f}."
        )
        if outcome.passed:
            channel.open(
                title, f"Promotion proposed. {detail} A person must approve it in the registry."
            )
        else:
            set_role(layout.registry, record["id"], "retired")
            channel.open(title, f"Rejected by the gate: {'; '.join(outcome.reasons)}. {detail}")
            channel.close(title, "Retired by the pipeline.")
        actions.append(f"gate: {record['id']} {record['status']}")

    in_shadow = {r["kind"] for r in records if r["status"] == "shadow"}
    wanted = [("refit", t) for code, t in decisions if code == "refit"]
    wanted += [("recalibration", t) for code, t in decisions if code == "recalibrate"]
    final_matches = matches.filter(pl.col("week").is_in(final))
    known = {r["trigger"] for r in records}
    for patch, first_week in result.events.select("patch", "week").iter_rows():
        trigger = f"patch {patch}"
        on_patch = final_matches.filter(pl.col("patch") == patch).height
        is_first_patch = first_week == result.events["week"].min()
        if not is_first_patch and trigger not in known and on_patch >= gate["new_patch_matches"]:
            wanted.append(("refit", trigger))
    for kind, trigger in wanted:
        if kind in in_shadow:
            actions.append(f"trigger ignored ({kind} challenger already in shadow): {trigger}")
            continue
        records.append(
            _create_challenger(
                layout,
                store,
                loader,
                kind,
                trigger,
                now,
                champion,
                gate,
                final,
                live,
            )
        )
        in_shadow.add(kind)
        actions.append(f"challenger created: {records[-1]['id']} ({trigger})")
    layout.challengers.write_text(json.dumps(records, indent=1) + "\n")

    summary["actions"] = actions
    (layout.monitoring / "last_run.json").write_text(json.dumps(summary, indent=1) + "\n")
    summary["judged_matches_scored"] = int(sum(o.counts["judged"] for o in outputs.values()))
    summary["issue_titles_open"] = sorted(channel.open_titles())
    return summary


def replay(layout: Layout, first_week: str, last_week: str, channel: Channel) -> list[dict]:
    """Run the job for every Monday from the one after `first_week` to the one after
    `last_week`, each run seeing only weeks that ended before it. No pull: the store is read
    as is. Two further runs follow when the last week needs the usual 14 days to become final."""
    start = week_bounds(first_week)[0] + WEEK_SECONDS
    end = week_bounds(last_week)[0] + WEEK_SECONDS
    if not is_final(last_week, end):
        end += 2 * WEEK_SECONDS
    summaries = []
    for monday in range(start, end + 1, WEEK_SECONDS):
        summaries.append(
            run_job(layout, monday + 3 * 3600 + 17 * 60, channel, until_week=last_week)
        )
    return summaries
