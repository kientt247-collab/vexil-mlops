import gzip
import json

import pytest

from vexil.data.pull import check_rows, pull_weeks
from vexil.data.remote import release_tag, upload_week
from vexil.data.schema import match_problems
from vexil.data.store import Store
from vexil.hashing import sha256_file
from vexil.labels import LabelLoader
from vexil.weeks import CUTOFF, week_bounds

from .synthetic import make_match

WEEK = "2026-11-16"  # holds days before and after the phase 2 cutoff


def rows(seeds, start_time):
    matches, players = [], []
    for seed in seeds:
        match, match_players = make_match(seed, start_time=start_time + seed)
        matches.append(match)
        players += match_players
    return matches, players


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store", tmp_path / "manifest.json")


def test_write_read_and_revisions(store):
    matches, players = rows(range(3), week_bounds(WEEK)[0])
    assert store.write_week(WEEK, matches, players, source="test") is True
    assert store.write_week(WEEK, matches, players, source="test") is False  # same content
    assert store.manifest["weeks"][WEEK]["revision"] == 1
    more_matches, more_players = rows(range(4), week_bounds(WEEK)[0])
    assert store.write_week(WEEK, more_matches, more_players, source="test") is True
    assert store.manifest["weeks"][WEEK]["revision"] == 2
    assert store.file_path(WEEK, 1, "matches").exists()  # the old revision is still there
    read = list(store.read_week(WEEK))
    assert [m["match_id"] for m, _ in read] == sorted(m["match_id"] for m in more_matches)
    assert all(len(match_players) == 10 for _, match_players in read)


def test_reader_never_returns_the_result(store):
    matches, players = rows(range(3), week_bounds(WEEK)[0])
    store.write_week(WEEK, matches, players, source="test")
    assert all("radiant_win" not in match for match, _ in store.read_week(WEEK))


def test_a_changed_file_is_detected(store):
    matches, players = rows(range(2), week_bounds(WEEK)[0])
    store.write_week(WEEK, matches, players, source="test")
    store.file_path(WEEK, 1, "matches").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        list(store.read_week(WEEK))


def test_label_rule_withholds_results_after_the_cutoff_until_the_lock_exists(store, tmp_path):
    before, players_before = rows(range(3), CUTOFF - 5000)
    after, players_after = rows(range(3, 7), CUTOFF + 5000)
    store.write_week(WEEK, before + after, players_before + players_after, source="test")
    lock = tmp_path / "lock.json"
    loader = LabelLoader(store, tmp_path / "label_reads.log", lock)
    labels = loader.week_labels(WEEK, purpose="test")
    assert set(labels) == {m["match_id"] for m in before}
    record = json.loads((tmp_path / "label_reads.log").read_text().splitlines()[-1])
    assert record["withheld"] == 4 and record["after_cutoff"] == 0
    lock.write_text("{}")
    assert len(loader.week_labels(WEEK, purpose="test")) == 7


def test_planted_bug_in_the_label_rule_is_caught(store, tmp_path, monkeypatch):
    """A loader whose cutoff is moved one day later would leak results; the test sees it."""
    after, players_after = rows(range(3), CUTOFF + 5000)
    store.write_week(WEEK, after, players_after, source="test")
    monkeypatch.setattr("vexil.labels.CUTOFF", CUTOFF + 86400)
    leaky = LabelLoader(store, tmp_path / "log", tmp_path / "lock.json")
    assert len(leaky.week_labels(WEEK, purpose="planted bug")) == 3  # the rule above would give 0


def test_shape_checks():
    match, players = make_match(1)
    assert match_problems(match, players) == []
    assert match_problems({k: v for k, v in match.items() if k != "start_time"}, players)
    assert match_problems({**match, "radiant_gold_adv": "oops"}, players)
    assert match_problems({**match, "radiant_xp_adv": match["radiant_xp_adv"][:-1]}, players)
    assert match_problems(match, players[:9])
    unparsed = {**match, "radiant_gold_adv": None, "radiant_xp_adv": None}
    assert match_problems(unparsed, players[:9]) == []  # unparsed matches still feed the ratings


def test_check_rows_drops_bad_matches_and_stops_on_a_structural_break():
    matches, players = rows(range(40), week_bounds(WEEK)[0])
    matches[0] = {**matches[0], "radiant_gold_adv": "oops"}
    kept, kept_players, report = check_rows(matches, players)
    assert len(kept) == 39 and len(kept_players) == 390 and report["rejected"] == 1
    broken = [{**m, "radiant_gold_adv": "oops"} for m in matches]
    with pytest.raises(RuntimeError):
        check_rows(broken, players)


class FakeExplorer:
    """Answers the SQL of ``pull_week`` from a list of matches, like OpenDota would."""

    def __init__(self, matches, players):
        self.matches, self.players = matches, players

    def __call__(self, sql):
        if sql.startswith("SELECT m.match_id"):
            return [
                {"match_id": m["match_id"]}
                for m in sorted(self.matches, key=lambda m: m["match_id"])
            ]
        if "match_patch" in sql:
            return [{"match_id": m["match_id"], "patch": "7.42"} for m in self.matches]
        if "FROM leagues" in sql:
            return [{"leagueid": 17, "name": "Test League", "tier": "professional"}]
        if "FROM player_matches" in sql:
            return [dict(p) for p in self.players]
        return [dict(m) for m in sorted(self.matches, key=lambda m: m["match_id"])]


def test_pull_keeps_history_before_the_cutoff_and_adds_later_matches(store):
    old, old_players = rows(range(3), CUTOFF - 5000)
    store.write_week(WEEK, old, old_players, source="phase2-seed")
    # OpenDota now returns a changed copy of an old match and two new ones after the cutoff
    changed = {**old[0], "radiant_gold_adv": [1] * len(old[0]["radiant_gold_adv"])}
    new, new_players = rows(range(3, 5), CUTOFF + 5000)
    explorer = FakeExplorer([changed, *old[1:], *new], old_players + new_players)
    report = pull_weeks(store, explorer, [WEEK])
    assert report[WEEK]["changed"] is True and report[WEEK]["matches"] == 5
    stored = {m["match_id"]: m for m, _ in store.read_week(WEEK)}
    assert stored[old[0]["match_id"]]["radiant_gold_adv"] == old[0]["radiant_gold_adv"]
    assert stored[new[0]["match_id"]]["patch"] == "7.42"
    assert pull_weeks(store, explorer, [WEEK])[WEEK]["changed"] is False  # second run: no change


def test_release_tag_uses_the_iso_year():
    assert release_tag("2025-12-29") == "data-2026"
    assert release_tag("2026-11-16") == "data-2026"


def test_same_rows_compressed_differently_are_not_a_change(store):
    """Another machine can compress the same rows to other bytes (seen on a GitHub runner)."""
    matches, players = rows(range(3), week_bounds(WEEK)[0])
    store.write_week(WEEK, matches, players, source="test")
    entry = store.manifest["weeks"][WEEK]
    for name in ("matches", "players"):
        path = store.file_path(WEEK, 1, name)
        content = gzip.decompress(path.read_bytes())
        path.write_bytes(gzip.compress(content, compresslevel=1, mtime=0))
        assert sha256_file(path) != entry["sha256"][name]
        entry["sha256"][name] = sha256_file(path)
    assert store.write_week(WEEK, matches, players, source="test") is False
    assert store.manifest["weeks"][WEEK]["revision"] == 1


def test_upload_gives_each_file_its_asset_name(store, monkeypatch):
    """The download builds its address from the asset name, so the upload must use it."""
    matches, players = rows(range(2), week_bounds(WEEK)[0])
    store.write_week(WEEK, matches, players, source="test")
    uploaded = []

    def fake_run(command, **_):
        if command[:3] == ["gh", "release", "upload"]:
            uploaded.append(command[4].rsplit("/", 1)[-1])
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr("vexil.data.remote.subprocess.run", fake_run)
    upload_week(store, WEEK, "owner/repository")
    assert uploaded == [store.asset_name(WEEK, 1, name) for name in ("matches", "players")]
    assert uploaded[0] == f"week-{WEEK}.r1.matches.jsonl.gz"
