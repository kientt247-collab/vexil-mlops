import math
import random

import pytest

from vexil import ratings
from vexil.ratings import MatchMeta, RatingState, reference_input


def league(seed: int, n_matches: int = 300, n_players: int = 40):
    """Random schedule: overlapping matches among a pool of players, with results."""
    rng = random.Random(seed)
    metas, results, clock = [], {}, 1_770_000_000
    for i in range(n_matches):
        clock += rng.randint(200, 4000)
        players = rng.sample(range(1, n_players + 1), 10)
        radiant, dire = tuple(players[:5]), tuple(players[5:])
        if rng.random() < 0.05:
            radiant = (None, *radiant[1:])  # a missing account id: the match is not rated
        metas.append(MatchMeta(i, clock, clock + rng.randint(1200, 4200), radiant, dire))
        if rng.random() < 0.97:
            results[i] = rng.randint(0, 1)
    return metas, results


def test_hand_worked_example():
    """Two matches, numbers worked out by hand."""
    a = MatchMeta(1, 0, 100, (1, 2, 3, 4, 5), (6, 7, 8, 9, 10))
    b = MatchMeta(2, 200, 300, (1, 2, 3, 4, 5), (6, 7, 8, 9, 10))
    state = RatingState()
    inputs = state.process([a, b], {1: 1, 2: 0})
    assert inputs[1] == 0.0
    # after match 1 every Radiant player is at 1500 + 64 * 0.5 and every Dire player at 1468
    assert inputs[2] == pytest.approx(64 * math.log(10) / 400)
    state.advance_to(10**9)
    expected = 1 / (1 + 10 ** (-64 / 400))
    assert state.ratings[1] == pytest.approx(1532 - 64 * expected)
    assert state.games[1] == 2


def test_a_match_never_sees_a_result_that_was_not_final_at_its_start():
    first = MatchMeta(1, 0, 1000, (1, 2, 3, 4, 5), (6, 7, 8, 9, 10))
    overlapping = MatchMeta(2, 500, 1500, (1, 2, 3, 4, 5), (6, 7, 8, 9, 10))
    later = MatchMeta(3, 1001, 2000, (1, 2, 3, 4, 5), (6, 7, 8, 9, 10))
    inputs = RatingState().process([first, overlapping, later], {1: 1, 2: 1, 3: 1})
    assert inputs[2] == 0.0  # match 1 was still running
    assert inputs[3] > 0.0  # match 1 had ended, match 2 had not


@pytest.mark.parametrize("seed", range(5))
def test_pass_equals_from_scratch_reference(seed):
    metas, results = league(seed)
    inputs = RatingState().process(metas, results)
    for meta in random.Random(seed).sample(metas, 25):
        assert inputs[meta.match_id] == pytest.approx(
            reference_input(metas, results, meta), abs=1e-9
        )


@pytest.mark.parametrize("seed", range(5))
def test_planted_bug_is_caught(seed):
    """A rating that includes the match's own result must disagree with the reference."""
    metas, results = league(seed)

    def leaky_pass():
        state, inputs = RatingState(), {}
        for meta in sorted(metas, key=lambda m: (m.start, m.match_id)):
            state.advance_to(meta.start)
            trial = RatingState(dict(state.ratings), dict(state.games), [])
            if meta.match_id in results and meta.rateable:
                trial._apply(meta.radiant, meta.dire, results[meta.match_id])
            inputs[meta.match_id] = trial.read(meta)
            if meta.match_id in results:
                state.add_result(meta, results[meta.match_id])
        return inputs

    leaky = leaky_pass()
    rated = [m for m in metas if m.match_id in results and m.rateable]
    caught = sum(
        abs(leaky[m.match_id] - reference_input(metas, results, m)) > 1e-9
        for m in random.Random(seed).sample(rated, 25)
    )
    assert caught == 25


@pytest.mark.parametrize("seed", range(5))
def test_resume_from_a_snapshot_equals_one_pass(seed, tmp_path):
    metas, results = league(seed)
    ordered = sorted(metas, key=lambda m: (m.start, m.match_id))
    full = RatingState().process(ordered, results)
    cut = random.Random(seed).randint(50, 250)
    first = RatingState()
    inputs = first.process(ordered[:cut], results)
    first.save(tmp_path / "state.json.gz")
    resumed = RatingState.load(tmp_path / "state.json.gz")
    inputs.update(resumed.process(ordered[cut:], results))
    assert inputs == full  # exact equality, not approximate


def test_a_late_result_inside_the_recomputed_window_changes_nothing_else(tmp_path):
    """Rerunning from the snapshot with one more result equals a pass that always had it."""
    metas, results = league(11)
    ordered = sorted(metas, key=lambda m: (m.start, m.match_id))
    late_id = ordered[200].match_id
    results[late_id] = 1
    without = {k: v for k, v in results.items() if k != late_id}
    state = RatingState()
    state.process(ordered[:150], without)
    state.save(tmp_path / "before.json.gz")
    RatingState.load(tmp_path / "before.json.gz").process(ordered[150:], without)  # first run
    rerun = RatingState.load(tmp_path / "before.json.gz").process(ordered[150:], results)
    full = RatingState().process(ordered, results)
    assert all(rerun[m.match_id] == full[m.match_id] for m in ordered[150:])


def test_unrated_matches_and_missing_ids():
    odd = MatchMeta(1, 0, 100, (None, 2, 3, 4, 5), (6, 7, 8, 9, 10))
    state = RatingState()
    state.process([odd], {1: 1})
    state.advance_to(10**9)
    assert state.ratings == {} and not odd.rateable
    assert state.new_player_share(odd) == 1.0


def test_rating_points_are_conserved_when_k_is_the_same_for_everyone(monkeypatch):
    monkeypatch.setattr(ratings, "K_NEW", ratings.K_ESTABLISHED)
    metas, results = league(3)
    state = RatingState()
    state.process(metas, results)
    state.advance_to(10**12)
    assert sum(v - 1500 for v in state.ratings.values()) == pytest.approx(0, abs=1e-6)
