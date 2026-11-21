import numpy as np

from vexil import alerts


def test_rolling_sum_and_ratio():
    values = np.arange(1.0, 11.0)
    rolled = alerts.rolling_sum(values, window=3)
    assert np.isnan(rolled[:2]).all()
    assert rolled[2:].tolist() == [6, 9, 12, 15, 18, 21, 24, 27]
    assert np.isnan(alerts.ratio(np.array([1.0]), np.array([0.0]))[0])
    assert np.isnan(
        alerts.rolling_sum(np.ones(5), window=8)
    ).all()  # history shorter than the window


def test_missing_values_never_fire():
    statistic = np.array([np.nan, 0.5, -0.5])
    assert alerts.below(statistic, reference=0.0, limit=0.1).tolist() == [False, False, True]
    assert alerts.outside(statistic, reference=0.0, limit=0.1).tolist() == [False, True, True]


def test_cusum_by_hand():
    shortfall = np.array([1.0, 1.0, np.nan, 1.0, -5.0, 3.0])
    # steps with slack 0.5: +0.5, +0.5, (missing), +0.5 -> 1.5 passes 1.2 and restarts
    firing = alerts.cusum(shortfall, slack=0.5, limit=1.2)
    assert firing.tolist() == [False, False, False, True, False, True]


def test_an_alarm_is_the_first_firing_week():
    firing = np.array([[False, True, True, False, True], [True, True, False, False, False]])
    assert alerts.alarms(firing).sum(axis=-1).tolist() == [2, 1]


def test_cusum_finds_a_lasting_shift_sooner_than_a_single_week_rule():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, (4000, 40))
    shifted = noise + 1.0  # every week is one standard deviation worse
    limit_single, limit_cusum = 2.6, 4.0  # both give well under one false alarm per 40 weeks
    assert alerts.alarms(noise > limit_single).sum(-1).mean() < 0.3
    assert alerts.alarms(alerts.cusum(noise, 0.5, limit_cusum)).sum(-1).mean() < 0.3
    first = lambda firing: np.where(firing.any(-1), firing.argmax(-1), 40).mean()  # noqa: E731
    assert first(alerts.cusum(shifted, 0.5, limit_cusum)) < first(shifted > limit_single)


def test_one_step_slope_is_one_for_calibrated_predictions():
    rng = np.random.default_rng(1)
    log_odds = rng.normal(0, 1.5, 400_000)
    p = 1 / (1 + np.exp(-log_odds))
    y = (rng.random(400_000) < p).astype(float)
    slope = alerts.one_step_slope((log_odds * (y - p)).sum(), (log_odds**2 * p * (1 - p)).sum())
    assert abs(slope - 1) < 0.01
    overconfident = 1 / (1 + np.exp(-log_odds / 0.85))
    x = log_odds / 0.85
    slope = alerts.one_step_slope(
        (x * (y - overconfident)).sum(), (x**2 * overconfident * (1 - overconfident)).sum()
    )
    assert 0.80 < slope < 0.90
