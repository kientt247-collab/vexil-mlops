# How this repository is organised, and why

This guide is for a reader who knows modelling and wants to see how a small production
system around a model is put together. It says where things are, which rules the code
follows, and how to change it. It assumes no knowledge of the project's history; words
that need explaining are in the glossary at the end.

## 1. What the system does

The models predict, at each minute of a professional Dota 2 match, the probability that
one side ("Radiant") wins. Every Monday at 03:17 UTC a job on GitHub does this, in order:

1. **Pull.** Fetch the matches of the three most recent ended weeks from OpenDota, a free
   public database, and store what changed.
2. **Score.** Compute, for every match and game minute, the prediction of every model
   listed in the registry (`models/registry.yaml`).
3. **Evaluate.** Apply the frozen alarm rules to all final weeks (a week is final 14
   days after it ends; see the calendar below).
4. **Alarm.** For the week that became final in this run: open a GitHub issue when a
   rule starts firing, with an investigation and a recommended action; comment on an
   issue that is already open; close one that has been quiet for 4 final weeks.
5. **Challengers.** A challenger is a model that is scored every week next to the
   champion (the model in use) without being used. The job fits one when an alarm
   recommends it or when a new game patch has 300 final matches: either a refit of the
   champion's recipe or a recalibration of the champion. After 8 weeks it applies the
   gate (section 4). A model that a person adds to the registry by hand with the role
   `challenger` is scored the same way.
6. **Dashboard.** Rebuild the static page and commit the week's tables.

In normal operation a person decides two things only: whether a proposed challenger
replaces the champion, and whether a locked rule may change. Committing the tables and
publishing the page are steps of the workflow file (`.github/workflows/weekly.yml`), not
of the Python job.

### One week on the calendar

A week is named by its Monday and runs to the next Monday, 00:00 UTC. Take the week
`2026-11-23`:

| Run on Monday | What happens to week 2026-11-23 |
|---|---|
| 2026-11-30 | the week has just ended; pulled and scored for the first time (provisional) |
| 2026-12-07 | pulled and scored again, because OpenDota processes some replays late (provisional) |
| 2026-12-14 | pulled and scored a third time; 14 days have passed since it ended, so it is now final: its numbers enter the alarm rules and are not recomputed by later runs |

So each run pulls three weeks, and those three are exactly the weeks that are not final
before the run. If a Monday run fails, nothing is committed. The following Monday pulls
its own three weeks; the week that should have become final in the failed run is no
longer among them, so it keeps the numbers of its second pull. It still gets its alarm
evaluation: the job remembers which weeks it has already treated as final
(`monitoring/final_weeks.json`) and handles every week that is final and not yet in that
list. Starting the job by hand before the next Monday avoids the missed third pull
(section 6). A match that OpenDota processes more than 14 days
late is never scored.

## 2. Map of the repository

```
models/                 model files (JSON) and registry.yaml, the list of monitored models
  challengers/          model files the job fitted itself
monitoring/             tables the job writes, and the two frozen rule files
  predictions/<week>.parquet   one row per match, minute and model
  matches.parquet              one row per scored match
  match_scores.parquet         one row per match and model: the sums metrics are built from
  alert_states.parquet         one row per week, alarm family and subject
  label_reads.log              one line per read of match results
  rules.yaml, gate.yaml        the frozen limits
data/manifest.json      which weekly data files exist, with their fingerprints
state/                  rating snapshots, so the job does not replay the whole year
lock.json               fingerprints of the rules, the gate, the starting models and the code
src/vexil/              the package (section 3)
tests/                  tests on synthetic matches; no real match data
scripts/                session scripts: comparisons with the research code, simulations
docs/evidence/          the output of those scripts, kept as proof
.github/workflows/      ci.yml (checks on every push), weekly.yml (the job)
```

The job also publishes the dashboard (built into a folder `site/` that is not committed)
to GitHub Pages.

Raw match data is not in the repository. It is a set of weekly files attached to the
GitHub release `data-2026` of this repository (a release is a place to attach files to a
repository; they download without credentials). `data/manifest.json` has one entry per
week:

```json
"2026-11-09": {
  "revision": 1,
  "matches": 185, "usable": 184, "source": "phase2-seed",
  "bytes":  {"matches": 696707, "players": 2265095},
  "sha256": {"matches": "14fed354b6...", "players": "6d3efc0b19..."}
}
```

A SHA-256 is a fingerprint of a file: change one byte and it changes. A data file is
never overwritten. When a new pull of a week differs from the stored one, it is written
as a new revision (`r2`, `r3`) and the manifest points to it. A result is tied to its
data through git: the commit that holds a table also holds the manifest that was current
when the table was computed.

## 3. The package, in three layers

Each module of `src/vexil/` has a layer number. **A module may import modules with the
same number or a smaller one, never a larger one.** Layer 1 knows nothing about the
others; layer 3 may use everything.

The rule is enforced by `tests/test_layers.py`, which reads the imports of every module
and holds the assignment:

```python
LAYER = {
    "weeks": 1, "matchdata": 1, "hashing": 1, "alerts": 1, "evaluate": 1, "ratings": 1,
    "features.a1": 1, "models.linear": 1,
    "scoring": 2, "registry": 2, "monitor": 2, "investigate": 2, "retrain": 2, ...
    "data.store": 3, "labels": 3, "pipeline": 3, "run": 3, "job": 3, ...
}
```

A new module must be added there, and an import that points to a larger number fails
the test.

**Layer 1: arithmetic and rules.** Functions take numbers or records and return numbers.
No network, no clock, no knowledge of where the project's files are. A few functions read
or write a file whose path they are handed: `hashing.sha256_file`, the save and load of
`ratings.py`, and `models.linear.load_model`.

| Module | What it holds |
|---|---|
| `weeks.py` | the calendar: week names, when a week is final, data regions |
| `matchdata.py` | helpers on one raw match record; the rule "minute t sees nothing after t" |
| `features/a1.py` | the ten in-game model inputs (differences between the two sides in gold, experience, kills, towers and similar), twice: a clear version used by tests and a fast version used by the job; a test compares the two at every minute |
| `models/linear.py` | from a model file and inputs to probabilities, with NumPy only |
| `ratings.py` | the player rating, the one input that is not in-game, as a state that can be saved and resumed |
| `evaluate.py` | loss sums per match, weekly summaries, paired gains, calibration slope |
| `alerts.py` | alarm statistics: rolling sums, CUSUM, one-step slope |
| `hashing.py` | the fingerprint of a file |

**Layer 2: one step each.** They combine layer 1 into a decision.

| Module | What it holds | Files it touches |
|---|---|---|
| `scoring.py` | which matches are judged; assembles all inputs of one match and predicts. The inputs are the ten from `features/a1.py`, the rating from `ratings.py`, and the raw gold lead, which only the anchor uses and which is read straight from the match record here | none |
| `registry.py` | reads and checks `models/registry.yaml` | that file and the model files |
| `monitor.py` | applies the rules, passed in as a dictionary, to weekly tables | none |
| `investigate.py` | recommendation and report text for an alarm | none |
| `retrain.py` | challenger specifications and the gate | none |
| `models/fit.py` | fitting; the only module that imports scikit-learn | none |
| `leakcheck.py` | the "poison" tool used by leak tests | none |

**Layer 3: the edges.** Everything that decides where files go, calls the network or
talks to GitHub. (`registry.py` in layer 2 is the one module outside this layer that
opens a project file by itself.)

| Module | What it holds |
|---|---|
| `data/opendota.py`, `data/pull.py`, `data/schema.py` | fetch weeks, check their shape |
| `data/store.py`, `data/remote.py` | weekly files, manifest, release |
| `labels.py` | the only door to match results (section 4, choice 2) |
| `pipeline.py`, `run.py` | one week through ratings, inputs, predictions; tables on disk; training rows for a refit |
| `issues.py` | alarms as GitHub issues, or as lines in a file during a replay |
| `job.py` | `run_job` (one Monday) and `replay` (many Mondays) |
| `dashboard.py` | builds the static page from the tables |
| `cli.py` | the commands of `python -m vexil` |

Why layers matter: when a number is wrong, the search is short. If an in-game input looks
wrong at minute 12 of a match, its definition is in `features/a1.py`; if
the rating input looks wrong, it is `ratings.py`; if the right inputs were put together
wrongly, it is `scoring.match_inputs`. And layer 1 can be used in a notebook without
GitHub or a data store.

## 4. Six choices that shape the code

1. **Frozen things are files with fingerprints.** A model file, a data file revision and
   a version of the rules are never changed silently. The registry stores the SHA-256 of
   each model; the manifest stores it for each data file; `lock.json` (shown below)
   stores it for the rules, the gate, the three starting models and the code. A run
   stops if a fingerprint differs. Changing a locked file is allowed, but only together
   with a new lock version that a person approves.
2. **One door to the labels.** A label is a match result. The reader of the data store
   (`Store.read_week`) deletes the result from every row it returns. Code that needs
   results calls the method `LabelLoader.week_labels(week, purpose)`, which appends a
   line to `monitoring/label_reads.log`:

   ```json
   {"time": "2026-11-21T11:43:59+00:00", "purpose": "weekly job", "kind": "labels",
    "matches": 225, "after_cutoff": 0, "withheld": 0}
   ```

   It also applies the cutoff rule: results of matches from 2026-11-20 on are held back
   until `lock.json` exists, so that no rule could be tuned on them. What stops other
   code from opening a raw file directly is convention plus a search: the only function
   that returns unfiltered rows is the private `Store._read_raw`, and searching for
   `_read_raw` and `week_labels` lists every caller. There is no stronger barrier, and
   none is claimed.
3. **Sums, not averages, are stored.** `match_scores` keeps, per match and model, the
   number of minute rows and the sum of their losses. Any weekly, rolling or per-league
   number is then a group-by, add the sums, add the counts, divide. A number over many weeks is therefore exactly
   consistent with the weekly ones.
   One metric does not add up: the calibration slope (glossary) is a fitted coefficient.
   The alarms therefore use a one-step approximation,
   `1 + sum(x * (y - p)) / sum(x^2 * p * (1 - p))` with `x = ln(p / (1 - p))`, `p` the
   prediction and `y` the outcome (one Newton step from a slope of 1); its two sums are
   stored per match like the losses. The gate, which runs rarely, fits the slope exactly.
4. **The replay is the job.** `replay(first_week, last_week)` takes two data weeks and
   calls `run_job` once for every Monday from the one after `first_week` to the one
   after `last_week` (a run on Monday reads the weeks that ended before it, so the first
   run reads `first_week`). Inside `run_job` three things differ from a live Monday, and
   only these: no pull (the weekly files already in the store are read), alarms go to
   `monitoring/replay_issues.jsonl` instead of GitHub issues, and the clock is the
   replayed Monday. Nothing is committed or published by a replay, because those are
   workflow steps. Each replayed run
   reads only weeks that ended before its Monday. What a replay cannot reproduce is the
   state of OpenDota on that past Monday: it sees each week as it is stored today. The
   history from week 2025-12-29 to week 2026-11-16 (47 weeks) was produced this way.
5. **Limits come from simulation and are written down.** `scripts/m6_alert_rules.py` and
   `scripts/m8_gate.py` measured how often each rule fires on noise and how fast it finds
   a planted fault. Their output is in `docs/evidence/`; the limits are in
   `monitoring/rules.yaml` and `monitoring/gate.yaml`. The job only applies them.
6. **Prediction needs NumPy only.** `models/linear.py` never imports scikit-learn; only
   `models/fit.py` does, and it is loaded only in a run that fits a challenger. A model
   file can therefore be scored anywhere NumPy runs.

### What the frozen files look like

A registry entry (`models/registry.yaml`). The role `champion` requires the approval
fields. `baseline` is a second, simpler model shown for comparison. `anchor` is a ruler
for weekly noise (glossary).

```yaml
  - id: final-model
    role: champion
    kind: ridge_a1                  # which class in models/linear.py reads the file
    file: models/candidate.json
    sha256: a24854dc32cb6e5b2e1eb37c8f2560ba0b868f10455d2cd6d8862cdbadaff072
    description: R* plus the player rating; confirmed in phase 2 (PASS-RESOLVED).
    approval:
      by: Kien
      date: "2026-11-21"
```

One rule of `monitoring/rules.yaml`, the performance alarm of the champion:

```yaml
performance:
  arm: B                 # CUSUM (glossary)
  cusum_slack: 0.5
  models:
    final-model:
      reference_gain: 0.01745   # usual gain over the anchor, in log loss
      weekly_sd: 0.01943
      typical_rows: 3714.6
      limit: 0.8011             # the alarm fires when the CUSUM level passes this
```

The gate (`monitoring/gate.yaml`). After a shadow period of 8 weeks a challenger is
proposed for promotion only if all four conditions hold:

```yaml
shadow_weeks: 8
min_matches: 600         # 1. judged matches in the shadow period
min_gain: 0.002          # 2. log loss of the champion minus that of the challenger
                         # 3. the 95% interval of that gain lies above 0
slope_low: 0.9           # 4. calibration slope of the challenger between 0.9 and 1.1
slope_high: 1.1
```

The lock (`lock.json`), written by `python -m vexil lock`. The lock version is the number
in its first field. The weekly job compares every fingerprint before it does anything.

```json
{"version": 1, "locked_at": "2026-11-22 09:00 UTC",
 "files": {"monitoring/rules.yaml": "...", "monitoring/gate.yaml": "...",
           "models/candidate.json": "...", "models/rstar.json": "...", "models/anchor.json": "..."},
 "source_tree_sha256": "...",
 "approved_by": "Kien", "reason": "go-live"}
```

`source_tree_sha256` covers every `.py` and `.html` file under `src/vexil`. Tests,
scripts and documents are not locked. Challenger files and the registry are not locked
either; the registry has its own check (the approval fields).

## 5. How the tests are built

- **Hand-built cases.** A match of twelve minute rows (minutes 0 to 11) whose ten inputs
  were computed by hand at minutes 3, 7 and 11 (`tests/test_a1.py`); two matches whose
  ratings were computed by hand.
- **Cut and poison tests.** Inputs are recomputed on a copy of the match cut at minute
  t, and on a copy where everything after t is random. Equal results prove the input
  does not read the future.
- **Planted bugs.** A deliberately wrong copy of the code (one that reads minute t+1,
  one that includes a match's own result in its rating, one that moves the label cutoff
  by a day) must make a test fail. A test that cannot fail proves nothing.
- **Planted faults.** Hand-built weekly tables with one fault each check that the right
  alarm fires and the right action is recommended (`tests/test_monitor.py`).
- **Pipeline-level leak test.** The replay is run on a store cut after week k; everything
  written for weeks up to k must be byte for byte what the full replay wrote
  (`tests/test_job.py` on synthetic data, `scripts/m10_replay_checks.py` on real data).
- **Comparison scripts.** The package was rewritten from research scripts.
  `scripts/m1_parity.py`, `m3_parity.py` and `m5_parity.py` check that it gives the same
  numbers as those scripts on the real data. They need files that are not in the
  repository, so they do not run in CI; their results are in `docs/evidence/`. The `m`
  numbers are the steps of the project plan in which each script was written.

`uv run pytest -q` runs everything in `tests/` in about ten seconds.

## 6. How to do common things

Each recipe says whether it needs a new lock version.

**Set up and run the checks.** Needs Python 3.12 or newer and `uv`, an installer and
environment manager (https://docs.astral.sh/uv/).

```
git clone https://github.com/kientt247-collab/vexil-mlops && cd vexil-mlops
uv sync                      # creates .venv with the pinned dependencies
uv run pytest -q             # expect "... passed"
uv run ruff check .          # lint: expect "All checks passed!"
```

**Get the data and replay three weeks.** No credentials are needed.

```
uv run python -m vexil verify-remote     # downloads 94 files (220 MB) into ./_store, checks every fingerprint
mkdir -p /tmp/scratch/monitoring /tmp/scratch/data
cp -r models /tmp/scratch/ && cp data/manifest.json /tmp/scratch/data/
cp monitoring/rules.yaml monitoring/gate.yaml /tmp/scratch/monitoring/
uv run python -m vexil --root /tmp/scratch --store _store replay --first-week 2026-10-26 --last-week 2026-11-09
```

It prints the number of runs and the actions taken, and writes its tables under
`/tmp/scratch/monitoring`. `lock.json` and `state/` are not needed: without a lock the
job runs unlocked, and without a snapshot it rebuilds the ratings from the first stored
week. Without a lock the cutoff rule also holds back results from 2026-11-20 on, so a
scratch replay scores no match after that date. The full history is `--first-week 2025-12-29 --last-week 2026-11-16` (about six
minutes).

**Put a model under monitoring.** No new lock version if the model is of a known kind.
Save the file under `models/`, compute its fingerprint with `sha256sum models/<file>`,
and add an entry like the one in section 4, with the role `challenger` or `baseline`.
The known kinds are `ridge_a1` (the format of `models/candidate.json`), `gold_time`
(`models/anchor.json`) and `recalibrated`. The model is scored from the next run on, for
the three open weeks; earlier weeks are not back-filled. A new kind needs code: a class
with `input_names` and `predict(inputs)` in `models/linear.py`, a branch in the function
`load_model` of the same file, and any new input in `scoring.match_inputs`. That is a
code change, so it needs a new lock version.

**Take a model out.** No new lock version. Set its role to `retired`. Its past
predictions stay in the tables. The champion cannot simply be retired: the registry
requires exactly one champion.

**Promote a challenger.** New lock version, because the alarm rules in `rules.yaml` are
keyed by model id: copy the old champion's entries under `performance` and `calibration`
to the new id, then make the lock. Without those entries the new champion would run
with no performance or calibration alarm. Do it only after the pipeline has opened an
issue "[challenger] <id>" that says "Promotion proposed"; the registry does not check
that such an issue exists, so this is a rule for people. In a pull request, set the old
champion to `retired`, the challenger to `champion`, and add `approval` with `by` and
`date` (a quoted `"YYYY-MM-DD"`). The owner of the repository approves.

**Change an alarm limit or the gate.** New lock version. A limit is the result of a
simulation, so first make the change in `scripts/m6_alert_rules.py` (or `m8_gate.py`),
rerun it, and keep its output in `docs/evidence/`. These scripts need the data store
and, for the development region, tables of the research project that only the
repository owner holds. Then put into one pull request: the edited `rules.yaml` or
`gate.yaml`, the new evidence, and the new `lock.json` (next recipe).

**Make a new lock version.** The current number is the `version` field of `lock.json`.
In the same pull request as the change, run

```
uv run python -m vexil lock --version <n+1> --approved-by <name> --reason "<why>"
```

and commit `lock.json`. Run it after the last edit to any locked file or to any file
under `src/vexil`; an edit after it makes the lock stale. CI does not read the lock, so
a stale lock passes review. The weekly job does, and it stops
with `LockError` if a locked file or the code differs from the lock. The dashboard shows
the version and its date. Numbers from before and after a lock change were produced
under different rules; say so when quoting them.

**Add an alarm rule.** New lock version. Write the statistic in `alerts.py` (layer 1, on
arrays). Measure its false-alarm rate with a script like `scripts/m6_alert_rules.py`;
the budget used so far is at most 1 false alarm per 52 weeks per model and alarm family.
Add its limit to `rules.yaml`, apply it in `monitor.evaluate_rules`, decide in
`investigate.recommend` which action it leads to, and add a planted-fault test in
`tests/test_monitor.py`.

**Start the job by hand.** In the repository's "Actions" tab choose the workflow
"weekly" and "Run workflow". Tick "dry run" to run everything without committing or
publishing.

**Read a failed run.** Open the run in the repository's "Actions" tab; the failing step
names the stage. What each message means and what to do about it is in
`docs/runbook.md`. The job commits its tables only in its last step, so a failed run
leaves the repository as it was.

## 7. Six exercises (optional)

1. In `tests/test_a1.py`, change one event time in the hand-built match so that
   `towers_diff` at minute 7 becomes 1, and work out the new expected values by hand.
2. Write a planted bug of your own: a copy of `RatingState.read` (in `ratings.py`) that
   skips its first line, `self.advance_to(meta.start)`. Which existing test catches it,
   and why?
3. Add a fifth total to the input alarm (for example total last hits at minute 10). List
   every file you must touch, in layer order, before you write any code.
4. `evaluate.summarise` returns log loss by minute phase. Add the mean number of minute
   rows per match and a test for it.
5. Replay four weeks into a scratch folder, delete one rating snapshot under `state/`,
   run the last Monday again, and explain from `job._rating_state` why the result is the
   same.
6. Draft (do not merge) a new model kind that predicts a constant 0.5. What is the
   smallest set of changes that lets the registry score it?

## Glossary

- **Alarm family and subject.** Four families of rules exist: performance, calibration,
  inputs, data. The subject is what the rule watches: a model id, `economy` (the input
  totals) or `pull` (the data pull).
- **Anchor.** A logistic regression with three coefficients, on the gold lead, time and
  their product. It is not a product. Weekly log loss depends heavily on how close the
  matches of that week were; the difference between a model and the anchor on the same
  matches removes most of that, so the performance alarm watches this difference.
- **Baseline.** A second, simpler model (`rstar`) shown for comparison with the champion.
- **Calibration slope.** Fit a logistic regression of the outcome on the model's
  log-odds; the slope is its coefficient. 1 is ideal; below 1 the model is overconfident.
- **Challenger, champion, gate.** Section 1, and the gate file in section 4.
- **CI.** The checks GitHub runs on every push (`.github/workflows/ci.yml`): lint (a
  style and error check by `ruff`) and the tests.
- **CUSUM.** A running sum of how far each week falls short of a reference, never below
  zero; it fires when it passes a limit and then restarts. It finds small lasting drops
  sooner than a rolling average.
- **Cutoff rule.** Results of matches that start on or after 2026-11-20 are withheld
  until `lock.json` exists.
- **Data regions.** Development (to 2026-07-30; the models were fitted there), holdout
  (2026-08-01 to 2026-11-19; used once to judge the models), prospective (from
  2026-11-23). Numbers mean different things in each, and the dashboard marks them. Two
  gaps are never scored: 2026-07-31 (a day left between fitting and judging) and
  2026-11-20 to 2026-11-22 (between the end of the research data and the first full
  prospective week).
- **Final week.** A week whose data will not be pulled again: 14 days after it ends.
- **Game patch.** A new version of the game. It can change how much gold or experience
  players earn, and so shift the model inputs.
- **Judged match.** Processed by OpenDota, at least 10 minutes long, played in the
  tournament draft mode, in a league OpenDota marks `professional` or `premium`. Only
  these enter metrics.
- **Log loss.** Mean of minus the log of the probability given to what happened, over
  game minutes 0 to 29; a coin flip scores 0.693; lower is better. Later minutes are
  left out because few matches reach them and they are easy to predict.
- **Minute phase.** Game minutes 0-9, 10-19, 20-29, and 30 or later. Losses are stored
  per phase; the headline log loss uses the first three.
- **Paired gain.** Log loss of a reference minus log loss of a model on the same matches.
- **PASS-RESOLVED, R\*.** Names from the research phase. R* was its strongest simple
  baseline (`rstar` here). PASS-RESOLVED was its verdict that the final model beat R* on
  the holdout by more than the required margin.
- **Pull request.** A proposed change on GitHub that is reviewed before it is merged.
- **Registry.** `models/registry.yaml`: the list of models the job scores, with roles.
- **Research phase, phase 2.** The earlier project that built and judged the models with
  scripts. This repository is its engineered version.
- **Starting models.** The three models present at go-live: `final-model`, `rstar`, `anchor`.
- **Usable match.** Processed by OpenDota (it has per-minute data) and at least 10
  minutes long. In the manifest, `"source": "phase2-seed"` marks a week copied from the
  research data instead of pulled by this pipeline.
