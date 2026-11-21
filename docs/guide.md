# How this repository is organised, and why

This guide is for a reader who knows modelling and wants to see how a small production
system around a model is put together. It explains where things are, which rules the code
follows, and how to change it. It assumes no knowledge of the project's history.

## 1. What the system does

Every Monday a job on GitHub does five things for professional Dota 2 matches:

1. pulls the matches of the last three ended weeks from OpenDota (a free public database);
2. scores every match, minute by minute, with every model listed in the registry;
3. measures performance on the weeks whose results are complete;
4. raises an alarm, as a GitHub issue, when a frozen rule says performance, calibration,
   the inputs or the data have moved, and writes what it found and what it recommends;
5. rebuilds a static dashboard.

It also fits *challengers* (a refit or a recalibration of the model in use) when an alarm
or a new game patch asks for one, scores them next to the model in use, and proposes a
replacement only if a gate is passed. A person approves any replacement.

Words used below. *Champion*: the model in use. *Anchor*: a three-number model that serves
as a ruler. *Minute row*: one row per match and game minute. *Log loss*: the metric, over
minutes 0 to 29, lower is better. *Final week*: a week whose data will no longer change
(14 days after it ends). *Label*: the match result.

## 2. Map of the repository

```
models/                 model files (JSON) and registry.yaml, the list of monitored models
monitoring/             everything the job produces: tables, rules.yaml, gate.yaml
  predictions/<week>.parquet   one row per match, minute and model
  matches.parquet              one row per scored match
  match_scores.parquet         one row per match and model: sums needed for metrics
  alert_states.parquet         one row per week, alarm family and subject
  rules.yaml, gate.yaml        frozen limits (set by simulation, see docs/evidence)
data/manifest.json      which weekly data files exist, with their SHA-256
state/                  rating snapshots, so the job does not replay the whole year
lock.json               hashes of the rules, the gate, the starting models and the code
src/vexil/              the package (section 3)
tests/                  tests on synthetic matches; no real match data
scripts/                one-off session scripts: parity with the research phase, simulations
docs/evidence/          the output of those scripts, kept as proof
.github/workflows/      ci.yml (lint and tests on every push), weekly.yml (the job)
```

Raw match data is not in the repository. It lives as weekly files attached to a GitHub
release; `data/manifest.json` records the hash of each file.

## 3. The package, in three layers

The rule: a module may import from its own layer or from a layer with a lower number,
never from a higher one. Code that reads or writes files sits in the bottom layer. The
arithmetic sits in the top layer and can be tested with plain numbers.

**Layer 1: pure logic.** No network, no clock, no project files. Two small helpers do
read or write a file they are handed: `hashing.py` fingerprints a file, and `ratings.py`
can save and load its own state.

| Module | What it holds |
|---|---|
| `weeks.py` | the calendar: week names, when a week is final, data regions |
| `matchdata.py` | helpers on one raw match record; the rule "minute t sees nothing after t" |
| `features/a1.py` | the ten model inputs, twice: a clear version and a fast version |
| `models/linear.py` | model files to probabilities, with NumPy only |
| `ratings.py` | player ratings as a state that can be saved and resumed |
| `evaluate.py` | log loss sums per match, weekly summaries, paired gains, calibration slope |
| `alerts.py` | alarm statistics: rolling sums, CUSUM, one-step slope |
| `hashing.py` | SHA-256 of a file |

**Layer 2: one step each.** They combine layer 1 and still do no file work of their own,
with two exceptions that read small configuration files (`registry.py`, `models/fit.py`
imports scikit-learn).

| Module | What it holds |
|---|---|
| `scoring.py` | which matches are judged; inputs and predictions of one match |
| `registry.py` | reads and checks `models/registry.yaml` |
| `monitor.py` | applies `rules.yaml` to weekly tables, returns alarm states |
| `investigate.py` | recommendation and report text for an alarm |
| `retrain.py` | challenger specifications and the promotion gate |
| `leakcheck.py` | the "poison" tool used by leak tests |

**Layer 3: the edges.** Everything that touches disk, network or GitHub.

| Module | What it holds |
|---|---|
| `data/opendota.py`, `data/pull.py`, `data/schema.py` | fetch weeks, check their shape |
| `data/store.py`, `data/remote.py` | weekly files, manifest, release assets |
| `labels.py` | the only door to match results, with a log and the cutoff rule |
| `pipeline.py`, `run.py` | one week through ratings, inputs, predictions; tables on disk; training rows for a refit |
| `issues.py` | alarms as GitHub issues (or a log file in a replay) |
| `job.py` | the weekly job and the replay, in one function |
| `dashboard.py` | builds the static page from the tables |
| `cli.py` | `python -m vexil <command>` |

The rule is enforced by a test: `tests/test_layers.py` reads the imports of every module
and fails if one reaches into a higher-numbered layer. A new module must be given a layer
there before the tests pass.

Why layers matter: when a number is wrong, the search is short. A wrong input is in layer
1 and has a test with a hand-built match. A wrong file is in layer 3. And layer 1 can be
reused elsewhere (a notebook, another pipeline) without dragging GitHub along.

## 4. Six choices that shape the code

1. **Frozen things are files with hashes.** Model files, the rules, the gate and the data
   files are never edited in place. The registry stores the SHA-256 of each model; the
   manifest stores it for each data file; `lock.json` stores it for the rules, the gate,
   the starting models and the code. A run stops if a hash differs. This replaces trust
   ("nobody changed it") with a check.
2. **One door to the labels.** Code that needs a match result calls `LabelLoader`. The
   reader of the data store removes the result from every row. So a search for
   `week_labels` lists every place that can see an outcome, and the log lists every time
   it happened.
3. **Sums, not averages, are stored.** `match_scores` keeps, per match and model, the
   number of minute rows and the sum of their losses. Any weekly, rolling or per-league
   number is then a group-by, and pooled numbers are exactly the sum of weekly ones. The
   calibration slope does not add up, which is why the alarms use a one-step version
   built from two sums.
4. **The replay is the job.** `replay` calls `run_job` once per past Monday. There is no
   second code path for history, so a year of replay is a test of the live job.
5. **Limits come from simulation and are written down.** `scripts/m6_alert_rules.py` and
   `scripts/m8_gate.py` measured how often each rule fires on noise and how fast it finds
   a planted fault. Their output is in `docs/evidence/`; the limits they chose are in
   `monitoring/rules.yaml` and `monitoring/gate.yaml`. The job only applies them.
6. **Prediction needs NumPy only.** Fitting imports scikit-learn; scoring does not. A
   model file can be scored anywhere, and the weekly job installs less.

## 5. How the tests are built

- *Hand-built cases.* A 12-minute match whose ten inputs were computed by hand
  (`tests/test_a1.py`); two matches whose ratings were computed by hand.
- *Cut and poison tests.* Inputs are recomputed on a copy of the match cut at minute t,
  and on a copy where everything after t is random. Equal results prove the input does
  not read the future.
- *Planted bugs.* A deliberately wrong copy of the code (reads minute t+1; includes a
  match's own result in its rating; moves the label cutoff by a day) must make a test
  fail. A test that cannot fail proves nothing.
- *Planted faults.* Hand-built weekly tables with one fault each check that the right
  alarm fires and the right action is recommended (`tests/test_monitor.py`).
- *Pipeline-level leak test.* The replay is run on a store cut after week k; everything
  written for weeks up to k must be byte-identical to the full-year replay
  (`tests/test_job.py` on synthetic data, `scripts/m10_replay_checks.py` on real data).
- *Parity scripts.* `scripts/m1_parity.py`, `m3_parity.py`, `m5_parity.py` compare this
  package with the research code it came from, on real data that is not in the
  repository. Their results are in `docs/evidence/`.

Tests in `tests/` use synthetic matches only and run in a few seconds.

## 6. How to do common things

**Run it locally.**

```
uv sync
uv run pytest -q
uv run python -m vexil verify-remote            # download the weekly data files, check hashes
uv run python -m vexil --root /tmp/scratch replay --first-week 2026-10-26 --last-week 2026-11-16
```

(For a replay into a scratch folder, copy `models/`, `data/manifest.json`,
`monitoring/rules.yaml` and `monitoring/gate.yaml` there first.)

**Put a model under monitoring.** Save its file under `models/`, add an entry to
`models/registry.yaml` with its SHA-256 and the role `challenger` (or `baseline`). If the
file is of a kind the code knows (`ridge_a1`, `gold_time`, `recalibrated`), nothing else
changes. A new kind needs one class with `input_names` and `predict(inputs)` in
`models/linear.py`, one branch in `load_model`, and any new input in
`scoring.match_inputs`.

**Take a model out.** Set its role to `retired`. Its past predictions stay in the tables.

**Promote a challenger.** Only after the pipeline has opened a "[challenger] ... Promotion
proposed" issue. Set the old champion to `retired`, the challenger to `champion`, and add
`approval: {by: <name>, date: <date>}` to its entry. The registry refuses a champion
without those fields. The starting models are in the lock, a challenger file is not, so
no new lock version is needed for a promotion.

**Add an alarm rule.** Write the statistic in `alerts.py` (pure, on arrays), measure its
false-alarm rate with a script like `scripts/m6_alert_rules.py`, put the limit into
`rules.yaml`, apply it in `monitor.evaluate_rules`, add a planted-fault test. Because
`rules.yaml` and the code are locked, this needs a new lock version (next item).

**Change anything that is locked.** Make the change in a pull request, then run
`uv run python -m vexil lock --version <n+1> --approved-by <name> --reason "<why>"` and
commit the new `lock.json`. The dashboard shows the version and its date. Numbers before
and after a lock change come from different rules; say so when you quote them.

**Read a failed run.** Open the run in the Actions tab; the failing step names the stage.
`RegistryError`: a model file or the registry is inconsistent. `LockError`: a locked file
or the code changed without a new lock. "fail the shape checks": OpenDota changed its
tables or returned broken rows. A failed run changes nothing: the next run processes
every week that is not final yet.

## 7. Six exercises (optional)

1. In `tests/test_a1.py`, change one event time in the hand-built match so that
   `towers_diff` at minute 7 becomes 1, and work out the new expected values by hand.
2. Write a planted bug of your own: a copy of `RatingState.read` that forgets
   `advance_to`. Which existing test catches it, and why?
3. Add a fourth watched total to the input alarm (for example total last hits at minute
   10). List every file you must touch, in layer order, before you write any code.
4. `summarise` returns log loss by minute phase. Add the mean number of minute rows per
   match and a test for it.
5. Run the replay for four weeks into a scratch folder, delete one rating snapshot, run
   the last Monday again, and explain from `job._rating_state` why the result is the same.
6. Draft (do not merge) a new model kind that predicts a constant 0.5. What is the
   smallest set of changes that lets the registry score it?
