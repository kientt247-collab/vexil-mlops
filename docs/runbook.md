# Runbook: what to do when something happens

The weekly job runs on Mondays at 03:17 UTC (GitHub may start it late). It needs no action
when nothing fires. This page says what to check when it does.

## An issue titled `[alert] <family> / <subject>` opens

The issue body holds the alarm, a recommendation, and breakdowns of the last 8 final
weeks. Read the recommendation first, then check it against the tables below it.

| Family | What fired | What to check | Usual action |
|---|---|---|---|
| `data / pull` | under 90% of judged-tier matches are usable | the Actions log of the pull; whether OpenDota is behind on parsing | none on the models; the next run pulls the week again |
| `inputs / economy` | weekly mean of a gold or experience total is more than 4.2% from its development mean | the "By patch" table: is there a new patch? | the pipeline fits a refit challenger once 300 judged matches on the patch are final |
| `calibration / <model>` | rolling 8-week slope or mean of p minus outcome left its band | whether the performance alarm is quiet (then confidence is off, ranking is fine) | the pipeline fits a recalibration challenger |
| `performance / <model>` | CUSUM of the gain over the anchor passed its limit | the breakdown by league and patch: one league, or everywhere? rating coverage | the pipeline fits a refit challenger |

An alarm on `rstar` (the baseline) asks for no action: it is a yardstick.

Later alarms of the same rule add a comment to the same issue. The pipeline closes the
issue after 4 quiet final weeks. You may close it earlier by hand.

Keep in mind how little one week says: a week has about 100 judged matches. The limits
were set so that each family gives at most one false alarm per 52 weeks per model, and
the performance rule needs a lasting drop of about 0.010 log loss to be found within 12
weeks.

## An issue titled `[challenger] <id>` opens

- "Rejected by the gate": nothing to do. The challenger is retired in the registry. This
  is the usual outcome.
- "Promotion proposed": the challenger beat the champion over its shadow period (8 weeks,
  at least 600 judged matches, gain of at least 0.002 with a 95% interval above 0, slope
  between 0.9 and 1.1). Decide whether to promote. If yes, edit `models/registry.yaml` in
  a pull request: old champion to `retired`, challenger to `champion`, and add
  `approval: {by: <name>, date: <date>}`. If no, set the challenger to `retired`.

## The weekly run failed

Open the run in the Actions tab. A failed run commits nothing, and the next run processes
every week that is not final, so one missed Monday loses nothing.

| Message | Meaning | Fix |
|---|---|---|
| `LockError` | a locked file or the code differs from `lock.json` | if the change is intended, make a new lock version (below); if not, revert it |
| `RegistryError` | a model file does not match its registered hash, or roles are inconsistent | restore the file or correct the registry |
| `ClockError` | a real run was started with a clock earlier than the last recorded run (`monitoring/last_run.json`); it would move the stored state back | wait for the next Monday, or use "dry run" |
| `RulesError` | the champion has no entry under `performance` or `calibration` in `monitoring/rules.yaml` (this happens after a promotion) | add the entries for the new champion and make a new lock version |
| "fail the shape checks" | more than 5% of pulled matches are malformed | look at one rejected match; OpenDota may have changed a table |
| "giving up on the OpenDota explorer" | OpenDota did not answer after retries | start the workflow again by hand later |
| `git push` rejected | someone pushed while the job ran | start the workflow again |

## Changing a rule, the gate, a starting model or the code

These are locked. A change is a new lock version:

1. make the change in a pull request and let CI pass;
2. run `uv run python -m vexil lock --version <n+1> --approved-by <name> --reason "<why>"`;
3. commit `lock.json` with the change.

The dashboard shows the lock version and its date. Numbers from before and after a lock
change were produced under different rules.

## Starting the job by hand

Actions tab, workflow "weekly", "Run workflow". Tick "dry run" to run everything on the
runner only: nothing is uploaded, committed or published, no issue is opened, and the log
shows what would have changed. Tick "publish only" to rebuild the page from the committed
tables without running the job.


## History of the lock

| Version | Date | What changed |
|---|---|---|
| 1 | 2026-11-21 13:18 UTC | go-live: rules, gate, three starting models, code. No real run happened under it |
| 2 | 2026-11-21 | code only, after a dry run under version 1: weekly files are uploaded under their asset names (they were uploaded under the bare file name, so a later run would not have found them); a week counts as changed only when its rows change (compressed bytes differ between machines); a dry run uploads nothing and opens no issue; a real run earlier than the last recorded run is refused. Rules, gate and models are the same files |
| 3 | 2026-11-23 10:59 UTC (approved by Kien 17:59 UTC+7) | four frozen candidates (A1, A3, A4, A5 of the vexil-auto loop) enter as challengers with kind `frozen_train`, scored from week 2026-11-23, the shadow clock starting then. The locked set grows by `models/constants/items.json`, `models/candidates/<tag>/train.py` and `model.json`, and `uv.lock`. Rules, gate, champion, baseline and anchor are the same files. |

## Candidates in the shadow period

A candidate is an audited `train.py` run by the pipeline through `CandidateCtx`: it sees the
rows of a week, the pregame facts and `results_before` (results that ended strictly before
the match starts), and gets the label only inside `fit`. Its fitted state lives in
`models/candidates/<tag>/model.json`, with the sha256 of the train file. A candidate is
scored from `scored_from` onwards; earlier weeks are left alone. The result history behind
`results_before` is kept as `state/history-before-<week>.json.gz` snapshots, the same way as
the rating snapshots, and is replayed from the store when a snapshot is missing.

Candidates are compared with the champion by the same gate as any challenger (8 final
weeks, at least 600 judged matches, gain at least 0.002 with the 95 percent interval above
0, slope 0.9 to 1.1). The gate only watches challengers of kind `refit` and `recalibration`
for its own collision check, so candidates of kind `frozen_train` do not clash with them.
The job opens a "Promotion proposed" Issue; a person approves it in the registry.
