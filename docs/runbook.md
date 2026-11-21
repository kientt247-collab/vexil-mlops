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

Actions tab, workflow "weekly", "Run workflow". Tick "dry run" to run everything without
committing or publishing; the log then shows what would have changed.
