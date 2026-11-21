# vexil-mlops

Weekly monitoring of live Dota 2 win-probability models on professional matches.

Every Monday a GitHub Actions job pulls the last weeks from [OpenDota](https://www.opendota.com),
scores them with every model in `models/registry.yaml`, measures performance, raises alarms
as GitHub issues with a short investigation and a recommendation, fits and gates challenger
models, and rebuilds a static dashboard.

- Dashboard: https://kientt247-collab.github.io/vexil-mlops/
- How the code is organised and why: [docs/guide.md](docs/guide.md)
- What to do when an alarm opens or a run fails: [docs/runbook.md](docs/runbook.md)
- Proof behind the frozen rules and the parity with the research code: [docs/evidence](docs/evidence)

## What is monitored

| Model | Role | What it is |
|---|---|---|
| `final-model` | champion | ridge logistic regression on ten in-game inputs, time, and a player rating |
| `rstar` | baseline | the same without the rating |
| `anchor` | anchor | logistic regression on the gold lead and time; a ruler for weekly noise |

The models come from an earlier research project and are frozen: their files are checked
by SHA-256 on every run.

## How to read the numbers

Log loss over game minutes 0 to 29; lower is better; a coin flip scores 0.693. Three data
regions appear on every chart. Development (to 2026-07-30): the models were fitted here,
so out-of-fold predictions are shown. Holdout (2026-08-01 to 2026-11-19): its labels were
read once for the research verdict, so every weekly or alarm number shown for it was
computed afterwards (marked POST-HOC) and only demonstrates the system. Prospective (from
2026-11-23): scored after the rules were locked; the only clean evidence, and a thin one
for the first 8 weeks at about 100 judged matches a week.

## Run it

```
uv sync && uv run pytest -q
uv run python -m vexil replay --first-week 2026-10-26 --last-week 2026-11-09 --scratch /tmp/scratch
```

The first line installs and runs the tests. The second downloads the weekly match files
(about 220 MB, no credentials) and replays three weeks into `/tmp/scratch`: five Monday
runs, because a week becomes final 14 days after it ends.
`uv run python -m vexil --help` lists the other commands; `docs/guide.md` explains the
code.

Match data: weekly files derived from the OpenDota API, attached to the release
`data-2026`; `data/manifest.json` lists them with their hashes.
