# Advanced matchup candidates

Adds four input groups to a separate market + EPA experiment and to isolated prospective challengers. Main forecasts are not replaced.

| Group | Inputs |
|---|---|
| Pressure / protection | Prior eight-game team pressure allowed/created rates using PFR pressured-dropback counts and PBP dropbacks; sack rates; pressure matchup interaction. These are team/QB outcomes, not isolated offensive-line grades. |
| Explosive plays | Offensive and defensive pass gains of at least 20 yards per dropback and non-dropback rush gains of at least 10 yards per rush; opponent matchup interactions. |
| Special teams | Net EPA on punts, kickoffs, field goals and extra points, per special-teams play; cumulative home EPA changes give a consistent team perspective. |
| Non-QB injuries | Separate Out, Doubtful and Questionable prior-snap-weighted burdens for OL, skill players, defensive front, secondary and specialists. Prior three team games; at least 50% usage for non-specialists and positive special-teams usage for specialists. No invented play probabilities or point penalties. |

## Evaluation

One fixed specification: current-season eight-game histories, annual prior-season-only training, ridge alpha 1000, 25% adjustment, two-point cap. All evaluated games are retained; missing values use training-only median imputation with explicit coverage flags. The baseline, four individual additions and combined model are compared on identical games. 2020 is an extra fold; the five-season summary is 2021–2025.

Historical pressure data begins in 2018. Historical injury reports without an individual modification timestamp are unavailable, including the inspected 2025 feed. Other historical injury rows must precede the game's date in UTC; this intentionally excludes same-day reports. Their historical first observation time is still unverified, so the injury backtest is a retrospective development diagnostic. Report presence means partial observed coverage, not a complete healthy roster. IR/reserve lists and player replacement quality are not modeled.

## Prospective recording

On this branch, auto_market.py calls advanced_shadow.record after preserving the original core/EPA predictions. It uses the same quote and records each candidate separately. Exact derived inputs, source metadata, usage and injury rows are archived. Current-season files refresh at every new forecast batch; completed-season aggregates are cached and hash checked.

Candidates with unavailable required feature groups are skipped explicitly. The combined candidate requires all four groups. Freshly retrieved live injury reports can be observed without a publication timestamp, but are never backdated into historical forecasts. Published timestamps, when present, cannot exceed observation time. Partial coverage and unknown player usage remain explicit.

grade_market.py adds paired grading against both the core forecast and the same-run EPA-only candidate. Core hashes, quote identity, prices and pre-kickoff forecast times are checked. Predictions are append-only and never backfilled. Workflow changes become active only if this branch is merged; this PR does not merge or promote any candidate.

## Reproduce

Install requirements-market.txt, then run:

```
python -m unittest -v test_advanced_matchups test_advanced_shadow test_epa_shadow
python advanced_matchups.py
python -m unittest -v test_advanced_matchups.ResultTests
```

The PR/manual workflow uploads full predictions, yearly comparisons, training folds, coverage rates, exact inputs and source hashes. Historical source values may be revised, and all evaluation seasons have been inspected before. This does not establish prospective improvement or betting profit.

## Existing code audit

The older V3 EPA model includes pass/rush EPA and success rates. Archived automatic reports already provide categorical injury reports and prior-use proxies, but do not fit those effects into the market model. This experiment connects independently defined advanced features to that market/EPA baseline without duplicating the older EPA inputs.
