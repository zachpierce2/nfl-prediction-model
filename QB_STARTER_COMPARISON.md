# Prospective starting-QB comparison

Compare three frozen forecasts on exactly the same future games and sportsbook quotes:

1. Existing market + EPA baseline.
2. QB passing model using the last observed team QB (the QB with most dropbacks in the team's most recent earlier completed game).
3. The same QB passing model using the timestamped expected/confirmed starter selection.

The two QB predictions use one fitted estimator per game. Historical training rows, coefficients, residual settings, EPA inputs, source snapshots and quote are identical. Only the live QB identities and their existing prior-performance features change. This isolates starter selection; it does not retrain a separate historical last-observed model. Prior eight QB appearances, shrinkage definitions and prior-season-only training stay fixed.

`qb_passing.record` optionally computes the last-observed prediction during creation of a new expected-QB forecast. It uses the same in-memory PBP aggregates, requiring no additional download. The archived common input hash includes the expected and last-observed feature snapshots. Both forecasts have the same creation timestamp, recorded before kickoff. The control references a hash of the complete expected forecast. A missing prior QB identity skips the control explicitly. Previously saved expected forecasts can never acquire a new control later.

`qb_starter_comparison.py` validates quote prices, line, teams, kickoff, snapshot ID, core identity, shared QB inputs, expected-row hash, model versions and timing. Daily grading uses only complete EPA/control/expected triples. It separately reports missing control and EPA coverage. Pending games remain pending; zero-game accuracy and error stay null. Existing archives remain independent.

Saved outputs in `data/auto_market/qb_starter_comparison/`:

- `forecasts.csv`: frozen last-observed-QB predictions.
- `summary.json` and `report.md`: same-game ATS records and mean absolute errors, plus a predeclared changed-starter subgroup and pairwise disagreement results.
- `changed_picks.csv`: QB identities, identity-change flag, three predictions, selected sides, results and prediction errors for each complete forecast triple, including pending games.
- Three paired-grade CSVs, each covering the identical graded games.

The daily main grading report also displays the three-model comparison. Automatic collection and grading workflows include validation and persist the new archive. The PR workflow uses fixtures; it never accesses odds or writes live forecasts.

Validation: 75 integrated checks and six existing grading checks passed locally. New checks cover same-estimator fitting, identical non-QB model inputs, prior-season training, prior-day QB identity isolation, source/hash/quote/timing corruption, equal three-model grading coverage, pending and missing forecasts, changed-starter results, disagreement wins and errors, and no historical backfill.

There are currently zero complete prospective triples and zero graded games. This implementation adds a measurement, not a demonstrated accuracy improvement. No historical starter expectations are reconstructed, no tuning thresholds are selected, and no primary-model promotion or executable-price profitability is established. Early results and the changed-starter subgroup can be noisy.
