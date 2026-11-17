# Market/EPA with existing V3 QB passing features

Reuses model_v3_epa.aggregate_pbp and its QB profile definitions: prior eight QB appearances, QB EPA sum divided by dropbacks + 100, success shrinkage with 50 successes in 100 prior dropbacks, and dropback count capped at 400. All games on a date are featured before that date's results update history. Unknown QBs use the same prior with explicit missing flags.

The frozen comparison uses the existing EPA baseline and two QB arms. Last-observed QB is available from earlier game records; actual-starter identities form a retrospective diagnostic because historical announcement times are unverified. Both train only on earlier seasons with fixed ridge alpha 1000, shrinkage .25 and adjustment cap 2. All games are retained; there is no pick filter or parameter search.

Future candidate forecasts reuse nfl_data_inputs/qb_starters.csv and its existing publication/observation validation. Both starters must be confirmed. Current-game schedule IDs and depth-chart guesses never substitute for live confirmation. The older archived reports currently contain unconfirmed depth-chart candidates; the confirmation file is empty. This work connects the existing report input, not an invented automatic confirmation service. Source URLs and timestamps are validated; humans or a trusted upstream feed must verify the linked factual claim.

The automatic pipeline records a separate QB passing candidate using the identical core quote, source hashes, exact QB stats/features and confirmed IDs. It records missing confirmations explicitly, preserves first forecasts, and blocks post-kickoff writes. It refreshes current-season QB PBP whenever needed; compact historical aggregates are cached with hashes.

Run python -m unittest -v test_qb_passing test_qb_market test_epa_shadow, then python qb_passing.py. The PR workflow runs this on GitHub and retains predictions, features, compact QB statistics, comparison, fold boundaries and source hashes. Current source PBP is revised development data, not a historical point-in-time archive.

No automatic model promotion or betting-profit claim is made. This experiment remains separate from main until reviewed.
