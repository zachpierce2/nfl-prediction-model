# EPA error audit and targeted test

The frozen 2021–2025 baseline reproduces 714–677–33 (51.33%) across all 1,424 games.

## Main diagnostic

Games with absolute market spreads greater than seven points: 329 games, 145–177–7 (45.03%). Actual home margins exceed model predictions by 2.27 points on average. The signed error is positive in every evaluated season: 2.18 points across 2021–2023 versus 2.42 across 2024–2025. Both periods are previously examined development data, not independent validation.

The baseline makes very small market adjustments: median 0.124 points; 90th percentile 0.328; maximum 0.644. A directional pick is often a slight lean. Aggregate model MAE is 9.76509 versus market MAE 9.76229. The week-clustered interval for the model's MAE gain spans zero. Subgroup intervals are descriptive and not adjusted for multiple comparisons.

## Targeted experiment

Add one binary input for absolute market spread strictly greater than seven points. Refit the existing ridge on prior seasons only. Keep alpha 1000, 25% adjustment, two-point cap and all-game coverage. No hardcoded 2.27-point bonus, threshold search or bet filtering.

| Scope | Baseline W–L–P | Baseline ATS | Candidate W–L–P | Candidate ATS |
|---|---:|---:|---:|---:|
| All 1,424 games | 714–677–33 | 51.33% | 702–689–33 | 50.47% |
| Spread greater than seven | 145–177–7 | 45.03% | 136–186–7 | 42.24% |

The candidate and baseline disagree on 70 non-push games: candidate wins 29 and baseline wins 41. The candidate loses 12 net wins. Its MAE worsens by 0.00333 points. The week-clustered interval for the ATS percentage-point difference is approximately [-2.12, +0.34]. No improvement is established; do not promote.

The feature was chosen after inspecting these test seasons. Prior-season training prevents fitting directly on each future game, but does not make the feature-selection process an untouched validation.

## Prospective sample

The inspected EPA archive contains zero timestamped EPA forecasts and zero graded EPA games. No historical results or reconstructed 2026 forecasts are added to that record. Existing automatic collection remains unchanged.

## Reproduction and verification

Install requirements-market.txt, then run:

```
python -m unittest -v test_epa_error_audit test_epa_targeted_correction
python epa_error_audit.py
python epa_targeted_correction.py
```

[Actions run 43989850187](https://github.com/zachpierce2/nfl-prediction-model/actions/runs/43989850187) passed ten tests and both complete analyses. All 1,693 baseline predictions, including the extra 2020 fold, reproduce within 1e-10 points. Tests cover outcome-independent bucket membership, sign/push handling, actual prior-season training membership, equal game coverage, paired comparison and rejection of post-kickoff prospective records.

The run artifact contains every game error, 25 largest errors, subgroup summaries by season, stability tables, full candidate predictions, annual folds, comparisons and source hashes. Checked-in JSON summaries preserve the reported audit and experiment.

## Next research

Keep collecting the already-enabled prospective comparisons. Before another structural change, inspect whether wide-spread errors are concentrated in unexpected QB changes or missing availability information using evidence recorded before kickoff. The available retrospective audit does not establish that cause. Do not turn the best-looking historical subgroup into a betting rule.
