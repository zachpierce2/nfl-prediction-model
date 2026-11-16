# Direct cover-probability experiment

Fixed regularized logistic regression versus the existing core + EPA margin model. All 1,424 games in 2021–2025 were evaluated, with 33 pushes excluded from ATS win-rate denominators and probability scoring.

| Version | Wins | Losses | Pushes | ATS win rate |
|---|---:|---:|---:|---:|
| Current core + EPA margin model | 714 | 677 | 33 | 51.33% |
| Direct cover-probability classifier | 693 | 698 | 33 | 49.82% |

The classifier records 21 fewer wins (1.51 percentage points lower). On the 303 non-push games where the two disagree, the classifier wins 141 and the margin model wins 162. No live model or prospective tracking configuration is changed.

## Probability evaluation

Lower scores are better. Scores use all 1,391 non-push evaluation games.

| Probability forecast | Brier score | Log loss | Calibration error (10 fixed bins) |
|---|---:|---:|---:|
| Direct cover classifier | 0.251813 | 0.696813 | 0.031112 |
| Constant 50% | 0.250000 | 0.693147 | 0.000359 |
| Prior training home-cover rate | 0.250221 | 0.693590 | 0.016403 |

The classifier did not outperform either simple probability benchmark. Calibration error is descriptive and depends on bins; a constant 50% forecast can have a low calibration error while providing no game-specific discrimination. The margin model has no validated cover-probability output, so no invented probability scores are attributed to it. Classifier reliability bins are saved in COVER_PROBABILITY_CALIBRATION.csv; the sparsest bins have only one or a few games and should not guide filters.

## Yearly ATS results

| Season | Margin W–L–P | Margin ATS | Classifier W–L–P | Classifier ATS |
|---|---:|---:|---:|---:|
| 2021 | 150–131–4 | 53.38% | 130–151–4 | 46.26% |
| 2022 | 130–144–10 | 47.45% | 133–141–10 | 48.54% |
| 2023 | 144–127–14 | 53.14% | 140–131–14 | 51.66% |
| 2024 | 153–128–4 | 54.45% | 153–128–4 | 54.45% |
| 2025 | 137–147–1 | 48.24% | 137–147–1 | 48.24% |

## Fixed method

Both models use exactly the existing core inputs plus four opponent-adjusted EPA/success differences, the same feature snapshots, and prior-season-only annual training. The classifier uses standardized inputs, median imputation, logistic regression with fixed C=0.01, lbfgs, max_iter=2000, tol=1e-8, and no class weighting or calibration fit. No regularization grid or pick-threshold search was performed. Pick home above 0.5, away below 0.5; exactly 0.5 makes no pick. All candidate evaluation games receive probabilities, including pushes; only probability scoring and classifier training exclude pushes.

Probabilities estimate P(home covers given no push), not unconditional cover probability. They do not model push mass and are not used to claim betting expected value. The unchanged margin baseline uses ridge alpha 1000, 25% adjustment and a 2-point cap. 2020 is an additional validation fold, excluded from the reported five-year total to match prior experiments.

## Verification and reproduction

The margin baseline reproduces all 1,693 prior validation predictions within 1e-10 points. Equal game coverage, prior-season training boundaries and valid probability ranges were checked. Three automated checks verify push exclusions, the 50% benchmark and tie handling, and absence of fabricated probability scores for the margin model.

With requirements-market.txt installed, run from the repository root:

```
python market_cover_probability.py
python -m unittest -v verify_cover_probability.py
```

Defaults use existing strength_games_snapshot.csv and strength_epa_features.csv. Outputs go to cover_probability_results/.

All evaluated seasons are previously examined development data, not an untouched holdout. Historical EPA/spreads may be revised and lack verified pregame publication timestamps. No executable-price profitability is established.

## Next proposed improvement

Test the sportsbook pregame total_line as one additional input to the existing margin + EPA model, keeping the current settings fixed. This field is available in the schedule snapshot. Only total_line is eligible; total (the actual combined score) must never be used as a pregame feature. That experiment is not implemented or promoted here.

## Branch workflow

The `experiment/direct-cover-probability` branch adds `.github/workflows/nfl-direct-cover.yml`.
It runs automatically when experiment code, inputs, or the workflow changes on that branch,
and on relevant pull requests. Manual dispatch is also declared; GitHub normally exposes
that button after the workflow exists on the default branch. No merge is needed for the
branch push run.

The workflow uses Python 3.12 and `requirements-cover.txt`, runs seven unit/integration
checks, reruns the frozen 2020–2025 annual walk-forward experiment, and uploads
`nfl-direct-cover-results-<run-id>-<attempt>`. The run summary shows the 2021–2025
comparison. The artifact contains `predictions.csv`, `comparison.csv`, `calibration.csv`,
`training_folds.csv`, `provenance.json`, `summary.md`, dependency versions and commit SHA.
Training folds now include the final training date and first evaluation date; overlapping
dates fail the run. Each method must cover exactly the same games. Integration checks
inspect actual fit rows, reproduce the frozen margin reference within 1e-10 points, and
confirm same-day/future score changes cannot alter earlier core features. EPA inputs
remain the existing frozen snapshots with the provenance limitations above.

Reproduce from the repository root:

```bash
python -m pip install -r requirements-cover.txt
python -m unittest -v verify_cover_probability.py test_direct_cover_experiment.py
python market_cover_probability.py --out cover_probability_results
```

This is an isolated research branch. The production forecast scripts are unchanged.
