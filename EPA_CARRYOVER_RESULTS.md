# Reduced-weight prior-season EPA carryover

Fixed 50% prior-season weight; no grid search. Existing core + opponent-adjusted EPA model otherwise unchanged. All 1,424 games in 2021–2025 included.

| Version | Wins | Losses | Pushes | ATS win rate | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Current season reset | 714 | 677 | 33 | 51.33% | 9.76509 | 12.66123 |
| Prior-season carryover at 50% | 687 | 704 | 33 | 49.39% | 9.76574 | 12.66270 |

Carryover lost 27 wins and 1.94 percentage points, with slightly worse prediction errors. It is not promoted; the live model and prospective comparison are unchanged.

## Early season diagnostic (regular-season Weeks 1–4)

All-game evaluation above remains primary. This predeclared diagnostic checks the proposed reason for carryover.

| Version | Wins | Losses | Pushes | ATS win rate | MAE |
|---|---:|---:|---:|---:|---:|
| Current season reset | 176 | 135 | 9 | 56.59% | 9.81373 |
| Carryover at 50% | 161 | 150 | 9 | 51.77% | 9.81763 |

## Yearly all-game results

| Season | Reset W–L–P | Reset ATS | Carryover W–L–P | Carryover ATS |
|---|---:|---:|---:|---:|
| 2021 | 150–131–4 | 53.38% | 153–128–4 | 54.45% |
| 2022 | 130–144–10 | 47.45% | 130–144–10 | 47.45% |
| 2023 | 144–127–14 | 53.14% | 134–137–14 | 49.45% |
| 2024 | 153–128–4 | 54.45% | 143–138–4 | 50.89% |
| 2025 | 137–147–1 | 48.24% | 127–157–1 | 44.72% |

## Method

Each team's most recent eight offense games, with previous-season games allowed to fill the window. Only the immediately preceding season is eligible. Prior-season play weights are multiplied by 0.5; current-season weights remain 1.0. Previous-season observations disappear as current-season games fill the window. Both EPA and success ratings retain opponent ridge alpha 200. The residual model retains all existing core inputs, alpha 1000, a 25% adjustment and a two-point cap. Feature history excludes same-day games. Each test season trains only on prior seasons.

Carryover changes historical training features too, so forecasts later in the season can differ even after old team games leave the current feature window. No teams or betting thresholds were selected using reported outcomes. 2020 is retained as an extra validation fold but excluded from the five-year total, matching prior experiments.

## Verification and reproduction

The reset variant reproduces previous EPA predictions across all 1,693 validation games within 1e-10 points and the previous feature matrix within 1e-12. Identical test games and training boundaries were verified. Three automated checks cover prior information, same-day/too-old exclusion, and exact 50% sample weighting.

From the repository root, install requirements-market.txt, then run:

```
python market_epa_carryover.py
python -m unittest -v verify_carryover.py
```

Defaults use existing epa_games_snapshot.csv and included carryover_team_game_epa.csv. Outputs go to epa_carryover_results/. Source team-game values are the same archived EPA aggregates used by the prior experiment, not newly downloaded or reconstructed from future results. The CSV summary is EPA_CARRYOVER_COMPARISON.csv.

## Limits and next experiment

All evaluated seasons are previously examined development data, not an untouched holdout. Historical EPA and spreads may be revised and lack verified pregame publication times. These are hypothetical ATS direction results, not executed betting profits.

Next proposed test: a fixed regularized model targeting cover probability rather than margin residual, using the same point-in-time features and annual folds. Push handling must be explicit; compare ATS accuracy and probability calibration separately from spread error. No such model is implemented or promoted in this experiment.
