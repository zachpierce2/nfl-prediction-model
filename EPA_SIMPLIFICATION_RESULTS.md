# Market + EPA only comparison

Fixed-setting development experiment, 2021–2025, all 1,424 games.

| Model | Wins | Losses | Pushes | ATS win rate | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Core | 673 | 718 | 33 | 48.38% | 9.76685 | 12.66182 |
| Core + EPA | 714 | 677 | 33 | 51.33% | 9.76509 | 12.66123 |
| Market + EPA only | 715 | 676 | 33 | 51.40% | 9.76120 | 12.65897 |
| Market baseline | No picks | No picks | No picks | N/A | 9.76229 | 12.65644 |

The simplification gains one win versus core + EPA (0.0719 percentage points). The models disagree on 237 non-push games: the simpler version wins 119, while core + EPA wins 118. This is not a persuasive advantage. Average absolute error improves by about 0.00390 points versus core + EPA and 0.00109 versus the market, while RMSE remains slightly worse than the market. No live model is changed.

## Yearly market + EPA only results

| Season | Wins | Losses | Pushes | ATS win rate |
|---|---:|---:|---:|---:|
| 2021 | 142 | 139 | 4 | 50.53% |
| 2022 | 133 | 141 | 10 | 48.54% |
| 2023 | 148 | 123 | 14 | 54.61% |
| 2024 | 149 | 132 | 4 | 53.02% |
| 2025 | 143 | 141 | 1 | 50.35% |

## Method and reproduction

Run `python market_epa_simplification.py` from the repository root with the market requirements installed. Defaults use the existing strength_games_snapshot.csv and strength_epa_features.csv. Outputs go to epa_simplification_results/. These are fixed historical snapshots through 2025, not current-season data.

All methods train an annual ridge on prior seasons only, targeting actual home margin minus the market expected home margin. Residual alpha 1000, standardized inputs, median imputation, 25% adjustment and 2-point cap remain fixed. No tuning or edge threshold filtering. The EPA-only candidate uses spread_line and the four tested opponent-adjusted EPA/success differences. Historical EPA construction remains unchanged (last eight team games within each season, opponent ridge alpha 200, same-day results excluded). A separate market-only baseline makes no directional picks.

2020 provides additional validation predictions but is excluded from the reported five-year total, matching prior experiments. The existing core and core + EPA predictions reproduced the previous artifact across all 1,693 validation games within 1e-10 points. Game coverage and prior-season fold boundaries were checked.

All evaluated seasons have already been examined in previous experiments. These are development results, not an untouched holdout. Historical statistics and spread data may be revised and lack verified pregame publication times. No executable-price profitability is established. Existing prospective core/EPA tracking stays unchanged.
