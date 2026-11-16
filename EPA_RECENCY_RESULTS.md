# Fixed EPA recency experiment

2021–2025 development backtest; all 1,424 games, 33 pushes.

| Model | Wins | Losses | ATS win rate | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Existing EPA baseline | 714 | 677 | 51.33% | 9.76509 | 12.66123 |
| Four-game half-life | 705 | 686 | 50.68% | 9.76620 | 12.66200 |

The candidate loses nine wins and worsens both error metrics. Do not promote it. Live forecasts are unchanged.

Uses the latest eight offense games within the current season, with exponentially declining weights (half-life four appearances). Normalization preserves each offense's total play weight, keeping the existing ridge penalty comparable. Opponent ridge alpha 200, residual ridge alpha 1000, 25% adjustment and two-point cap remain fixed. No grid search or game filtering. Each annual test fold trains on prior seasons only; feature construction excludes same-day results.

Validation: three checks pass for exact half-life and weight mass, same-day/future/prior-season exclusion, identical game coverage and training boundaries. The reset baseline reproduces all 1,693 archived validation predictions within 1e-10 points. 2020 remains an extra fold outside the five-year summary.

Reproduce with `pip install -r requirements-market.txt`, `python market_epa_recency.py`, and `python -m unittest -v test_market_epa_recency.py`. The Actions workflow runs the experiment and uploads full predictions, feature matrices, fold records, comparison and source hashes.

These seasons have already been examined. Historical EPA and spreads may be revised and lack verified pregame publication timestamps. This is development evidence, not an untouched holdout or verified betting profit. Early-week tables are inherited exploratory diagnostics and do not justify filtering or promotion.

Next priority: populate and validate the existing timestamped confirmed-starter input so the QB candidate can accumulate paired prospective forecasts. More recency parameter searching on these same seasons is not justified by this test.
