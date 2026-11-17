# Fixed nonlinear cover experiment

Tests a shallow gradient boosted classifier on the same existing market and EPA inputs as the direct logistic classifier. All games remain included. Settings are fixed in source; there is no parameter or pick-threshold search. Each test season trains on earlier seasons, excludes training pushes, and forecasts home cover conditional on no push. Test pushes remain in ATS records and are excluded from binary probability scores.

## Development results, 2021–2025

| Method | Wins | Losses | Pushes | Win rate | Brier score |
|---|---:|---:|---:|---:|---:|
| EPA margin baseline | 714 | 677 | 33 | 51.33% | unavailable |
| Direct logistic | 693 | 698 | 33 | 49.82% | 0.251813 |
| Shallow trees | 703 | 688 | 33 | 50.54% | 0.253055 |
| Constant 50% probability | — | — | — | — | 0.250000 |

The challenger fails to beat the margin baseline and worsens probability scores versus the logistic classifier and constant 50%. Do not promote it. These previously examined seasons are development evidence, not an untouched validation sample or proof of profitability. Source statistics and lines lack verified historical publication timestamps.

Run `python -m unittest verify_cover_probability verify_cover_nonlinear` then `python market_cover_nonlinear.py`. The workflow also retains predictions, training folds, baseline results, and source hashes.

Next research priority: improve the information available before kickoff, starting with timestamped quarterback availability and passing-strength changes, then evaluate a frozen candidate prospectively. Do not repeatedly tune against these same historical outcomes.
