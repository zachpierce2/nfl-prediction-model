# Advanced matchup results

Fixed 2021–2025 development test: all 1,424 games, 33 pushes per version. No thresholds or parameter search.

| Version | Wins | Losses | ATS win rate | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Existing market + EPA | 714 | 677 | 51.33% | 9.76509 | 12.66123 |
| Add pressure / protection | 689 | 702 | 49.53% | 9.76655 | 12.66323 |
| Add explosive plays | 691 | 700 | 49.68% | 9.76552 | 12.66178 |
| Add special teams | 707 | 684 | 50.83% | 9.76495 | 12.65958 |
| Add non-QB injuries | 694 | 697 | 49.89% | 9.77173 | 12.66163 |
| All four groups | 710 | 681 | 51.04% | 9.77393 | 12.66239 |

None improves the baseline's all-game ATS record. Special teams slightly improves margin MAE and RMSE, but wins fewer spread picks. The combined version loses four wins and worsens both error measures. No candidate is promoted.

The injury/combined arms are incomplete historical diagnostics: 2025 injury publication times are unavailable and those inputs are withheld, with explicit missingness and training-only imputation. Partial injury reports never establish a healthy roster. Pressure is a team/QB outcome proxy for protection, not an isolated line grade. Sources may be revised and these seasons have already been examined.

Validation: final [Actions run 43986198396](https://github.com/zachpierce2/nfl-prediction-model/actions/runs/43986198396) passed the feature/prospective tests, existing grading tests, all 36 annual model folds and final artifact verification. All 1,693 EPA baseline validation predictions reproduce within 1e-10 points. Full predictions, annual comparisons, source hashes, injury/snap snapshots, derived feature matrices, coverage and fold records are in the run artifact.

The branch also connects isolated prospective forecast archiving and paired core/EPA grading. It skips candidates lacking required groups and rejects after-kickoff results. These integrations are tested with controlled fixtures; no new real live forecasts were created by this PR. Workflow integrations do not activate until merge.

Next step: collect paired prospective forecasts with measured input coverage. Do not tune these additions against the same historical seasons until they look profitable.
