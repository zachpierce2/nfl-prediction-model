# QB passing experiment results

GitHub Actions run 43956958065 tested the source and completed the 2020–2025 prior-season folds. The table pools 2021–2025; all 1,424 games are included and win rates exclude 33 pushes.

| Method | Wins | Losses | Pushes | ATS win rate | MAE |
|---|---:|---:|---:|---:|---:|
| Existing EPA baseline | 714 | 677 | 33 | 51.33% | 9.76509 |
| EPA + last observed QB passing | 710 | 681 | 33 | 51.04% | 9.76586 |
| EPA + actual starter passing diagnostic | 717 | 674 | 33 | 51.55% | 9.76698 |

The starter diagnostic adds three net wins relative to the EPA baseline; its MAE is slightly worse. This is not convincing evidence of improvement. Actual historical starter identities were recorded retrospectively without verified announcement times; prior QB performance excludes same-day/future games, but historical identity availability is unverified.

The prospective connector requires valid entries in the existing confirmation input; that input currently has zero rows. No confirmed-news collector or source-content verification service has been installed. Unconfirmed depth-chart candidates remain excluded. The implementation is on a separate branch and does not change production forecasts.

Detailed input/source hashes and per-year records are in QB_PASSING_RESULTS.json. Full predictions, features, fold boundaries and compact QB stats are retained in the run artifact.
