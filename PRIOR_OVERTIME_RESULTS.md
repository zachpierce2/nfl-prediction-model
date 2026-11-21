# Prior overtime and short recovery

Across all 1,424 games in 2021–2025 the overtime model went 716–675–33 (51.47% ATS). The market-plus-EPA primary went 714–677–33 (51.33%). Eighteen sides changed. The gain of two wins on previously studied seasons does not establish a durable advantage. Keep the feature experimental.

| Season | Primary W–L–P | Overtime W–L–P |
|---|---|---|
| 2021 | 150–131–4 | 150–131–4 |
| 2022 | 130–144–10 | 128–146–10 |
| 2023 | 144–127–14 | 146–125–14 |
| 2024 | 153–128–4 | 154–127–4 |
| 2025 | 137–147–1 | 138–146–1 |

Each test season uses only earlier seasons for fitting, starting in 2017. The feature is home minus away prior-game overtime exposure within seven days. Both versions use identical folds, archived historical spreads, ridge settings, and EPA features. The baseline reproduces archived predictions to within 1e-10 points. Pushes are excluded from percentages. Historical spreads have no verified opening timestamp.

Run from the repository root with the market requirements installed using `python prior_overtime_rest.py`.
