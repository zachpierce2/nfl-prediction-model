# Fixed passing/rushing EPA experiment

One predeclared ablation replaces the four combined EPA/success differences with eight separate passing/rushing differences. All other core inputs, annual prior-season training, eight-game season-reset windows, opponent ridge alpha 200, residual ridge alpha 1000, 25% adjustment and two-point cap stay fixed. No edge filters or parameter search.

| Model, 2021–2025 | Wins | Losses | Pushes | ATS | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Existing core + EPA | 714 | 677 | 33 | 51.33% | 9.765093 | 12.661227 |
| Separate pass/run EPA | 707 | 684 | 33 | 50.83% | 9.764498 | 12.657237 |

Reject promotion for ATS prediction. The candidate loses seven wins overall. Of 237 non-push disagreements, baseline wins 122 and candidate 115. Tiny error improvements do not establish a better betting model.

| Season | Baseline W–L–P | Candidate W–L–P |
|---|---|---|
| 2021 | 150–131–4 | 143–138–4 |
| 2022 | 130–144–10 | 128–146–10 |
| 2023 | 144–127–14 | 145–126–14 |
| 2024 | 153–128–4 | 157–124–4 |
| 2025 | 137–147–1 | 134–150–1 |

Run `python market_pass_rush.py`, then `python -m unittest -v test_pass_rush`. The read-only PR/manual workflow reproduces the full run and publishes predictions, feature snapshots, annual training boundaries, comparisons, source hashes and metadata. Source caches store compact per-team/game/type summaries and download hashes.

The 2020 fold is retained for reproduction; the five-year report includes all 1,424 games in 2021–2025. All 1,693 baseline validation predictions must reproduce the frozen reference within 1e-10 points. Fixture tests verify temporal exclusion, season reset, split isolation, duplicate handling, play filtering, success definition and missing source coverage. Result checks verify equal coverage and prior-season fold boundaries.

These previously examined seasons are development data, not untouched validation. Historical PBP and lines may be revised and lack verified pregame publication timestamps. Pass/run uses nflverse play_type, so scrambles remain runs; this experiment does not isolate designed rushing from all dropbacks. No executable prices or profitability are established. Live forecasts are unchanged.
