# Dropback component results

Reject this candidate for ATS prediction. Separating pass attempts, sacks, scrambles and designed runs into opponent-adjusted EPA/success contributions and frequencies loses 24 wins across the same 1,424 games in 2021–2025. Existing live forecasting is unchanged.

| Model | Wins | Losses | Pushes | ATS | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Frozen core + EPA baseline | 714 | 677 | 33 | 51.33% | 9.765093 | 12.661227 |
| Original EPA features, refreshed source control | 714 | 677 | 33 | 51.33% | 9.765015 | 12.661258 |
| Four play components | 690 | 701 | 33 | 49.60% | 9.768032 | 12.659685 |

ATS and MAE worsen. The tiny RMSE improvement does not establish a better betting model. This tests one fixed representation; it does not establish that sacks or scrambles contain no useful information.

| Season | Frozen baseline W–L–P | Refreshed control W–L–P | Candidate W–L–P |
|---|---|---|---|
| 2021 | 150–131–4 | 151–130–4 | 146–135–4 |
| 2022 | 130–144–10 | 128–146–10 | 131–143–10 |
| 2023 | 144–127–14 | 142–129–14 | 145–126–14 |
| 2024 | 153–128–4 | 157–124–4 | 141–140–4 |
| 2025 | 137–147–1 | 136–148–1 | 127–157–1 |

On 362 non-push disagreements with the frozen baseline, baseline wins 193 and candidate 169. On 360 disagreements with the identical-source control, control wins 192 and candidate 168. The decline therefore also appears when source vintage is held constant.

## Source revision and reproducibility

The raw source check failed before candidate model evaluation. Current PBP differs from the earlier frozen aggregates in 4,988 offense-game EPA means, 1,557 success means and 27 play counts. A protocol amendment added the original-model control rebuilt from the same new data, before inspecting candidate results. All games retain identical schedule and spread snapshots. No data revision is credited as feature improvement.

`dropback_team_components.csv.gz` freezes the compact 2017–2025 aggregates (121,047 designed runs, 169,421 pass attempts, 11,966 sacks, 8,607 scrambles). Its manifest preserves raw and aggregate source hashes. Default reproduction uses this checked-in snapshot without network requests. The original frozen EPA predictions must still reproduce across all 1,693 validation games within 1e-10 points. Summed category EPA/success ratings must recover the rebuilt combined ratings within 1e-12.

Run `python market_dropback_components.py`, then `python -m unittest -v test_dropback_components`. The read-only PR/manual Actions workflow uploads full predictions, yearly tables, changed picks, feature snapshots, source revision diagnostics and training folds.

Eight checks cover disjoint classification, EPA/success/count conservation, replay reversals, no-event categories, malformed flags, kneels/spikes, duplicate/missing coverage, strict prior-date history, season resets, exactly eight recent games including zeros, actual model fit-row membership, equal game coverage and baseline reproduction. Validation uses 2020–2025; the report uses all 2021–2025 games and no edge filter. All other model settings stay fixed, as documented in DROPBACK_PROTOCOL.md.

These are previously examined development seasons, not untouched validation. Historical source publication times and executable prices are not verified. There is no automatic promotion, live workflow change or profitability claim. Existing prospective QB comparisons continue independently.
