# Benchmark sensitivity and capped-margin test

The 2021–2025 benchmark reproduced the archived pregame-total win, loss and push counts in every season. Rebuilding the baseline with refreshed EPA changed 42 picks on the same games and spreads. Those picks went 26–16 with the archived inputs and 16–26 with refreshed inputs. Every changed archived projection was within 0.026 points of the spread. The average absolute projection change across 1,424 games was 0.0105 points.

One fixed change clipped the historical training target (actual home margin minus spread) to ±21 points. All other model inputs, annual training boundaries, ridge and prediction settings were held fixed. Every game was graded against its actual score and archived spread. Recent training begins in 2017 and earlier-era training begins in 1999. Each test year uses only earlier seasons.

| Period and EPA source | Pregame total W–L–P | Capped W–L–P |
|---|---:|---:|
| 2021–2025 archived | 725–666–33 | 703–688–33 |
| 2021–2025 refreshed | 715–676–33 | 706–685–33 |
| 2003–2016 refreshed | 1,809–1,828–101 | 1,805–1,832–101 |

Reject the capped training target. These are historical development results. Historical spreads have no verified opening timestamp. Neither the historical replay nor the source sensitivity comparison changes a previously timestamped pregame forecast.

Run from the repository root with the market dependencies installed using `python capped_margin.py CACHE_DIR capped_margin_results`. The recent source sensitivity comparison uses the unweighted training feature file produced by `game_state_epa.py`.
