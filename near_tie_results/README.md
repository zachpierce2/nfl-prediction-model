# Independent football spread as the near-tie decision

Use the archived pregame-total side for every game except when its absolute edge is at most 0.10 points. In those games, use the side indicated by the independent predicted home margin versus the same archived spread. An exact football tie retains the original pick. This single cutoff was fixed before scoring.

Across all 1,424 games in 2021–2025 the original went 725–666–33. The tie-break went 707–684–33. Of 585 eligible games, 269 switched sides. On those games the original went 140–122–7 and the tie-break went 122–140–7. The tie-break lost wins in four of five seasons. In 2004–2016 it gained 15 wins across 3,471 games. Keep the pregame-total benchmark.

Both sources use annual prior-season fits. These are historical replays on archived spreads without verified opening timestamps. The 2004–2016 window covers the full earlier seasons shared by the input files.

To reproduce, run capped_margin.py as described in capped_margin_results/README.md. Export football_spread_results/games.csv from commit b10b4615fc97c3133d36a9d8f35a2ff59fbf1874. Then run the command below. The output includes every game and an annual summary.

```sh
python near_tie_comparison.py capped_margin_results/games.csv football_spread_results/games.csv near_tie_results
```
