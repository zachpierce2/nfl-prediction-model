# NFL prospective grading

Updated: 2026-11-28T10:30:49.812106+00:00

| Metric | Result |
|---|---:|
| Forecasts | 32 |
| Graded games | 20 |
| Pending | 12 |
| Hypothetical wins | 9 |
| Hypothetical losses | 10 |
| Pushes | 1 |
| No lean | 0 |
| Win rate excluding pushes | 47.37% |
| Sportsbook MAE | 8.6500 |
| Model MAE | 8.6307 |

Grades use the frozen forecast and recorded spread. Any nonzero adjustment counts as a hypothetical lean, however tiny. This is not an executed betting record. No profit or ROI is claimed.

Only provider-confirmed completed games are graded. Pending games are not losses. Scores are fetched daily with a three-day lookback; outages longer than that may leave missing results.

## DraftKings opening-line comparison

Matched finals: 0

| Forecast | Wins | Losses | Pushes | No lean |
|---|---:|---:|---:|---:|
| opening | 0 | 0 | 0 | 0 |
| regular | 0 | 0 | 0 | 0 |


## Line movement against later sportsbook prices

Near-kickoff comparisons (within 60 minutes): 16. Average point advantage: +0.000.

Earlier last observations: 0 (excluded from that average).

Positive means the saved pick had a better spread than the later same-book quote. Each CSV row shows the quote time and minutes before kickoff. Three-hour polling may miss the closing price. These are proxies, not certified closing lines; odds/juice are shown separately and are not included in point value.

Status counts: {"NEAR_KICKOFF_PROXY": 16, "NO_LATER_SAME_BOOK_QUOTE": 4, "PENDING_KICKOFF": 12}

## Prospective core versus EPA (paired games only)

Paired pregame forecasts: 30. Historical predictions are not backfilled.

| Model | Graded | Wins | Losses | Pushes | ATS win rate | MAE |
|---|---:|---:|---:|---:|---:|---:|
| core | 18 | 9 | 8 | 1 | 52.94% | 8.2537 |
| epa | 18 | 10 | 7 | 1 | 58.82% | 8.2558 |

Matched same-book near-kickoff comparisons: 15. Separate line-value means are in epa_shadow/summary.json. These are point-spread proxies, not certified closing prices or profit.
# Starting QB comparison

All three models use the same games and sportsbook quotes.

| Model | Graded | W–L–P | ATS | MAE |
|---|---:|---|---:|---:|
| epa | 8 | 4–3–1 | 57.14% | 10.7996 |
| last_observed | 8 | 4–3–1 | 57.14% | 10.7934 |
| expected | 8 | 4–3–1 | 57.14% | 10.7934 |

Complete forecast triples: 20. Starter identity changes: 3. Expected forecasts missing a control: 0. Controls missing EPA: 0.

The changed-starter subgroup and pairwise disagreements are in summary.json; per-game IDs, picks, results and errors are in changed_picks.csv. A small early sample does not establish improvement.
