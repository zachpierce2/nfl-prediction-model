# NFL prospective grading

Updated: 2026-11-24T10:33:49.490591+00:00

| Metric | Result |
|---|---:|
| Forecasts | 15 |
| Graded games | 15 |
| Pending | 0 |
| Hypothetical wins | 8 |
| Hypothetical losses | 7 |
| Pushes | 0 |
| No lean | 0 |
| Win rate excluding pushes | 53.33% |
| Sportsbook MAE | 8.4333 |
| Model MAE | 8.3868 |

Grades use the frozen forecast and recorded spread. Any nonzero adjustment counts as a hypothetical lean, however tiny. This is not an executed betting record. No profit or ROI is claimed.

Only provider-confirmed completed games are graded. Pending games are not losses. Scores are fetched daily with a three-day lookback; outages longer than that may leave missing results.

## DraftKings opening-line comparison

Matched finals: 0

| Forecast | Wins | Losses | Pushes | No lean |
|---|---:|---:|---:|---:|
| opening | 0 | 0 | 0 | 0 |
| regular | 0 | 0 | 0 | 0 |


## Line movement against later sportsbook prices

Near-kickoff comparisons (within 60 minutes): 14. Average point advantage: +0.036.

Earlier last observations: 0 (excluded from that average).

Positive means the saved pick had a better spread than the later same-book quote. Each CSV row shows the quote time and minutes before kickoff. Three-hour polling may miss the closing price. These are proxies, not certified closing lines; odds/juice are shown separately and are not included in point value.

Status counts: {"NEAR_KICKOFF_PROXY": 14, "NO_LATER_SAME_BOOK_QUOTE": 1}

## Prospective core versus EPA (paired games only)

Paired pregame forecasts: 13. Historical predictions are not backfilled.

| Model | Graded | Wins | Losses | Pushes | ATS win rate | MAE |
|---|---:|---:|---:|---:|---:|---:|
| core | 13 | 8 | 5 | 0 | 61.54% | 7.8272 |
| epa | 13 | 8 | 5 | 0 | 61.54% | 7.8260 |

Matched same-book near-kickoff comparisons: 13. Separate line-value means are in epa_shadow/summary.json. These are point-spread proxies, not certified closing prices or profit.
# Starting QB comparison

All three models use the same games and sportsbook quotes.

| Model | Graded | W–L–P | ATS | MAE |
|---|---:|---|---:|---:|
| epa | 3 | 2–1–0 | 66.67% | 13.1768 |
| last_observed | 3 | 2–1–0 | 66.67% | 13.1790 |
| expected | 3 | 2–1–0 | 66.67% | 13.1790 |

Complete forecast triples: 3. Starter identity changes: 0. Expected forecasts missing a control: 0. Controls missing EPA: 0.

The changed-starter subgroup and pairwise disagreements are in summary.json; per-game IDs, picks, results and errors are in changed_picks.csv. A small early sample does not establish improvement.
