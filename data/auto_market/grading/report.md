# NFL prospective grading

Updated: 2026-11-23T10:42:20.103974+00:00

| Metric | Result |
|---|---:|
| Forecasts | 15 |
| Graded games | 14 |
| Pending | 1 |
| Hypothetical wins | 8 |
| Hypothetical losses | 6 |
| Pushes | 0 |
| No lean | 0 |
| Win rate excluding pushes | 57.14% |
| Sportsbook MAE | 8.4643 |
| Model MAE | 8.4072 |

Grades use the frozen forecast and recorded spread. Any nonzero adjustment counts as a hypothetical lean, however tiny. This is not an executed betting record. No profit or ROI is claimed.

Only provider-confirmed completed games are graded. Pending games are not losses. Scores are fetched daily with a three-day lookback; outages longer than that may leave missing results.

## DraftKings opening-line comparison

Matched finals: 0

| Forecast | Wins | Losses | Pushes | No lean |
|---|---:|---:|---:|---:|
| opening | 0 | 0 | 0 | 0 |
| regular | 0 | 0 | 0 | 0 |


## Line movement against later sportsbook prices

Near-kickoff comparisons (within 60 minutes): 13. Average point advantage: +0.038.

Earlier last observations: 0 (excluded from that average).

Positive means the saved pick had a better spread than the later same-book quote. Each CSV row shows the quote time and minutes before kickoff. Three-hour polling may miss the closing price. These are proxies, not certified closing lines; odds/juice are shown separately and are not included in point value.

Status counts: {"NEAR_KICKOFF_PROXY": 13, "NO_LATER_SAME_BOOK_QUOTE": 1, "PENDING_KICKOFF": 1}

## Prospective core versus EPA (paired games only)

Paired pregame forecasts: 13. Historical predictions are not backfilled.

| Model | Graded | Wins | Losses | Pushes | ATS win rate | MAE |
|---|---:|---:|---:|---:|---:|---:|
| core | 12 | 8 | 4 | 0 | 66.67% | 7.8044 |
| epa | 12 | 8 | 4 | 0 | 66.67% | 7.7980 |

Matched same-book near-kickoff comparisons: 12. Separate line-value means are in epa_shadow/summary.json. These are point-spread proxies, not certified closing prices or profit.
# Starting QB comparison

All three models use the same games and sportsbook quotes.

| Model | Graded | W–L–P | ATS | MAE |
|---|---:|---|---:|---:|
| epa | 2 | 2–0–0 | 100.00% | 15.6845 |
| last_observed | 2 | 2–0–0 | 100.00% | 15.6919 |
| expected | 2 | 2–0–0 | 100.00% | 15.6919 |

Complete forecast triples: 3. Starter identity changes: 0. Expected forecasts missing a control: 0. Controls missing EPA: 0.

The changed-starter subgroup and pairwise disagreements are in summary.json; per-game IDs, picks, results and errors are in changed_picks.csv. A small early sample does not establish improvement.
