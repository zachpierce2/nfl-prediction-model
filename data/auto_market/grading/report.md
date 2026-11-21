# NFL prospective grading

Updated: 2026-11-21T10:27:19.771495+00:00

| Metric | Result |
|---|---:|
| Forecasts | 3 |
| Graded games | 3 |
| Pending | 0 |
| Hypothetical wins | 1 |
| Hypothetical losses | 2 |
| Pushes | 0 |
| No lean | 0 |
| Win rate excluding pushes | 33.33% |
| Sportsbook MAE | 11.5000 |
| Model MAE | 11.4508 |

Grades use the frozen forecast and recorded spread. Any nonzero adjustment counts as a hypothetical lean, however tiny. This is not an executed betting record. No profit or ROI is claimed.

Only provider-confirmed completed games are graded. Pending games are not losses. Scores are fetched daily with a three-day lookback; outages longer than that may leave missing results.

## DraftKings opening-line comparison

Matched finals: 0

| Forecast | Wins | Losses | Pushes | No lean |
|---|---:|---:|---:|---:|
| opening | 0 | 0 | 0 | 0 |
| regular | 0 | 0 | 0 | 0 |


## Line movement against later sportsbook prices

Near-kickoff comparisons (within 60 minutes): 2. Average point advantage: +0.000.

Earlier last observations: 0 (excluded from that average).

Positive means the saved pick had a better spread than the later same-book quote. Each CSV row shows the quote time and minutes before kickoff. Three-hour polling may miss the closing price. These are proxies, not certified closing lines; odds/juice are shown separately and are not included in point value.

Status counts: {"NEAR_KICKOFF_PROXY": 2, "NO_LATER_SAME_BOOK_QUOTE": 1}

## Prospective core versus EPA (paired games only)

Paired pregame forecasts: 1. Historical predictions are not backfilled.

| Model | Graded | Wins | Losses | Pushes | ATS win rate | MAE |
|---|---:|---:|---:|---:|---:|---:|
| core | 1 | 1 | 0 | 0 | 100.00% | 10.3047 |
| epa | 1 | 1 | 0 | 0 | 100.00% | 10.3613 |

Matched same-book near-kickoff comparisons: 1. Separate line-value means are in epa_shadow/summary.json. These are point-spread proxies, not certified closing prices or profit.
# Starting QB comparison

All three models use the same games and sportsbook quotes.

| Model | Graded | W–L–P | ATS | MAE |
|---|---:|---|---:|---:|
| epa | 1 | 1–0–0 | 100.00% | 10.3613 |
| last_observed | 1 | 1–0–0 | 100.00% | 10.3900 |
| expected | 1 | 1–0–0 | 100.00% | 10.3900 |

Complete forecast triples: 1. Starter identity changes: 0. Expected forecasts missing a control: 0. Controls missing EPA: 0.

The changed-starter subgroup and pairwise disagreements are in summary.json; per-game IDs, picks, results and errors are in changed_picks.csv. A small early sample does not establish improvement.
