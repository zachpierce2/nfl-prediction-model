# Sportsbook pricing research

This branch adds all-supported-book collection and a separate, experimental pricing archive. Existing preferred-book core forecasts and EPA/QB predictions retain their selection rules. No bet is placed or selected, and the probability model is not promoted.

## What is collected and priced

The existing odds request is reused; no additional API request or secret is needed. `all_book_quotes.csv` retains every supported book's eligible, fresh, two-sided spread. Existing `quotes.csv` still contains the preferred-book quotes, preserving the existing grading and QB comparison cohorts.

When a new EPA forecast is made, every same-snapshot book quote for that game receives a home and away pricing row. The fixed EPA predicted margin is held constant across books. Historical prior-season out-of-fold prediction errors are added to this margin and rounded to integer final margins (nearest integer; ties toward positive infinity). This creates unconditional home-cover, push and away-cover probabilities. Only residual seasons earlier than the forecast season, and dates before forecast creation, are eligible. At least 200 residual games are required.

The output includes the exact quote and forecast hashes, source/code hashes, timestamps, residual count, American price, conditional break-even probability, conditional model cover probability, normalized two-sided market implied probability and estimated net return per unit risked. For win probability W, push probability P and net payout b, the calculation is `W*b - (1-W-P)`. Normalized market implied probabilities condition on no push; they are descriptive, not a true-probability guarantee.

The rank compares payout only within the same game, side and spread. All alternatives remain visible. Different spreads are priced separately; a better price at the same spread does not imply a better wager across different spreads. Supported means included in the collector, not verified account access or executable availability.

Pricing runs only when quotes and a newly recorded EPA forecast share a snapshot. It never applies an old forecast to new quotes, invents missing forecasts or backfills old games. Pricing failures are explicit in the snapshot and leave core forecasts saved. The pricing report shows only the latest capture; `prices.csv` preserves all earlier captures. Existing data collected before this feature is not reclassified as prospective pricing evidence.

## Evaluation and outcome

Run `python sportsbook_pricing.py` to reproduce `pricing_results/`. All 1,424 games in 2021–2025 are evaluated, including pushes. Each season's residual pool contains only earlier seasons' out-of-fold errors. The control applies the same method to market-only margins and market residuals on identical games. Both three-outcome Brier score and log loss are lower-is-better.

The EPA probability candidate scores **0.521894 Brier / 0.771995 log loss**, versus **0.521566 / 0.771845** for the market-only control. It does not demonstrate improvement. Predicted push frequency also differs from observed frequency. Do not label these probabilities calibrated or use positive estimated EV as evidence of profitable betting. There is no threshold tuning, historical price fabrication or ROI claim. These previously examined historical snapshots can contain revisions; this is development evidence.

Daily grading checks original quote/forecast identities and timestamps, then scores archived probabilities once per two-sided quote, against provider-confirmed finals. `pricing_shadow/grading_summary.json` distinguishes quote count, unique game count and pending games. Quotes from multiple books are correlated; the quote-weighted average is not independent evidence. The existing score-fetch three-day lookback limitation also applies here. No betting portfolio or historical/live profit is calculated.

`NFL Sportsbook Pricing Research` runs tests and reproduces the frozen-source experiment on pull requests/manual runs, with read-only permissions and no odds key. After merge, the existing automatic workflow collects the additional archive and daily grading scores it. No primary prediction changes. Artifacts include per-game probabilities, annual folds, score comparison and source provenance.

Validation: `python -m unittest -v test_sportsbook_pricing` and existing paired-forecast tests. Tests check pushes, half-point lines, American odds, payout ranking, prior-season exclusion, identical coverage, stale/wrong-identity rejection, unchanged preferred-book selection, prospective pending/completed grading and archive corruption.

Next: accumulate prospective probability scores and compare against same-quote market probabilities before considering probability calibration or any betting decision rule. Historical development results do not justify promoting this estimator.
