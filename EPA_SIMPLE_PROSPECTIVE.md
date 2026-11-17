# Prospective market + EPA only comparison

Historical development results are 715–676–33 (51.40%) for market + EPA only versus 714–677–33 (51.33%) for core + EPA across 2021–2025. One extra win is insufficient evidence to promote a model. These seasons have already informed development.

This challenger freezes the existing simplification specification: spread_line and the four opponent-adjusted EPA/success differences, prior-season training, median imputation, standardization, Ridge alpha 1000, 25% adjustment and a two-point cap. It omits the core team's other inputs. No threshold or parameter search is added.

After merge, automatic collection creates this challenger only alongside new core/EPA forecasts, using the identical saved feature snapshot, quote, sportsbook and prices. A missing EPA pair or incomplete input prevents the challenger. Both calculations must finish before kickoff. Existing games are never backfilled and archived rows are never replaced. There is no extra odds API request. The primary forecast stays unchanged.

Daily grading compares simple versus existing EPA on exactly the same candidate games. Summary JSON, paired grade CSVs and a Markdown table are persisted under data/auto_market/epa_simple_shadow/ and included in the main grading report and artifact. Empty samples have null accuracy. These are hypothetical picks, not executed betting records.

Review all-game ATS accuracy, margin MAE, season consistency and paired disagreements as future results accumulate; do not promote based on a handful of wins or repeatedly retune against the same games. No improved prospective accuracy has been established at implementation time.

Validation: python -m unittest -q test_epa_simple_shadow test_epa_shadow test_advanced_shadow test_qb_market test_qb_passing test_qb_news test_qb_announcements. Tests check the fixed numerical formula, identical inputs and quotes, pregame boundaries, immutable archives, no backfill, same-subset grading and null empty-sample accuracy. PR checks validate fixtures without fetching odds or writing live archives.
