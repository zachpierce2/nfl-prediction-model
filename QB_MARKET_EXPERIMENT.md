# Existing QB input connected to the market/EPA candidate

The existing input is nfl_data_inputs/qb_starters.csv. At inspection it contained only its header: no confirmed weekly starters were recorded. The older V3 comparison workflow passes this file to its forecast tools. The automatic market/EPA pipeline did not read it.

qb_market.py reuses this exact schema. It accepts only status=confirmed, nonempty QB IDs and source URLs, with publication <= observation <= calculation cutoff < kickoff. A later uncertain report revokes an earlier confirmation. Both teams must have valid reports. It never substitutes the retrospective schedule starter for a missing live confirmation.

The candidate adds the existing V3 qb_margin_diff and qb_experience_diff inputs to the market/EPA regression. These summarize prior eight starts' team margins and prior starts, not individual passing EPA or injury severity. A confirmed replacement's own history is used, with zero history for a new QB. No arbitrary point penalty is applied.

The automatic recorder saves separate QB forecasts and status under data/auto_market/qb_shadow and each run snapshot. Core and EPA outputs remain unchanged. Forecasts use the identical core quote and archive source hashes and starter IDs. Missing confirmations are reported, not guessed, and old forecasts are not backfilled. The module runs after the EPA snapshot is prepared; unavailable EPA inputs also block QB forecasts.

Run python -m unittest -v test_qb_market test_epa_shadow and python qb_market.py. The new PR-triggered workflow performs both without odds credentials and retains comparisons, predictions, fold boundaries, and provenance.

The historical comparison trains on prior seasons and compares identical games. It uses retrospectively recorded actual starter IDs as a development diagnostic only: it does not establish what starter information was known at a historical betting time. Missing IDs retain the original zero-history convention. No candidate is automatically promoted. Prospective coverage will remain zero until timestamped confirmations are supplied through the existing input.

The workflow will show whether these existing QB proxies add value before building a more detailed passing-efficiency model. No improvement is claimed until results are inspected.
