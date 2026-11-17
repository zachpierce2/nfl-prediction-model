# Expected starting QB integration

The confirmation-only adapter previously produced zero accepted starters for 13 upcoming games. A live capture on November 17, 2026 now supplies 26 expected starters across those 13 games. All 26 come from current nflverse/ESPN depth charts. None is relabeled confirmed. Matching game-week injury rows are absent in this capture, so all 26 injury statuses remain unknown; this does not establish health or starter accuracy.

`qb_expected.py` collects independent source snapshots and records game/team identity, name, expected/confirmed/uncertain/unknown status, reason, source URL, publication time when supplied, actual observation time, kickoff, injury/practice categories and exact supporting rows. The current source URLs and SHA-256 hashes are in `data/auto_market/qb_expected/latest.json`; selected evidence is in `latest.csv`. Raw compressed feeds are retained as workflow artifacts for 90 days.

Selection rules:

- An eligible explicit starter report retains confirmed status. A newer current-week Out/Doubtful feed can flag a conflict rather than silently reuse it.
- A unique QB on a depth snapshot from the last 48 hours is expected. A current-week Out report can select the next unambiguous ranked QB. Missing IDs and tied ranks abstain.
- Without a current depth candidate, a completed earlier start within 35 days is a fallback, restricted to the current season and matching current QB team identity. Same-day and future games are excluded.
- Questionable/Doubtful, conflicting injury statuses, limited/no practice or a revoked report produce uncertainty. These cases skip the expected-QB forecast. Missing injury rows stay unknown, never healthy.
- Starter snapshots must have been observed before forecast calculation and kickoff, must be at most six hours old, and must match the exact game's kickoff. All source snapshots are prospective; no historical starter expectations are reconstructed.

`expected_qb_shadow.py` reuses the existing QB passing efficiency formula and prior-season fitting with separate expected-ID fields and archive. It accepts fresh expected or confirmed selections, preserves the exact new core quote, freezes input evidence, excludes same-day/future QB performance, and never backfills existing forecasts. Core/EPA and confirmed-only QB records remain separate. Daily grading compares the candidate with identical-game core and EPA forecasts and persists its own reports. No primary-model promotion or win-rate improvement is claimed.

Automatic collection runs before the existing three-hour market workflow. New candidate forecasts follow the existing first-forecast-within-24-hours policy. Daily grading and evidence uploads are connected in the workflows; missing collectors explicitly skip the candidate. The read-only PR validation uses fixtures and makes no live archive or odds requests.

Validation: 69 integrated fixture checks plus six existing grading checks pass locally. These include source/future/stale guards, team/week/season identity, depth rank ambiguity and missing IDs, replacement selection, injuries/practice, confirmation conflicts, fallback constraints, distinct archives, same-quote forecast creation and grading, no backfill, and prior-season training. Live coverage establishes collection, not real-world correctness or profitability. Injury feeds are partial and reserve lists are not comprehensive.
