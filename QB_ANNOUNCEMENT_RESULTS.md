# Official announcement adapter validation

Local live validation on November 17, 2026. All 32 official club RSS feeds were fetched. Current schedule has 13 upcoming regular-season games over eight days; the existing odds archive has 12. Identity filtering retained 111 current-season QB rows. The first completed article pass attempted 57 pages and parsed 54 supported official NewsArticle responses. No explicit announcement matched an upcoming game's week, opponent and current QB identity, so no automatic confirmations or QB forecasts were invented. Full scheduled-game confirmation coverage remains zero.

46 tests passed across the announcement adapter, news collector, confirmation gates, QB passing forecasts and EPA paired forecasts. Positive matching and negative/retraction cases use controlled fixtures. Live validation with zero accepted announcements does not establish real-world parser precision or comprehensive starter coverage.

Code now connects the adapter before automatic forecast creation and preserves its ledger independently from manual inputs. This describes the initial pre-activation validation; see the activation evidence below for current status. Previously archived predictions are unchanged.

Activation validation: resolved conflicts against newer main while preserving advanced shadow forecasts. Added persistence for paired QB grading reports. 61 integrated tests and six line-comparison checks pass locally. The expanded parser accepts the official team/week/season designation form only for a unique matching scheduled game; controlled tests reject old weeks, wrong seasons, conditions and ambiguous schedules. Same-quote EPA/QB grading is now explicit, alongside the existing core comparison. No current live confirmation or win-rate improvement is inferred from these fixture tests.

## Activation verified November 17, 2026

PR #4 and PR #8 merged, retaining main's advanced-stat tracking. The final PR head passed all three triggered Actions checks. Local validation passed 61 integrated tests and six line-comparison checks.

[Live collection run 44147903472](https://github.com/zachpierce2/nfl-prediction-model/actions/runs/44147903472) succeeded: 32 official feeds fetched, 57 article sources attempted, 54 supported articles parsed, zero matched decisions and zero confirmations across 13 upcoming scheduled games. Refreshed odds for all 13; no new forecasts because no game was within the existing 24-hour first-forecast window. Full evidence and forecast artifacts were uploaded, and the small ledger and coverage summary were committed.

[Live grading run 44147903459](https://github.com/zachpierce2/nfl-prediction-model/actions/runs/44147903459) succeeded and persisted the QB summary, paired EPA baseline/QB tables and grading artifact. Both paired forecast and graded-game counts are zero. ATS accuracy and MAE correctly remain null; no improvement or executed betting profit is established.

The primary forecasting model is unchanged. The QB model remains an isolated challenger. Automatic collection runs every three hours and grading daily. Announcements outside the explicit parser contract can still be missed; this remains partial coverage.
