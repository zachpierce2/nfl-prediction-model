# Official announcement adapter validation

Local live validation on November 17, 2026. All 32 official club RSS feeds were fetched. Current schedule has 13 upcoming regular-season games over eight days; the existing odds archive has 12. Identity filtering retained 111 current-season QB rows. The first completed article pass attempted 57 pages and parsed 54 supported official NewsArticle responses. No explicit announcement matched an upcoming game's week, opponent and current QB identity, so no automatic confirmations or QB forecasts were invented. Full scheduled-game confirmation coverage remains zero.

46 tests passed across the announcement adapter, news collector, confirmation gates, QB passing forecasts and EPA paired forecasts. Positive matching and negative/retraction cases use controlled fixtures. Live validation with zero accepted announcements does not establish real-world parser precision or comprehensive starter coverage.

Code now connects the adapter before automatic forecast creation and preserves its ledger independently from manual inputs. The branch remains unmerged, so scheduled integration is not active. Previously archived predictions are unchanged.

Activation validation: resolved conflicts against newer main while preserving advanced shadow forecasts. Added persistence for paired QB grading reports. 61 integrated tests and six line-comparison checks pass locally. The expanded parser accepts the official team/week/season designation form only for a unique matching scheduled game; controlled tests reject old weeks, wrong seasons, conditions and ambiguous schedules. Same-quote EPA/QB grading is now explicit, alongside the existing core comparison. No current live confirmation or win-rate improvement is inferred from these fixture tests.
