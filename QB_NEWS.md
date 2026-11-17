# QB starter evidence collection

This addition collects official club RSS responses and records publication and observation times, source URLs, exact response hashes, feed failures, and recent news. All 32 club URLs are attempted; availability is measured rather than assumed. Only dated items linking to the corresponding club domain are retained. Headlines are searchable evidence, never confirmed-starter inputs. Raw responses survive parse failures.

The coverage report uses the existing candidate's confirmation eligibility rules and includes every upcoming quoted game in the next eight days, including missing confirmations. This is odds-archive coverage, not a claim to cover the full NFL schedule. Each game's quote timestamp is included so stale quotes are visible. Source errors do not become empty successful feeds. Reports count invalid confirmations separately.

Run `python qb_news.py` and `python -m unittest -v test_qb_news test_qb_market test_qb_passing`. Outputs are under `data/qb_news/`; the independent manual/scheduled workflow saves them as 90-day Actions artifacts. Scheduled collection activates only when the workflow reaches the default branch. Artifacts must be retained externally for longer studies.

## Remaining blocker

No validated automatic confirmation feed is connected. The existing `nfl_data_inputs/qb_starters.csv` remains unchanged. Depth charts, injury availability, news keywords, and in-game starters do not establish a timestamped pregame confirmation. This collector cannot claim to have solved that missing input, and it creates zero confirmations. An authoritative structured feed or a validated source-specific announcement parser is still required, with player-ID mapping, game matching, corrections/revocations and measured coverage.

The branch builds on the existing QB passing experiment. Its baseline and QB forecasts/grading remain unchanged. It neither merges that experiment nor promotes a model. No historical test is rerun because prediction behavior does not change.
