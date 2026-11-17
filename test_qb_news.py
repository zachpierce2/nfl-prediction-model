import unittest
from datetime import datetime, timezone
from qb_news import parse_feed, coverage

NOW = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)


class NewsTests(unittest.TestCase):
    def test_news_is_never_a_confirmation(self):
        raw = b'<rss><channel><item><title>QB Example will start</title><link>https://www.chiefs.com/news/example</link><pubDate>Tue, 17 Nov 2026 10:00:00 GMT</pubDate></item></channel></rss>'
        rows = parse_feed(raw, 'KC', NOW)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'UNREVIEWED_NEWS_NOT_A_STARTER_CONFIRMATION')
        self.assertTrue(rows[0]['qb_headline'])
        self.assertEqual(parse_feed(raw.replace(b'chiefs.com', b'other.com'), 'KC', NOW), [])
        self.assertEqual(parse_feed(raw.replace(b'10:00:00', b'13:00:00'), 'KC', NOW), [])
        self.assertEqual(parse_feed(raw.replace(b'17 Nov', b'01 Nov'), 'KC', NOW), [])

    def test_full_quote_coverage_counts_missing_and_deduplicates(self):
        q = dict(game_id='g',home_team='KC',away_team='BUF',kickoff_utc='2026-11-18T12:00:00Z',captured_at_utc='2026-11-17T10:00:00Z')
        r = coverage([q, q], [], NOW)
        self.assertEqual(r['quoted_upcoming_games'], 1)
        self.assertEqual(r['both_confirmed_games'], 0)
        self.assertEqual(r['games'][0]['home']['status'], 'MISSING')

    def test_future_observation_does_not_count(self):
        q = dict(game_id='g',home_team='KC',away_team='BUF',kickoff_utc='2026-11-18T12:00:00Z',captured_at_utc='2026-11-17T10:00:00Z')
        reports = [dict(game_id='g',team=t,qb_id='id',status='confirmed',source_url='https://example.com',published_at_utc='2026-11-17T10:00:00Z',observed_at_utc='2026-11-17T13:00:00Z') for t in ['KC','BUF']]
        self.assertEqual(coverage([q], reports, NOW)['both_confirmed_games'], 0)
        for r in reports:
            r['observed_at_utc']='2026-11-17T11:00:00Z'
        self.assertEqual(coverage([q], reports, NOW)['both_confirmed_games'], 1)


if __name__ == '__main__':
    unittest.main()
