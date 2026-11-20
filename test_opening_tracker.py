import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import auto_market as auto
import grade_market as grader
import opening_tracker as tracker


class OpeningTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        for target, value in [('ROOT', self.folder/'open'), ('model_version', lambda: 'v1')]:
            swap = patch.object(tracker, target, value)
            swap.start()
            self.addCleanup(swap.stop)
        self.time = datetime.now(timezone.utc)
        self.kickoff = self.time + timedelta(days=10)
        self.stamp = auto.stamp(self.time)
        self.schedule = pd.DataFrame([dict(game_id='g', home_team='HOU', away_team='IND',
            gameday=pd.Timestamp(self.kickoff).tz_convert('America/New_York').date().isoformat())])
        self.market = dict(key='spreads', last_update=self.stamp, outcomes=[
            dict(name='Houston Texans', point=-3, price=-110),dict(name='Indianapolis Colts',point=3,price=-110)])
        self.book = dict(key='draftkings', title='DraftKings', markets=[self.market])
        self.event = dict(id='e', home_team='Houston Texans', away_team='Indianapolis Colts',
                          commence_time=auto.stamp(self.kickoff), bookmakers=[self.book])

    def observe(self, events):
        return tracker.capture(events, self.schedule, None, self.time, self.folder)

    def test_first_sighting_is_never_replaced(self):
        self.observe([self.event])
        self.event['id'] = 'second'
        self.observe([self.event])
        self.book['markets'].append(dict(key='totals', last_update=self.stamp,
            outcomes=[dict(name='Over', point=45), dict(name='Under', point=45)]))
        self.assertEqual(self.observe([self.event])['first_seen'], 0)
        statuses = [r['status'] for r in tracker.read(tracker.ROOT/'first_seen.csv')]
        self.assertEqual(statuses, ['already_listed_at_start','missing_total'])

    def test_distinct_frozen_forecasts_and_line_move(self):
        self.observe([])
        self.book['markets'].append(dict(key='totals', last_update=self.stamp,
            outcomes=[dict(name='Over',point=45),dict(name='Under',point=45)]))
        games = pd.DataFrame([dict(game_id='g', actual_margin=float('nan'))])
        features = pd.DataFrame([dict(game_id='g', **dict.fromkeys(tracker.epa.ADJ, 0))])
        features.to_csv(self.folder/'epa_features_asof.csv.gz', index=False)
        (self.folder/'epa_provenance.json').write_text('{}')
        with patch.object(tracker.model,'read_games',return_value=games), patch.object(
                tracker.total_model, 'predict_margin', return_value=4):
            self.assertEqual(self.observe([self.event])['saved'], 1)
        early = tracker.read(tracker.ROOT/'forecasts.csv')[0]
        late = dict(early, snapshot_id='later', home_line=-5, predicted_home_margin=6,
                    captured_at_utc=auto.stamp(self.kickoff-timedelta(hours=18)),
                    forecast_created_at_utc=auto.stamp(self.kickoff-timedelta(hours=17)))
        auto.archive_rows(tracker.ROOT/'regular.csv', list(late), [late], 'game_id')
        final = dict(early, home_score=21, away_score=17)
        report = tracker.report(grader,[final],self.kickoff+timedelta(days=1))
        self.assertEqual(report['opening']['WIN'], 1)
        self.assertEqual(report['regular']['LOSS'], 1)
        self.assertEqual(report['categories'], {'same_team_line_changed_grade': 1})
