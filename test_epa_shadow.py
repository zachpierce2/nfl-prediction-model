import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime,timezone,timedelta
import numpy as np
import pandas as pd
import epa_shadow as e
import grade_market as grader

class PairedTests(unittest.TestCase):
    def setUp(self):
        self.core={k:'' for k in e.auto.FORECAST_FIELDS}
        self.core.update(game_id='future',quote_id='same',sportsbook='Book',home_team='H',away_team='A',
            home_line=-3.,home_price_american=-110,away_price_american=-110,
            captured_at_utc='2099-01-01T10:00:00Z',forecast_created_at_utc='2099-01-01T10:01:00Z',
            kickoff_utc='2099-01-01T20:00:00Z',predicted_home_margin=4.)
        self.epa=dict(self.core,core_forecast_sha256=e.core_hash(self.core),forecast_created_at_utc='2099-01-01T10:02:00Z')
    def test_csv_hash_is_stable(self):
        loaded={k:str(v) for k,v in self.core.items()}
        self.assertEqual(e.core_hash(self.core),e.core_hash(loaded))
        self.assertEqual(len(e.verified_pairs([self.epa],[loaded])),1)
    def test_reject_changed_quote_or_core(self):
        with self.assertRaises(ValueError):e.verified_pairs([dict(self.epa,home_line=-4.)],[self.core])
        with self.assertRaises(ValueError):e.verified_pairs([self.epa],[dict(self.core,predicted_home_margin=5.)])
    def test_reject_postkickoff_epa(self):
        with self.assertRaises(ValueError):e.verified_pairs([dict(self.epa,forecast_created_at_utc=self.core['kickoff_utc'])],[self.core])
    def test_no_backfill_without_new_core(self):
        with tempfile.TemporaryDirectory() as d,patch.object(e,'ROOT',Path(d)),patch.object(e,'live_features') as features:
            r=e.record('unused',[],Path(d))
            self.assertEqual(r['new_epa_forecasts'],0);features.assert_not_called()
    def test_feature_builder_ignores_same_day(self):
        games=pd.DataFrame(dict(game_id=['x','y'],season=[2099,2099],gameday=pd.to_datetime(['2099-01-01','2099-01-02'],utc=True),home_team=['H','H'],away_team=['A','A']))
        team=pd.DataFrame(dict(game_id=['x','x'],season=[2099]*2,gameday=pd.to_datetime(['2099-01-01']*2,utc=True),posteam=['H','A'],defteam=['A','H'],plays=[60,60],epa=[1.,-1.],success=[1.,0.]))
        features=e.build_features(games,team)
        np.testing.assert_array_equal(features.loc[features.game_id.eq('x'),e.ADJ],0.)
        self.assertGreater(abs(features.loc[features.game_id.eq('y'),'adj_off_epa_diff'].iloc[0]),0)
    def test_future_outcomes_do_not_change_features(self):
        games=pd.DataFrame(dict(game_id=['x'],season=[2099],gameday=pd.to_datetime(['2099-01-01'],utc=True),home_team=['H'],away_team=['A']))
        future=pd.DataFrame(dict(game_id=['z'],season=[2099],gameday=pd.to_datetime(['2099-01-02'],utc=True),posteam=['H'],defteam=['A'],plays=[60],epa=[99.],success=[1.]))
        np.testing.assert_array_equal(e.build_features(games,future)[e.ADJ],0.)
    def test_paired_grading_separates_models(self):
        self.core.update(captured_at_utc='2026-01-01T10:00:00Z',forecast_created_at_utc='2026-01-01T10:01:00Z',kickoff_utc='2026-01-01T20:00:00Z')
        self.epa=dict(self.core,predicted_home_margin=2.,core_forecast_sha256=e.core_hash(self.core),forecast_created_at_utc='2026-01-01T10:02:00Z')
        final=dict(provider_event_id='event',home_team='H',away_team='A',kickoff_utc=self.core['kickoff_utc'],home_score=24,away_score=20,snapshot_id='final')
        with tempfile.TemporaryDirectory() as d,patch.object(e,'ROOT',Path(d)),patch.object(grader,'read_rows',return_value=[self.epa]):
            report,_=e.grade_shadow(grader,[self.core],[final],[],datetime(2026,1,2,tzinfo=timezone.utc))
            self.assertEqual(report['core']['hypothetical_wins'],1)
            self.assertEqual(report['epa']['hypothetical_losses'],1)
            self.assertEqual(report['core']['graded_games'],report['epa']['graded_games'])
    def test_paired_forecast_uses_identical_quote(self):
        now=datetime.now(timezone.utc)
        self.core.update(captured_at_utc=e.auto.stamp(now-timedelta(minutes=2)),
            forecast_created_at_utc=e.auto.stamp(now-timedelta(minutes=1)),
            kickoff_utc=e.auto.stamp(now+timedelta(hours=12)))
        n=600;rng=np.random.default_rng(1)
        f=pd.DataFrame(rng.normal(size=(n+1,len(e.model.INPUTS))),columns=e.model.INPUTS)
        f['game_id']=[str(i) for i in range(n)]+['future'];f['season']=[2025]*n+[2099]
        f['actual_margin']=rng.normal(size=n+1);f.loc[n,'actual_margin']=np.nan
        features=pd.DataFrame(rng.normal(size=(n+1,len(e.ADJ))),columns=e.ADJ);features['game_id']=f.game_id
        with tempfile.TemporaryDirectory() as d,patch.object(e,'ROOT',Path(d)),patch.object(e.model,'read_games',return_value=f),patch.object(e,'live_features',return_value=(features,'hash')):
            report=e.record('unused',[self.core],Path(d))
            self.assertEqual(report['new_epa_forecasts'],1)
            row=pd.read_csv(Path(d)/'forecasts.csv',keep_default_na=False).iloc[0].to_dict()
            self.assertEqual(row['quote_id'],self.core['quote_id'])
            self.assertEqual(row['home_line'],self.core['home_line'])
            self.assertEqual(row['captured_at_utc'],self.core['captured_at_utc'])
            self.assertLess(abs(row['adjustment']),2.0000001)

if __name__=='__main__':unittest.main()
