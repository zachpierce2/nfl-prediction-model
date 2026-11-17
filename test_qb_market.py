import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import qb_market as q

class QBTests(unittest.TestCase):
    def reports(self):
        return pd.DataFrame([dict(game_id='g',team='H',qb_id='qb1',status='confirmed',
            source_url='https://example.com/report',published_at_utc='2026-11-15T10:00:00Z',
            observed_at_utc='2026-11-15T11:00:00Z')])
    def select(self,rows):
        return q.confirmed_ids(rows,'g','H','2026-11-15T12:00:00Z','2026-11-15T20:00:00Z')
    def test_confirmed_before_cutoff(self):
        self.assertEqual(self.select(self.reports()),'qb1')
    def test_late_reports_excluded(self):
        r=self.reports();r['observed_at_utc']='2026-11-15T13:00:00Z'
        self.assertIsNone(self.select(r))
    def test_missing_or_unconfirmed_not_guessed(self):
        self.assertIsNone(self.select(self.reports().iloc[:0]))
        r=self.reports();r['status']='questionable'
        self.assertIsNone(self.select(r))
    def test_later_uncertainty_revokes_confirmation(self):
        r=self.reports();late=r.copy()
        late['observed_at_utc']='2026-11-15T11:30:00Z';late['status']='questionable'
        self.assertIsNone(self.select(pd.concat([r,late])))
    def test_conflicting_reports_rejected(self):
        r=self.reports();other=r.copy();other['qb_id']='qb2'
        with self.assertRaises(ValueError):self.select(pd.concat([r,other]))
    def test_publication_and_timezone_validated(self):
        r=self.reports();r['published_at_utc']='2026-11-15T11:30:00Z'
        with self.assertRaises(ValueError):self.select(r)
        r=self.reports();r['observed_at_utc']='2026-11-15T11:00:00'
        with self.assertRaises(ValueError):self.select(r)
    def test_prior_day_only_and_new_starter(self):
        frame=pd.DataFrame(dict(game_id=['old','same','future'],gameday=pd.to_datetime(
            ['2026-11-01','2026-11-15','2026-11-16'],utc=True),
            actual_margin=[7.,99.,-999.],home_qb_id=['qb1']*3,away_qb_id=['qb2']*3))
        values=q.features_for_ids(frame,pd.Timestamp('2026-11-15',tz='UTC'),'qb1','qb2')
        self.assertEqual(values,dict(qb_margin_diff=14.,qb_experience_diff=0.))
        self.assertEqual(q.features_for_ids(frame,pd.Timestamp('2026-11-15',tz='UTC'),'rookie','qb2')['qb_margin_diff'],7.)
    def test_live_candidate_uses_confirmed_ids_and_same_quote(self):
        from datetime import datetime,timezone,timedelta
        now=datetime.now(timezone.utc)
        core={k:'' for k in q.auto.FORECAST_FIELDS}
        core.update(game_id='g',home_team='H',away_team='A',quote_id='same',
            home_line=-3.,captured_at_utc=q.auto.stamp(now-timedelta(minutes=3)),
            forecast_created_at_utc=q.auto.stamp(now-timedelta(minutes=2)),
            kickoff_utc=q.auto.stamp(now+timedelta(hours=12)))
        reports=self.reports()
        reports['published_at_utc']=q.auto.stamp(now-timedelta(minutes=5))
        reports['observed_at_utc']=q.auto.stamp(now-timedelta(minutes=4))
        away=reports.copy();away['team']='A';away['qb_id']='qb2'
        reports=pd.concat([reports,away])
        cols=list(dict.fromkeys(q.base.INPUTS+q.ADJ+q.QB))
        frame=pd.DataFrame(np.zeros((601,len(cols))),columns=cols)
        frame['game_id']=[str(i) for i in range(600)]+['g']
        frame['season']=[2025]*600+[2026]
        frame['actual_margin']=[7.]*600+[np.nan]
        frame['gameday']=pd.to_datetime(['2025-01-01']*600+[now.date().isoformat()],utc=True)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'confirmations.csv';reports.to_csv(path,index=False)
            schedule=Path(d)/'schedule.csv';schedule.write_text('snapshot')
            (Path(d)/'epa_features_asof.csv.gz').write_bytes(b'epa')
            with patch.object(q,'ROOT',Path(d)/'archive'),patch.object(q,'CONFIRMATIONS',path),patch.object(q,'load_frame',return_value=frame),patch.object(q,'features_for_ids',return_value=dict(qb_margin_diff=1.,qb_experience_diff=2.)) as features,patch.object(q,'fit') as fit:
                fit.return_value.predict.return_value=np.array([1.])
                result=q.record(schedule,[core],Path(d))
                self.assertEqual(result['new_qb_forecasts'],1)
                self.assertEqual(features.call_args.args[2:],('qb1','qb2'))
                saved=pd.read_csv(Path(d)/'archive/forecasts.csv').iloc[0]
                self.assertEqual(saved.quote_id,'same')
                self.assertEqual(saved.home_line,-3.)
                self.assertEqual(saved.home_confirmed_qb_id,'qb1')
                self.assertEqual(saved.predicted_home_margin,3.25)
                self.assertTrue(fit.call_args.args[0].season.lt(2026).all())

    def test_no_backfill(self):
        with tempfile.TemporaryDirectory() as d,patch.object(q,'ROOT',Path(d)),patch.object(q,'load_frame') as load:
            self.assertEqual(q.record('unused',[],Path(d))['new_qb_forecasts'],0)
            load.assert_not_called()
    def test_empty_confirmations_do_not_use_schedule_starter(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'confirmations.csv';self.reports().iloc[:0].to_csv(path,index=False)
            with patch.object(q,'ROOT',Path(d)/'archive'),patch.object(q,'CONFIRMATIONS',path),patch.object(q,'load_frame') as load:
                result=q.record('unused',[dict(game_id='g',home_team='H',away_team='A',
                    kickoff_utc='2099-01-01T20:00:00Z')],Path(d))
                self.assertEqual(result['status'],'MISSING_CONFIRMED_STARTERS')
                self.assertEqual(result['new_qb_forecasts'],0);load.assert_not_called()

if __name__=='__main__':unittest.main()
