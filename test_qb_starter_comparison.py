import json
import os
import tempfile
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import qb_starter_comparison as m
import qb_passing as q
import expected_qb_shadow as expected
import grade_market as grader
import epa_shadow

class ComparisonTests(unittest.TestCase):
    def fixture(self):
        core={k:'' for k in m.auto.FORECAST_FIELDS}
        core.update(game_id='g',quote_id='same',sportsbook='Book',home_team='H',away_team='A',home_line=-3.,home_price_american=-110,away_line=3.,away_price_american=-110,captured_at_utc='2026-01-01T10:00:00Z',forecast_created_at_utc='2026-01-01T10:01:00Z',kickoff_utc='2026-01-01T20:00:00Z',predicted_home_margin=3.,snapshot_id='shared')
        e=dict(core,model_version=epa_shadow.VERSION,core_forecast_sha256=epa_shadow.core_hash(core),predicted_home_margin=3.1,forecast_created_at_utc='2026-01-01T10:02:00Z')
        x=dict(core,model_version=expected.VERSION,core_forecast_sha256=epa_shadow.core_hash(core),qb_inputs_sha256='inputs',home_expected_qb_id='backup',away_expected_qb_id='away',predicted_home_margin=4.,forecast_created_at_utc='2026-01-01T10:03:00Z')
        c=dict(core,model_version=m.VERSION,core_forecast_sha256=epa_shadow.core_hash(core),qb_inputs_sha256='inputs',home_last_qb_id='starter',away_last_qb_id='away',predicted_home_margin=2.,forecast_created_at_utc=x['forecast_created_at_utc'],expected_forecast_sha256=m.row_hash(x))
        final=dict(provider_event_id='event',home_team='H',away_team='A',kickoff_utc=core['kickoff_utc'],home_score=24,away_score=20,snapshot_id='final')
        return core,e,x,c,final
    def save(self,root,core,e,x,c):
        for path,row in [(root/'epa/forecasts.csv',e),(root/'expected/forecasts.csv',x),(root/'comparison/forecasts.csv',c)]:
            path.parent.mkdir(parents=True,exist_ok=True);m.write_csv(path,list(row),[row])
    def test_threeway_same_games_changed_starter_and_disagreements(self):
        core,e,x,c,final=self.fixture()
        with tempfile.TemporaryDirectory() as d,patch.object(m,'ROOT',Path(d)/'comparison'),patch.object(expected,'ROOT',Path(d)/'expected'),patch.object(epa_shadow,'ROOT',Path(d)/'epa'):
            self.save(Path(d),core,e,x,c)
            report,text=m.grade_shadow(grader,[core],[final],[],datetime(2026,1,2,tzinfo=timezone.utc))
            self.assertEqual(report['complete_triples'],1);self.assertEqual(report['starter_changed_forecasts'],1)
            self.assertEqual(report['same_game_results']['expected']['hypothetical_wins'],1)
            self.assertEqual(report['same_game_results']['last_observed']['hypothetical_losses'],1)
            self.assertEqual(report['changed_starter_results']['epa']['graded_games'],1)
            self.assertEqual(report['disagreements'][0]['first_wins_other_loses'],1)
            self.assertEqual(report['disagreements'][0]['mean_absolute_error_improvement'],2.)
            self.assertIn('expected',text)
            changes=pd.read_csv(m.ROOT/'changed_picks.csv');self.assertTrue(changes.loc[0,'starter_changed'])
    def test_common_cohort_excludes_unpaired_games_and_pending_is_not_loss(self):
        core,e,x,c,final=self.fixture()
        with tempfile.TemporaryDirectory() as d,patch.object(m,'ROOT',Path(d)/'comparison'),patch.object(expected,'ROOT',Path(d)/'expected'),patch.object(epa_shadow,'ROOT',Path(d)/'epa'):
            self.save(Path(d),core,e,x,c)
            report,_=m.grade_shadow(grader,[core],[],[],datetime(2026,1,1,12,tzinfo=timezone.utc))
            for r in report['same_game_results'].values():
                self.assertEqual(r['pending_games'],1);self.assertIsNone(r['hypothetical_win_pct']);self.assertIsNone(r['model_mae'])
            (m.ROOT/'forecasts.csv').unlink()
            report,_=m.grade_shadow(grader,[core],[final],[],datetime(2026,1,2,tzinfo=timezone.utc))
            self.assertEqual(report['expected_missing_control'],1);self.assertEqual(report['complete_triples'],0)
            for r in report['same_game_results'].values():self.assertEqual(r['graded_games'],0)
    def test_missing_epa_is_explicit_and_duplicate_archives_fail(self):
        core,e,x,c,final=self.fixture()
        with tempfile.TemporaryDirectory() as d,patch.object(m,'ROOT',Path(d)/'comparison'),patch.object(expected,'ROOT',Path(d)/'expected'),patch.object(epa_shadow,'ROOT',Path(d)/'epa'):
            self.save(Path(d),core,e,x,c);(epa_shadow.ROOT/'forecasts.csv').unlink()
            report,_=m.grade_shadow(grader,[core],[],[],datetime(2026,1,2,tzinfo=timezone.utc))
            self.assertEqual(report['controls_missing_epa'],1);self.assertEqual(report['complete_triples'],0)
            m.write_csv(m.ROOT/'forecasts.csv',list(c),[c,c])
            with self.assertRaisesRegex(ValueError,'Duplicate'):m.grade_shadow(grader,[core],[],[],datetime(2026,1,2,tzinfo=timezone.utc))
    def test_quote_identity_input_hash_and_timing_tampering_fail(self):
        core,e,x,c,final=self.fixture();m.validate(core,x,c,e)
        for field,value in [('home_price_american',-120),('qb_inputs_sha256','other'),('expected_forecast_sha256','changed'),('forecast_created_at_utc',core['kickoff_utc']),('model_version','wrong')]:
            with self.assertRaises(ValueError):m.validate(core,x,dict(c,**{field:value}),e)
        with self.assertRaises(ValueError):m.validate(core,dict(x,home_expected_qb_id='changed'),c,e)
    def test_last_observed_ids_exclude_current_day_and_future(self):
        games=pd.DataFrame(dict(game_id=['old','target','future'],gameday=pd.to_datetime(['2026-01-01','2026-01-08','2026-01-15'],utc=True),home_team=['H']*3,away_team=['A']*3,actual_margin=[7.,np.nan,np.nan]))
        stats=pd.DataFrame([dict(game_id='old',posteam=t,passer_player_id=qid,epa_sum=v,success_sum=5.,dropbacks=n) for t,qid,v,n in [('H','starter',5.,20),('H','backup',-5.,10),('A','away',1.,20)]])
        a=q.features(games,stats,include_ids=True)
        newer=stats.assign(game_id='target',passer_player_id='future',dropbacks=999)
        b=q.features(games,pd.concat([stats,newer]),include_ids=True)
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(a.loc[1,'home_last_qb_id'],'starter');self.assertIsNone(a.loc[0,'home_last_qb_id'])

class RecordingTests(unittest.TestCase):
    def test_same_fit_same_inputs_only_live_starter_changes_and_no_backfill(self):
        now=datetime.now(timezone.utc)
        core={k:'' for k in m.auto.FORECAST_FIELDS}
        core.update(game_id='target',home_team='H',away_team='A',quote_id='same',home_line=-3.,captured_at_utc=m.auto.stamp(now-timedelta(minutes=3)),forecast_created_at_utc=m.auto.stamp(now-timedelta(minutes=2)),kickoff_utc=m.auto.stamp(now+timedelta(hours=12)))
        cols=list(dict.fromkeys(q.base.INPUTS+q.ADJ))
        frame=pd.DataFrame(np.zeros((601,len(cols))),columns=cols);frame['game_id']=[str(i) for i in range(600)]+['target'];frame['season']=[2025]*600+[2026]
        frame['actual_margin']=[7.]*600+[np.nan];frame['gameday']=pd.to_datetime(['2025-01-01']*600+[now.date().isoformat()],utc=True)
        frame['home_team']='H';frame['away_team']='A';frame['home_qb_id']='starter';frame['away_qb_id']='away'
        stats=pd.DataFrame([dict(game_id=str(i),posteam=t,passer_player_id=qid,epa_sum=v,success_sum=5.,dropbacks=n) for i in range(600) for t,qid,v,n in [('H','starter',5.,20),('H','backup',-5.,10),('A','away',1.,20)]])
        reports=pd.DataFrame([dict(game_id='target',team=t,qb_id=qid,status='confirmed',source_url='https://example.com/qb',published_at_utc=m.auto.stamp(now-timedelta(minutes=6)),observed_at_utc=m.auto.stamp(now-timedelta(minutes=5))) for t,qid in [('H','backup'),('A','away')]])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);schedule=root/'schedule.csv';schedule.write_text('snapshot');(root/'epa_features_asof.csv.gz').write_bytes(b'epa');path=root/'reports.csv';reports.to_csv(path,index=False)
            with patch.object(q.confirmations,'load_frame',return_value=frame),patch.object(q,'load_stats',return_value=(stats,[])),patch.object(q.confirmations,'fit') as fit:
                fitted=fit.return_value;calls=[]
                def predict(test):calls.append(test.copy());return test.qb_epa_diff.to_numpy()
                fitted.predict.side_effect=predict
                result=q.record(schedule,[core],root,reports_path=path,root=root/'expected',version=expected.VERSION,id_fields=['home_expected_qb_id','away_expected_qb_id'],tag='expected_qb',control_root=root/'comparison')
                self.assertEqual(result['control_comparison']['new_last_observed_forecasts'],1)
                self.assertEqual(fit.call_count,1);self.assertEqual(len(calls),2)
                self.assertTrue(fit.call_args.args[0].season.lt(2026).all())
                pd.testing.assert_frame_equal(calls[0].drop(columns=q.QB),calls[1].drop(columns=q.QB))
                self.assertNotEqual(calls[0].qb_epa_diff.iloc[0],calls[1].qb_epa_diff.iloc[0])
                x=grader.read_rows(root/'expected/forecasts.csv')[0];c=grader.read_rows(root/'comparison/forecasts.csv')[0]
                self.assertEqual(c['home_last_qb_id'],'starter');self.assertEqual(x['home_expected_qb_id'],'backup')
                self.assertEqual(c['qb_inputs_sha256'],x['qb_inputs_sha256']);self.assertEqual(c['forecast_created_at_utc'],x['forecast_created_at_utc']);m.validate(core,x,c)
                before=(root/'comparison/forecasts.csv').read_bytes()
                q.record(schedule,[],root,root=root/'expected',control_root=root/'comparison')
                self.assertEqual(before,(root/'comparison/forecasts.csv').read_bytes())
                # A preexisting expected forecast cannot acquire a new control later.
                (root/'comparison/forecasts.csv').unlink()
                result=q.record(schedule,[core],root,reports_path=path,root=root/'expected',version=expected.VERSION,id_fields=['home_expected_qb_id','away_expected_qb_id'],tag='expected_qb',control_root=root/'comparison')
                self.assertEqual(result['control_comparison']['new_last_observed_forecasts'],0)
                self.assertFalse((root/'comparison/forecasts.csv').exists())

if __name__=='__main__':unittest.main()
