import json
import os
import tempfile
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import qb_expected as q
import expected_qb_shadow as shadow

class ExpectedTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,11,17,20,tzinfo=timezone.utc)
        self.game=dict(game_id='target',season='2026',week='12',game_type='REG',home_team='KC',away_team='BUF',kickoff_utc='2026-11-20T20:00:00Z')
        self.depth=pd.DataFrame([dict(dt='2026-11-17T16:00:00Z',team=t,gsis_id=i,player_name=n,pos_abb='QB',pos_rank=str(rank)) for t,i,n,rank in [('KC','00-0000001','Starter',1),('KC','00-0000002','Backup',2),('BUF','00-0000003','Away',1)]])
        self.inj=pd.DataFrame(columns=['team','season','week','season_type','gsis_id','report_status','practice_status'])
        self.reports=pd.DataFrame(columns=q.announcements.FIELDS)
        self.meta={k:dict(url='https://example.com/'+k,observed_at_utc='2026-11-17T19:00:00Z',last_modified_utc='2026-11-17T18:00:00Z') for k in ['depth','injuries','schedule','players']}
        self.schedule=[];self.players=[]
    def derive(self):return q.derive([self.game],self.schedule,self.depth,self.inj,self.players,self.reports,self.meta,self.now)
    def injury(self,status,practice='Full Participation in Practice',week='12',qid='00-0000001'):
        return dict(team='KC',season='2026',week=week,season_type='REG',gsis_id=qid,report_status=status,practice_status=practice)
    def test_depth_is_expected_never_confirmed(self):
        rows=self.derive();self.assertEqual([r['status'] for r in rows],['expected','expected'])
        self.assertEqual(rows[0]['qb_id'],'00-0000001');self.assertTrue(json.loads(rows[0]['evidence_json']))
        self.assertEqual(q.resolve(pd.DataFrame(rows),'target','KC',q.auto.stamp(self.now),self.game['kickoff_utc']),'00-0000001')
    def test_out_promotes_only_ranked_replacement(self):
        self.inj=pd.DataFrame([self.injury('Out')]);row=self.derive()[0]
        self.assertEqual(row['qb_id'],'00-0000002');self.assertEqual(row['reason'],'DEPTH_REPLACEMENT_FOR_OUT')
    def test_questionable_doubtful_and_practice_do_not_create_usable_forecast(self):
        for status,practice in [('Questionable','Full Participation in Practice'),('Doubtful',''),('','Did Not Participate In Practice'),('','Limited Participation in Practice')]:
            self.inj=pd.DataFrame([self.injury(status,practice)])
            rows=self.derive();self.assertEqual(rows[0]['status'],'uncertain')
            self.assertIsNone(q.resolve(pd.DataFrame(rows),'target','KC',q.auto.stamp(self.now),self.game['kickoff_utc']))
    def test_wrong_week_and_stale_injuries_cannot_remove_starter(self):
        self.inj=pd.DataFrame([self.injury('Out',week='11')]);self.assertEqual(self.derive()[0]['qb_id'],'00-0000001')
        self.inj=pd.DataFrame([self.injury('Out')]);self.meta['injuries']['last_modified_utc']='2026-11-01T00:00:00Z'
        self.assertEqual(self.derive()[0]['qb_id'],'00-0000001');self.assertEqual(self.derive()[0]['injury_status'],'unknown')
    def test_conflicting_injury_rows_are_uncertain(self):
        self.inj=pd.DataFrame([self.injury('Out'),self.injury('Questionable')]);r=self.derive()[0]
        self.assertEqual(r['qb_id'],'00-0000001');self.assertEqual(r['status'],'uncertain')
    def test_ambiguous_depth_fails(self):
        self.depth.loc[1,'pos_rank']='1';r=self.derive()[0]
        self.assertEqual(r['status'],'unknown');self.assertEqual(r['reason'],'AMBIGUOUS_DEPTH_RANK')
    def test_future_and_stale_depth_are_not_current(self):
        for dt in ['2026-11-18T00:00:00Z','2026-11-01T00:00:00Z']:
            self.depth['dt']=dt;self.assertEqual(self.derive()[0]['status'],'unknown')
    def test_recent_start_fallback_requires_current_team_and_prior_date(self):
        self.depth=self.depth.iloc[:0]
        self.players=[dict(gsis_id='00-0000001',display_name='Starter',position='QB',latest_team='KC',last_season='2026')]
        old=dict(game_id='old',season='2026',gameday='2026-11-10',home_team='KC',away_team='BUF',home_qb_id='00-0000001',away_qb_id='00-0000003',home_score='20',away_score='10')
        self.schedule=[old];r=self.derive()[0];self.assertEqual(r['reason'],'RECENT_START');self.assertEqual(r['published_at_utc'],'')
        self.schedule=[dict(old,gameday='2026-11-17')];self.assertEqual(self.derive()[0]['status'],'unknown')
        self.schedule=[dict(old,season='2025')];self.assertEqual(self.derive()[0]['status'],'unknown')
        self.schedule=[old];self.players[0]['latest_team']='BUF';self.assertEqual(self.derive()[0]['status'],'unknown')
    def test_explicit_confirmation_overrides_depth_and_keeps_label(self):
        self.reports=pd.DataFrame([dict(game_id='target',team='KC',qb_id='00-0000002',qb_name='Backup',status='confirmed',source_url='https://example.com/coach',published_at_utc='2026-11-17T17:00:00Z',observed_at_utc='2026-11-17T18:00:00Z')])
        row=self.derive()[0];self.assertEqual(row['qb_id'],'00-0000002');self.assertEqual(row['status'],'confirmed')
        self.reports.loc[0,'status']='unknown';self.reports.loc[0,'qb_id']='00-0000001';self.assertEqual(self.derive()[0]['status'],'uncertain')
    def test_resolver_rejects_stale_future_kickoff_and_duplicate(self):
        rows=pd.DataFrame(self.derive())
        self.assertIsNone(q.resolve(rows,'target','KC',q.auto.stamp(self.now+timedelta(hours=7)),self.game['kickoff_utc']))
        self.assertIsNone(q.resolve(rows,'target','KC','2026-11-17T19:00:00Z',self.game['kickoff_utc']))
        with self.assertRaisesRegex(ValueError,'kickoff changed'):q.resolve(rows,'target','KC',q.auto.stamp(self.now),'2026-11-21T20:00:00Z')
        with self.assertRaisesRegex(ValueError,'Ambiguous'):q.resolve(pd.concat([rows,rows]),'target','KC',q.auto.stamp(self.now),self.game['kickoff_utc'])
        rows.loc[0,'published_at_utc']='2026-11-18T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'Future'):q.resolve(rows,'target','KC',q.auto.stamp(self.now),self.game['kickoff_utc'])
    def test_unresolved_first_rank_does_not_silently_choose_backup(self):
        self.depth.loc[0,'gsis_id']=''
        row=self.derive()[0]
        self.assertEqual(row['status'],'unknown');self.assertEqual(row['reason'],'UNRESOLVED_DEPTH_ID')
    def test_new_out_report_blocks_older_confirmation(self):
        self.reports=pd.DataFrame([dict(game_id='target',team='KC',qb_id='00-0000001',qb_name='Starter',status='confirmed',source_url='https://example.com/coach',published_at_utc='2026-11-17T17:00:00Z',observed_at_utc='2026-11-17T17:30:00Z')])
        self.inj=pd.DataFrame([self.injury('Out')])
        row=self.derive()[0]
        self.assertEqual(row['status'],'uncertain');self.assertEqual(row['reason'],'CONFIRMATION_CONFLICTS_WITH_NEWER_INJURY_FEED')
    def test_future_source_observation_rejected(self):
        self.meta['depth']['observed_at_utc']='2026-11-18T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'Source observation'):self.derive()
    def test_shadow_no_backfill_and_no_stale_global_fallback(self):
        with patch.object(shadow.qb_passing,'record') as record,patch.dict(os.environ,{},clear=True):
            self.assertEqual(shadow.record('unused',[],Path('.'))['new_qb_forecasts'],0)
            self.assertEqual(shadow.record('unused',[{}],Path('.'))['status'],'EXPECTED_QB_COLLECTION_UNAVAILABLE')
            record.assert_not_called()
    def test_shadow_uses_distinct_archive_and_identity_fields(self):
        with patch.object(shadow.qb_passing,'record',return_value={}) as record,patch.dict(os.environ,{'QB_EXPECTATIONS_PATH':'asof.csv'}):
            shadow.record('schedule',[{}],Path('.'))
            self.assertEqual(record.call_args.kwargs['root'],shadow.ROOT)
            self.assertEqual(record.call_args.kwargs['id_fields'],['home_expected_qb_id','away_expected_qb_id'])
            self.assertIs(record.call_args.kwargs['resolver'],q.resolve)


class IntegrationTests(unittest.TestCase):
    def test_expected_profile_archives_separately_and_grades(self):
        import numpy as np
        import qb_passing as passing
        import grade_market as grader
        import epa_shadow
        now=datetime.now(timezone.utc)
        core={k:'' for k in q.auto.FORECAST_FIELDS}
        core.update(game_id='target',home_team='KC',away_team='BUF',quote_id='same',home_line=-3.,sportsbook='Book',
            captured_at_utc=q.auto.stamp(now-timedelta(minutes=3)),forecast_created_at_utc=q.auto.stamp(now-timedelta(minutes=2)),kickoff_utc=q.auto.stamp(now+timedelta(hours=12)),predicted_home_margin=3.)
        reports=[]
        for team,qid in [('KC','00-0000002'),('BUF','00-0000003')]:
            reports.append(dict(game_id='target',team=team,qb_id=qid,qb_name='QB',status='expected',reason='DEPTH_CHART',source_url='https://example.com/depth',published_at_utc=q.auto.stamp(now-timedelta(minutes=5)),observed_at_utc=q.auto.stamp(now-timedelta(minutes=4)),kickoff_utc=core['kickoff_utc'],injury_status='unknown',practice_status='unknown',evidence_json='[]'))
        cols=list(dict.fromkeys(passing.base.INPUTS+passing.ADJ))
        frame=pd.DataFrame(np.zeros((601,len(cols))),columns=cols)
        frame['game_id']=[str(i) for i in range(600)]+['target'];frame['season']=[2025]*600+[2026]
        frame['actual_margin']=[7.]*600+[np.nan];frame['gameday']=pd.to_datetime(['2025-01-01']*600+[now.date().isoformat()],utc=True)
        frame['home_team']='KC';frame['away_team']='BUF';frame['home_qb_id']='00-0000001';frame['away_qb_id']='00-0000003'
        stats=pd.DataFrame([dict(game_id=str(i),posteam=team,passer_player_id=qid,epa_sum=epa,success_sum=10.,dropbacks=20.) for i in range(600) for team,qid,epa in [('KC','00-0000002',-2.),('BUF','00-0000003',2.)]])
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);path=folder/'reports.csv';pd.DataFrame(reports).to_csv(path,index=False)
            schedule=folder/'schedule.csv';schedule.write_text('snapshot');(folder/'epa_features_asof.csv.gz').write_bytes(b'epa')
            with patch.object(shadow,'ROOT',folder/'expected'),patch.dict(os.environ,{'QB_EXPECTATIONS_PATH':str(path)}),patch.object(passing.confirmations,'load_frame',return_value=frame),patch.object(passing,'load_stats',return_value=(stats,[])),patch.object(passing.confirmations,'fit') as fit:
                fit.return_value.predict.return_value=np.array([1.])
                result=shadow.record(schedule,[core],folder)
                self.assertEqual(result['new_qb_forecasts'],1)
                row=pd.read_csv(folder/'expected/forecasts.csv').iloc[0]
                self.assertEqual(row.home_expected_qb_id,'00-0000002');self.assertNotIn('home_confirmed_qb_id',row.index)
                self.assertEqual(row.quote_id,'same');self.assertEqual(row.predicted_home_margin,3.25)
                self.assertAlmostEqual(row.qb_epa_diff,-32/260)
                self.assertTrue(fit.call_args.args[0].season.lt(2026).all())
                before=(folder/'expected/forecasts.csv').read_bytes()
                shadow.record(schedule,[],folder);self.assertEqual(before,(folder/'expected/forecasts.csv').read_bytes())
                with patch.object(epa_shadow,'ROOT',folder/'epa'):
                    report=shadow.grade_shadow(grader,[core],[],[],now)
                    self.assertEqual(report['paired_forecasts'],1);self.assertEqual(report['qb_passing']['graded_games'],0)
                    with self.assertRaisesRegex(ValueError,'archive mismatch|quote mismatch'):
                        shadow.grade_shadow(grader,[dict(core,quote_id='changed')],[],[],now)
                self.assertFalse((folder/'confirmed').exists())


if __name__=='__main__':unittest.main()
