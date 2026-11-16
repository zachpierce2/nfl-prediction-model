import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import qb_passing as q

class PassingTests(unittest.TestCase):
    def games(self):
        return pd.DataFrame(dict(game_id=['old','target','future'],
            gameday=pd.to_datetime(['2026-01-01','2026-01-08','2026-01-15'],utc=True),
            home_team=['H']*3,away_team=['A']*3,actual_margin=[7.,np.nan,np.nan]))
    def stats(self):
        return pd.DataFrame([dict(game_id='old',posteam='H',passer_player_id='starter',
                epa_sum=10.,success_sum=12.,dropbacks=20.),
            dict(game_id='old',posteam='H',passer_player_id='backup',
                epa_sum=-2.,success_sum=4.,dropbacks=10.),
            dict(game_id='old',posteam='A',passer_player_id='away',
                epa_sum=2.,success_sum=5.,dropbacks=10.)])
    def test_matches_v3_shrinkage(self):
        h={'x':[dict(epa_sum=10.,success_sum=12.,dropbacks=20.)]}
        r=q.profile(h,'x')
        self.assertAlmostEqual(r['epa'],10/120)
        self.assertAlmostEqual(r['success'],62/120)
        self.assertEqual(r['dropbacks'],20)
    def test_eight_games_and_dropback_cap(self):
        h={'x':[dict(epa_sum=10000.,success_sum=1.,dropbacks=1.)]+
           [dict(epa_sum=10.,success_sum=20.,dropbacks=60.)]*8}
        r=q.profile(h,'x')
        self.assertAlmostEqual(r['epa'],80/580)
        self.assertEqual(r['dropbacks'],400)
    def test_unknown_starter_uses_prior_and_missing_flag(self):
        r=q.profile({},'rookie')
        self.assertEqual(r,dict(epa=0.,success=.5,dropbacks=0,missing=1.))
    def test_no_same_day_or_future_pbp(self):
        games=self.games();stats=self.stats()
        before=q.features(games,stats)
        late=stats.copy();late['game_id']='future';late['epa_sum']=999999.
        after=q.features(games,pd.concat([stats,late]))
        pd.testing.assert_frame_equal(before,after)
        self.assertEqual(before.iloc[0].qb_epa_diff,0.)
        self.assertAlmostEqual(before.iloc[1].qb_epa_diff,10/120-2/110)
    def test_confirmed_replacement_uses_own_history(self):
        f=q.features(self.games(),self.stats(),{'target':('backup','away')})
        r=f[f.game_id.eq('target')].iloc[0]
        self.assertAlmostEqual(r.qb_epa_diff,-2/110-2/110)
        self.assertEqual(r.home_qb_missing,0.)
    def test_missing_prior_game_feed_blocks_prediction(self):
        with self.assertRaises(ValueError):
            q.features(self.games(),self.stats().iloc[:0])
    def test_no_backfill_or_pbp_request_without_new_forecast(self):
        with tempfile.TemporaryDirectory() as d,patch.object(q,'ROOT',Path(d)),patch.object(q,'load_stats') as fetch:
            self.assertEqual(q.record('unused',[],Path(d))['new_qb_forecasts'],0)
            fetch.assert_not_called()
    def test_empty_confirmations_do_not_download_pbp(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'qb.csv'
            p.write_text('game_id,team,qb_id,qb_name,status,source_url,published_at_utc,observed_at_utc\n')
            with patch.object(q,'ROOT',Path(d)/'archive'),patch.object(q.confirmations,'CONFIRMATIONS',p),patch.object(q,'load_stats') as fetch:
                r=q.record('unused',[dict(game_id='g',home_team='H',away_team='A',kickoff_utc='2099-01-01T00:00:00Z')],Path(d))
                self.assertEqual(r['status'],'MISSING_CONFIRMED_STARTERS')
                fetch.assert_not_called()
    def test_live_confirmed_profile_is_archived_on_identical_quote(self):
        from datetime import datetime,timezone,timedelta
        now=datetime.now(timezone.utc)
        core={k:'' for k in q.auto.FORECAST_FIELDS}
        core.update(game_id='target',home_team='H',away_team='A',quote_id='same',home_line=-3.,
            captured_at_utc=q.auto.stamp(now-timedelta(minutes=3)),
            forecast_created_at_utc=q.auto.stamp(now-timedelta(minutes=2)),
            kickoff_utc=q.auto.stamp(now+timedelta(hours=12)))
        reports=pd.DataFrame([dict(game_id='target',team=team,qb_id=qid,status='confirmed',
            source_url='https://example.com/report',
            published_at_utc=q.auto.stamp(now-timedelta(minutes=5)),
            observed_at_utc=q.auto.stamp(now-timedelta(minutes=4)))
            for team,qid in [('H','backup'),('A','away')]])
        cols=list(dict.fromkeys(q.base.INPUTS+q.ADJ))
        frame=pd.DataFrame(np.zeros((601,len(cols))),columns=cols)
        frame['game_id']=[str(i) for i in range(600)]+['target']
        frame['season']=[2025]*600+[2026]
        frame['actual_margin']=[7.]*600+[np.nan]
        frame['gameday']=pd.to_datetime(['2025-01-01']*600+[now.date().isoformat()],utc=True)
        frame['home_team']='H';frame['away_team']='A'
        frame['home_qb_id']='starter';frame['away_qb_id']='away'
        stats=pd.DataFrame([dict(game_id=str(i),posteam=team,passer_player_id=qid,
            epa_sum=epa,success_sum=10.,dropbacks=20.)
            for i in range(600) for team,qid,epa in [('H','backup',-2.),('A','away',2.)]])
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);path=folder/'reports.csv';reports.to_csv(path,index=False)
            schedule=folder/'schedule.csv';schedule.write_text('snapshot')
            (folder/'epa_features_asof.csv.gz').write_bytes(b'epa')
            with patch.object(q,'ROOT',folder/'archive'),patch.object(q.confirmations,'CONFIRMATIONS',path),patch.object(q.confirmations,'load_frame',return_value=frame),patch.object(q,'load_stats',return_value=(stats,[])),patch.object(q.confirmations,'fit') as fit:
                fit.return_value.predict.return_value=np.array([1.])
                result=q.record(schedule,[core],folder)
                self.assertEqual(result['new_qb_forecasts'],1)
                row=pd.read_csv(folder/'archive/forecasts.csv').iloc[0]
                self.assertEqual(row.home_confirmed_qb_id,'backup')
                self.assertEqual(row.quote_id,'same')
                self.assertEqual(row.predicted_home_margin,3.25)
                self.assertAlmostEqual(row.qb_epa_diff,(-16-16)/260)
                self.assertTrue(fit.call_args.args[0].season.lt(2026).all())
    def test_paired_grading_rejects_altered_core(self):
        from datetime import datetime,timezone
        import grade_market as grader
        from epa_shadow import core_hash
        core={k:'' for k in q.auto.FORECAST_FIELDS}
        core.update(game_id='g',quote_id='same',sportsbook='Book',home_team='H',away_team='A',
            home_line=-3.,predicted_home_margin=4.,captured_at_utc='2026-01-01T10:00:00Z',
            forecast_created_at_utc='2026-01-01T10:01:00Z',kickoff_utc='2026-01-01T20:00:00Z')
        candidate=dict(core,core_forecast_sha256=core_hash(core),
            predicted_home_margin=2.,forecast_created_at_utc='2026-01-01T10:02:00Z')
        final=dict(provider_event_id='event',home_team='H',away_team='A',
            kickoff_utc=core['kickoff_utc'],home_score=24,away_score=20,snapshot_id='final')
        with tempfile.TemporaryDirectory() as d,patch.object(q,'ROOT',Path(d)),patch.object(grader,'read_rows',return_value=[candidate]):
            report=q.grade_shadow(grader,[core],[final],[],datetime(2026,1,2,tzinfo=timezone.utc))
            self.assertEqual(report['core']['hypothetical_wins'],1)
            self.assertEqual(report['qb_passing']['hypothetical_losses'],1)
            with self.assertRaises(ValueError):
                q.grade_shadow(grader,[dict(core,predicted_home_margin=5.)],[final],[],datetime(2026,1,2,tzinfo=timezone.utc))

    def test_source_aggregate_reuses_existing_function(self):
        p=pd.DataFrame(dict(game_id=['x']*3,posteam=['H']*3,defteam=['A']*3,
            epa=[1.,-1.,999.],play_type=['pass']*3,qb_dropback=[1.]*3,
            rush_attempt=[0.]*3,qb_kneel=[0.,0.,1.],qb_spike=[0.]*3,
            passer_player_id=['fallback']*3,passer_id=['preferred']*3,qb_epa=[2.,np.nan,999.]))
        _,stats=q.old.aggregate_pbp(p)
        self.assertEqual(stats.iloc[0].passer_player_id,'preferred')
        self.assertEqual(stats.iloc[0].dropbacks,2)
        self.assertEqual(stats.iloc[0].epa_sum,1.)

if __name__=='__main__':unittest.main()
