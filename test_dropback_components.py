import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import market_dropback_components as m

class FeatureTests(unittest.TestCase):
    def plays(self):
        rows=[]
        for gid in ['g1','g2','g3']:
            for team,other,sign in [('A','B',1),('B','A',-1)]:
                for i,kind,db,sc,sack,epa in [(1,'pass',1,0,0,2.),(2,'pass',1,0,1,-3.),(3,'run',1,1,0,1.),(4,'run',0,0,0,-1.),(5,'run',0,0,0,0.)]:
                    rows.append(dict(game_id=gid,play_id=(i if team=='A' else i+10),posteam=team,defteam=other,play_type=kind,epa=epa*sign,qb_dropback=db,qb_scramble=sc,sack=sack,qb_kneel=0,qb_spike=0))
        return pd.DataFrame(rows)
    def games(self):
        return pd.DataFrame(dict(game_id=['g1','g2','g3'],season=[2021,2021,2022],gameday=pd.to_datetime(['2021-09-01','2021-09-08','2022-09-01'],utc=True),home_team=['A']*3,away_team=['B']*3,actual_margin=[3.,7.,0.]))
    def test_classification_conserves_plays_epa_success_and_zeros(self):
        p=self.plays();t=m.aggregate(p)
        for (gid,team),g in t.groupby(['game_id','posteam']):
            raw=p[p.game_id.eq(gid)&p.posteam.eq(team)]
            self.assertEqual(g.category_plays.sum(),len(raw));self.assertAlmostEqual(g.rate.sum(),1.)
            self.assertAlmostEqual(g.epa.sum(),raw.epa.mean());self.assertAlmostEqual(g.success.sum(),raw.epa.gt(0).mean())
        g=t[t.game_id.eq('g1')&t.posteam.eq('A')].set_index('category')
        self.assertEqual(g.loc['sack','epa'],-3/5);self.assertEqual(g.loc['scramble','epa'],1/5)
        # No sack events must still produce a row with zero contribution.
        no_sacks=m.aggregate(p[p.sack.ne(1)]).query("category == 'sack'")
        self.assertEqual(len(no_sacks),6);self.assertTrue(no_sacks.rate.eq(0).all());self.assertTrue(no_sacks.epa.eq(0).all())
    def test_kneels_spikes_duplicates_missing_and_inconsistent_flags(self):
        p=self.plays();kneel=p.iloc[:1].copy();kneel.play_id=99;kneel.qb_kneel=1;kneel.epa=9999
        pd.testing.assert_frame_equal(m.aggregate(p),m.aggregate(pd.concat([p,kneel])))
        spike=kneel.assign(qb_kneel=0,qb_spike=1)
        pd.testing.assert_frame_equal(m.aggregate(p),m.aggregate(pd.concat([p,spike])))
        with self.assertRaisesRegex(ValueError,'Duplicate'):m.aggregate(pd.concat([p,p]))
        for flag in ['sack','qb_scramble','qb_dropback']:
            bad=p.copy();bad.loc[0,flag]=np.nan
            with self.assertRaisesRegex(ValueError,'flag'):m.aggregate(bad)
        bad=p.copy();bad.loc[0,['sack','qb_dropback']]=[1,0]
        with self.assertRaisesRegex(ValueError,'classification'):m.aggregate(bad)
    def test_replay_reversal_sack_flag_precedes_scramble(self):
        p=self.plays();p.loc[p.sack.eq(1),'qb_scramble']=1
        actual=m.aggregate(p);original=m.aggregate(self.plays())
        pd.testing.assert_frame_equal(actual,original)
    def test_same_day_future_history_excluded_and_season_resets(self):
        games=self.games();p=self.plays();a=m.build_features(games,m.aggregate(p))
        p.loc[p.game_id.ne('g1'),'epa']=99999
        b=m.build_features(games,m.aggregate(p))
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(np.allclose(a.loc[[0,2],m.EXTRA],0))
        self.assertNotEqual(a.loc[1,'sack_off_epa_diff'],0.)
    def test_missing_team_category_or_invalid_rate_fails(self):
        games=self.games();t=m.aggregate(self.plays())
        with self.assertRaisesRegex(ValueError,'coverage'):m.prepare(games,t.iloc[1:])
        bad=t.copy();bad.loc[0,'rate']=2
        with self.assertRaisesRegex(ValueError,'contributions'):m.prepare(games,bad)
    def test_eight_game_window_counts_games_without_events(self):
        # A sack nine prior team games ago must not enter the eight-game window.
        games=pd.concat([self.games().iloc[[0]].assign(game_id='g'+str(i),gameday=pd.Timestamp('2021-09-01',tz='UTC')+pd.Timedelta(days=7*i)) for i in range(10)],ignore_index=True)
        raw=self.plays().query("game_id == 'g1'")
        plays=[]
        for i in range(10):
            part=raw.assign(game_id='g'+str(i))
            if i>0:part=part[part.sack.eq(0)]
            plays.append(part)
        f=m.build_features(games,m.aggregate(pd.concat(plays,ignore_index=True)))
        self.assertNotEqual(f.loc[1,'sack_off_epa_diff'],0.)
        self.assertAlmostEqual(f.loc[9,'sack_off_epa_diff'],0.)

class FoldTests(unittest.TestCase):
    def test_actual_fitted_rows_are_from_prior_seasons(self):
        columns=list(dict.fromkeys(m.base.INPUTS+m.original.ADJ+m.REFRESH+m.EXTRA))
        frame=pd.DataFrame(np.zeros((506,len(columns))),columns=columns)
        frame['game_id']=[str(i) for i in range(506)]
        frame['season']=[2017]*500+list(range(2020,2026))
        frame['actual_margin']=3.;seen=[]
        class Estimator:
            def fit(self,x,y):self.indices=x.index;return self
            def predict(self,x):
                year=int(frame.loc[x.index,'season'].iloc[0])
                self_case.assertTrue(frame.loc[self.indices,'season'].lt(year).all())
                seen.append((year,len(self.indices)))
                return np.zeros(len(x))
        self_case=self
        with patch.object(m.base,'make_pipeline',side_effect=lambda *args:Estimator()):
            predictions,folds=m.evaluate(frame)
        self.assertEqual(len(seen),18)
        self.assertEqual(len(predictions),18)
        self.assertEqual(seen[:2],[(2020,500),(2020,500)])

class ResultTests(unittest.TestCase):
    def test_coverage_baseline_reproduction_and_temporal_folds(self):
        p=pd.read_csv('dropback_results/predictions.csv')
        self.assertFalse(p.duplicated(['game_id','method']).any())
        self.assertEqual(p.groupby('method').size().tolist(),[1693,1693,1693])
        self.assertEqual(p[p.season.ge(2021)].groupby('method').size().tolist(),[1424,1424,1424])
        a=p.query("method == 'epa_baseline'").set_index('game_id')
        b=pd.read_csv('strength_epa_reference_predictions.csv').query("method == 'opponent_epa'").set_index('game_id')
        np.testing.assert_allclose(a.predicted_margin,b.loc[a.index].predicted_margin,atol=1e-10,rtol=0)
        folds=pd.read_csv('dropback_results/training_folds.csv')
        self.assertTrue((folds.train_last_season<folds.test_season).all())
        f=pd.read_csv('dropback_results/features.csv');self.assertEqual(len(f.columns),29)
        self.assertTrue(np.isfinite(f[m.EXTRA].to_numpy()).all())

if __name__=='__main__':unittest.main()
