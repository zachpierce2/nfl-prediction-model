import unittest
import numpy as np
import pandas as pd
import market_pass_rush as m
from test_market_epa_experiment import Tests as Fixture

class Tests(unittest.TestCase):
    def fixture(self):
        games,t=Fixture().fixture();t=t.drop(columns=['gameday','season'])
        return games,pd.concat([t.assign(play_type=k) for k in ['pass','run']],ignore_index=True)

    def test_temporal_exclusion_and_reset(self):
        games,t=self.fixture();before=m.features(games,t)
        t.loc[t.game_id.ne('g1'),['epa','success']]=[99.,0.]
        after=m.features(games,t)
        pd.testing.assert_frame_equal(before,after)
        self.assertTrue(np.allclose(before.loc[[0,2],m.SPLIT],0))

    def test_passing_and_rushing_remain_separate(self):
        games,t=self.fixture();before=m.features(games,t)
        t.loc[t.play_type.eq('run'),'epa']*=4
        after=m.features(games,t)
        cols=['pass_'+c for c in m.e.ADJ]
        pd.testing.assert_frame_equal(before[cols],after[cols])
        self.assertGreater(after.loc[1,'run_adj_off_epa_diff'],before.loc[1,'run_adj_off_epa_diff'])

    def test_missing_type_coverage_fails(self):
        games,t=self.fixture()
        with self.assertRaisesRegex(ValueError,'Incomplete PBP coverage'):
            m.features(games,t[~(t.game_id.eq('g1') & t.play_type.eq('run'))])

    def test_aggregation_filters_and_success(self):
        p=pd.DataFrame([dict(game_id='g',play_id=i,posteam='A',defteam='B',play_type=k,epa=v,qb_kneel=n,qb_spike=0) for i,k,v,n in [(1,'pass',2,0),(2,'pass',-1,0),(3,'run',10,1),(4,'run',0,0)]])
        a=m.aggregate(p).set_index('play_type')
        self.assertEqual(a.loc['pass','epa'],.5)
        self.assertEqual(a.loc['pass','success'],.5)
        self.assertEqual(a.loc['run','plays'],1)
        with self.assertRaisesRegex(ValueError,'Duplicate plays'):m.aggregate(pd.concat([p,p]))

class ResultTests(unittest.TestCase):
    def test_coverage_folds_and_reference(self):
        p=pd.read_csv('pass_rush_results/predictions.csv')
        self.assertFalse(p.duplicated(['game_id','method']).any())
        self.assertEqual(p.groupby('method').size().tolist(),[1693,1693])
        a=p[p.method.eq('epa_baseline')].set_index('game_id')
        b=pd.read_csv('strength_epa_reference_predictions.csv').query("method == 'opponent_epa'").set_index('game_id')
        np.testing.assert_allclose(a.predicted_margin,b.loc[a.index].predicted_margin,atol=1e-10,rtol=0)
        f=pd.read_csv('pass_rush_results/training_folds.csv')
        self.assertTrue((f.train_last_season<f.test_season).all())
        self.assertEqual(p[p.season.ge(2021)].groupby('method').size().tolist(),[1424,1424])

if __name__=='__main__':unittest.main()
