import unittest
import numpy as np
import pandas as pd
from market_epa_strength import choose, score

class StrengthTests(unittest.TestCase):
    def test_future_outcomes_cannot_change_selection(self):
        p=pd.DataFrame(dict(season=[2020,2020,2021,2021],strength=[0.,.25,0.,.25],actual_margin=[1.,1.,10.,10.],spread_line=[0.]*4,predicted_margin=[0.,1.,0.,1.],adjustment=[0.,1.,0.,1.]))
        self.assertEqual(choose(p,2021),.25)
        p.loc[p.season.eq(2021),'actual_margin']=-1000
        self.assertEqual(choose(p,2021),.25)
    def test_market_has_no_picks(self):
        g=pd.DataFrame(dict(actual_margin=[1.,-1.,0.],spread_line=[0.]*3,predicted_margin=[0.]*3,adjustment=[0.]*3))
        s=score(g)
        self.assertEqual((s['wins'],s['losses'],s['pushes'],s['no_pick']),(0,0,0,3))
        self.assertIsNone(s['win_pct'])
    def test_replay_and_fold_boundaries(self):
        p=pd.read_csv('epa_strength_results/candidate_predictions.csv')
        reference=pd.read_csv('strength_epa_reference_predictions.csv')
        ref=reference[reference.method.eq('opponent_epa')]
        now=p[p.strength.eq(.25)].merge(ref,on=['game_id','season'],validate='one_to_one',suffixes=('_new','_old'))
        self.assertEqual(len(now),len(ref))
        np.testing.assert_allclose(now.predicted_margin_new,now.predicted_margin_old,rtol=0,atol=1e-10)
        folds=pd.read_csv('epa_strength_results/training_folds.csv')
        self.assertTrue((folds.train_last_season<folds.test_season).all())
        for _,g in p.groupby('game_id'):
            self.assertEqual(g[g.strength.gt(0)].adjustment.map(np.sign).nunique(),1)

if __name__=='__main__':unittest.main()
