import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import epa_targeted_correction as t

class TargetedTests(unittest.TestCase):
    def test_threshold_and_side_symmetry(self):
        f=t.add_feature(pd.DataFrame({'spread_line':[-7.5,-7,0,7,7.5]}))
        self.assertEqual(f[t.FEATURE].tolist(),[1.,0.,0.,0.,1.])

    def test_feature_is_independent_of_result(self):
        a=t.add_feature(pd.DataFrame(dict(spread_line=[8.,-8.],actual_margin=[1.,2.])))
        b=t.add_feature(pd.DataFrame(dict(spread_line=[8.,-8.],actual_margin=[100.,-100.])))
        pd.testing.assert_series_equal(a[t.FEATURE],b[t.FEATURE])

    def test_actual_training_rows_and_equal_game_coverage(self):
        f=pd.DataFrame(dict(game_id=[str(i) for i in range(800)],season=[2017]*500+[2020]*50+[2021]*50+[2022]*50+[2023]*50+[2024]*50+[2025]*50,
            spread_line=np.tile([3.,8.],400),actual_margin=0.))
        for c in set(t.base.INPUTS+t.ADJ)-set(f):f[c]=0.
        used=[]
        class Fake:
            named_steps={'ridge':type('R',(),{'coef_':np.array([1.])})()}
            def predict(self,x):return np.zeros(len(x))
        def fitted(train,columns):
            used.append((set(train.game_id),columns));return Fake()
        with patch.object(t,'fit',side_effect=fitted):
            p,folds=t.evaluate(f)
        for i,year in enumerate(range(2020,2026)):
            expected=set(f.loc[f.season.lt(year),'game_id'])
            self.assertEqual(used[2*i][0],expected);self.assertEqual(used[2*i+1][0],expected)
            self.assertNotIn(t.FEATURE,used[2*i][1]);self.assertIn(t.FEATURE,used[2*i+1][1])
        self.assertTrue((folds.train_last_season<folds.test_season).all())
        self.assertEqual(set(p[p.method.eq('epa_baseline')].game_id),set(p[p.method.eq('epa_wide_spread')].game_id))

    def test_paired_disagreements_and_common_push(self):
        rows=[]
        for name,adj in [('epa_baseline',[.1,-.1,.1]),('epa_wide_spread',[-.1,-.1,-.1])]:
            for gid,margin,a in zip(['x','y','z'],[1.,-1.,0.],adj):
                rows.append(dict(game_id=gid,season=2021,actual_margin=margin,spread_line=0.,
                    adjustment=a,predicted_margin=a,method=name))
        result=t.paired(pd.DataFrame(rows),pd.DataFrame(dict(game_id=['x','y','z'],week=[1,2,3])))
        self.assertEqual(result['candidate_only_wins'],0)
        self.assertEqual(result['baseline_only_wins'],1)
        self.assertEqual(result['disagreements'],1)

if __name__=='__main__':unittest.main()
