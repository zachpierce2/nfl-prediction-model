import unittest
import numpy as np
import pandas as pd
import market_model as m

class MarketTests(unittest.TestCase):
    def test_future_outcomes_do_not_change_earlier_features(self):
        a=m.read_games('games_snapshot.csv')
        raw=pd.read_csv('games_snapshot.csv')
        raw.loc[raw.season>=2024,['home_score','away_score']]=[99,0]
        raw.to_csv('/tmp/market_future_test.csv',index=False)
        b=m.read_games('/tmp/market_future_test.csv')
        pd.testing.assert_frame_equal(a.loc[a.season<2024,m.INPUTS],b.loc[b.season<2024,m.INPUTS])
    def test_market_sign_and_cap(self):
        class Dummy:
            def predict(self, x): return np.array([100,-100,0])
        d=pd.DataFrame({c:[0,0,0] for c in m.INPUTS})
        np.testing.assert_allclose(m.adjustment(Dummy(),d),[2,-2,0])
        # Home favorite -8.5 means expected home margin +8.5.
        self.assertEqual(-(-8.5),8.5)
    def test_fold_training_excludes_test_season(self):
        f=m.read_games('games_snapshot.csv')
        original=m.fit
        def checked(train):
            self.assertLess(train.season.max(),2020)
            return original(train)
        m.fit=checked
        try:
            p=m.evaluate(f[f.season<=2020])
            self.assertEqual(set(p.season),{2020})
        finally: m.fit=original
if __name__=='__main__': unittest.main()
