import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import market_epa_recency as model
import market_epa_experiment as epa

class RecencyTests(unittest.TestCase):
    def test_half_life_and_weight_mass(self):
        f=pd.DataFrame({'posteam':['H']*5+['A']*2,'plays':[60]*5+[40,80]})
        w=model.recency_weights(f)
        self.assertAlmostEqual(w[4]/w[0],2.)
        self.assertAlmostEqual(w[:5].sum(),300.)
        self.assertAlmostEqual(w[5:].sum(),120.)

    def test_excludes_same_day_future_and_previous_season(self):
        games=pd.DataFrame(dict(game_id=['test'],season=[2025],gameday=pd.to_datetime(['2025-09-08'],utc=True),home_team=['H'],away_team=['A']))
        history=pd.DataFrame(dict(game_id=['prior','today','future','old'],season=[2025,2025,2025,2024],gameday=pd.to_datetime(['2025-09-01','2025-09-08','2025-09-09','2024-12-01'],utc=True),posteam=['H']*4,defteam=['A']*4,plays=[60]*4,epa=[.2,100,100,100],success=[.5,100,100,100]))
        a=model.features(games,history,True)
        b=model.features(games,history.iloc[:1],True)
        np.testing.assert_allclose(a[epa.ADJ],b[epa.ADJ],atol=0,rtol=0)

    def test_replay_and_fold_coverage(self):
        p=pd.read_csv('epa_recency_results/predictions.csv')
        baseline=p[p.method.eq('season_reset')]
        reference=pd.read_csv('strength_epa_reference_predictions.csv')
        reference=reference[reference.method.eq('opponent_epa')]
        pair=baseline.merge(reference,on='game_id',suffixes=('_new','_ref'),validate='one_to_one')
        self.assertEqual(len(pair),1693)
        np.testing.assert_allclose(pair.predicted_margin_new,pair.predicted_margin_ref,atol=1e-10,rtol=0)
        a=p[p.method.eq('recency_half_life_4')]
        self.assertEqual(set(a.game_id),set(baseline.game_id))
        folds=pd.read_csv('epa_recency_results/training_folds.csv')
        self.assertTrue((folds.train_last_season<folds.test_season).all())

if __name__=='__main__': unittest.main()
