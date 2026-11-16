import unittest
import numpy as np
import pandas as pd
from market_epa_carryover import features
import market_epa_experiment as epa

class CarryoverTests(unittest.TestCase):
 def setUp(self):
  self.games=pd.DataFrame(dict(game_id=['new'],season=[2025],gameday=pd.to_datetime(['2025-09-01'],utc=True),home_team=['H'],away_team=['A']))
  self.history=pd.DataFrame(dict(game_id=['old','old'],season=[2024,2024],gameday=pd.to_datetime(['2024-12-01','2024-12-01'],utc=True),posteam=['H','A'],defteam=['A','H'],plays=[60,60],epa=[1.,-1.],success=[1.,0.]))
 def test_carry_has_prior_information_reset_has_none(self):
  reset=features(self.games,self.history,False);carry=features(self.games,self.history,True)
  np.testing.assert_array_equal(reset[epa.ADJ],0.)
  self.assertGreater(carry.adj_off_epa_diff.iloc[0],0.)
 def test_no_same_day_or_older_than_previous_season(self):
  for column,value in [('season',2023),('gameday',pd.Timestamp('2025-09-01',tz='UTC'))]:
   h=self.history.copy();h[column]=value
   np.testing.assert_array_equal(features(self.games,h,True)[epa.ADJ],0.)
 def test_half_weight_matches_explicit_ridge(self):
  # Two offense/defense rows with 30 effective plays each and ridge alpha 200.
  x=np.array([[0,1,1,0],[1,0,0,1]]) # teams sorted A,H
  fitted=epa.Ridge(alpha=200.).fit(x,self.history[epa.METRICS].to_numpy(),sample_weight=np.array([30.,30.]))
  expected=fitted.coef_[0,1]-fitted.coef_[0,0]
  self.assertAlmostEqual(features(self.games,self.history,True).adj_off_epa_diff.iloc[0],expected)

if __name__=='__main__':unittest.main()
