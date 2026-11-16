import unittest
import numpy as np
import pandas as pd
from market_cover_probability import score

class CoverTests(unittest.TestCase):
 def test_pushes_excluded_from_probability_scores(self):
  g=pd.DataFrame(dict(actual_margin=[4.,2.,3.],spread_line=[3.]*3,direction=[1.,-1.,1.],home_cover_probability=[.8,.2,.99]))
  s=score(g)
  self.assertEqual((s['games'],s['wins'],s['losses'],s['pushes'],s['probability_games']),(3,2,0,1,2))
  self.assertAlmostEqual(s['brier'],.04)
  self.assertAlmostEqual(s['log_loss'],-np.log(.8))
 def test_half_probability_has_no_pick(self):
  g=pd.DataFrame(dict(actual_margin=[4.,2.],spread_line=[3.]*2,direction=[0.,0.],home_cover_probability=[.5,.5]))
  s=score(g)
  self.assertEqual(s['no_pick'],2);self.assertIsNone(s['win_pct'])
  self.assertAlmostEqual(s['brier'],.25);self.assertAlmostEqual(s['log_loss'],np.log(2))
 def test_margin_has_no_fabricated_probability_score(self):
  g=pd.DataFrame(dict(actual_margin=[4.],spread_line=[3.],direction=[1.],home_cover_probability=[np.nan]))
  s=score(g)
  self.assertEqual(s['wins'],1);self.assertEqual(s['probability_games'],0);self.assertIsNone(s['brier'])

if __name__=='__main__':unittest.main()
