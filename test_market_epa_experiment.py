import unittest
import numpy as np
import pandas as pd
import market_epa_experiment as e

class Tests(unittest.TestCase):
    def fixture(self):
        games=pd.DataFrame({'game_id':['g1','g2','g3'],'gameday':pd.to_datetime(['2021-09-01','2021-09-08','2022-09-01'],utc=True),'season':[2021,2021,2022],'home_team':['A']*3,'away_team':['B']*3,'actual_margin':[3,7,0]})
        rows=[]
        for g in games.itertuples():
            for a,b,v in [('A','B',1.),('B','A',-1.)]:
                rows.append(dict(game_id=g.game_id,gameday=g.gameday,season=g.season,posteam=a,defteam=b,epa=v,success=float(v>0),plays=60))
        return games,pd.DataFrame(rows)

    def test_same_day_future_values_do_not_change_earlier_features(self):
        games,t=self.fixture();a=e.build_features(games,t)
        t.loc[t.game_id.ne('g1'),['epa','success']]=[99.,0.]
        b=e.build_features(games,t)
        pd.testing.assert_frame_equal(a.iloc[:2],b.iloc[:2])
        self.assertTrue(np.allclose(a.loc[0,e.RAW+e.ADJ].to_numpy(dtype=float),0))
        self.assertTrue(np.allclose(a.loc[2,e.RAW+e.ADJ].to_numpy(dtype=float),0))

    def test_prior_results_influence_next_game_and_signs(self):
        games,t=self.fixture();a=e.build_features(games,t)
        self.assertGreater(a.loc[1,'adj_off_epa_diff'],0)
        self.assertGreater(a.loc[1,'adj_def_epa_diff'],0)
        self.assertEqual(a.loc[1,'raw_off_epa_diff'],2.)

    def test_missing_pbp_game_fails(self):
        games,t=self.fixture()
        with self.assertRaisesRegex(ValueError,'Incomplete PBP coverage'):
            e.prepare_team_games(t[t.game_id.ne('g2')].drop(columns=['gameday','season']),games)

    def test_market_has_no_betting_record(self):
        result=e.score(pd.DataFrame({'actual_margin':[7.],'spread_line':[3.],'adjustment':[0.],'predicted_margin':[3.]}))
        self.assertEqual(result['wins'],0);self.assertIsNone(result['win_pct'])

if __name__=='__main__':unittest.main()
