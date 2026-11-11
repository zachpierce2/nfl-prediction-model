"""Offline tests for historical information boundaries and spread grading."""
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from model_v3_epa import epa_features, spread_report

class Boundaries(unittest.TestCase):
    def setUp(self):
        self.games = pd.DataFrame([
            dict(game_id=f'g{i}', gameday=pd.Timestamp(f'2020-09-{day:02d}',tz='UTC'),home_team='A',away_team='B',home_score=20+i,away_score=17)
            for i,day in enumerate([1,8,15])])
        self.teams = pd.DataFrame([
            dict(game_id=f'g{i}',posteam=t,defteam=o,off_epa=v,pass_epa=v,rush_epa=v,success=.5,plays=60)
            for i in range(3) for t,o,v in [('A','B',.2+i*.1),('B','A',-.1)]])
        self.qbs = pd.DataFrame([
            dict(game_id=f'g{i}',posteam=t,passer_player_id=q,epa_sum=e,success_sum=15,dropbacks=30)
            for i in range(3) for t,q,e in [('A','qa',6),('B','qb',-3)]])

    def test_current_and_future_data_do_not_change_current_features(self):
        original=epa_features(self.games,self.teams,self.qbs)
        teams=self.teams.copy(); qbs=self.qbs.copy(); games=self.games.copy()
        teams.loc[teams.game_id.isin(['g1','g2']),'off_epa']=999
        qbs.loc[qbs.game_id.isin(['g1','g2']),'epa_sum']=999
        qbs.loc[qbs.game_id.isin(['g1','g2']),'passer_player_id']='surprise_starter'
        games.loc[games.game_id.isin(['g1','g2']),'home_score']=999
        changed=epa_features(games,teams,qbs)
        assert_frame_equal(original.iloc[:2],changed.iloc[:2])
        self.assertNotEqual(original.iloc[2].off_epa_diff,changed.iloc[2].off_epa_diff)
        self.assertAlmostEqual(original.iloc[1].qb_epa_diff,9/130)

    def test_same_day_does_not_update_history(self):
        self.games.loc[1,'gameday']=self.games.loc[0,'gameday']
        result=epa_features(self.games,self.teams,self.qbs)
        self.assertEqual(result.iloc[1].home_epa_missing,1)
        self.assertEqual(result.iloc[1].home_qb_missing,1)

    def test_missing_completed_game_fails(self):
        with self.assertRaisesRegex(ValueError,'Missing PBP coverage'):
            epa_features(self.games,self.teams[self.teams.game_id!='g1'],self.qbs)

    def test_spread_sign_and_push(self):
        p=pd.DataFrame(dict(model=['epa_ridge']*3,season=[2020]*3,
                            predicted_margin=[6,0,6],spread_line=[3,3,3],actual_margin=[4,2,3]))
        report=spread_report(p)
        row=report[(report.period=='all') & (report.minimum_edge==3)].iloc[0]
        self.assertEqual((row.wins,row.losses,row.pushes),(2,0,1))
        self.assertEqual(row.hypothetical_profit,200)

if __name__=='__main__': unittest.main()
