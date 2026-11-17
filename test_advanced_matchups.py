import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import advanced_matchups as m

def games():
    return pd.DataFrame(dict(game_id=['old','same','next'],season=[2021]*3,week=[1,2,3],
        game_type=['REG']*3,gameday=pd.to_datetime(['2021-09-01','2021-09-08','2021-09-15'],utc=True),
        home_team=['H']*3,away_team=['A']*3,actual_margin=[1.,2.,np.nan]))

class AdvancedTests(unittest.TestCase):
    def stats(self):
        rows=[]
        for gid in ['old','same','next']:
            for team,opp in [('H','A'),('A','H')]:
                rows.append(dict(game_id=gid,posteam=team,defteam=opp,dropbacks=20,rushes=10,
                    pass_explosives=4 if team=='H' else 2,rush_explosives=1,sacks=2,
                    times_pressured=6 if team=='H' else 4,st_epa=1 if team=='H' else -1,st_plays=5))
        return pd.DataFrame(rows)

    def test_pregame_uses_only_prior_dates_and_completed_games(self):
        g=games();s=self.stats()
        a=m.matchup_features(g,s)
        s.loc[s.game_id.ne('old'),['pass_explosives','times_pressured','st_epa']]=9999
        b=m.matchup_features(g,s)
        cols=m.PRESSURE+m.EXPLOSIVE+m.SPECIAL
        np.testing.assert_allclose(a.loc[:1,cols],b.loc[:1,cols],equal_nan=True)
        self.assertEqual(a.loc[1,'explosive_pass_off_diff'],.1)
        self.assertAlmostEqual(a.loc[1,'pressure_allowed_diff'],.1)

    def test_no_history_is_missing_and_resets_each_season(self):
        g=games();g.loc[1:,'season']=2022
        r=m.matchup_features(g,self.stats())
        self.assertEqual(r.loc[1,'home_pressure_missing'],1)
        self.assertTrue(pd.isna(r.loc[1,'pressure_allowed_diff']))

    def test_special_epa_is_home_relative_and_not_double_counted(self):
        p=pd.DataFrame([
            dict(game_id='x',play_id=1,home_team='H',away_team='A',posteam='A',defteam='H',play_type='kickoff',qb_dropback=0,qb_kneel=0,qb_spike=0,yards_gained=0,sack=0,total_home_epa=1.),
            dict(game_id='x',play_id=2,home_team='H',away_team='A',posteam='H',defteam='A',play_type='pass',qb_dropback=1,qb_kneel=0,qb_spike=0,yards_gained=20,sack=0,total_home_epa=3.),
            dict(game_id='x',play_id=3,home_team='H',away_team='A',posteam='A',defteam='H',play_type='run',qb_dropback=0,qb_kneel=0,qb_spike=0,yards_gained=10,sack=0,total_home_epa=4.),
            dict(game_id='x',play_id=4,home_team='H',away_team='A',posteam='H',defteam='A',play_type='field_goal',qb_dropback=0,qb_kneel=0,qb_spike=0,yards_gained=0,sack=0,total_home_epa=6.)])
        t=m.aggregate_plays(p).set_index('posteam')
        self.assertEqual(t.loc['H','st_epa'],3.)
        self.assertEqual(t.loc['A','st_epa'],-3.)
        self.assertEqual(t.loc['H','pass_explosives'],1)
        self.assertEqual(t.loc['A','rush_explosives'],1)

    def injury_data(self):
        inj=pd.DataFrame([dict(season=2021,week=2,game_type='REG',team='H',gsis_id='gsis',
            position='T',report_status='Out',date_modified='2021-09-07T12:00:00Z')])
        sn=pd.DataFrame([dict(game_id=gid,team='H',pfr_player_id='pfr',offense_pct=share,defense_pct=0.,st_pct=0.) for gid,share in [('old',.8),('same',0.),('next',0.)]])
        ids=pd.DataFrame([dict(pfr_id='pfr',gsis_id='gsis')])
        return inj,sn,ids

    def test_injury_uses_prior_usage_and_partial_coverage(self):
        inj,sn,ids=self.injury_data()
        # Supply an away report so a difference is defined, without implying roster completeness.
        away=inj.copy();away['team']='A';away['report_status']=''
        r=m.injury_features(games(),pd.concat([inj,away]),sn,ids)
        self.assertEqual(r.loc[1,'out_OL_diff'],.8)
        self.assertEqual(r.loc[1,'home_injury_missing'],0)
        sn.loc[sn.game_id.ne('old'),'offense_pct']=1.
        changed=m.injury_features(games(),pd.concat([inj,away]),sn,ids)
        self.assertEqual(changed.loc[1,'out_OL_diff'],.8)

    def test_missing_and_future_injury_timestamps_are_withheld(self):
        inj,sn,ids=self.injury_data()
        for value in [None,'2021-09-08T00:00:00Z','2021-09-09T00:00:00Z']:
            inj['date_modified']=value
            r=m.injury_features(games(),inj,sn,ids)
            self.assertEqual(r.loc[1,'home_injury_missing'],1)
            self.assertTrue(pd.isna(r.loc[1,'out_OL_diff']))

    def test_conflicting_player_ids_are_not_matched(self):
        inj,sn,ids=self.injury_data()
        ids=pd.concat([ids,pd.DataFrame([dict(pfr_id='pfr',gsis_id='other')])])
        r=m.injury_features(games(),inj,sn,ids)
        self.assertEqual(r.loc[1,'home_injury_missing'],0)
        self.assertTrue(pd.isna(r.loc[1,'injury_unknown_usage_diff']))

    def test_pbp_duplicates_fail(self):
        p=pd.DataFrame([{c:0 for c in m.PBP}]*2)
        with self.assertRaisesRegex(ValueError,'Duplicate'):m.aggregate_plays(p)

    def test_actual_fit_rows_are_prior_seasons(self):
        f=pd.DataFrame(dict(game_id=[str(i) for i in range(800)],season=[2017]*500+[2020]*50+[2021]*50+[2022]*50+[2023]*50+[2024]*50+[2025]*50,
            actual_margin=np.arange(800)%7,spread_line=0.))
        for c in set(m.base.INPUTS+m.ADJ+m.PRESSURE+m.EXPLOSIVE+m.SPECIAL+m.INJURY)-set(f):f[c]=0.
        seen=[]
        class Fit:
            def predict(self,x):return np.zeros(len(x))
        def fake(train,columns):
            seen.append(set(train.game_id));return Fit()
        with patch.object(m,'fit',side_effect=fake):
            p,folds=m.evaluate(f)
        for i,year in enumerate(range(2020,2026)):
            expected=set(f.loc[f.season.lt(year),'game_id'])
            for used in seen[i*len(m.ARMS):(i+1)*len(m.ARMS)]:self.assertEqual(used,expected)
        self.assertEqual(len(p),300*len(m.ARMS))
        self.assertTrue((folds.train_last_season<folds.test_season).all())

class ResultTests(unittest.TestCase):
    @unittest.skipUnless(Path('advanced_matchup_results/predictions.csv').exists(),'Run full backtest first')
    def test_identical_all_game_coverage_and_baseline(self):
        p=pd.read_csv('advanced_matchup_results/predictions.csv')
        baseline=p[p.method.eq('epa_baseline')]
        for name,g in p.groupby('method'):
            self.assertEqual(set(g.game_id),set(baseline.game_id),name)
            self.assertEqual(len(g),1693,name)
        reference=pd.read_csv('strength_epa_reference_predictions.csv')
        pair=baseline.merge(reference[reference.method.eq('opponent_epa')],on='game_id',validate='one_to_one',suffixes=('_new','_ref'))
        self.assertEqual(len(pair),1693)
        np.testing.assert_allclose(pair.predicted_margin_new,pair.predicted_margin_ref,atol=1e-10,rtol=0)

if __name__=='__main__':unittest.main()
