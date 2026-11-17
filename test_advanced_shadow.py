import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import advanced_matchups as advanced
import advanced_shadow as shadow
import auto_market as auto
import epa_shadow

class ShadowTests(unittest.TestCase):
    def core(self):
        now=datetime.now(timezone.utc)
        r={c:'' for c in auto.FORECAST_FIELDS}
        r.update(game_id='live',quote_id='fixed',home_team='H',away_team='A',sportsbook='Book',
            home_line=-3,away_line=3,home_price_american=-110,away_price_american=-110,
            captured_at_utc=auto.stamp(now-timedelta(minutes=2)),
            forecast_created_at_utc=auto.stamp(now-timedelta(minutes=1)),
            kickoff_utc=auto.stamp(now+timedelta(days=1)))
        return r

    def test_missing_groups_skip_only_affected_candidates(self):
        row={side+'_'+g+'_missing':0 for side in ['home','away'] for g in ['pressure','explosive','special','injury']}
        row['home_injury_missing']=1
        self.assertFalse(shadow.eligible('epa_all',row))
        self.assertFalse(shadow.eligible('epa_injury',row))
        self.assertTrue(shadow.eligible('epa_pressure',row))
        self.assertTrue(shadow.eligible('epa_baseline',row))

    def test_immutable_quote_hash_and_time_checks(self):
        c=self.core();r=dict(c,arm='epa_pressure',core_forecast_sha256=epa_shadow.core_hash(c))
        self.assertEqual(len(shadow.verified_pairs([r],[c])),1)
        for key,value in [('quote_id','changed'),('home_price_american',-120),
                          ('forecast_created_at_utc',auto.stamp(datetime.now(timezone.utc)-timedelta(days=1)))]:
            with self.assertRaises(ValueError):shadow.verified_pairs([dict(r,**{key:value})],[c])

    def test_record_preserves_quotes_and_never_backfills(self):
        c=self.core()
        f=pd.DataFrame(dict(game_id=[str(i) for i in range(500)]+['live'],
            season=[2017]*500+[2026],actual_margin=[1.]*500+[np.nan],spread_line=[0.]*501))
        for col in advanced.base.INPUTS:
            if col not in f:f[col]=0.
        x=pd.DataFrame({'game_id':f.game_id})
        for col in advanced.PRESSURE+advanced.EXPLOSIVE+advanced.SPECIAL:x[col]=0.
        injury=pd.DataFrame({'game_id':f.game_id})
        for col in advanced.INJURY:injury[col]=0.
        injury.loc[injury.game_id.eq('live'),'home_injury_missing']=1.
        class Fixed:
            def predict(self,x):return np.ones(len(x))*4
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'snapshot';folder.mkdir()
            schedule=folder/'schedule.csv';schedule.write_text('fixture')
            epa=pd.DataFrame({'game_id':f.game_id})
            for col in advanced.ADJ:epa[col]=0.
            epa.to_csv(folder/'epa_features_asof.csv.gz',index=False)
            with patch.object(shadow,'ROOT',Path(tmp)/'archive'),patch.object(shadow.base,'read_games',return_value=f),patch.object(advanced,'sources',return_value=(pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(columns=['gsis_id','pfr_id']),[])),patch.object(advanced,'matchup_features',return_value=x),patch.object(advanced,'injury_features',return_value=injury),patch.object(advanced,'fit',return_value=Fixed()):
                status=shadow.record(schedule,[c],folder)
                self.assertEqual(status['arms']['epa_all']['new_forecasts'],0)
                self.assertEqual(status['arms']['epa_pressure']['new_forecasts'],1)
                p=pd.read_csv(Path(tmp)/'archive/epa_pressure/forecasts.csv').iloc[0]
                self.assertEqual(p.quote_id,c['quote_id'])
                self.assertEqual(p.home_line,c['home_line'])
                self.assertEqual(p.home_price_american,c['home_price_american'])
                self.assertEqual(p.core_forecast_sha256,epa_shadow.core_hash(c))
                self.assertEqual(shadow.record(schedule,[c],folder)['arms']['epa_pressure']['new_forecasts'],0)
                late=dict(c,kickoff_utc=auto.stamp(datetime.now(timezone.utc)-timedelta(seconds=1)))
                with self.assertRaisesRegex(ValueError,'kickoff'):shadow.record(schedule,[late],folder)

    def test_grading_pairs_candidate_with_epa_on_same_games(self):
        c=self.core()
        baseline=dict(c,arm='epa_baseline',core_forecast_sha256=epa_shadow.core_hash(c))
        pressure=dict(c,arm='epa_pressure',core_forecast_sha256=epa_shadow.core_hash(c))
        class Grader:
            def read_rows(self,path):
                return [baseline] if path.parent.name=='epa_baseline' else [pressure] if path.parent.name=='epa_pressure' else []
            def match(self,row,finals,quotes):return {'final':True}
            def grade(self,row,final,now):
                return dict(game_id=row['game_id'],forecast_archive=row.get('forecast_archive','core'))
            def summary(self,grades,rows,now):return dict(graded=len(grades),forecasts=len(rows))
        with tempfile.TemporaryDirectory() as tmp,patch.object(shadow,'ROOT',Path(tmp)):
            result=shadow.grade_shadow(Grader(),[c],[],[],datetime.now(timezone.utc))
            self.assertEqual(result['epa_pressure']['candidate']['graded'],1)
            self.assertEqual(result['epa_pressure']['epa']['graded'],1)
            p=pd.read_csv(Path(tmp)/'epa_pressure/graded_predictions.csv')
            self.assertEqual(p.forecast_archive.iloc[0],str(Path(tmp)/'epa_pressure/forecasts.csv'))
            e=pd.read_csv(Path(tmp)/'epa_pressure/paired_epa_grades.csv')
            self.assertEqual(e.forecast_archive.iloc[0],str(Path(tmp)/'epa_baseline/forecasts.csv'))

if __name__=='__main__':unittest.main()
