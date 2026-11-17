import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import epa_error_audit as a

class AuditTests(unittest.TestCase):
    def frames(self):
        p=pd.DataFrame(dict(game_id=['x','y'],season=[2021,2021],week=[1,2],actual_margin=[7.,-3.],
            spread_line=[3.,-3.],adjustment=[.2,.1],predicted_margin=[3.2,-2.9]))
        g=pd.DataFrame(dict(game_id=['x','y'],week=[1,2],game_type=['REG','REG'],home_team=['H','H'],
            away_team=['A','A'],rest_diff=[0.,1.],location=['Home','Home'],actual_margin=[7.,-3.],spread_line=[3.,-3.]))
        return p.drop(columns='week'),g

    def test_signs_and_pushes(self):
        p,g=self.frames();d=a.enrich(p,g);s=a.metrics(d)
        self.assertEqual(s['wins'],1);self.assertEqual(s['pushes'],1)
        self.assertEqual(d.pick_type.tolist(),['favorite','underdog'])
        self.assertAlmostEqual(s['mae_gain'],.05)

    def test_group_membership_does_not_use_outcomes(self):
        p,g=self.frames();d=a.enrich(p,g)
        p['actual_margin']+=100;g['actual_margin']+=100
        changed=a.enrich(p,g)
        pd.testing.assert_frame_equal(d[a.AXES],changed[a.AXES])

    def test_bad_reference_and_duplicates_rejected(self):
        p,g=self.frames()
        with self.assertRaises(ValueError):a.enrich(pd.concat([p,p]),g)
        g.loc[0,'spread_line']=-3
        with self.assertRaisesRegex(ValueError,'disagrees'):a.enrich(p,g)

    def test_cluster_interval_reproducible(self):
        p,g=self.frames();d=a.enrich(p,g)
        self.assertEqual(a.interval(d,'mae_gain'),a.interval(d,'mae_gain'))
        lo,hi=a.interval(d,'mae_gain');self.assertLessEqual(lo,hi)

    def test_no_prospective_sample_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=a.prospective(tmp)
            self.assertEqual(s['graded_epa_games'],0)
            self.assertEqual(s['status'],'NO_GRADED_PROSPECTIVE_EPA_SAMPLE')

    def test_postkickoff_forecast_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'epa_shadow';d.mkdir()
            pd.DataFrame([dict(game_id='x',captured_at_utc='2026-11-01T12:00:00Z',
                forecast_created_at_utc='2026-11-01T15:00:00Z',kickoff_utc='2026-11-01T14:00:00Z')]).to_csv(d/'forecasts.csv',index=False)
            with self.assertRaisesRegex(ValueError,'timing'):a.prospective(tmp)

if __name__=='__main__':unittest.main()
