import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import epa_shadow as epa
import epa_simple_shadow as simple
import grade_market as grader
import test_epa_shadow


class SimpleTests(unittest.TestCase):
    def setUp(self):
        fixture = test_epa_shadow.PairedTests(); fixture.setUp()
        self.core = fixture.core
        now = datetime.now(timezone.utc)
        self.core.update(captured_at_utc=epa.auto.stamp(now-timedelta(minutes=3)),
                         forecast_created_at_utc=epa.auto.stamp(now-timedelta(minutes=2)),
                         kickoff_utc=epa.auto.stamp(now+timedelta(hours=12)))
        self.baseline = dict(self.core, core_forecast_sha256=epa.core_hash(self.core),
                             forecast_created_at_utc=epa.auto.stamp(now-timedelta(minutes=1)),
                             epa_inputs_sha256='hash')

    def test_fixed_formula_quote_pairs_and_no_overwrite(self):
        rng = np.random.default_rng(7); n = 600
        frame = pd.DataFrame(rng.normal(size=(n+1,len(epa.model.INPUTS))), columns=epa.model.INPUTS)
        frame['game_id'] = [str(i) for i in range(n)]+['future']
        frame['season'] = [2025]*n+[2026]
        frame['actual_margin'] = rng.normal(size=n+1); frame.loc[n,'actual_margin']=np.nan
        features = pd.DataFrame(rng.normal(size=(n+1,len(epa.ADJ))),columns=epa.ADJ)
        features['game_id']=frame.game_id
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); path=folder/'epa_features_asof.csv.gz'
            features.to_csv(path,index=False)
            self.baseline['epa_inputs_sha256']=epa.auto.digest(path.read_bytes())
            def reads(path):
                return [self.baseline] if path.parent==epa.ROOT else [self.core]
            with patch.object(simple,'ROOT',folder/'simple'),patch.object(simple,'read_rows',side_effect=reads),patch.object(epa.model,'read_games',return_value=frame):
                result=simple.record('unused',[self.core],folder)
                self.assertEqual(result['new_forecasts'],1)
                row=pd.read_csv(folder/'simple/forecasts.csv').iloc[0].to_dict()
                merged=frame.merge(pd.read_csv(path),on='game_id');train=merged.iloc[:n]
                fitted=epa.model.make_pipeline(epa.model.SimpleImputer(strategy='median'),epa.model.StandardScaler(),simple.Ridge(alpha=1000.))
                fitted.fit(train[simple.COLUMNS],train.actual_margin-train.spread_line)
                test=merged.iloc[[n]].copy();test['spread_line']=3.
                expected=float(np.clip(.25*fitted.predict(test[simple.COLUMNS])[0],-2,2))
                self.assertAlmostEqual(row['adjustment'],expected)
                self.assertEqual(row['quote_id'],self.core['quote_id'])
                self.assertEqual(row['epa_inputs_sha256'],self.baseline['epa_inputs_sha256'])
                self.assertEqual(simple.record('unused',[self.core],folder)['new_forecasts'],0)
                before=(folder/'simple/forecasts.csv').read_bytes()
                self.core['kickoff_utc']=epa.auto.stamp(datetime.now(timezone.utc)-timedelta(seconds=1))
                self.baseline['kickoff_utc']=self.core['kickoff_utc']
                self.baseline['core_forecast_sha256']=epa.core_hash(self.core)
                with self.assertRaisesRegex(ValueError,'kickoff'):
                    simple.record('unused',[self.core],folder)
                self.assertEqual(before,(folder/'simple/forecasts.csv').read_bytes())

    def test_no_backfill(self):
        with patch.object(simple,'read_rows') as read:
            self.assertEqual(simple.record('unused',[],Path('unused'))['new_forecasts'],0)
            read.assert_not_called()

    def test_grade_same_subset_and_reject_wrong_inputs(self):
        row=dict(self.baseline,model_version=simple.VERSION,predicted_home_margin=2.,forecast_created_at_utc=epa.auto.stamp(datetime.now(timezone.utc)))
        self.baseline['predicted_home_margin']=4.
        final=dict(provider_event_id='event',home_team='H',away_team='A',kickoff_utc=self.core['kickoff_utc'],home_score=24,away_score=20,snapshot_id='final')
        with tempfile.TemporaryDirectory() as tmp,patch.object(simple,'ROOT',Path(tmp)),patch.object(grader,'read_rows',side_effect=lambda p:[row] if p.parent==Path(tmp) else [self.baseline]):
            result,_=simple.grade_shadow(grader,[self.core],[final],[],datetime.now(timezone.utc))
            self.assertEqual(result['simple']['hypothetical_losses'],1)
            self.assertEqual(result['epa']['hypothetical_wins'],1)
            row['epa_inputs_sha256']='different'
            with self.assertRaisesRegex(ValueError,'same-input'):
                simple.grade_shadow(grader,[self.core],[final],[],datetime.now(timezone.utc))

    def test_zero_games_have_no_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(simple,'ROOT',Path(tmp)):
            result,_=simple.grade_shadow(grader,[],[],[],datetime.now(timezone.utc))
            self.assertEqual(result['paired_forecasts'],0)
            self.assertIsNone(result['simple']['hypothetical_win_pct'])

if __name__=='__main__':unittest.main()
