"""Integration checks against the frozen historical snapshots (no network)."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import market_cover_probability as experiment


class WalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.out = Path(cls.temp.name)
        cls.fits = []
        original = experiment.base.make_pipeline

        def tracked_pipeline(*steps):
            model = original(*steps)
            fit = model.fit

            def tracked_fit(x, y, **kwargs):
                cls.fits.append(set(x.index))
                return fit(x, y, **kwargs)

            model.fit = tracked_fit
            return model

        with patch.object(experiment.base, 'make_pipeline', tracked_pipeline), contextlib.redirect_stdout(io.StringIO()):
            cls.predictions, cls.summary = experiment.run(
                'strength_games_snapshot.csv', 'strength_epa_features.csv', cls.out)

    def test_actual_fit_rows_use_only_prior_seasons(self):
        f = experiment.base.read_games('strength_games_snapshot.csv')
        e = pd.read_csv('strength_epa_features.csv')
        f = f[f.season.between(2017, 2025)].merge(e[['game_id'] + experiment.EPA], on='game_id')
        self.assertEqual(len(self.fits), 12)
        for i, year in enumerate(range(2020, 2026)):
            train = f[(f.season < year) & f.actual_margin.notna() & f.spread_line.notna()]
            binary = train[(train.actual_margin - train.spread_line).ne(0)]
            self.assertEqual(self.fits[2*i], set(binary.index))
            self.assertEqual(self.fits[2*i+1], set(train.index))
        folds = pd.read_csv(self.out / 'training_folds.csv')
        self.assertTrue((folds.train_last_season < folds.test_season).all())
        self.assertTrue((pd.to_datetime(folds.train_last_date) < pd.to_datetime(folds.test_first_date)).all())

    def test_equal_game_coverage_and_probabilities(self):
        groups = list(self.predictions.groupby('method'))
        expected = set(groups[0][1].game_id)
        self.assertEqual(len(groups), 4)
        for name, g in groups:
            self.assertFalse(g.game_id.duplicated().any())
            self.assertEqual(set(g.game_id), expected)
            if name != 'margin_core_epa':
                self.assertTrue(np.isfinite(g.home_cover_probability).all())
                self.assertTrue(g.home_cover_probability.between(0, 1).all())
        self.assertEqual(len(self.predictions.query("method == 'cover_logistic' and season >= 2021")), 1424)

    def test_margin_baseline_reproduces_frozen_reference(self):
        reference = pd.read_csv('strength_epa_reference_predictions.csv')
        reference = reference[reference.method.eq('opponent_epa')].set_index('game_id')
        self.assertGreater(len(reference), 0)
        actual = self.predictions[self.predictions.method.eq('margin_core_epa')].set_index('game_id')
        self.assertEqual(set(actual.index), set(reference.index))
        np.testing.assert_allclose(actual.predicted_margin, reference.loc[actual.index].predicted_margin, atol=1e-10, rtol=0)

    def test_same_day_and_future_scores_do_not_enter_features(self):
        raw = pd.read_csv('strength_games_snapshot.csv', low_memory=False)
        original = experiment.base.read_games('strength_games_snapshot.csv').set_index('game_id')
        cutoff = '2024-10-06'
        raw.loc[raw.gameday.ge(cutoff), ['home_score', 'away_score']] = [99, 0]
        path = self.out / 'perturbed.csv'
        raw.to_csv(path, index=False)
        changed = experiment.base.read_games(path).set_index('game_id')
        ids = original.index[original.gameday.le(pd.Timestamp(cutoff, tz='UTC'))]
        np.testing.assert_allclose(original.loc[ids, experiment.base.INPUTS], changed.loc[ids, experiment.base.INPUTS], equal_nan=True)


if __name__ == '__main__':
    unittest.main()
