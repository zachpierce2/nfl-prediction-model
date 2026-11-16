import unittest
import pandas as pd
from market_cover_nonlinear import split_season


class TemporalBoundaryTests(unittest.TestCase):
    def test_future_outcomes_cannot_enter_training(self):
        frame = pd.DataFrame({'season': [2019]*501+[2020,2021],
                              'actual_margin': [4.]*500+[3.,4.,99.],
                              'spread_line': [3.]*503})
        train, test = split_season(frame, 2020)
        self.assertEqual(len(train), 500)
        self.assertEqual(list(test.season), [2020])
        changed = frame.copy()
        changed.loc[changed.season.ge(2020), 'actual_margin'] = -999.
        changed_train, _ = split_season(changed, 2020)
        pd.testing.assert_frame_equal(train, changed_train)


if __name__ == '__main__':
    unittest.main()
