import unittest
from publish_primary import same_price_timing


class TimingTests(unittest.TestCase):
    def test_comparable_denominator_and_direction(self):
        def row(edge, start=-110, end=-110, status='NEAR_KICKOFF_PROXY'):
            return dict(status=status, initial_selected_price=start, latest_selected_price=end,
                        line_value_points=edge)
        result = same_price_timing([row(.5),row(-1),row(0,100,-100),row(3,-110,-115),
                                    row(2,'',-110),row(3,status='EARLY_LAST_OBSERVATION')])
        self.assertEqual(result,dict(better=1,equal=1,worse=1,changed_price=1,missing_price=1,
                                     comparable=3,primary_forecasts=6,mean_points=-.5/3))
        self.assertIsNone(same_price_timing([])['mean_points'])
