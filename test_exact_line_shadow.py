import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import exact_line_shadow as shadow


class ExactLineTest(unittest.TestCase):
    def test_capture_and_grade_same_quote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            game = dict(game_id='future', snapshot_id='tick', captured_at_utc=(now-timedelta(minutes=2)).isoformat(),
                        forecast_created_at_utc=(now-timedelta(minutes=1)).isoformat(),
                        kickoff_utc=(now+timedelta(hours=1)).isoformat(), home_team='H', away_team='A',
                        provider_event_id='event', quote_id='quote', home_line=-3, predicted_home_margin=3.1)
            schedule = root/'schedule.csv'
            pd.DataFrame(dict(game_id=['a','b','c','d','future'], season=[2025]*4+[2026],
                              home_score=[2,2,3,15,None], away_score=[0,0,0,0,None],
                              spread_line=[3]*5)).to_csv(schedule,index=False)
            with patch.object(shadow,'ROOT',root/'output'), patch.object(shadow.total_shadow,'ROOT',root):
                pd.DataFrame([game]).to_csv(root/'pairs.csv',index=False)
                self.assertEqual(shadow.capture(schedule,root/'other')['saved'],0)
                self.assertEqual(shadow.capture(schedule,root/'tick')['saved'],1)
                self.assertEqual(shadow.capture(schedule,root/'tick')['saved'],0)
                pick = shadow.grader.read_rows(shadow.ROOT/'forecasts.csv')[0]
                self.assertEqual((pick['primary_side'],pick['exact_side']),('HOME','AWAY'))
                self.assertAlmostEqual(sum(float(pick[k]) for k in ['home_cover','push','away_cover']),1)
                for score, expected in [(4,('WIN','LOSS')),(3,('PUSH','PUSH')),(2,('LOSS','WIN'))]:
                    report = shadow.grade([dict(game,home_score=score,away_score=0)])
                    self.assertEqual(report['primary_on_changed'][expected[0]],1)
                    self.assertEqual(report['exact_on_changed'][expected[1]],1)
                pd.DataFrame([dict(game,game_id='future',kickoff_utc=game['captured_at_utc'])]).to_csv(root/'pairs.csv',index=False)
                with patch.object(shadow,'ROOT',root/'late'):
                    self.assertEqual(shadow.capture(schedule,root/'tick')['saved'],0)
