import copy
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import auto_market as auto
import sportsbook_pricing as pricing


class PricingTests(unittest.TestCase):
    def test_price_and_push_math(self):
        r = pricing.price_side(.5, .1, -110)
        self.assertAlmostEqual(r['conditional_break_even'], 110/210)
        self.assertAlmostEqual(r['expected_net_per_unit'], .5*100/110-.4)
        self.assertAlmostEqual(r['conditional_model_cover'], .5/.9)
        self.assertEqual(pricing.payout(150), 1.5)
        self.assertEqual(pricing.price_side(0, 1, -110)['expected_net_per_unit'], 0)
        for price in [0, 99, -99, float('nan'), 110.5]:
            with self.assertRaises(ValueError): pricing.payout(price)
        for win,push in [(.8,.3),(-.1,0),(float('nan'),0)]:
            with self.assertRaises(ValueError): pricing.price_side(win,push,-110)

    def test_discrete_pushes_and_half_points(self):
        p = pricing.probabilities(3, -3, [-1,0,1])
        self.assertEqual(list(p.values()), [1/3,1/3,1/3])
        self.assertEqual(pricing.probabilities(3,-3.5,[-1,0,1])['push_probability'],0)
        self.assertEqual(pricing.probabilities(3,-2.5,[-1,0,1])['push_probability'],0)
        for residuals in [[], [float('nan')], [[1,2]]]:
            with self.assertRaises(ValueError): pricing.probabilities(3,-3,residuals)
        with self.assertRaises(ValueError): pricing.probabilities(3,-3.25,[0])

    def test_prior_seasons_and_dates_only(self):
        p = pricing.source_frame()
        errors,h = pricing.residual_history(p,2022,'2022-09-01T00:00:00Z','opponent_epa')
        self.assertTrue(h.season.lt(2022).all())
        modified = p.copy()
        modified.loc[modified.season.ge(2022),'actual_margin'] += 1000
        np.testing.assert_array_equal(errors, pricing.residual_history(modified,2022,'2022-09-01T00:00:00Z','opponent_epa')[0])
        with self.assertRaises(ValueError): pricing.residual_history(p,2020,'2020-09-01T00:00:00Z','opponent_epa')

    def fixtures(self):
        now = datetime(2026,11,17,12,2,tzinfo=timezone.utc)
        q = dict(game_id='2026_11_A_B', quote_id='q', snapshot_id='s', provider_event_id='event',
            home_team='B',away_team='A', bookmaker_key='draftkings',sportsbook='DK',
            home_line=-3,away_line=3,home_price_american=-110,away_price_american=-110,
            captured_at_utc='2026-11-17T12:00:00+00:00',provider_updated_at_utc='2026-11-17T11:59:00+00:00',
            kickoff_utc='2026-11-18T12:00:00+00:00')
        f = dict(q, forecast_created_at_utc='2026-11-17T12:01:00+00:00',predicted_home_margin=3.5,
                 model_version='opponent_epa_alpha1000_shrink025_cap2_v1')
        return q,f,now

    def test_same_line_price_ranking_and_complete_sides(self):
        q,f,now=self.fixtures();better=dict(q,quote_id='q2',bookmaker_key='fanduel',sportsbook='FD',home_price_american=100)
        different=dict(q,quote_id='q3',home_line=-3.5,away_line=3.5)
        rows=pricing.build_prices([q,better,different],[f],pricing.source_frame(),now)
        self.assertEqual(len(rows),6)
        home={r['quote_id']:r for r in rows if r['side']=='home'}
        self.assertEqual(home['q']['same_line_price_rank'],2)
        self.assertEqual(home['q2']['same_line_price_rank'],1)
        self.assertEqual(home['q3']['same_line_price_rank'],1)
        self.assertGreater(home['q2']['expected_net_per_unit'],home['q']['expected_net_per_unit'])
        for r in rows:
            self.assertAlmostEqual(sum(r[c] for c in ['home_win_probability','push_probability','away_win_probability']),1)

    def test_no_stale_pairing_or_invalid_identity(self):
        q,f,now=self.fixtures();p=pricing.source_frame()
        self.assertEqual(pricing.build_prices([dict(q,snapshot_id='old')],[f],p,now),[])
        self.assertEqual(pricing.build_prices([q],[],p,now),[])
        corruptions=[dict(q,home_team='C'),dict(q,away_line=4),
            dict(q,provider_updated_at_utc='2026-11-17T10:00:00Z'),
            dict(q,captured_at_utc='2026-11-17T12:02:00Z'),dict(q,home_price_american=99)]
        for bad in corruptions:
            with self.assertRaises(ValueError): pricing.build_prices([bad],[f],p,now)
        with self.assertRaises(ValueError): pricing.build_prices([q,q],[f],p,now)
        with self.assertRaises(ValueError): pricing.build_prices([q],[dict(f,model_version='different')],p,now)
        with self.assertRaises(ValueError): pricing.build_prices([q],[f],p,datetime(2026,11,19,tzinfo=timezone.utc))

    def test_all_book_capture_preserves_primary(self):
        now=datetime(2026,11,17,12,tzinfo=timezone.utc)
        schedule=pd.DataFrame([dict(game_id='g',home_team='KC',away_team='BUF',gameday='2026-11-18')])
        def book(key,price):
            return dict(key=key,title=key,markets=[dict(key='spreads',last_update='2026-11-17T11:59:00Z',
                outcomes=[dict(name='Kansas City Chiefs',point=-3,price=price),dict(name='Buffalo Bills',point=3,price=-110)])])
        event=dict(id='e',commence_time='2026-11-18T12:00:00Z',home_team='Kansas City Chiefs',away_team='Buffalo Bills',
                   bookmakers=[book('fanduel',100),book('draftkings',-110),book('unsupported',-110)])
        selected,_=auto.match_quotes([event],schedule,now,'s')
        all_quotes,_=auto.match_quotes([event],schedule,now,'s',all_books=True)
        self.assertEqual(len(all_quotes),2)
        reconstructed=list({q['game_id']:q for q in sorted(all_quotes,key=lambda q:auto.BOOK_ORDER.index(q['bookmaker_key']),reverse=True)}.values())
        self.assertEqual(selected,reconstructed)
        self.assertEqual(selected[0]['bookmaker_key'],'draftkings')
        self.assertEqual(len({r['quote_id'] for r in all_quotes}),2)

    def test_source_rejects_mismatched_controls(self):
        p=pd.read_csv(pricing.SOURCE)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.csv'
            p.iloc[1:].to_csv(path,index=False)
            with self.assertRaises(ValueError): pricing.source_frame(path)

    def test_prospective_grading_and_archive_corruption(self):
        import grade_market as grader
        import epa_shadow
        import csv
        q,f,now=self.fixtures()
        q={k:str(v) for k,v in q.items()};f={k:str(v) for k,v in f.items()}
        rows=pricing.build_prices([q],[f],pricing.source_frame(),now)
        def write(path, data):
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        final=dict(provider_event_id='event',home_team='B',away_team='A',kickoff_utc=q['kickoff_utc'],home_score=24,away_score=21)
        after=datetime(2026,11,19,tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d, patch.object(pricing,'ROOT',Path(d)/'pricing'), patch.object(auto,'ROOT',Path(d)/'auto'), patch.object(epa_shadow,'ROOT',Path(d)/'epa'):
            write(pricing.ROOT/'prices.csv',rows)
            write(auto.ROOT/'all_book_quotes.csv',[q]);write(epa_shadow.ROOT/'forecasts.csv',[f])
            pending=pricing.grade_prices(grader,[],now)
            self.assertEqual(pending['pending_games'],1)
            self.assertIsNone(pending['brier_three_outcome'])
            graded=pricing.grade_prices(grader,[final],after)
            self.assertEqual(graded['graded_quotes'],1)
            self.assertEqual(graded['graded_games'],1)
            self.assertAlmostEqual(graded['brier_three_outcome'],sum((rows[0][c]-int(c=='push_probability'))**2 for c in ['home_win_probability','push_probability','away_win_probability']))
            bad=copy.deepcopy(rows);bad[0]['line']=-4;write(pricing.ROOT/'prices.csv',bad)
            with self.assertRaises(ValueError): pricing.grade_prices(grader,[final],after)
            write(pricing.ROOT/'prices.csv',rows);write(epa_shadow.ROOT/'forecasts.csv',[dict(f,predicted_home_margin='100')])
            with self.assertRaises(ValueError): pricing.grade_prices(grader,[final],after)

    def test_full_walk_forward_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            p,s=pricing.evaluate(out=d)
            for method,g in p.groupby('method'):
                self.assertEqual(len(g),1424)
                self.assertFalse(g.game_id.duplicated().any())
                self.assertEqual(set(g.season),set(range(2021,2026)))
            self.assertEqual(set(p[p.method.eq('market')].game_id),set(p[p.method.eq('opponent_epa')].game_id))
            folds=pd.read_csv(Path(d)/'folds.csv')
            self.assertTrue((folds.history_last_season<folds.season).all())
            self.assertTrue(np.allclose(p[['home_win_probability','push_probability','away_win_probability']].sum(axis=1),1))


if __name__=='__main__': unittest.main()
