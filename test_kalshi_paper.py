import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
import kalshi_paper as k


class KalshiTests(unittest.TestCase):
    def test_prices_size_and_fee(self):
        book={'no_dollars':[['.52','10']], 'yes_dollars':[['.46','4']]}
        offers,side=k.offers(book,.51,Decimal('1'))
        self.assertEqual(side,'yes')
        self.assertEqual(Decimal(offers['yes']['ask']),Decimal('.48'))
        self.assertEqual(Decimal(offers['yes']['fee']),Decimal('.0175'))
        self.assertEqual(k.offers(book,.49,Decimal('1'))[1],'pass')
        book['no_dollars'][0][1]='.75'
        self.assertEqual(k.offers(book,.51,Decimal('1'))[1],'pass')

    def test_frozen_blend_and_matched_accounting(self):
        book={'yes_dollars':[['.48','5']], 'no_dollars':[['.48','5']]}
        arms=k.freeze_comparison(.56,book,Decimal('1'))
        self.assertAlmostEqual(arms['blend']['p_yes'],.53)
        self.assertEqual([arms[a]['decision'] for a in ('model','market','blend')],['yes','pass','pass'])
        scores=k.settle_comparison(arms,'no')
        report=k.summarize_comparison([dict(comparison=scores),dict(comparison=None)])
        self.assertEqual((report['comparable'],report['excluded_settlements']),(1,1))
        self.assertAlmostEqual(report['model']['net_paper_dollars'],-.5375)
        self.assertAlmostEqual(report['blend']['brier'],.53**2)
        self.assertEqual(report['market']['net_paper_dollars'],0)
        self.assertIsNone(k.freeze_comparison(.56,{'no_dollars':[['.48','5']]},Decimal('1')))

    def test_agreement_rule_and_legacy_settlements(self):
        for midpoint, model, expected in [(.60, .70, .65), (.40, .30, .35),
                                         (.60, .30, .60), (.40, .70, .40),
                                         (.50, .70, .50), (.60, .50, .60)]:
            book = {'yes_dollars': [[f'{midpoint-.01:.2f}', '5']],
                    'no_dollars': [[f'{1-midpoint-.01:.2f}', '5']]}
            arms = k.freeze_comparison(model, book, Decimal('1'))
            self.assertAlmostEqual(arms['agreement_v1']['p_yes'], expected)
        scores = k.settle_comparison(arms, 'no')
        legacy = {name: score for name, score in scores.items() if name != 'agreement_v1'}
        results = [dict(comparison=r) for r in (scores, legacy, None)]
        report = k.summarize_comparison(results, ('agreement_v1', 'blend'))
        self.assertEqual((report['comparable'], report['excluded_settlements']), (1, 2))
        self.assertAlmostEqual(report['agreement_v1']['brier'], .60**2)
        self.assertEqual(report['agreement_v1']['passes'], 1)
        self.assertEqual(report['blend']['trades'], 1)
        self.assertAlmostEqual(report['blend']['net_paper_dollars'], scores['blend']['net_dollars'])
        self.assertEqual(k.summarize_comparison(results)['comparable'], 2)

    def test_match_threshold_before_considering_prices(self):
        e='KXNFLSPREAD-26NOV19INDHOU'
        def market(team,line):
            return dict(event_ticker=e,status='active',market_type='binary',strike_type='greater',
                        notional_value_dollars='1',floor_strike=line,rules_primary='example',
                        ticker=e+'-'+team+str(int(line+.5)))
        markets=[market('HOU',3.5),market('IND',3.5),market('HOU',4.5)]
        r,sign=k.select({'home_line':3.5},markets,'IND','HOU',e)
        self.assertEqual((r['ticker'],sign),(e+'-IND4',-1))

    def test_pass_and_no_side_settlement(self):
        with tempfile.TemporaryDirectory() as d, patch.object(k,'ROOT',Path(d)):
            for game,side in [('a','no'),('b','pass')]:
                k.save(k.ROOT/'decisions'/(game+'.json'),dict(market={'ticker':'X'},decision=side,
                       offers=({'yes':{'ask':'.63'},'no':{'ask':'.41','fee':'.0170'}} if side=='pass' else {'no':{'ask':'.41','fee':'.0170'}}),p_yes=.4))
            with patch.object(k,'get',return_value={'market':dict(ticker='X',status='finalized',result='no')}):
                k.grade()
            r=json.loads((k.ROOT/'report.json').read_text())
            self.assertEqual((r['graded'],r['graded_passes'],r['net_paper_dollars']),(2,1,.573))
            self.assertAlmostEqual(r['mean_brier'],.16)
            bench=r['probability_benchmark']
            self.assertEqual((bench['comparable'],bench['excluded']),(1,1))
            self.assertAlmostEqual(bench['kalshi_midpoint_brier'],.61**2)
            self.assertAlmostEqual(bench['model_advantage'],.61**2-.4**2)
            self.assertIsNone(k.yes_midpoint({'yes':{'ask':'.40'},'no':{'ask':'.40'}}))
