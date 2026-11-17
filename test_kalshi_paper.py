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
                       offers={'no':{'ask':'.41','fee':'.0170'}},p_yes=.4))
            with patch.object(k,'get',return_value={'market':dict(ticker='X',status='finalized',result='no')}):
                k.grade()
            r=json.loads((k.ROOT/'report.json').read_text())
            self.assertEqual((r['graded'],r['graded_passes'],r['net_paper_dollars']),(2,1,.573))
            self.assertAlmostEqual(r['mean_brier'],.16)
