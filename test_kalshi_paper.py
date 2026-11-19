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


class ExactSpreadTests(unittest.TestCase):
    def test_exact_home_and_away_and_never_round(self):
        import exact_spread as x
        event='KXNFLSPREAD-26NOV19INDHOU'
        def market(team,line):
            return dict(event_ticker=event,status='active',market_type='binary',strike_type='greater',
                        notional_value_dollars='1',floor_strike=line,rules_primary='example',
                        ticker=event+'-'+team+str(int(line+.5)))
        markets=[market('IND',3.5),market('HOU',3.5),market('HOU',4.5)]
        choice=x.choose({3.5:[1,2],-4.5:[3]},markets,'IND','HOU',event,-4.5)
        self.assertEqual((choice[0]['ticker'],choice[1]),(event+'-IND4',-1))
        self.assertEqual(x.choose({-3.5:[1]},markets,'IND','HOU',event,-3.5)[1],1)
        self.assertIsNone(x.choose({-4:[1],-2.5:[2]},markets,'IND','HOU',event,-4))

    def test_remove_vig_then_compare_full_cost(self):
        import exact_spread as x
        quotes=[dict(home_price_american=-120,away_price_american=100)]
        book={'no_dollars':[['.49','2']], 'yes_dollars':[['.47','1']]}
        arms=x.freeze(quotes,1,.60,book,Decimal('1'))
        fair=(120/220)/(120/220+.5)
        self.assertAlmostEqual(arms['sportsbooks']['p_yes'],fair)
        self.assertEqual(arms['sportsbooks']['decision'],'pass')
        self.assertEqual(arms['sportsbooks_epa_blend']['decision'],'yes')
        self.assertAlmostEqual(x.freeze(quotes,-1,.4,book,Decimal('1'))['sportsbooks']['p_yes'],1-fair)

    def test_exact_grading_counts_pass_and_cached_settlement(self):
        import exact_spread as x
        with tempfile.TemporaryDirectory() as d, patch.object(x,'ROOT',Path(d)):
            q=[dict(home_price_american=-120,away_price_american=100)]
            book={'no_dollars':[['.49','2']], 'yes_dollars':[['.47','1']]}
            arms=x.freeze(q,1,.60,book,Decimal('1'))
            k.save(x.ROOT/'decisions'/'A.json',dict(ticker='K',game='A',arms=arms))
            k.save(x.ROOT/'settlements'/'K.json',dict(ticker='K',status='finalized',result='yes'))
            x.grade()
            report=json.loads((x.ROOT/'report.json').read_text())
            self.assertEqual(report['games_with_positive_sportsbook_edge'],0)
            self.assertEqual(report['paired']['sportsbooks']['passes'],1)
            self.assertGreater(report['paired']['sportsbooks_epa_blend']['net_paper_dollars'],0)


class ClosingTests(unittest.TestCase):
    def test_strict_pregame_quote_and_wide_flag(self):
        import exact_close as c
        def bar(end,bid,ask):
            return dict(end_period_ts=end,yes_bid={'close_dollars':bid},yes_ask={'close_dollars':ask})
        bars=[bar(999,'.51','.53'),bar(1000,'.80','.81'),bar(998,'.81','.80')]
        self.assertEqual(c.closing_quote(bars,900,1000)['yes_midpoint'],'0.52')
        self.assertEqual(c.closing_quote(bars,999,1000)['status'],'missing')
        self.assertEqual(c.closing_quote([bar(699,'.4','.5')],500,1000)['status'],'missing')
        self.assertEqual(c.closing_quote([bar(990,'.4','.5')],500,1000)['status'],'wide')

    def test_side_costs_and_switch_groups(self):
        import exact_close as c
        self.assertEqual([c.group_for(a,b) for a,b in [('pass','yes'),('no','pass'),('pass','pass'),('yes','yes')]],
                         ['EPA adds','EPA skips','both pass','same trade'])
        arms={name:dict(decision=s,offers={s:dict(ask='.45',fee='.02',available='2')},p_yes=.5)
              for name,s in zip(c.ARMS,['yes','no'])}
        e=dict(ticker='X',game='A',arms=arms)
        grades=[dict(ticker='X',comparison={name:dict(net_dollars=v) for name,v in zip(c.ARMS,[.53,-.47])})]
        with patch.object(c,'get_close',return_value=dict(status='usable',yes_midpoint='.52')):
            g=c.report([e],grades,Path('.'))['groups']['opposite sides']
        self.assertEqual([g[n]['mean_advantage_cents'] for n in c.ARMS],[5,1])
        self.assertEqual(g[c.ARMS[1]]['net_paper_dollars'],-.47)
        with patch.object(c,'get_close',return_value=dict(status='wide',yes_midpoint='.52')):
            g=c.report([e],grades,Path('.'))['groups']['opposite sides']
        self.assertIsNone(g[c.ARMS[0]]['mean_advantage_cents'])
        self.assertEqual(g[c.ARMS[0]]['net_paper_dollars'],.53)


class MovementTests(unittest.TestCase):
    def test_three_book_move_and_orientation(self):
        import spread_movement as m
        def quotes(prices):
            return [dict(bookmaker_key=key,home_price_american=price,away_price_american=-110,
                         provider_event_id='1',home_team='HOU',away_team='IND',home_line=-3.5,
                         kickoff_utc='kick',quote_id=key) for key,price in prices]
        keys=['draftkings','caesars','fanduel']
        old=quotes(zip(keys,[-110]*3))
        new=quotes(zip(['draftkings','williamhill_us','fanduel'],[-120]*3))
        self.assertEqual(m.compare(old,new,1)['direction'],'yes')
        self.assertEqual(m.compare(old,new,-1)['direction'],'no')
        self.assertIsNone(m.compare(old,old,1)['direction'])
        new[-1]['home_line']=-4.5
        self.assertEqual(m.compare(old,new,1)['status'],'insufficient_books')
        new[-1]['home_line']=-3.5
        new[-1]['home_price_american']=-110
        self.assertIsNone(m.compare(old,new,1)['direction'])

    def test_settled_filtered_pass_and_missing_close(self):
        import spread_movement as m
        r=m.report([dict(game='A',ticker='X',movement_v1=dict(status='comparable',decision='pass')),
                    dict(game='B',ticker='Y')],
            {'games':[dict(ticker='X',arms={'sportsbooks':dict(side='no',net_paper_dollars=-.48)},
                           closing_quote={'status':'missing'})]})
        self.assertEqual(r['coverage'],{'comparable':1,'pre_experiment':1})
        self.assertEqual(r['three_book_pairs']['baseline']['net_paper_dollars'],-.48)
        self.assertEqual(r['three_book_pairs']['filtered']['net_paper_dollars'],0)
        self.assertIsNone(r['three_book_pairs']['baseline']['mean_advantage_cents'])
