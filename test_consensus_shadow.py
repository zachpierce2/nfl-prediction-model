import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import auto_market as auto
import consensus_shadow as cs


class ConsensusTest(unittest.TestCase):
    def test_same_snapshot_and_opposite_picks_at_original_prices(self):
        now=datetime.now(timezone.utc)
        kickoff=auto.stamp(now+timedelta(hours=2))
        stamp=auto.stamp(now-timedelta(minutes=1))
        home,away='Buffalo Bills','Miami Dolphins'
        def book(key,line,updated=stamp):
            return dict(key=key,title=key,markets=[dict(key='spreads',last_update=updated,outcomes=[
                dict(name=home,point=line,price=-115),dict(name=away,point=-line,price=105)])])
        event=dict(id='e',commence_time=kickoff,home_team=home,away_team=away,bookmakers=[
            book('draftkings',-3),book('fanduel',-3.5),book('caesars',-3.5),book('williamhill_us',-3.5),
            book('betmgm',-4),book('betrivers',-6,auto.stamp(now-timedelta(hours=1)))])
        q=dict(game_id='fixture',provider_event_id='e',quote_id='q',bookmaker_key='draftkings',sportsbook='draftkings',
               captured_at_utc=stamp,kickoff_utc=kickoff,snapshot_id='s',home_team='BUF',away_team='MIA',
               home_line=-3,away_line=3,home_price_american=-115,away_price_american=105)
        f=dict(quote_id='q',game_id='fixture',snapshot_id='s',forecast_created_at_utc=stamp,adjustment=-.2,predicted_home_margin=2.8)
        with tempfile.TemporaryDirectory() as d, patch.object(cs,'ROOT',Path(d)/'c'),patch.object(cs.epa_shadow,'ROOT',Path(d)/'e'):
            auto.archive_rows(cs.epa_shadow.ROOT/'forecasts.csv',list(f),[f],'game_id')
            self.assertEqual(set(cs.comparison(event,q)),{'fanduel','caesars','betmgm'})
            self.assertEqual(cs.record([dict(event,bookmakers=event['bookmakers'][:4])],[q],'s')['paired'],0)
            self.assertEqual(cs.record([event],[q],'s')['paired'],1)
            self.assertEqual(cs.grade([],now)['pending'],1)
            final=dict(provider_event_id='e',home_team='BUF',away_team='MIA',kickoff_utc=kickoff,home_score=24,away_score=20,snapshot_id='final')
            result=cs.grade([final],now+timedelta(days=1))
            self.assertEqual(result['changed'],1)
            self.assertEqual(result['primary']['LOSS'],1)
            self.assertEqual(result['consensus']['changed_wins'],1)
            self.assertAlmostEqual(result['consensus']['hypothetical_net_units'],100/115)
