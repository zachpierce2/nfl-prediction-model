"""Capture the original sportsbook within 55 minutes of a published EPA pick's kickoff."""
import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path('data/auto_market')
ARCHIVE = ROOT/'near_kickoff'


def rows(path):
    if path.exists():
        with path.open(newline='') as src:
            return list(csv.DictReader(src))
    return []


def due(now):
    anchors = {r['quote_id']: r for r in rows(ROOT/'quotes.csv')}
    observed = {r['game_id'] for r in rows(ARCHIVE/'quotes.csv')}
    picks = rows(ROOT/'primary/forecasts.csv')
    return [(p,anchors[p['quote_id']]) for p in picks
            if p['game_id'] not in observed and p['quote_id'] in anchors
            and datetime.fromisoformat(p['forecast_created_at_utc']) <= now
            and timedelta(0) < datetime.fromisoformat(p['kickoff_utc'])-now <= timedelta(minutes=55)]


def capture():
    import auto_market as auto
    import pandas as pd
    pending = due(datetime.now(timezone.utc))
    if not pending:
        return
    params = urlencode(dict(apiKey=os.environ['ODDS_API_KEY'],regions='us',markets='spreads',
                            oddsFormat='american',dateFormat='iso'))
    raw,_ = auto.download('https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?'+params,'Near-kickoff spread')
    observed = datetime.now(timezone.utc)
    events = {e['id']:e for e in json.loads(raw)}
    sid = observed.strftime('%Y%m%dT%H%M%S%fZ')
    result = []
    for p,a in pending:
        event = events.get(a['provider_event_id'])
        kickoff = auto.utc(p['kickoff_utc'])
        if not event or observed >= kickoff or abs(auto.utc(event['commence_time'])-kickoff)>timedelta(minutes=5):
            continue
        event = dict(event,bookmakers=[b for b in event['bookmakers'] if b['key']==a['bookmaker_key']])
        schedule = pd.DataFrame([dict(game_id=p['game_id'],home_team=p['home_team'],away_team=p['away_team'],
                                      gameday=pd.Timestamp(kickoff).tz_convert('America/New_York').date().isoformat())])
        quotes,_ = auto.match_quotes([event],schedule,observed,sid)
        for q in quotes:
            if q['sportsbook']!=p['sportsbook']:
                continue
            # An identical still-fresh offer observed later is a distinct observation.
            q['quote_id']=auto.digest((q['quote_id']+sid).encode())[:24]
            result.append(q)
    auto.archive_rows(ARCHIVE/'quotes.csv',auto.QUOTE_FIELDS,result,'game_id')
    print(f'{len(result)} same-book quotes captured for {len(pending)} eligible games')


if __name__=='__main__':
    import sys
    if '--check' in sys.argv:
        print('yes' if due(datetime.now(timezone.utc)) else 'no')
    else:
        capture()
