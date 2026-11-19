"""Paper comparison of identical sportsbook and Kalshi NFL spread contracts."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import gzip
import json
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import auto_market as auto
import kalshi_paper as k
import exact_close
import spread_movement

ROOT = k.ROOT/'exact_spread'
ARMS = ('sportsbooks','sportsbooks_epa_blend')


def sportsbook_lines(event, forecast, captured, snapshot):
    kickoff = auto.utc(forecast['kickoff_utc'])
    if auto.utc(event['commence_time']) != kickoff or any(
        auto.NAMES.get(event[side]) != forecast[side] for side in ('home_team','away_team')):
        return {}
    schedule = pd.DataFrame([dict(game_id=forecast['game_id'],home_team=forecast['home_team'],
        away_team=forecast['away_team'],gameday=kickoff.astimezone(ZoneInfo('America/New_York')).date().isoformat())])
    by_brand = {}
    for book in event['bookmakers']:
        rows,_ = auto.match_quotes([dict(event,bookmakers=[book])],schedule,captured,snapshot)
        if not rows:
            continue
        q=rows[0]
        brand='caesars' if q['bookmaker_key']=='williamhill_us' else q['bookmaker_key']
        if brand not in by_brand or q['provider_updated_at_utc']>by_brand[brand]['provider_updated_at_utc']:
            by_brand[brand]=q
    lines=defaultdict(list)
    for q in by_brand.values():
        if abs(q['home_line'])%1==.5:
            lines[q['home_line']].append(q)
    return lines


def choose(lines,markets,away,home,event,anchor):
    eligible=[]
    for line,quotes in lines.items():
        choice=k.select({'home_line':line},markets,away,home,event)
        if choice:
            m,sign=choice
            if -sign*float(m['floor_strike'])==line:
                eligible.append((-len(quotes),abs(line-anchor),m['ticker'],m,sign,quotes))
    return min(eligible,key=lambda v:v[:3])[3:] if eligible else None


def implied(odds):
    return 100/(100+odds) if odds>0 else -odds/(100-odds)


def freeze(quotes,sign,model_probability,book,multiplier):
    probs=[]
    for q in quotes:
        h,a=[implied(q[s+'_price_american']) for s in ('home','away')]
        probs.append((h if sign==1 else a)/(h+a))
    fair=median(probs)
    arms={}
    for name,p in zip(ARMS,[fair,(fair+model_probability)/2]):
        offers,decision=k.offers(book,p,multiplier)
        arms[name]=dict(p_yes=p,offers=offers,decision=decision)
    return arms


def observe(f,lines,history,captured):
    kick=auto.utc(f['kickoff_utc'])
    date=kick.astimezone(ZoneInfo('America/New_York')).strftime('%y%b%d').upper()
    for away in k.ALIASES.get(f['away_team'],[f['away_team']]):
        for home in k.ALIASES.get(f['home_team'],[f['home_team']]):
            event=f'{k.SERIES}-{date}{away}{home}'
            page=k.get(f'markets?event_ticker={event}&limit=1000')
            if page.get('cursor'):
                raise ValueError('Incomplete Kalshi event')
            choice=choose(lines,page['markets'],away,home,event,float(f['home_line']))
            if choice:
                break
        else:
            continue
        break
    else:
        return 'NO_IDENTICAL_CONTRACT',None
    market,sign,quotes=choice
    series=k.get('series/'+k.SERIES)['series']
    details=k.get('events/'+event)['event']
    if details['event_ticker']!=event or details['series_ticker']!=k.SERIES or series['ticker']!=k.SERIES:
        raise ValueError('Unexpected fee source')
    fee_type=details.get('fee_type_override') or series['fee_type']
    override=details.get('fee_multiplier_override')
    multiplier=Decimal(str(series['fee_multiplier'] if override is None else override))
    if fee_type not in ('quadratic','quadratic_with_maker_fees') or not multiplier.is_finite() or multiplier<0:
        return 'UNSUPPORTED_FEE',None
    book=k.get('markets/'+market['ticker']+'/orderbook?depth=1')['orderbook_fp']
    now=datetime.now(timezone.utc)
    if not captured<=now<kick or now-captured>timedelta(minutes=5) or auto.utc(f['forecast_created_at_utc'])>now:
        return 'OUTSIDE_CAPTURE_WINDOW',None
    quotes=[q for q in quotes if now-auto.utc(q['provider_updated_at_utc'])<=timedelta(minutes=30)]
    if not quotes:
        return 'STALE_BOOKS',None
    prior=history[(history.season<int(f['game_id'][:4])) & history.method.eq('opponent_epa')]
    if len(prior)<200 or prior.game_id.duplicated().any():
        raise ValueError('Invalid EPA residual history')
    margins=np.floor(float(f['predicted_home_margin'])+(prior.actual_margin-prior.predicted_margin)+.5)
    if not np.isfinite(margins).all():
        raise ValueError('Invalid margin')
    model=float((sign*margins>float(market['floor_strike'])).mean())
    arms=freeze(quotes,sign,model,book,multiplier)
    offers=arms[ARMS[0]]['offers']
    if k.yes_midpoint(offers) is None or not any(Decimal(x['available'])>=1 for x in offers.values()):
        return 'NO_USABLE_BOOK',None
    return 'RECORDED',dict(game=f['game_id'],ticker=market['ticker'],forecast=f,market=market,
        sign=sign,sportsbooks=quotes,kalshi_book=book,kalshi_observed_at=now.isoformat(),
        fee_type=fee_type,fee_multiplier=str(multiplier),model_yes_probability=model,arms=arms)


def capture():
    run=json.loads((auto.ROOT/'latest_run.json').read_text())
    sid=run['snapshot_id']
    captured=auto.utc(run['captured_at_utc'])
    stamp=datetime.now(timezone.utc)
    status={}
    history=None
    with gzip.open(auto.ROOT/'snapshots'/sid/'odds.json.gz','rt') as file:
        events=json.load(file)
    for f in k.read_rows(auto.ROOT/'primary/forecasts.csv'):
        dest=ROOT/'decisions'/(f['game_id']+'.json')
        kick=auto.utc(f['kickoff_utc'])
        if dest.exists() or not stamp<kick<=stamp+timedelta(hours=24):
            continue
        try:
            candidates=[e for e in events if auto.NAMES.get(e['home_team'])==f['home_team']
                        and auto.NAMES.get(e['away_team'])==f['away_team'] and auto.utc(e['commence_time'])==kick]
            lines=sportsbook_lines(candidates[0],f,captured,sid) if len(candidates)==1 else {}
            if not lines or stamp-captured>timedelta(minutes=5):
                status[f['game_id']]='NO_FRESH_EXACT_LINE_CANDIDATE'
                continue
            if history is None:
                history=pd.read_csv(k.SOURCE)
            reason,record=observe(f,lines,history,captured)
            status[f['game_id']]=reason
            if record:
                record['movement_v1']=spread_movement.freeze(auto.ROOT,sid,record,sportsbook_lines)
                k.save(dest,record)
        except (OSError,ValueError,KeyError,ArithmeticError) as exc:
            status[f['game_id']]='UNAVAILABLE: '+type(exc).__name__
    k.save(ROOT/'capture_status.json',dict(snapshot=sid,games=status))


def grade():
    results=[]
    entries=[json.loads(p.read_text()) for p in sorted((ROOT/'decisions').glob('*.json'))]
    for entry in entries:
        ticker=entry['ticker']
        path=ROOT/'settlements'/(ticker+'.json')
        if path.exists():
            market=json.loads(path.read_text())
        else:
            try:
                market=k.get('markets/'+ticker)['market']
            except (OSError,ValueError,KeyError):
                continue
            if market.get('ticker')!=ticker or market.get('status')!='finalized' or market.get('result') not in ('yes','no'):
                continue
            k.save(path,market)
        results.append(dict(game=entry['game'],ticker=ticker,comparison=k.settle_comparison(entry['arms'],market['result'])))
    cheaper=sum(e['arms']['sportsbooks']['decision']!='pass' for e in entries)
    closing=exact_close.report(entries,results,ROOT)
    k.save(ROOT/'closing_report.json',closing)
    k.save(ROOT/'movement_report.json',spread_movement.report(entries,closing))
    k.save(ROOT/'report.json',dict(matched_games=len(entries),games_with_positive_sportsbook_edge=cheaper,
           awaiting_settlement=len(entries)-len(results),paired=k.summarize_comparison(results,ARMS),games=results))
