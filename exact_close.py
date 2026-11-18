"""Closing midpoint and decision attribution for frozen exact-spread paper entries."""
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from urllib.error import HTTPError
from urllib.parse import urlencode
import json

import kalshi_paper as k

ARMS=('sportsbooks','sportsbooks_epa_blend')
GROUPS=('EPA adds','EPA skips','opposite sides','same trade','both pass')


def group_for(a,b):
    if a==b:
        return 'both pass' if a=='pass' else 'same trade'
    if a=='pass':
        return 'EPA adds'
    if b=='pass':
        return 'EPA skips'
    return 'opposite sides'


def closing_quote(bars,entered,kickoff):
    for bar in sorted(bars,key=lambda c:c['end_period_ts'],reverse=True):
        end=bar['end_period_ts']
        if not (entered<end<kickoff and kickoff-end<=300):
            continue
        try:
            sides=[bar[key] for key in ('yes_bid','yes_ask')]
            bid,ask=[Decimal(str(side['close_dollars'] if 'close_dollars' in side else side['close'])) for side in sides]
            if not (0<bid<=ask<1):
                continue
        except (InvalidOperation,KeyError,TypeError,ValueError):
            continue
        return dict(status='wide' if ask-bid>Decimal('.05') else 'usable',end_period_ts=end,
                    yes_bid=str(bid),yes_ask=str(ask),yes_midpoint=str((bid+ask)/2),
                    seconds_before_kickoff=kickoff-end)
    return dict(status='missing')


def get_close(entry,root,stamp):
    ticker=entry['ticker']
    if 'forecast' not in entry or 'kalshi_observed_at' not in entry:
        return dict(status='missing_entry_time')
    kick=datetime.fromisoformat(entry['forecast']['kickoff_utc']).timestamp()
    entered=datetime.fromisoformat(entry['kalshi_observed_at']).timestamp()
    if stamp<kick+60:
        return dict(status='pending')
    path=root/'closing_quotes'/(ticker+'.json')
    if path.exists():
        return json.loads(path.read_text())
    query=urlencode(dict(start_ts=int(kick)-300,end_ts=int(kick)-1,period_interval=1))
    for prefix in ('series/'+k.SERIES+'/markets','historical/markets'):
        route=f'{prefix}/{ticker}/candlesticks?{query}'
        try:
            data=k.get(route)
            if data.get('ticker')!=ticker:
                return dict(status='ticker_mismatch')
            close=closing_quote(data['candlesticks'],entered,kick)
        except HTTPError as exc:
            if exc.code==404:
                continue
            return dict(status='source_unavailable')
        except (OSError,KeyError,TypeError,ValueError):
            return dict(status='source_unavailable')
        if close['status']!='missing':
            close.update(ticker=ticker,source=route)
            k.save(path,close)
            return close
    # Allow the feed time to populate; cache confirmed absence after one day.
    close=dict(status='missing',ticker=ticker)
    if stamp>=kick+86400:
        k.save(path,close)
    return close


def report(entries,settlements,root):
    grades={r['ticker']:r['comparison'] for r in settlements}
    groups={name:[] for name in GROUPS}
    games=[]
    stamp=datetime.now(timezone.utc).timestamp()
    for entry in entries:
        ticker=entry['ticker']
        arms=entry['arms']
        label=group_for(*(arms[name]['decision'] for name in ARMS))
        close=get_close(entry,root,stamp)
        sides={}
        for name in ARMS:
            side=arms[name]['decision']
            row=dict(side=side,net_paper_dollars=grades[ticker][name]['net_dollars'] if ticker in grades else None)
            if side!='pass':
                offer=arms[name]['offers'][side]
                cost=Decimal(offer['ask'])+Decimal(offer['fee'])
                row.update(entry_cost=float(cost),available=offer['available'],p_yes=arms[name]['p_yes'])
                if close['status'] in ('usable','wide'):
                    mid=Decimal(close['yes_midpoint'])
                    advantage=(mid if side=='yes' else 1-mid)-cost
                    row['advantage_cents']=float(100*advantage)
            sides[name]=row
        item=dict(game=entry['game'],ticker=ticker,entry_time=entry.get('kalshi_observed_at'),
                  group=label,closing_quote=close,arms=sides)
        games.append(item)
        groups[label].append(item)
    summary={}
    for label,items in groups.items():
        stats=dict(games=len(items),closing_coverage=dict(Counter(r['closing_quote']['status'] for r in items)))
        for name in ARMS:
            arms=[r['arms'][name] for r in items]
            advantages=[r['arms'][name]['advantage_cents'] for r in items if
                        r['closing_quote']['status']=='usable' and r['arms'][name]['side']!='pass']
            nets=[r['net_paper_dollars'] for r in arms if r['net_paper_dollars'] is not None]
            stats[name]=dict(trades=sum(r['side']!='pass' for r in arms),usable_closes=len(advantages),
                             beat_close=sum(a>0 for a in advantages),
                             mean_advantage_cents=sum(advantages)/len(advantages) if advantages else None,
                             settled_games=len(nets),net_paper_dollars=sum(nets) if nets else None)
        summary[label]=stats
    return dict(groups=summary,games=games)
