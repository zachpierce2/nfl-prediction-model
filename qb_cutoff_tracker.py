"""Archive observed starters and public Kalshi books in pre-cutoff windows."""
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import sys
from urllib.request import urlopen
from zoneinfo import ZoneInfo

ROOT=Path('data/auto_market/qb_cutoffs')
SCHEDULE='https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
CUTS=(1440,360,90)
WINDOW=timedelta(minutes=35)


def now():
    return datetime.now(timezone.utc)


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def kickoff(g):
    local=datetime.fromisoformat(g['gameday']+'T'+g['gametime'])
    return local.replace(tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc)


def path_for(g):
    return ROOT/'games'/g['game_id']/kickoff(g).strftime('%Y%m%dT%H%MZ')


def targets(games,stamp):
    for g in games:
        for minutes in CUTS:
            cutoff=kickoff(g)-timedelta(minutes=minutes)
            if cutoff-WINDOW <= stamp <= cutoff:
                yield g,minutes,cutoff


def quote(game,ticker):
    import kalshi_paper as k
    home,away=[{'LA':'LAR'}.get(game[s],game[s]) for s in ('home_team','away_team')]
    if ticker:
        market=k.get('markets/'+ticker)['market']
        if market['ticker']!=ticker or market['status']!='active':
            return {'status':'LOCKED_CONTRACT_UNAVAILABLE','ticker':ticker}
    else:
        day=kickoff(game).astimezone(ZoneInfo('America/New_York')).strftime('%y%b%d').upper()
        for a in k.ALIASES.get(away,[away]):
            for h in k.ALIASES.get(home,[home]):
                event=f'{k.SERIES}-{day}{a}{h}'
                page=k.get('markets?event_ticker='+event+'&limit=1000')
                if page.get('cursor'):
                    raise ValueError('Incomplete event page')
                chosen=k.select({'home_line':-float(game['spread_line'] or 0)},page['markets'],a,h,event)
                if chosen:
                    market,_=chosen
                    break
            else:
                continue
            break
        else:
            return {'status':'NO_MATCHING_MARKET'}
    series=k.get('series/'+k.SERIES)['series']
    if series['ticker']!=k.SERIES or series['fee_type'] not in ('quadratic','quadratic_with_maker_fees'):
        raise ValueError('Unsupported series fee')
    multiplier=Decimal(str(series['fee_multiplier']))
    if not multiplier.is_finite() or multiplier<0:
        raise ValueError('Invalid fee multiplier')
    book=k.get('markets/'+market['ticker']+'/orderbook?depth=1')['orderbook_fp']
    prices,_=k.offers(book,.5,multiplier)
    offers={s:{field:p[field] for field in ('ask','available','fee')} for s,p in prices.items()}
    return dict(status='OBSERVED',ticker=market['ticker'],market=market,book=book,offers=offers,
                fee_type=series['fee_type'],fee_multiplier=str(multiplier))


def capture(games):
    import pandas as pd
    import qb_expected
    path,report=qb_expected.collect(Path('qb_cutoff_evidence')/now().strftime('%Y%m%dT%H%M%S%fZ'))
    starters=pd.read_csv(path,dtype=str,keep_default_na=False)
    for g,m,cut in targets(games,now()):
        folder=path_for(g)
        dest=folder/f'{m}.json'
        old=json.loads(dest.read_text()) if dest.exists() else None
        prior=sorted(folder.glob('*.json'))
        ticker=next((q['ticker'] for f in prior if (q:=json.loads(f.read_text())['quote']).get('status')=='OBSERVED'),None)
        selection={}
        for side in ('home','away'):
            team={'LA':'LAR'}.get(g[side+'_team'],g[side+'_team'])
            evidence=starters[(starters.game_id==g['game_id']) & (starters.team==team)]
            qid=qb_expected.resolve(evidence,g['game_id'],team,cut.isoformat(),kickoff(g).isoformat())
            selection[side]=dict(qb_id=qid,evidence=evidence.to_dict('records'))
        try:
            book=quote(g,ticker)
        except (OSError,KeyError,ValueError,ArithmeticError) as exc:
            book=dict(status='QUOTE_UNAVAILABLE',reason=type(exc).__name__)
        stamp=now()
        if stamp>cut:
            continue
        # Keep the most recent same-window pair; preserve a valid book if a retry fails.
        if old and old['quote'].get('offers') and not book.get('offers'):
            continue
        save(dest,dict(game=g['game_id'],kickoff=kickoff(g).isoformat(),minutes=m,cutoff=cut.isoformat(),
                       observed=stamp.isoformat(),qbs=selection,quote=book))


def grade(games,stamp):
    startpath=ROOT/'start.json'
    if not startpath.exists():
        save(startpath,{'observed':stamp.isoformat()})
    start=datetime.fromisoformat(json.loads(startpath.read_text())['observed'])
    summary={str(m):dict(due=0,captured=0,quoted=0,graded_teams=0,named=0,correct=0) for m in CUTS}
    paired=[]
    for g in games:
        books={}
        observations={}
        for m in CUTS:
            cut=kickoff(g)-timedelta(minutes=m)
            if not start<=cut<=stamp:
                continue
            s=summary[str(m)]
            s['due']+=1
            path=path_for(g)/f'{m}.json'
            if not path.exists():
                continue
            r=json.loads(path.read_text())
            s['captured']+=1
            observations[m]=r
            q=r['quote']
            if q.get('offers'):
                s['quoted']+=1
                books[m]=q
            if g['home_score'] and g['away_score'] and stamp>kickoff(g)+timedelta(hours=6):
                for side in ('home','away'):
                    actual=g[side+'_qb_id']
                    if actual:
                        s['graded_teams']+=1
                        named=r['qbs'][side]['qb_id']
                        s['named']+=int(bool(named))
                        s['correct']+=int(named==actual)
        for early,late in [(1440,360),(360,90),(1440,90)]:
            if early not in books or late not in books or books[early]['ticker']!=books[late]['ticker']:
                continue
            for side in ('yes','no'):
                a,b=[books[m]['offers'].get(side) for m in (early,late)]
                if a and b and min(Decimal(a['available']),Decimal(b['available']))>=1:
                    costs=[Decimal(v['ask'])+Decimal(v['fee']) for v in (a,b)]
                    paired.append(dict(game=g['game_id'],ticker=books[early]['ticker'],side=side,
                                       earlier=early,later=late,earlier_cost=str(costs[0]),later_cost=str(costs[1]),
                                       cost_change=str(costs[1]-costs[0]),
                                       starters={str(m):{s:observations[m]['qbs'][s]['qb_id'] for s in ('home','away')}
                                                 for m in (early,late)}))
    save(ROOT/'report.json',dict(cutoffs=summary,paired_one_contract_costs=paired))


def main():
    with urlopen(SCHEDULE,timeout=30) as resp:
        games=[g for g in csv.DictReader(io.StringIO(resp.read().decode())) if g['game_type']=='REG' and g['gametime']]
    if sys.argv[1]=='gate':
        active=any(targets(games,now()))
        with open(os.environ['GITHUB_OUTPUT'],'a') as out:
            print('active='+str(active).lower(),file=out)
        grade(games,now())
    else:
        capture(games)
        grade(games,now())


if __name__=='__main__':
    main()
