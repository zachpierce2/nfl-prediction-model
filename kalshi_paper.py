"""One hypothetical contract per primary forecast at the observed offer, held to settlement."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal as D, ROUND_CEILING
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from grade_market import read_rows

ROOT = Path('data/auto_market/kalshi_paper')
SERIES = 'KXNFLSPREAD'
BASE = 'https://api.elections.kalshi.com/trade-api/v2/'
SOURCE = Path('strength_epa_reference_predictions.csv')
# Common nflverse to Kalshi team symbols. Unmatched events are explicitly skipped.
ALIASES = {'JAX':['JAC','JAX'],'LAR':['LA','LAR'],'WAS':['WAS','WSH']}


def get(path):
    with urlopen(BASE+path, timeout=15) as response:
        return json.load(response)


def save(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def select(f,markets,away,home,event):
    target = -float(f['home_line'])
    candidates = []
    for m in markets:
        if (m.get('event_ticker')!=event or m.get('status')!='active' or m.get('market_type')!='binary'
                or m.get('strike_type')!='greater' or D(m['notional_value_dollars'])!=1):
            continue
        strike = float(m['floor_strike'])
        if strike<0 or strike%1!=.5 or not m.get('rules_primary'):
            continue
        for team,sign in [(home,1),(away,-1)]:
            if m['ticker']==f'{event}-{team}{int(strike+.5)}':
                candidates.append((abs(sign*strike-target),m['ticker'],m,sign))
    if candidates:
        _,_,market,sign=min(candidates,key=lambda x:x[:2])
        return market,sign
    return None


def offers(book, probability, multiplier):
    result = {}
    for side,reverse,prob in [('yes','no',probability),('no','yes',1-probability)]:
        levels = [(D(p),D(n)) for p,n in book.get(reverse+'_dollars',[]) if D(n)>0 and 0<D(p)<1]
        if not levels:
            continue
        bid,size=max(levels)
        price=1-bid
        fee=(D('.07')*multiplier*price*(1-price)).quantize(D('.0001'),rounding=ROUND_CEILING)
        result[side]=dict(ask=str(price),available=str(size),fee=str(fee),probability=prob,
                          net_expected_dollars=prob-float(price+fee))
    eligible=[s for s,v in result.items() if D(v['available'])>=1 and v['net_expected_dollars']>0]
    return result,max(eligible,key=lambda s:result[s]['net_expected_dollars']) if eligible else 'pass'


def observe(f,history):
    kickoff=datetime.fromisoformat(f['kickoff_utc'])
    day=kickoff.astimezone(ZoneInfo('America/New_York')).strftime('%y%b%d').upper()
    for away in ALIASES.get(f['away_team'],[f['away_team']]):
        for home in ALIASES.get(f['home_team'],[f['home_team']]):
            event=f'{SERIES}-{day}{away}{home}'
            page=get('markets?'+urlencode(dict(event_ticker=event,limit=1000)))
            if page.get('cursor'):
                raise ValueError('Incomplete market list')
            selected=select(f,page['markets'],away,home,event)
            if selected:
                break
        else:
            continue
        break
    else:
        return dict(status='NO_MATCHING_MARKET')
    market,sign=selected
    series=get('series/'+SERIES)['series']
    if series.get('fee_type') not in ['quadratic','quadratic_with_maker_fees'] or series['ticker']!=SERIES:
        raise ValueError('Unsupported fee schedule')
    multiplier=D(str(series['fee_multiplier']))
    if not multiplier.is_finite() or multiplier<0:
        raise ValueError('Invalid fee multiplier')
    season=int(f['game_id'].split('_')[0])
    prior=history[history.season.lt(season) & history.method.eq('opponent_epa')]
    if len(prior)<200 or prior.game_id.duplicated().any():
        raise ValueError('Invalid prior-season residual sample')
    margins=np.floor(float(f['predicted_home_margin'])+prior.actual_margin-prior.predicted_margin+.5)
    if not np.isfinite(margins).all():
        raise ValueError('Invalid margin data')
    prob=float((sign*margins>float(market['floor_strike'])).mean())
    book=get('markets/'+market['ticker']+'/orderbook?depth=1')['orderbook_fp']
    now=datetime.now(timezone.utc)
    created=datetime.fromisoformat(f['forecast_created_at_utc'])
    if not created<=now<kickoff:
        return dict(status='OUTSIDE_PREGAME_WINDOW')
    sides,decision=offers(book,prob,multiplier)
    comparison = freeze_comparison(prob, book, multiplier)
    if datetime.now(timezone.utc) >= kickoff:
        return dict(status='OUTSIDE_PREGAME_WINDOW')
    return dict(comparison=comparison,status='RECORDED',captured_at=now.isoformat(),forecast=f,market=market,
                book=book,fee_type=series['fee_type'],fee_multiplier=str(multiplier),
                residual_games=len(prior),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                p_yes=prob,offers=sides,decision=decision,contracts=int(decision!='pass'))


def capture():
    sid=json.loads(Path('data/auto_market/latest_run.json').read_text())['snapshot_id']
    forecasts=read_rows(Path('data/auto_market/primary/forecasts.csv'))
    status={}
    for f in forecasts:
        if f['snapshot_id']!=sid:
            continue
        path=ROOT/'decisions'/(f['game_id']+'.json')
        if path.exists() or datetime.now(timezone.utc)>=datetime.fromisoformat(f['kickoff_utc']):
            continue
        try:
            result=observe(f,pd.read_csv(SOURCE))
            status[f['game_id']]=result['status']
            if result['status']=='RECORDED':
                save(path,result)
        except Exception as exc:
            status[f['game_id']]='UNAVAILABLE: '+type(exc).__name__
    save(ROOT/'capture_status.json',dict(snapshot=sid,games=status))


def yes_midpoint(offers):
    """Recover the original Yes bid from the simultaneous No ask."""
    if not all(side in offers for side in ('yes','no')):
        return None
    ask = D(offers['yes']['ask'])
    bid = 1 - D(offers['no']['ask'])
    if not (0 < bid <= ask < 1):
        return None
    return float((bid + ask)/2)


def freeze_comparison(model_probability, book, multiplier):
    original, _ = offers(book, model_probability, multiplier)
    midpoint = yes_midpoint(original)
    if midpoint is None:
        return None
    forecasts = {'model': model_probability, 'market': midpoint,
                 'blend': (model_probability + midpoint) / 2}
    same_side = (model_probability - .5) * (midpoint - .5) > 0
    forecasts['agreement_v1'] = forecasts['blend'] if same_side else midpoint
    frozen = {}
    for name, probability in forecasts.items():
        prices, decision = offers(book, probability, multiplier)
        frozen[name] = dict(p_yes=probability, decision=decision, offers=prices)
    return frozen


def settle_comparison(frozen, result):
    scores = {}
    for name, arm in frozen.items():
        side = arm['decision']
        profit = D(0)
        if side != 'pass':
            offer = arm['offers'][side]
            profit = D(int(side == result)) - D(offer['ask']) - D(offer['fee'])
        scores[name] = dict(p_yes=arm['p_yes'], decision=side, net_dollars=float(profit),
                            brier=(arm['p_yes'] - int(result == 'yes')) ** 2)
    return scores


def summarize_comparison(results, names=('model', 'market', 'blend')):
    comparable = [r['comparison'] for r in results if r['comparison'] and all(name in r['comparison'] for name in names)]
    summary = dict(comparable=len(comparable), excluded_settlements=len(results)-len(comparable))
    for name in names:
        arms = [r[name] for r in comparable]
        trades = sum(a['decision'] != 'pass' for a in arms)
        summary[name] = dict(trades=trades, passes=len(arms)-trades,
                             net_paper_dollars=sum(a['net_dollars'] for a in arms) if arms else None,
                             brier=sum(a['brier'] for a in arms)/len(arms) if arms else None)
    return summary


def grade():
    results=[]
    for path in sorted((ROOT/'decisions').glob('*.json')):
        entry=json.loads(path.read_text())
        ticker=entry['market']['ticker']
        settled=ROOT/'settlements'/(ticker+'.json')
        if settled.exists():
            final=json.loads(settled.read_text())
        else:
            try:
                final=get('markets/'+ticker)['market']
                if final['ticker']!=ticker or final.get('status')!='finalized':
                    continue
                if final.get('result') not in ['yes','no']:
                    continue
                save(settled,final)
            except (OSError,ValueError,KeyError):
                continue
        won=final['result']=='yes'
        side=entry['decision']
        profit=D(0)
        if side!='pass':
            offer=entry['offers'][side]
            profit=D(int(side==final['result']))-D(offer['ask'])-D(offer['fee'])
        midpoint = yes_midpoint(entry['offers'])
        comparison = settle_comparison(entry['comparison'], final['result']) if entry.get('comparison') else None
        results.append(dict(comparison=comparison,model_yes_probability=entry['p_yes'],kalshi_yes_midpoint=midpoint,
                            midpoint_brier=(midpoint-int(won))**2 if midpoint is not None else None,
                            game=path.stem,ticker=ticker,decision=side,result=final['result'],
                            net_dollars=float(profit),brier=(entry['p_yes']-int(won))**2))
    recorded=len(list((ROOT/'decisions').glob('*.json')))
    trades=[r for r in results if r['decision']!='pass']
    paired = [r for r in results if r['midpoint_brier'] is not None]
    model_score = sum(r['brier'] for r in paired)/len(paired) if paired else None
    market_score = sum(r['midpoint_brier'] for r in paired)/len(paired) if paired else None
    benchmark = dict(comparable=len(paired), excluded=len(results)-len(paired),
                     model_brier=model_score, kalshi_midpoint_brier=market_score,
                     model_advantage=market_score-model_score if paired else None)
    save(ROOT/'report.json',dict(agreement_vs_blend=summarize_comparison(results, ('agreement_v1', 'blend')),
        three_way=summarize_comparison(results),probability_benchmark=benchmark,recorded=recorded,graded=len(results),awaiting_binary_settlement=recorded-len(results),
        graded_paper_trades=len(trades),graded_passes=len(results)-len(trades),
        net_paper_dollars=sum(r['net_dollars'] for r in trades) if trades else None,
        mean_brier=float(np.mean([r['brier'] for r in results])) if results else None, games=results))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['capture','grade'])
    import exact_spread
    if parser.parse_args().mode=='capture':
        capture()
        exact_spread.capture()
    else:
        grade()
        exact_spread.grade()
