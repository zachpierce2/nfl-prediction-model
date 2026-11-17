"""Same-snapshot leave-one-book-out research comparison."""
import json
from datetime import datetime, timezone
from statistics import median
from pathlib import Path
import pandas as pd
import auto_market as auto
import epa_shadow
import grade_market as grader

ROOT = auto.ROOT/'consensus_shadow'


def brand(key):
    return 'caesars' if key == 'williamhill_us' else key


def comparison(event, anchor):
    captured = auto.utc(anchor['captured_at_utc'])
    if event['id'] != anchor['provider_event_id'] or auto.utc(event['commence_time']) != auto.utc(anchor['kickoff_utc']):
        raise ValueError('Consensus event mismatch')
    date = pd.Timestamp(anchor['kickoff_utc']).tz_convert('America/New_York').date().isoformat()
    schedule = pd.DataFrame([dict(game_id=anchor['game_id'],home_team=anchor['home_team'],
                                  away_team=anchor['away_team'],gameday=date)])
    peers = {}
    for book in event['bookmakers']:
        key = brand(book['key'])
        if key == brand(anchor['bookmaker_key']):
            continue
        quotes,_ = auto.match_quotes([dict(event,bookmakers=[book])],schedule,captured,anchor['snapshot_id'])
        if quotes:
            q = quotes[0]
            previous = peers.get(key)
            if previous is None or q['provider_updated_at_utc'] > previous['provider_updated_at_utc']:
                peers[key] = q
    return peers


def record(events, quotes, snapshot):
    forecasts = {r['quote_id']:r for r in grader.read_rows(epa_shadow.ROOT/'forecasts.csv') if r['snapshot_id']==snapshot}
    events = {e['id']:e for e in events}
    pairs = []
    for q in quotes:
        f = forecasts.get(q['quote_id'])
        if f is None:
            continue
        peers = comparison(events[q['provider_event_id']],q)
        now = datetime.now(timezone.utc)
        if len(peers)<3 or not auto.utc(q['captured_at_utc'])<=auto.utc(f['forecast_created_at_utc'])<=now<auto.utc(q['kickoff_utc']):
            continue
        line = median(float(p['home_line']) for p in peers.values())
        pairs.append(dict(q,peer_quotes=json.dumps(peers,sort_keys=True),peer_median_home_line=line,
                          baseline_margin=f['predicted_home_margin'],predicted_home_margin=-line+float(f['adjustment']),
                          forecast_created_at_utc=auto.stamp(now)))
    if pairs:
        auto.archive_rows(ROOT/'pairs.csv',list(pairs[0]),pairs,'game_id')
    return dict(primary_forecasts=len(forecasts),paired=len(pairs))


def grade(finals,now):
    pairs = grader.read_rows(ROOT/'pairs.csv')
    results = []
    for r in pairs:
        final = grader.match(r,finals,[r])
        if final is None:
            continue
        row = dict(game_id=r['game_id'])
        for name, margin in [('primary',r['baseline_margin']),('consensus',r['predicted_home_margin'])]:
            g = grader.grade(dict(r,predicted_home_margin=margin),final,now)
            side = g['selected_side'].lower()
            status = g['ats_result']
            price = float(r[side+'_price_american']) if side!='none' else None
            units = (price/100 if price>0 else -100/price) if status=='WIN' else -1 if status=='LOSS' else 0
            row.update({name+'_side':side,name+'_grade':status,name+'_price':price,name+'_units':units})
        row['changed']=row['primary_side']!=row['consensus_side']
        results.append(row)
    ROOT.mkdir(parents=True,exist_ok=True)
    summary=dict(paired=len(pairs),graded=len(results),pending=len(pairs)-len(results),changed=sum(r['changed'] for r in results))
    for name in ['primary','consensus']:
        summary[name]={s:sum(r[name+'_grade']==s for r in results) for s in ['WIN','LOSS','PUSH','NO_LEAN']}
        summary[name]['changed_wins']=sum(r['changed'] and r[name+'_grade']=='WIN' for r in results)
        summary[name]['hypothetical_net_units']=sum(r[name+'_units'] for r in results) if results else None
    pd.DataFrame(results,columns=['game_id','primary_side','primary_grade','primary_price','primary_units','consensus_side',
                                 'consensus_grade','consensus_price','consensus_units','changed']).to_csv(ROOT/'grades.csv',index=False)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
