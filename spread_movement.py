"""A fixed directional filter for the existing one-contract exact-spread decision."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from statistics import median


def stamp(sid):
    return datetime.strptime(sid,'%Y%m%dT%H%M%S%fZ').replace(tzinfo=timezone.utc)


def read_events(folder):
    with gzip.open(folder/'odds.json.gz','rt') as f:
        return json.load(f)


def brand(q):
    key=q['bookmaker_key']
    return 'caesars' if key=='williamhill_us' else key


def probability(q,sign):
    def implied(odds):
        return Fraction(100,100+odds) if odds>0 else Fraction(-odds,100-odds)
    h=implied(q['home_price_american'])
    a=implied(q['away_price_american'])
    return (h if sign==1 else a)/(h+a)


def compare(old,new,sign):
    # Callers supply one exact line for one provider event; reject mismatched quote pairs too.
    def key(q):
        return (brand(q),q['provider_event_id'],q['home_team'],q['away_team'],q['kickoff_utc'],q['home_line'])
    prior={key(q):q for q in old}
    pairs=[(prior[key(q)],q) for q in new if key(q) in prior]
    before=[probability(a,sign) for a,b in pairs]
    after=[probability(b,sign) for a,b in pairs]
    up=sum(b>a for a,b in zip(before,after))
    down=sum(b<a for a,b in zip(before,after))
    delta=median(after)-median(before) if pairs else None
    status='insufficient_books' if len(pairs)<3 else 'comparable'
    direction=None
    if status=='comparable':
        if up>=3 and delta>0:
            direction='yes'
        elif down>=3 and delta<0:
            direction='no'
    return dict(status=status,direction=direction,common_books=len(pairs),up=up,down=down,
                median_change_pp=float(100*delta) if delta is not None else None,
                previous_quotes=[a for a,b in pairs],current_quote_ids=[b['quote_id'] for a,b in pairs])


def freeze(root,sid,entry,parse_lines):
    candidates=[p for p in (root/'snapshots').iterdir() if p.is_dir() and p.name<sid]
    evidence=dict(status='no_previous_snapshot',direction=None)
    if candidates:
        folder=max(candidates,key=lambda p:p.name)
        qs=entry['sportsbooks']
        try:
            events=[e for e in read_events(folder) if e['id']==qs[0]['provider_event_id']]
            lines=parse_lines(events[0],entry['forecast'],stamp(folder.name),folder.name) if len(events)==1 else {}
            evidence=compare(lines.get(qs[0]['home_line'],[]),qs,entry['sign'])
        except (OSError,ValueError,KeyError,EOFError):
            evidence=dict(status='previous_snapshot_unavailable',direction=None)
        evidence['previous_snapshot']=folder.name
    baseline=entry['arms']['sportsbooks']['decision']
    evidence['current_snapshot']=sid
    evidence['decision']=baseline if baseline==evidence['direction'] else 'pass'
    return evidence


def report(entries,closing):
    close={r['ticker']:r for r in closing['games']}
    coverage=Counter()
    rows=[]
    for entry in entries:
        evidence=entry.get('movement_v1')
        if evidence is None:
            coverage['pre_experiment']+=1
            continue
        coverage[evidence['status']]+=1
        mark=close[entry['ticker']]
        source=mark['arms']['sportsbooks']
        row=dict(game=entry['game'],ticker=entry['ticker'],status=evidence['status'],
                 close_status=mark['closing_quote']['status'])
        for arm,side in [('baseline',source['side']),('filtered',evidence['decision'])]:
            trade=side!='pass'
            net=source['net_paper_dollars']
            row[arm]=dict(side=side,net_dollars=(net if trade else 0) if net is not None else None,
                          advantage_cents=source.get('advantage_cents') if trade and row['close_status']=='usable' else None)
        rows.append(row)
    def summarize(group):
        out={'games':len(group)}
        for arm in ('baseline','filtered'):
            values=[r[arm] for r in group]
            scores=[v['advantage_cents'] for v in values if v['advantage_cents'] is not None]
            nets=[v['net_dollars'] for v in values if v['net_dollars'] is not None]
            out[arm]=dict(trades=sum(v['side']!='pass' for v in values),usable_closes=len(scores),
                          mean_advantage_cents=sum(scores)/len(scores) if scores else None,
                          beat_close=sum(x>0 for x in scores),settled_games=len(nets),
                          net_paper_dollars=sum(nets) if nets else None)
        return out
    return dict(coverage=dict(coverage),all_entries=summarize(rows),
                three_book_pairs=summarize([r for r in rows if r['status']=='comparable']),
                skipped_baseline_trades=summarize([r for r in rows if r['status']=='comparable'
                    and r['baseline']['side']!='pass' and r['filtered']['side']=='pass']),games=rows)
