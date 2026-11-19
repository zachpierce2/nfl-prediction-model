"""Replay coverage of identical-line, consecutive sportsbook snapshots; no exchange fills."""
from collections import Counter
from datetime import timedelta
from pathlib import Path
import json

import auto_market as auto
from exact_spread import sportsbook_lines
from spread_movement import compare,read_events,stamp


def audit(root):
    previous={}
    metrics=Counter()
    distinct=set()
    for folder in sorted(root.iterdir()):
        at=stamp(folder.name)
        current={}
        for event in read_events(folder):
            info=dict(game_id=event['id'],kickoff_utc=event['commence_time'],
                      home_team=auto.NAMES.get(event['home_team']),away_team=auto.NAMES.get(event['away_team']))
            for line,quotes in sportsbook_lines(event,info,at,folder.name).items():
                key=(event['id'],line)
                current[key]=quotes
                if key not in previous:
                    continue
                evidence=compare(previous[key],quotes,1)
                metrics['repeated_event_lines']+=1
                if evidence['status']=='comparable':
                    metrics['three_books']+=1
                    distinct.add(event['id'])
                    near=auto.utc(event['commence_time'])-at<=timedelta(hours=24)
                    metrics['three_books_within_24h']+=int(near)
                    if evidence['direction']:
                        metrics['directional_signals']+=1
                        metrics['directional_signals_within_24h']+=int(near)
        previous=current
        metrics['snapshots']+=1
    return dict(metrics,distinct_comparable_events=len(distinct))


if __name__=='__main__':
    print(json.dumps(audit(Path('data/auto_market/snapshots')),indent=2))
