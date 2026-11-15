"""Append a manually supplied pregame quote. Python standard library only."""
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ['quote_id', 'game_id', 'home_line', 'home_price_american', 'kickoff_utc',
          'sportsbook', 'captured_at_utc', 'recorded_at_utc', 'source', 'github_run_id']

def utc(value):
    dt = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Timestamp must include timezone, for example 2026-11-22T21:25:00Z')
    return dt.astimezone(timezone.utc)

def record(env, path, now=None):
    now = now or datetime.now(timezone.utc)
    game = env['GAME_ID'].strip()
    book = env['SPORTSBOOK'].strip()
    if not game or not book or any(c in game + book for c in '\r\n'):
        raise ValueError('Supply a game ID and sportsbook without line breaks')
    line = float(env['HOME_LINE'])
    price = int(env['HOME_PRICE_AMERICAN'])
    if not math.isfinite(line) or abs(line) > 100:
        raise ValueError('Home line must be finite and within +/-100')
    if abs(price) < 100:
        raise ValueError('American odds must be <= -100 or >= +100')
    captured = utc(env['CAPTURED_AT_UTC'])
    kickoff = utc(env['KICKOFF_UTC'])
    if captured > now:
        raise ValueError('Quote capture time cannot be in the future')
    if captured >= kickoff or now >= kickoff:
        raise ValueError('Both capture and recording must occur before kickoff')
    row = dict(game_id=game, home_line=str(line), home_price_american=str(price),
               kickoff_utc=kickoff.isoformat(), sportsbook=book,
               captured_at_utc=captured.isoformat(), recorded_at_utc=now.isoformat(),
               source='MANUAL_USER_SUPPLIED_UNVERIFIED', github_run_id=env.get('GITHUB_RUN_ID', 'local'))
    identity = {k: row[k] for k in ['game_id','home_line','home_price_american','kickoff_utc','sportsbook','captured_at_utc']}
    row['quote_id'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(path)
    if path.exists():
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise ValueError('Existing archive header differs; refusing to append')
            if any(r['quote_id'] == row['quote_id'] for r in reader):
                print('Identical quote already archived; no duplicate added')
                return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print('Saved pregame quote:', row['quote_id'])

if __name__ == '__main__':
    record(os.environ, 'data/market_quotes.csv')
