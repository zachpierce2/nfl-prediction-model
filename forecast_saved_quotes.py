"""Save prospective market-model forecasts for manually archived quotes."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import market_model as model

OUT = Path('data/market_forecasts.csv')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def select_quotes(q, now):
    required = {'quote_id','game_id','home_line','home_price_american','sportsbook',
                'captured_at_utc','recorded_at_utc','kickoff_utc','source'}
    if required - set(q):
        raise ValueError(f'Missing quote columns: {sorted(required-set(q))}')
    q = q.copy()
    for c in ['captured_at_utc','recorded_at_utc','kickoff_utc']:
        q[c] = pd.to_datetime(q[c], utc=True, errors='raise')
        if q[c].isna().any():
            raise ValueError(f'Missing {c}')
    if q.quote_id.isna().any() or q.quote_id.duplicated().any():
        raise ValueError('Unique quote IDs required')
    q['home_line'] = pd.to_numeric(q.home_line, errors='raise')
    q['home_price_american'] = pd.to_numeric(q.home_price_american, errors='raise')
    if not np.isfinite(q[['home_line','home_price_american']].to_numpy()).all():
        raise ValueError('Nonfinite line or price')
    if ((q.home_price_american.abs() < 100) | (q.home_price_american % 1 != 0)).any():
        raise ValueError('Invalid American price')
    if (q.captured_at_utc > q.recorded_at_utc).any():
        raise ValueError('Quote capture cannot follow recording')
    q = q[(q.recorded_at_utc <= now) & (q.kickoff_utc > now)]
    # Selection rule: latest observed eligible quote per game, across all books.
    return q.sort_values(['captured_at_utc','recorded_at_utc','quote_id']).drop_duplicates('game_id', keep='last')

def build(games_path, quotes_path, now):
    q = select_quotes(pd.read_csv(quotes_path), now)
    if q.empty:
        raise ValueError('No recorded quotes for games that have not started')
    raw = pd.read_csv(games_path)
    days = pd.to_datetime(raw.gameday, errors='raise')
    today = pd.Timestamp(now.tz_convert('America/New_York').date())
    # Fail closed: this daily schedule cannot establish intraday completion times.
    if ((days >= today) & (raw.home_score.notna() | raw.away_score.notna())).any():
        raise ValueError('Schedule has scores on/after forecast date; use a genuine pregame snapshot without same-day results')
    f = model.read_games(games_path)
    if not set(q.game_id) <= set(f.game_id):
        raise ValueError('An archived game ID is absent from games_snapshot.csv')
    live = f.merge(q, on='game_id', validate='one_to_one', suffixes=('', '_quote'))
    if live.actual_margin.notna().any():
        raise ValueError('Selected quote is for a scored game')
    scheduled_days = live.gameday.dt.date
    quote_days = live.kickoff_utc.dt.tz_convert('America/New_York').dt.date
    if not scheduled_days.eq(quote_days).all():
        raise ValueError('Quote kickoff date does not match schedule game date')
    live['spread_line'] = -live.home_line
    rows = []
    for season, group in live.groupby('season'):
        train = f[(f.season < season) & (f.gameday < now)].dropna(subset=['actual_margin','spread_line'])
        if len(train) < 500:
            raise ValueError('At least 500 completed prior-season training games required')
        group = group.copy()
        group['adjustment'] = model.adjustment(model.fit(train), group)
        group['predicted_home_margin'] = group.spread_line + group.adjustment
        group['home_model_spread'] = -group.predicted_home_margin
        rows.append(group)
    result = pd.concat(rows, ignore_index=True)
    result['forecast_cutoff_utc'] = now.isoformat()
    result['schedule_sha256'] = sha(games_path)
    result['model_sha256'] = sha(model.__file__)
    result['adapter_sha256'] = sha(__file__)
    result['history_last_scored_date'] = str(raw.loc[raw.home_score.notna() & raw.away_score.notna(), 'gameday'].max())
    result['status'] = 'RESEARCH_ONLY_UNVALIDATED'
    result['selection_rule'] = 'latest_observed_eligible_quote_per_game'
    return result

def main():
    cutoff = pd.Timestamp.now(tz='UTC')
    result = build('games_snapshot.csv', 'data/market_quotes.csv', cutoff)
    finished = pd.Timestamp.now(tz='UTC')
    if (result.kickoff_utc <= finished).any():
        raise ValueError('Kickoff occurred during forecast calculation; refusing to save')
    result['forecast_created_at_utc'] = finished.isoformat()
    result['github_run_id'] = os.environ.get('GITHUB_RUN_ID', 'local')
    # One prospective forecast per game. Later reruns do not rewrite predictions.
    columns = ['game_id','home_team','away_team','quote_id','sportsbook','source',
               'home_line','home_price_american','captured_at_utc','recorded_at_utc','kickoff_utc',
               'adjustment','predicted_home_margin','home_model_spread','forecast_cutoff_utc',
               'forecast_created_at_utc','history_last_scored_date','schedule_sha256','model_sha256',
               'adapter_sha256','github_run_id','selection_rule','status']
    result = result[columns]
    if OUT.exists():
        old = pd.read_csv(OUT)
        if old.columns.tolist() != columns or old.game_id.duplicated().any():
            raise ValueError('Existing forecast archive has unexpected schema or duplicate games')
        result = result[~result.game_id.isin(old.game_id)]
        previous_count = len(old)
    else:
        previous_count = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not result.empty:
        exists = OUT.exists()
        result.to_csv(OUT, mode='a' if exists else 'w', header=not exists, index=False)
    print(f'Saved {len(result)} new prospective forecasts; {previous_count + len(result)} total archived')
    if not result.empty:
        print(result[['game_id','home_line','adjustment','home_model_spread','history_last_scored_date']].to_string(index=False))
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a') as f:
            f.write(f'## Market forecast archive\n\nSaved {len(result)} new forecasts. Download the artifact to inspect them.\n\nResearch only. Quotes are manually supplied. Historical features use the included saved schedule, which is not refreshed automatically.\n')

if __name__ == '__main__':
    main()
