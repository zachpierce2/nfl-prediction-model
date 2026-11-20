"""Compare the first observed DraftKings spread and the regular 24-hour forecast."""
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import auto_market as auto
import epa_shadow as epa
import market_model as model
import total_shadow as total_model

ROOT = auto.ROOT / 'opening_comparison'


def read(path):
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def model_version():
    sources = ['market_model.py', 'market_epa_experiment.py', 'epa_shadow.py', 'total_shadow.py',
               'strength_epa_features.csv', 'requirements-market.txt']
    return auto.digest(''.join(auto.digest(Path(p).read_bytes()) for p in sources).encode())


def capture(events, schedule, schedule_path, captured, folder):
    ROOT.mkdir(parents=True, exist_ok=True)
    version = model_version()
    frozen = ROOT / 'model_version.txt'
    if not frozen.exists():
        frozen.write_text(version)
    if frozen.read_text() != version:
        raise ValueError('Frozen model changed')
    paired = [r for r in read(total_model.ROOT/'pairs.csv') if r['snapshot_id'] == folder.name
              and r['bookmaker_key'] == 'draftkings']
    if paired:
        rows = [dict(r, model_version=version) for r in paired]
        auto.archive_rows(ROOT/'regular.csv', list(rows[0]), rows, 'game_id')
    registry = ROOT / 'first_seen.csv'
    marker = ROOT / 'started_at.txt'
    startup = not marker.exists()
    seen = {row['event'] for row in read(registry)}
    observations = []
    fresh = []
    for event in events:
        if event['id'] in seen or auto.utc(event['commence_time']) <= captured:
            continue
        books = [b for b in event['bookmakers'] if b['key'] == 'draftkings'
                 and any(m['key'] == 'spreads' for m in b['markets'])]
        if not books:
            continue
        seen.add(event['id'])
        filtered = dict(event, bookmakers=books)
        quotes, _ = auto.match_quotes([filtered], schedule, captured, folder.name, max_days=None)
        quote = quotes[0] if quotes else None
        total = total_model.total_quote([filtered], quote) if quote else None
        reason = ('already_listed_at_start' if startup else 'invalid_spread_or_schedule' if not quote
                  else 'missing_total' if not total else 'eligible')
        observations.append(dict(event=event['id'], snapshot=folder.name, first_seen=auto.stamp(captured),
                                 game_id=quote['game_id'] if quote else '', status=reason))
        if reason == 'eligible':
            fresh.append((quote, total))
    # Record every first sighting even if features are missing. Later lines cannot replace it.
    if observations:
        auto.archive_rows(registry, list(observations[0]), observations, 'event')
    if startup:
        marker.write_text(auto.stamp(captured))
    if not fresh:
        return {'first_seen': len(observations), 'saved': 0}
    games = model.read_games(schedule_path)
    feature_path = folder / 'epa_features_asof.csv.gz'
    if not (feature_path.exists() and (folder/'epa_provenance.json').exists()):
        epa.live_features(games, folder)
    features = pd.read_csv(feature_path)
    games = games.merge(features[['game_id'] + epa.ADJ], on='game_id', how='left', validate='one_to_one')
    rows = []
    for quote, (total, updated) in fresh:
        game = games[games.game_id.eq(quote['game_id'])]
        if len(game) != 1 or game.actual_margin.notna().any() or game[epa.ADJ].isna().any().any():
            continue
        margin = total_model.predict_margin(games, quote, total)
        now = datetime.now(timezone.utc)
        if captured <= now < auto.utc(quote['kickoff_utc']):
            rows.append(dict(quote, predicted_home_margin=margin, total_line=total,
                             total_updated_at_utc=updated, forecast_created_at_utc=auto.stamp(now),
                             model_version=version))
    saved = auto.archive_rows(ROOT/'forecasts.csv', list(rows[0]), rows, 'game_id') if rows else []
    return {'first_seen': len(observations), 'saved': len(saved)}


def report(grader, finals, now):
    openings = read(ROOT/'forecasts.csv')
    regular = {r['game_id']: r for r in read(ROOT/'regular.csv')}
    grades = []
    coverage = Counter()
    for early in openings:
        late = regular.get(early['game_id'])
        if not late:
            coverage['no_regular_forecast'] += 1
            continue
        if (late['bookmaker_key'] != 'draftkings' or late['provider_event_id'] != early['provider_event_id']
                or late.get('snapshot_id') == early['snapshot_id']
                or auto.utc(late['captured_at_utc']) <= auto.utc(early['forecast_created_at_utc'])
                or late.get('model_version', '') != early['model_version']):
            coverage['incomparable'] += 1
            continue
        final = grader.match(early, finals, [early])
        if not final or grader.match(late, finals, [late]) != final:
            coverage['awaiting_matched_final'] += 1
            continue
        a, b = [grader.grade(row, final, now) for row in (early, late)]
        sides = {a['selected_side'], b['selected_side']}
        if 'NONE' in sides:
            category = 'no_lean'
        elif len(sides) == 2:
            category = 'switched_teams'
        elif a['ats_result'] != b['ats_result']:
            category = 'same_team_line_changed_grade'
        else:
            category = 'same_team_same_grade'
        grades.append(dict(game_id=early['game_id'], opening_side=a['selected_side'], regular_side=b['selected_side'],
                           opening_line=early['home_line'], regular_line=late['home_line'],
                           opening=a['ats_result'], regular=b['ats_result'], category=category))
    ROOT.mkdir(parents=True, exist_ok=True)
    fields = ['game_id','opening_side','regular_side','opening_line','regular_line','opening','regular','category']
    pd.DataFrame(grades, columns=fields).to_csv(ROOT/'grades.csv', index=False)
    result = dict(first_sightings=dict(Counter(r['status'] for r in read(ROOT/'first_seen.csv'))),
                  opening_forecasts=len(openings), paired_finals=len(grades), coverage=dict(coverage),
                  categories=dict(Counter(r['category'] for r in grades)))
    for arm in ('opening', 'regular'):
        counts = Counter(r[arm] for r in grades)
        decisions = counts['WIN'] + counts['LOSS']
        result[arm] = {k: counts[k] for k in ('WIN','LOSS','PUSH','NO_LEAN')}
        result[arm]['ats_pct'] = round(100 * counts['WIN']/decisions, 2) if decisions else None
    (ROOT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
