"""Prospective comparison at the preferred book's spread and total."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import auto_market as auto
import epa_shadow as epa
import market_model as model

ROOT = Path('data/auto_market/total_shadow')


def total_quote(events, quote):
    matches = []
    captured = auto.utc(quote['captured_at_utc'])
    for event in events:
        if event['id'] != quote['provider_event_id']:
            continue
        for book in event['bookmakers']:
            if book['key'] != quote['bookmaker_key']:
                continue
            for market in book['markets']:
                if market['key'] != 'totals':
                    continue
                updated = market.get('last_update') or book.get('last_update')
                if not updated or not 0 <= (captured-auto.utc(updated)).total_seconds() <= 1800:
                    continue
                outcomes = market['outcomes']
                if len(outcomes) != 2 or {o['name'] for o in outcomes} != {'Over','Under'}:
                    continue
                points = {float(o['point']) for o in outcomes}
                if len(points) == 1 and np.isfinite(min(points)) and min(points) > 0:
                    matches.append((min(points), updated))
    return matches[0] if len(matches) == 1 else None


def predict_margin(games, q, total, extra_columns=()):
    game = games[games.game_id.eq(q['game_id'])].copy()
    train = games[games.season.lt(game.season.item())].dropna(subset=['actual_margin','spread_line'])
    if not train.total_line.gt(0).all() or not np.isfinite(train[epa.ADJ+['total_line']]).all().all():
        raise ValueError('Incomplete training inputs')
    game['spread_line'] = -float(q['home_line'])
    game['total_line'] = total
    columns = model.INPUTS + epa.ADJ + ['total_line'] + list(extra_columns)
    estimator = model.make_pipeline(model.SimpleImputer(strategy='median'), model.StandardScaler(), Ridge(alpha=1000))
    estimator.fit(train[columns], train.actual_margin-train.spread_line)
    margin = float(game.spread_line.item()+np.clip(.25*estimator.predict(game[columns])[0],-2,2))
    return margin


def record(schedule, quotes, events, folder):
    if not quotes or not (epa.ROOT/'forecasts.csv').exists():
        return {'paired': 0}
    with (epa.ROOT/'forecasts.csv').open() as stream:
        baseline = {r['quote_id']: r for r in csv.DictReader(stream) if r['snapshot_id'] == folder.name}
    pairs = [(q, baseline[q['quote_id']], total_quote(events,q)) for q in quotes if q['quote_id'] in baseline]
    eligible = [(q,b,t) for q,b,t in pairs if t]
    if not eligible:
        return {'paired': 0, 'missing_total': len(pairs)}
    features = pd.read_csv(folder/'epa_features_asof.csv.gz')
    games = model.read_games(schedule).merge(features[['game_id']+epa.ADJ], on='game_id', how='left', validate='one_to_one')
    rows = []
    for q, b, (total, updated) in eligible:
        margin = predict_margin(games, q, total)
        now = datetime.now(timezone.utc)
        if not auto.utc(q['captured_at_utc']) <= auto.utc(b['forecast_created_at_utc']) <= now < auto.utc(q['kickoff_utc']):
            continue
        rows.append(dict(q, total_line=total,total_updated_at_utc=updated, baseline_margin=b['predicted_home_margin'],
                         predicted_home_margin=margin,forecast_created_at_utc=auto.stamp(now)))
    if rows:
        auto.archive_rows(ROOT/'pairs.csv',list(rows[0]),rows,'game_id')
    status = {'paired': len(rows), 'missing_total': len(pairs)-len(eligible)}
    if rows:
        from overtime_comparison import recent_overtime
        try:
            games["recent_ot"] = games.game_id.map(recent_overtime(games))
            challengers = []
            for row in rows:
                margin = predict_margin(games, row, row["total_line"], ["recent_ot"])
                now = datetime.now(timezone.utc)
                if now < auto.utc(row["kickoff_utc"]):
                    challengers.append(dict(row, baseline_margin=row["predicted_home_margin"],
                                            predicted_home_margin=margin, forecast_created_at_utc=auto.stamp(now)))
            saved = auto.archive_rows(ROOT/'overtime'/'pairs.csv', list(rows[0]), challengers, 'game_id')
            status["new_overtime_pairs"] = len(saved)
        except (KeyError, ValueError) as exc:
            status["overtime_error"] = str(exc)
    return status


def grade(grader, finals, now):
    report = grade_pairs(grader, finals, now, ROOT, "with_total")
    grade_pairs(grader, finals, now, ROOT/"overtime", "with_overtime")
    return report


def grade_pairs(grader, finals, now, folder, candidate_name):
    rows = grader.read_rows(folder/'pairs.csv')
    graded = []
    for r in rows:
        final = grader.match(r,finals,[r])
        if final is None:
            continue
        original = grader.grade(dict(r,predicted_home_margin=r['baseline_margin']),final,now)
        candidate = grader.grade(r,final,now)
        graded.append(dict(game_id=r['game_id'], baseline=original['ats_result'], **{candidate_name: candidate['ats_result']},
                           disagreement=original['selected_side']!=candidate['selected_side']))
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(graded,columns=['game_id','baseline',candidate_name,'disagreement']).to_csv(folder/'grades.csv',index=False)
    report = {'paired':len(rows),'graded':len(graded),'pending':len(rows)-len(graded),
              'disagreements':sum(g['disagreement'] for g in graded)}
    for arm in ['baseline',candidate_name]:
        report[arm] = {s:sum(g[arm]==s for g in graded) for s in ['WIN','LOSS','PUSH','NO_LEAN']}
        report[arm]['wins_on_disagreements'] = sum(g['disagreement'] and g[arm]=='WIN' for g in graded)
    (folder/'summary.json').write_text(json.dumps(report,indent=2))
    return report
