"""Prospective exact-line comparison using the archived spread + EPA + total projection."""
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import softmax

import auto_market as auto
import grade_market as grader
import total_shadow

ROOT = auto.ROOT / 'exact_line_shadow'
VERSION = 'signed_home_margin_gaussian_3pt_mean_tilt_v1'


def probabilities(lines, margins, line, projection):
    log_weights = -0.5 * ((lines - line) / 3.0) ** 2
    def mean_error(tilt):
        return softmax(log_weights + tilt * margins) @ margins - projection
    tilt = brentq(mean_error, -2, 2)
    weights = softmax(log_weights + tilt * margins)
    return tuple(float(weights[mask].sum()) for mask in (margins > line, margins == line, margins < line))


def capture(schedule_path, folder):
    rows = [r for r in grader.read_rows(total_shadow.ROOT/'pairs.csv') if r['snapshot_id'] == folder.name]
    if not rows:
        return {'eligible': 0, 'saved': 0}
    games = pd.read_csv(schedule_path, usecols=['game_id','season','home_score','away_score','spread_line'])
    seasons = games.set_index('game_id', verify_integrity=True).season
    complete = games.dropna().loc[lambda f: f.season >= 2017]
    forecasts = []
    for r in rows:
        past = complete[complete.season < seasons.at[r['game_id']]]
        line = -float(r['home_line'])
        margin = float(r['predicted_home_margin'])
        home, push, away = probabilities(past.spread_line.to_numpy(), (past.home_score-past.away_score).to_numpy(), line, margin)
        side = 'HOME' if margin > line else 'AWAY' if margin < line else 'NONE'
        exact = 'HOME' if home > away else 'AWAY' if away > home else side
        now = datetime.now(timezone.utc)
        if auto.utc(r['captured_at_utc']) <= auto.utc(r['forecast_created_at_utc']) <= now < auto.utc(r['kickoff_utc']):
            forecasts.append(dict(r, version=VERSION, exact_created_at_utc=auto.stamp(now),
                                  primary_side=side, exact_side=exact, home_cover=home, push=push, away_cover=away))
    saved = auto.archive_rows(ROOT/'forecasts.csv', list(forecasts[0]), forecasts, 'game_id') if forecasts else []
    return {'eligible': len(rows), 'saved': len(saved)}


def grade(finals):
    forecasts = grader.read_rows(ROOT/'forecasts.csv')
    records = []
    for r in forecasts:
        final = grader.match(r, finals, [r])
        if final is None or auto.utc(r['exact_created_at_utc']) >= auto.utc(final['kickoff_utc']):
            continue
        adjusted = int(final['home_score']) - int(final['away_score']) + float(r['home_line'])
        row = {'game_id': r['game_id'], 'quote_id': r['quote_id'], 'home_line': r['home_line'],
               'changed': r['primary_side'] != r['exact_side']}
        for arm in ('primary','exact'):
            side = r[arm+'_side']
            value = adjusted if side == 'HOME' else -adjusted
            row[arm] = 'NO_LEAN' if side == 'NONE' else 'WIN' if value > 0 else 'LOSS' if value < 0 else 'PUSH'
        records.append(row)
    ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=['game_id','quote_id','home_line','changed','primary','exact']).to_csv(ROOT/'grades.csv', index=False)
    report = {'version':VERSION, 'paired':len(forecasts), 'graded':len(records), 'pending':len(forecasts)-len(records)}
    for arm in ('primary','exact'):
        for name, group in [(arm, records), (arm+'_on_changed', [r for r in records if r['changed']])]:
            report[name] = {grade:sum(r[arm]==grade for r in group) for grade in ('WIN','LOSS','PUSH','NO_LEAN')}
    (ROOT/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
