"""Fixed market/EPA-only challenger, evaluated on future identical-quote pairs."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import auto_market as auto
import epa_shadow as epa

ROOT = Path('data/auto_market/epa_simple_shadow')
VERSION = 'market_epa_only_alpha1000_shrink025_cap2_v1'
COLUMNS = ['spread_line'] + epa.ADJ


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline='') as source:
        return list(csv.DictReader(source))


def record(schedule_path, cores, folder):
    if not cores:
        return dict(status='NO_NEW_CORE_FORECASTS', new_forecasts=0)
    # Require the successful baseline archive and its exact saved feature snapshot.
    baselines = read_rows(epa.ROOT / 'forecasts.csv')
    by_id = {r['game_id']: r for _, r in epa.verified_pairs(baselines, read_rows(auto.ROOT / 'forecasts.csv'))}
    path = Path(folder) / 'epa_features_asof.csv.gz'
    features = pd.read_csv(path)
    frame = epa.model.read_games(schedule_path).merge(features[['game_id'] + epa.ADJ], on='game_id', how='left', validate='one_to_one')
    rows = []
    for core in cores:
        baseline = by_id.get(core['game_id'])
        if baseline is None or baseline['core_forecast_sha256'] != epa.core_hash(core):
            raise ValueError('Missing identical EPA baseline; no simple forecast saved')
        if baseline['epa_inputs_sha256'] != auto.digest(path.read_bytes()):
            raise ValueError('EPA baseline feature snapshot mismatch')
        test = frame[frame.game_id.eq(core['game_id'])].copy()
        if len(test) != 1 or test.actual_margin.notna().any():
            raise ValueError('Invalid simple forecast game')
        train = frame[frame.season.lt(test.iloc[0].season)].dropna(subset=['actual_margin', 'spread_line'])
        if len(train) < 500 or not np.isfinite(train[epa.ADJ].to_numpy()).all() or not np.isfinite(test[epa.ADJ].to_numpy()).all():
            raise ValueError('Incomplete simple EPA inputs')
        test['spread_line'] = -float(core['home_line'])
        fitted = epa.model.make_pipeline(epa.model.SimpleImputer(strategy='median'), epa.model.StandardScaler(), Ridge(alpha=1000.))
        fitted.fit(train[COLUMNS], train.actual_margin - train.spread_line)
        adj = float(np.clip(.25 * fitted.predict(test[COLUMNS])[0], -2., 2.))
        margin = -float(core['home_line']) + adj
        row = dict(core, adjustment=adj, predicted_home_margin=margin, home_model_spread=-margin,
                   model_version=VERSION, core_forecast_sha256=epa.core_hash(core),
                   epa_inputs_sha256=baseline['epa_inputs_sha256'],
                   model_sha256=auto.digest(Path(__file__).read_bytes()),
                   pipeline_sha256=auto.digest(Path(epa.__file__).read_bytes()),
                   **{c: float(test.iloc[0][c]) for c in epa.ADJ})
        rows.append(row)
    finished = datetime.now(timezone.utc)
    if any(not auto.utc(c['forecast_created_at_utc']) <= auto.utc(by_id[c['game_id']]['forecast_created_at_utc']) <= finished < auto.utc(c['kickoff_utc']) for c in cores):
        raise ValueError('Simple calculation crossed kickoff or invalid pair time')
    for row in rows:
        row['forecast_created_at_utc'] = auto.stamp(finished)
    saved = auto.archive_rows(ROOT / 'forecasts.csv', epa.FIELDS, rows, 'game_id')
    return dict(status='PAIRED_PREGAME_FORECASTS_SAVED', new_forecasts=len(saved), model_version=VERSION)


def grade_shadow(grader, cores, finals, quotes, now):
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = grader.read_rows(ROOT / 'forecasts.csv')
    pairs = epa.verified_pairs(rows, cores)
    baselines = dict((r['game_id'], r) for _, r in epa.verified_pairs(grader.read_rows(epa.ROOT / 'forecasts.csv'), cores))
    candidate_grades, baseline_grades, baseline_rows = [], [], []
    for core, row in pairs:
        baseline = baselines.get(row['game_id'])
        if baseline is None or baseline['core_forecast_sha256'] != row['core_forecast_sha256'] or baseline['epa_inputs_sha256'] != row['epa_inputs_sha256']:
            raise ValueError('Simple candidate missing same-input EPA pair')
        if row['model_version'] != VERSION or not auto.utc(core['forecast_created_at_utc']) <= auto.utc(baseline['forecast_created_at_utc']) <= auto.utc(row['forecast_created_at_utc']):
            raise ValueError('Invalid simple model version or pair timing')
        baseline_rows.append(baseline)
        final = grader.match(row, finals, quotes)
        if final:
            candidate_grades.append(grader.grade(dict(row, forecast_archive=str(ROOT / 'forecasts.csv')), final, now))
            baseline_grades.append(grader.grade(baseline, final, now))
    for name, grades in [('graded_predictions', candidate_grades), ('paired_epa_grades', baseline_grades)]:
        pd.DataFrame(grades, columns=grader.GRADE_FIELDS).to_csv(ROOT / (name + '.csv'), index=False)
    report = dict(model_version=VERSION, paired_forecasts=len(pairs), updated_at_utc=auto.stamp(now),
                  simple=grader.summary(candidate_grades, rows, now), epa=grader.summary(baseline_grades, baseline_rows, now))
    (ROOT / 'summary.json').write_text(json.dumps(report, indent=2))
    text = '\n## Prospective simpler EPA versus existing EPA\n\n'
    text += f'Identical-quote forecast pairs: {len(pairs)}. Fixed settings; no historical backfill.\n\n'
    text += '| Model | Graded | Wins | Losses | Pushes | ATS win rate |\n|---|---:|---:|---:|---:|---:|\n'
    for name in ['simple', 'epa']:
        s = report[name]
        pct = 'N/A' if s['hypothetical_win_pct'] is None else f"{s['hypothetical_win_pct']:.2f}%"
        text += f"| {name} | {s['graded_games']} | {s['hypothetical_wins']} | {s['hypothetical_losses']} | {s['pushes']} | {pct} |\n"
    (ROOT / 'report.md').write_text(text)
    return report, text
