"""One fixed nonlinear challenger; development comparison, no live promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import market_model as base
import market_cover_probability as linear

# Locked before running this experiment; no tuning or early stopping on test data.
PARAMS = dict(max_iter=100, learning_rate=0.03, max_leaf_nodes=4,
              min_samples_leaf=100, l2_regularization=10.,
              early_stopping=False, random_state=2026)


def split_season(frame, year):
    train = frame[frame.season.lt(year)].copy()
    test = frame[frame.season.eq(year)].copy()
    train = train[(train.actual_margin-train.spread_line).ne(0)]
    if len(train) < 500 or test.empty:
        raise ValueError('Insufficient prior-season training or test games')
    return train, test


def run(games, features, out):
    out = Path(out)
    baseline, _ = linear.run(games, features, out/'linear_reference')
    frame = base.read_games(games)
    frame = frame[frame.season.between(2017, 2025)].copy()
    epa = pd.read_csv(features)
    frame = frame.merge(epa[['game_id']+linear.EPA], on='game_id', validate='one_to_one')
    frame = frame.dropna(subset=['actual_margin', 'spread_line'])
    if not np.isfinite(frame[linear.EPA].to_numpy()).all():
        raise ValueError('Missing EPA features')
    columns = base.INPUTS+linear.EPA
    predictions = []
    folds = []
    for year in range(2020, 2026):
        train, test = split_season(frame, year)
        expected = baseline[(baseline.method=='cover_logistic') & (baseline.season==year)]
        if set(test.game_id) != set(expected.game_id):
            raise ValueError('Challenger and baseline game coverage differs')
        model = HistGradientBoostingClassifier(**PARAMS)
        model.fit(train[columns], (train.actual_margin-train.spread_line).gt(0).astype(int))
        p = test[['game_id','season','actual_margin','spread_line']].copy()
        p['method'] = 'cover_shallow_trees'
        p['home_cover_probability'] = model.predict_proba(test[columns])[:, 1]
        p['direction'] = np.sign(p.home_cover_probability-.5)
        predictions.append(p)
        folds.append(dict(test_season=year, train_last_season=int(train.season.max()),
                          training_nonpush_games=len(train), test_games=len(test)))
    combined = pd.concat([baseline, *predictions], ignore_index=True)
    evaluated = combined[combined.season.ge(2021)]
    rows = []
    for method, g in evaluated.groupby('method'):
        rows.append(dict(period='2021-2025', method=method, **linear.score(g)))
        for year, season in g.groupby('season'):
            rows.append(dict(period=str(year), method=method, **linear.score(season)))
    comparison = pd.DataFrame(rows)
    combined.to_csv(out/'predictions.csv', index=False)
    comparison.to_csv(out/'comparison.csv', index=False)
    pd.DataFrame(folds).to_csv(out/'training_folds.csv', index=False)
    metadata = dict(status='DEVELOPMENT_ONLY_NO_LIVE_CHANGE', parameters=PARAMS,
                    features=columns, pick_threshold=.5,
                    probability='P(home covers | no push)',
                    limitations=['Previously examined seasons; no untouched holdout.',
                                 'Historical revised data; pregame publication times unverified.',
                                 'No tuning, selection filters, profit claim, or automatic promotion.'],
                    hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                            for p in [games, features, __file__, linear.__file__, base.__file__]})
    (out/'provenance.json').write_text(json.dumps(metadata, indent=2))
    print(comparison[comparison.period.eq('2021-2025')].to_string(index=False))
    return combined, comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', default='strength_games_snapshot.csv')
    parser.add_argument('--features', default='strength_epa_features.csv')
    parser.add_argument('--out', default='nonlinear_cover_results')
    args = parser.parse_args()
    run(args.games, args.features, args.out)
