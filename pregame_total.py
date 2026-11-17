"""Run from repository root. Compare one extra sportsbook input in matched annual folds."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import market_model as base
from market_epa_strength import EPA


def run():
    schedule = 'strength_games_snapshot.csv'
    games = base.read_games(schedule)
    if 'total_line' not in games:
        games = games.merge(pd.read_csv(schedule, usecols=['game_id', 'total_line']),
                            on='game_id', validate='one_to_one')
    games = games.merge(pd.read_csv('strength_epa_features.csv')[['game_id'] + EPA],
                        on='game_id', how='left', validate='one_to_one')
    games = games[games.season.between(2017, 2025)].dropna(subset=['actual_margin', 'spread_line'])
    assert games.total_line.gt(0).all() and np.isfinite(games[EPA + ['total_line']]).all().all()
    reference = pd.read_csv('strength_epa_reference_predictions.csv').query("method == 'opponent_epa'").set_index('game_id')
    history = {'baseline': [], 'with_total': []}
    rows = []
    for year in range(2020, 2026):
        train = games[games.season < year]
        test = games[games.season == year]
        for name, extra in [('baseline', []), ('with_total', ['total_line'])]:
            cols = base.INPUTS + EPA + extra
            model = base.make_pipeline(base.SimpleImputer(strategy='median'), base.StandardScaler(), Ridge(alpha=1000))
            model.fit(train[cols], train.actual_margin - train.spread_line)
            predicted = test.spread_line + np.clip(.25 * model.predict(test[cols]), -2, 2)
            if name == 'baseline':
                np.testing.assert_allclose(predicted, reference.loc[test.game_id].predicted_margin, atol=1e-10, rtol=0)
            past = np.asarray(history[name])
            for game, margin in zip(test.itertuples(), predicted):
                if year == 2020:
                    continue
                line = game.spread_line
                observed = np.sign(game.actual_margin - line)
                pick = int(np.sign(margin - line))
                simulated = np.sign(np.floor(margin + past + .5) - line)
                probs = np.array([(simulated == side).mean() for side in [-1, 0, 1]])
                brier = sum((probs - (np.arange(-1, 2) == observed)) ** 2)
                rows.append(dict(game_id=game.game_id, year=year, model=name, total_line=game.total_line,
                                 predicted_margin=margin, pick=pick, result=pick*observed,
                                 absolute_error=abs(game.actual_margin-margin), brier=brier))
            history[name].extend(test.actual_margin - predicted)
    results = pd.DataFrame(rows)
    assert results.groupby('model').game_id.nunique().eq(1424).all()
    out = Path('pregame_total_results')
    out.mkdir(exist_ok=True)
    results.to_csv(out/'games.csv', index=False)
    wide = results.pivot(index='game_id', columns='model', values='pick')
    changed = wide.index[wide.baseline != wide.with_total]
    summary = []
    for period, frame in [('all', results), *results.groupby('year')]:
        for model, group in frame.groupby('model'):
            assert group.pick.ne(0).all()
            wins = group.result.eq(1)
            summary.append(dict(period=period, model=model, wins=wins.sum(), losses=group.result.eq(-1).sum(),
                                pushes=group.result.eq(0).sum(), changed_wins=(wins & group.game_id.isin(changed)).sum(),
                                mae=group.absolute_error.mean(), brier=group.brier.mean()))
    summary = pd.DataFrame(summary)
    summary.to_csv(out/'summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    run()
