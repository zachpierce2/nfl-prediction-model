"""Test normalized home spread price as one extra input. Run from repository root."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from pregame_total import base, EPA


def implied(odds):
    valid = odds.abs().ge(100) & odds.mod(1).eq(0)
    return pd.Series(np.where(odds.gt(0), 100/(100+odds), -odds/(100-odds)), index=odds.index).where(valid)


def run():
    games = base.read_games('strength_games_snapshot.csv').merge(
        pd.read_csv('strength_epa_features.csv')[['game_id']+EPA], on='game_id', how='left', validate='one_to_one')
    games = games[games.season.between(2017,2025)].dropna(subset=['actual_margin','spread_line']).copy()
    home, away = implied(games.home_spread_odds), implied(games.away_spread_odds)
    games['home_price_probability'] = home/(home+away)
    inputs = base.INPUTS + EPA + ['total_line']
    folds = []
    for year in range(2021,2026):
        train = games[games.season.lt(year)]
        test = games[games.season.eq(year)].copy()
        for arm, columns in [('baseline', inputs), ('price', inputs+['home_price_probability'])]:
            estimator = base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),Ridge(alpha=1000))
            estimator.fit(train[columns],train.actual_margin-train.spread_line)
            test[arm] = np.clip(.25*estimator.predict(test[columns]),-2,2)
        test.loc[test.home_price_probability.isna(),'price'] = test.baseline
        folds.append(test)
    p = pd.concat(folds)
    reference = pd.read_csv('pregame_total_results/summary.csv').query("model=='with_total'").set_index('period')
    rows = []
    for period, g in [('all',p),*p.groupby('season')]:
        margin_error = g.actual_margin-g.spread_line
        grades = {a: np.sign(g[a])*np.sign(margin_error) for a in ['baseline','price']}
        assert all(g[a].abs().gt(1e-10).all() for a in grades)
        assert (grades['baseline']>0).sum()==reference.loc[str(period),'wins']
        rows.append(dict(period=period,games=len(g),priced=g.home_price_probability.notna().sum(),
            baseline_wins=(grades['baseline']>0).sum(),wins=(grades['price']>0).sum(),
            losses=(grades['price']<0).sum(), pushes=(grades['price']==0).sum(),
            gained=(grades['price']>grades['baseline']).sum(),lost=(grades['price']<grades['baseline']).sum(),
            baseline_mae=(margin_error-g.baseline).abs().mean(),price_mae=(margin_error-g.price).abs().mean()))
    assert len(p)==p.game_id.nunique()==1424
    out=Path('spread_price_results')
    out.mkdir(exist_ok=True)
    p[['game_id','season','spread_line','home_spread_odds','away_spread_odds','home_price_probability','baseline','price','actual_margin']].to_csv(out/'games.csv',index=False)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'summary.csv',index=False)
    print(summary.to_string(index=False))


if __name__=='__main__':
    run()
