"""Score frozen half-point agreement and unconditional blends using earlier seasons only."""
from pathlib import Path
import numpy as np
import pandas as pd


def score(games):
    assert games.game_id.is_unique
    games = games.assign(margin=games.spread_line + games.baseline)
    rows = []
    for season, current in games.groupby('season'):
        prior = games[games.season < season]
        if prior.empty:
            continue
        for g in current.itertuples():
            strike = np.floor(g.spread_line) + .5
            y = int(g.actual_margin > strike)
            def probability(margin, historical):
                errors = (prior.actual_margin - historical).to_numpy()
                return np.mean(np.floor(margin + errors + .5) > strike)
            model = probability(g.margin, prior.margin)
            market = probability(g.spread_line, prior.spread_line)
            blend = (model + market) / 2
            agreement = blend if (model-.5)*(market-.5) > 0 else market
            rows.append(dict(season=season, game_id=g.game_id, changed=agreement!=blend,
                             agreement=(agreement-y)**2, blend=(blend-y)**2, market=(market-y)**2))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    data = pd.read_csv('earlier_total_results/games.csv')
    rows = score(data)
    rows.to_csv('agreement_games.csv', index=False)
    def summarize(rows):
        return dict(games=len(rows), changed=int(rows.changed.sum()),
                    **rows[['agreement','blend','market']].mean().to_dict())
    report = pd.DataFrame([dict(season=s, **summarize(r)) for s,r in rows.groupby('season')]
                          + [dict(season='all', **summarize(rows))])
    report.to_csv('agreement_summary.csv', index=False)
    print(report.to_string(index=False))
