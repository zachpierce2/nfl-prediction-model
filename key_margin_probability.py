"""Compare integer-margin probabilities at one predetermined half-point strike per game."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def evaluate(data, label):
    rows = []
    for season, games in data.groupby('season'):
        prior = data[data.season < season]
        if len(prior) < 200:
            continue
        history = prior.predicted_margin.to_numpy()
        actual = prior.actual_margin.to_numpy()
        errors = actual-history
        # Mirror home/away orientation so favored and underdog margins share history.
        centers = np.r_[history,-history]
        outcomes = np.r_[actual,-actual]
        for g in games.itertuples():
            margin = g.predicted_margin
            # Nearest half point; for an integer anchor choose the higher home-margin threshold.
            strike = np.floor(g.spread_line)+.5
            baseline = np.mean(np.floor(margin+errors+.5)>strike)
            # Fixed 3-point bandwidth; weight historical integer scores without shifting them.
            weights = np.exp(-.5*((centers-margin)/3.)**2)
            candidate = np.average(outcomes>strike, weights=weights)
            yes = int(g.actual_margin>strike)
            rows.append(dict(era=label,season=season,game_id=g.game_id,strike=strike,yes=yes,
                             current_probability=baseline,candidate_probability=candidate,
                             current_brier=(baseline-yes)**2,candidate_brier=(candidate-yes)**2))
    return pd.DataFrame(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--earlier-games',help='games.csv from earlier_total.py (PR 18)')
    args=parser.parse_args()
    recent=pd.read_csv('strength_epa_reference_predictions.csv').query("method=='opponent_epa'")
    frames=[evaluate(recent,'recent')]
    if args.earlier_games:
        earlier=pd.read_csv(args.earlier_games)
        earlier['predicted_margin']=earlier.spread_line+earlier.baseline
        frames.append(evaluate(earlier,'earlier'))
    scores=pd.concat(frames)
    out=Path('key_margin_results')
    out.mkdir(exist_ok=True)
    scores.to_csv(out/'probabilities.csv',index=False)
    def summarize(group):
        return group.agg(games=('game_id','size'),current_brier=('current_brier','mean'),
                         candidate_brier=('candidate_brier','mean'))
    annual=summarize(scores.groupby(['era','season']))
    annual.to_csv(out/'annual.csv')
    overall=summarize(scores.groupby('era'))
    overall.to_csv(out/'summary.csv')
    print(overall)
    print(annual)
