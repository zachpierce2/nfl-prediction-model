"""Replay PR 29's fixed scale at four alternate thresholds and its original control.

Run from repo root with the 2003–2019 earlier_total_results/games.csv from the
earlier-era replay. Uses the two existing PR 29 annual summaries as control checks.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from advanced_spread_replay import pricing_inputs

OFFSETS = np.array([-7,-3,0,3,7])


def evaluate(games):
    assert games.game_id.is_unique
    assert np.isfinite(games[['margin','spread_line','actual_margin','total_line']]).all().all()
    rows = []
    for year in sorted(games.season.unique())[1:]:
        past = games[games.season < year]
        test = games[games.season == year]
        errors = (past.actual_margin - past.margin).to_numpy()
        scale = make_pipeline(StandardScaler(), Ridge(alpha=100))
        scale.fit(pricing_inputs(past), np.abs(errors))
        standardized = errors / np.maximum(1, scale.predict(pricing_inputs(past)))
        widths = np.maximum(1, scale.predict(pricing_inputs(test)))
        for g, width in zip(test.itertuples(), widths):
            strikes = np.floor(g.spread_line) + .5 + OFFSETS
            actual = g.actual_margin > strikes
            pooled = np.floor(g.margin + errors + .5)
            scaled = np.floor(g.margin + width * standardized + .5)
            for offset, strike, covered in zip(OFFSETS, strikes, actual):
                base_prob = np.mean(pooled > strike)
                new_prob = np.mean(scaled > strike)
                rows.append((year,g.game_id,offset,(base_prob-covered)**2,(new_prob-covered)**2))
    return pd.DataFrame(rows,columns=['season','game_id','offset','baseline','scaled'])


def main():
    old = pd.read_csv('earlier_total_results/games.csv')
    old['margin'] = old.spread_line + old.baseline
    recent = pd.read_csv('strength_epa_reference_predictions.csv').query("method=='opponent_epa'")
    lines = pd.read_csv('strength_games_snapshot.csv')[['game_id','total_line']]
    recent = recent.rename(columns={'predicted_margin':'margin'}).merge(lines,on='game_id',validate='one_to_one')
    reports = []
    for label, data, control in [('2004-2019',old,'advanced_spread_earlier'),
                                 ('2021-2025',recent,'advanced_spread_results')]:
        scores = evaluate(data)
        summary = scores.groupby(['season','offset']).agg(
            games=('game_id','size'),baseline=('baseline','mean'),scaled=('scaled','mean')).reset_index()
        annual = pd.read_csv(Path(control)/'summary.csv').query("period!='all'")
        centers = summary.query('offset==0').set_index('season')
        for arm,col in [('baseline','baseline'),('uncertainty','scaled')]:
            expected = annual[annual.arm==arm].copy()
            expected.index = expected.period.astype(int)
            np.testing.assert_allclose(centers[col],expected.loc[centers.index,'brier'],atol=1e-12,rtol=0)
        overall = scores.groupby('offset').agg(
            games=('game_id','size'),baseline=('baseline','mean'),scaled=('scaled','mean')).reset_index()
        reports.extend([summary.assign(era=label),overall.assign(season='all',era=label)])
    report = pd.concat(reports,ignore_index=True)
    report['gain'] = report.baseline - report.scaled
    report.to_csv('alternate_uncertainty_results.csv',index=False)
    print(report[report.season=='all'].to_string(index=False))
    print(report[report.season!='all'].groupby(['era','offset']).gain.apply(lambda x:(x>0).sum()))


if __name__=='__main__':
    main()
