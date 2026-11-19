"""Annual out-of-sample scale calibration. Run on the files from football_spread.py."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path('football_calibration_results')


def evaluate(forecasts,seeds):
    cols=['game_id','season','football','actual_margin']
    history=pd.concat([seeds[cols],forecasts[cols]],ignore_index=True)
    assert history.game_id.is_unique
    rows=[]
    for year,test in forecasts.groupby('season'):
        # The independent model's team history restarts in 2017 for the modern replay.
        start=2020 if year>=2021 else 2003
        prior=history[history.season.between(start,year-1)]
        x=prior.football.to_numpy()
        y=prior.actual_margin.to_numpy()
        # Symmetric least-squares calibration: zero stays zero; no sportsbook inputs.
        scale=np.dot(x,y)/np.dot(x,x)
        rows.append(test.assign(scale=scale,calibrated=test.football*scale,
                                prior_games=len(prior),last_training_season=prior.season.max()))
    return pd.concat(rows,ignore_index=True)


def score(g):
    rows=[]
    for name in ['spread_line','current','football','calibrated']:
        pred=g[name]
        sign=np.sign(pred-g.spread_line)
        result=sign*np.sign(g.actual_margin-g.spread_line)
        rows.append(dict(model=name,games=len(g),wins=result.gt(0).sum(),losses=result.lt(0).sum(),
                         pushes=(result.eq(0)&sign.ne(0)).sum(),no_pick=sign.eq(0).sum(),
                         mae=(g.actual_margin-pred).abs().mean(),
                         rmse=np.sqrt(((g.actual_margin-pred)**2).mean())))
    return pd.DataFrame(rows)


def main():
    g=pd.read_csv('football_spread_results/games.csv')
    s=pd.read_csv(OUT/'warmup.csv')
    p=evaluate(g,s)
    assert len(p)==5846 and p.game_id.is_unique and p.last_training_season.lt(p.season).all()
    summaries=[]
    for label,period in list(p.groupby('era'))+list(p.groupby('season')):
        for group,mask in [('all',period.season.notna()),
                           ('book_favorite_7',period.spread_line.abs().ge(7)),
                           ('football_favorite_7',period.football.abs().ge(7)),
                           ('changed',((period.football-period.spread_line)*(period.calibrated-period.spread_line)).lt(0))]:
            summaries.append(score(period[mask]).assign(period=str(label),group=group))
    summary=pd.concat(summaries)
    # The 2026 era and 2026 season are the same cohort.
    summary=summary.drop_duplicates(['model','period','group'])
    summary.to_csv(OUT/'summary.csv',index=False)
    p[['season','scale','prior_games','last_training_season']].drop_duplicates().to_csv(OUT/'scales.csv',index=False)
    p.to_csv(OUT/'games.csv',index=False)
    print(summary.query("period in ['2004-2019','2021-2025','2026'] and model in ['football','calibrated']").to_string(index=False))


if __name__=='__main__':
    main()
