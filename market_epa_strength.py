"""Development-only annual walk-forward EPA shrinkage comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import strength_reference_market_model as base

EPA = ['adj_off_epa_diff','adj_def_epa_diff','adj_off_success_diff','adj_def_success_diff']
STRENGTHS = (0., .05, .10, .15, .25, .50, 1.)

def score(g):
    error = g.actual_margin-g.predicted_margin
    market_error = g.actual_margin-g.spread_line
    active = g.adjustment.abs()>1e-10
    cover = market_error*np.sign(g.adjustment)
    w = int((active & cover.gt(0)).sum())
    l = int((active & cover.lt(0)).sum())
    return dict(games=len(g),mae=float(error.abs().mean()),rmse=float(np.sqrt((error**2).mean())),
        improvement_vs_market=float((market_error.abs()-error.abs()).mean()),
        wins=w,losses=l,pushes=int((active & cover.eq(0)).sum()),no_pick=int((~active).sum()),
        win_pct=100*w/(w+l) if w+l else None,
        mean_absolute_adjustment=float(g.adjustment.abs().mean()))

def choose(predictions, year):
    prior = predictions[predictions.season<year]
    if prior.empty:
        raise ValueError('Earlier out-of-sample validation required')
    # MAE selects strength; prefer less adjustment on an exact tie.
    return min((score(g)['mae'],float(s)) for s,g in prior.groupby('strength'))[1]

def run(games, features, out):
    f = base.read_games(games)
    f = f[f.season.between(2017,2025)].copy()
    e = pd.read_csv(features)
    if e.game_id.duplicated().any() or not set(f.game_id)<=set(e.game_id):
        raise ValueError('EPA features must cover every game exactly once')
    f = f.merge(e[['game_id']+EPA],on='game_id',validate='one_to_one')
    if not np.isfinite(f[EPA].to_numpy()).all():
        raise ValueError('EPA features must be finite')
    played = f.dropna(subset=['actual_margin','spread_line'])
    predictions=[];folds=[]
    for year in range(2020,2026):
        train=played[played.season<year];test=played[played.season.eq(year)]
        if len(train)<500 or test.empty:raise ValueError('Insufficient fold data')
        model=base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),Ridge(alpha=1000.))
        model.fit(train[base.INPUTS+EPA],train.actual_margin-train.spread_line)
        raw=model.predict(test[base.INPUTS+EPA])
        folds.append(dict(test_season=year,train_last_season=int(train.season.max()),train_games=len(train),test_games=len(test)))
        for strength in STRENGTHS:
            g=test[['game_id','season','gameday','actual_margin','spread_line']].copy()
            g['raw_adjustment']=raw
            g['adjustment']=np.clip(strength*raw,-2.,2.)
            g['predicted_margin']=g.spread_line+g.adjustment
            g['strength']=strength;predictions.append(g)
    p=pd.concat(predictions,ignore_index=True)
    selected=[];choices=[];summary=[]
    for year in range(2021,2026):
        s=choose(p,year)
        choices.append(dict(test_season=year,validation_first_season=2020,validation_last_season=year-1,selected_strength=s))
        now=p[p.season.eq(year)]
        g=now[now.strength.eq(s)].copy();g['method']='prior_year_selection';selected.append(g)
        for strength,c in now.groupby('strength'):
            summary.append(dict(period=str(year),method=f'strength_{strength:g}',**score(c)))
        summary.append(dict(period=str(year),method='prior_year_selection',**score(g)))
    selected=pd.concat(selected,ignore_index=True)
    for strength,g in p[p.season.between(2021,2025)].groupby('strength'):
        summary.append(dict(period='2021-2025',method=f'strength_{strength:g}',**score(g)))
    summary.append(dict(period='2021-2025',method='prior_year_selection',**score(selected)))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,data in [('candidate_predictions',p),('selected_predictions',selected),('comparison',pd.DataFrame(summary)),('selection_by_year',pd.DataFrame(choices)),('training_folds',pd.DataFrame(folds))]:
        data.to_csv(out/(name+'.csv'),index=False)
    meta=dict(status='DEVELOPMENT_ONLY_NO_LIVE_CHANGE',strength_grid=STRENGTHS,cap_points=2,alpha=1000,
        selection_metric='earlier out-of-sample MAE; lower strength breaks ties',
        next_research_strength=choose(p,2026),
        hashes={str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [games,features,__file__,base.__file__]},
        limitations=['All evaluated seasons have previously been examined; no untouched holdout.',
        'Historical spread and EPA timestamps are unverified and sources may be revised.',
        'Positive strength scaling preserves pick direction, so its all-game ATS record should be identical.',
        'Zero strength predicts the market and makes no picks; it has no ATS win rate.',
        'No executable prices, verified profit, fresh forecasts, or closing-line value are supplied by this experiment.'])
    (out/'provenance.json').write_text(json.dumps(meta,indent=2))
    print(pd.DataFrame(summary).query("period == '2021-2025'").to_string(index=False))
    print(pd.DataFrame(choices).to_string(index=False))
    return p,selected

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--games',default='strength_games_snapshot.csv')
    a.add_argument('--epa-features',default='strength_epa_features.csv')
    a.add_argument('--out',default='epa_strength_results')
    args=a.parse_args();run(args.games,args.epa_features,args.out)
