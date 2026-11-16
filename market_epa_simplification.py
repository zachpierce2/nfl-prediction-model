"""Fixed-setting development comparison; no automatic live model changes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import market_model as base

EPA=['adj_off_epa_diff','adj_def_epa_diff','adj_off_success_diff','adj_def_success_diff']
METHODS={'market':None,'core':base.INPUTS,'core_epa':base.INPUTS+EPA,'market_epa_only':['spread_line']+EPA}

def metrics(g):
    cover=(g.actual_margin-g.spread_line)*np.sign(g.adjustment)
    active=g.adjustment.abs()>1e-10
    w=int((active&cover.gt(0)).sum());l=int((active&cover.lt(0)).sum())
    return dict(games=len(g),wins=w,losses=l,pushes=int((active&cover.eq(0)).sum()),
        no_pick=int((~active).sum()),win_pct=100*w/(w+l) if w+l else None,
        mae=float((g.actual_margin-g.predicted_margin).abs().mean()),
        rmse=float(np.sqrt(((g.actual_margin-g.predicted_margin)**2).mean())))

def run(games,features,out):
    f=base.read_games(games)
    f=f[f.season.between(2017,2025)].copy()
    e=pd.read_csv(features)
    if e.game_id.duplicated().any():raise ValueError('Duplicate EPA game IDs')
    f=f.merge(e[['game_id']+EPA],on='game_id',how='left',validate='one_to_one')
    if not np.isfinite(f[EPA].to_numpy()).all():raise ValueError('Missing or invalid EPA features')
    f=f.dropna(subset=['actual_margin','spread_line'])
    predictions=[];folds=[]
    for year in range(2020,2026):
        train=f[f.season<year];test=f[f.season.eq(year)]
        if len(train)<500 or test.empty:raise ValueError('Insufficient fold data')
        folds.append(dict(test_season=year,train_last_season=int(train.season.max()),train_games=len(train),test_games=len(test)))
        for name,columns in METHODS.items():
            g=test[['game_id','season','actual_margin','spread_line']].copy()
            if columns is None:g['adjustment']=0.
            else:
                m=base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),Ridge(alpha=1000.))
                m.fit(train[columns],train.actual_margin-train.spread_line)
                g['adjustment']=np.clip(.25*m.predict(test[columns]),-2.,2.)
            g['predicted_margin']=g.spread_line+g.adjustment;g['method']=name
            predictions.append(g)
    p=pd.concat(predictions,ignore_index=True);rows=[]
    for (year,name),g in p[p.season.ge(2021)].groupby(['season','method']):
        rows.append(dict(period=str(year),method=name,**metrics(g)))
    for name,g in p[p.season.ge(2021)].groupby('method'):
        rows.append(dict(period='2021-2025',method=name,**metrics(g)))
    summary=pd.DataFrame(rows)
    # Paired win differences: pushes are common to all methods at the same line.
    wide=p[p.season.ge(2021)].pivot(index=['game_id','season'],columns='method',values='adjustment')
    actual=p[p.method.eq('market')&p.season.ge(2021)].set_index(['game_id','season'])
    residual=actual.actual_margin-actual.spread_line
    pairs=[]
    for opponent in ['core','core_epa']:
        a=np.sign(wide['market_epa_only']);b=np.sign(wide[opponent])
        eligible=(residual!=0)&(a!=0)&(b!=0)
        aw=residual*a>0;bw=residual*b>0
        pairs.append(dict(comparison='market_epa_only vs '+opponent,
            disagreements=int((eligible&(a!=b)).sum()),
            epa_only_wins_when_other_loses=int((eligible&aw&~bw).sum()),
            other_wins_when_epa_only_loses=int((eligible&~aw&bw).sum())))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p.to_csv(out/'predictions.csv',index=False);summary.to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'training_folds.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'paired_disagreements.csv',index=False)
    metadata=dict(status='DEVELOPMENT_DATA_NO_LIVE_CHANGE',alpha=1000,shrink=.25,cap=2,
        features=METHODS,hashes={str(x):hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in [games,features,__file__,base.__file__]},
        limitations=['All seasons were previously examined; this is not an untouched holdout.',
        'Historical EPA and spread data may be revised and lack verified pregame publication timestamps.',
        'No threshold filtering, new parameter search, or automatic promotion.',
        'ATS accuracy is a hypothetical direction diagnostic; no executable price or profit is established.'])
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2))
    print(summary[summary.period.eq('2021-2025')].to_string(index=False))
    print(summary[summary.method.eq('market_epa_only')].to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))
    return p,summary

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--games',default='strength_games_snapshot.csv');a.add_argument('--features',default='strength_epa_features.csv');a.add_argument('--out',default='epa_simplification_results')
    args=a.parse_args();run(args.games,args.features,args.out)
