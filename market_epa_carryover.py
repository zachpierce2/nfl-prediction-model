"""Fixed 50% prior-season EPA carryover experiment on development seasons."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import market_model as base
import market_epa_experiment as epa

def features(games,team_games,carry):
    rows=[]
    for season,season_games in games.groupby('season',sort=True):
        history=team_games[team_games.season.between(season-1 if carry else season,season)]
        for day,group in season_games.groupby('gameday',sort=True):
            prior=history[history.gameday<day].sort_values(['gameday','game_id'])
            recent=prior.groupby('posteam',sort=False).tail(8).copy()
            teams=sorted(set(group.home_team)|set(group.away_team)|set(recent.posteam)|set(recent.defteam))
            idx={t:i for i,t in enumerate(teams)};count=len(teams);off={};defense={}
            if not recent.empty:
                x=np.zeros((len(recent),2*count))
                x[np.arange(len(recent)),recent.posteam.map(idx).to_numpy()]=1
                x[np.arange(len(recent)),count+recent.defteam.map(idx).to_numpy()]=1
                weights=recent.plays.to_numpy()*np.where(recent.season.eq(season),1.,.5)
                fit=Ridge(alpha=200.).fit(x,recent[epa.METRICS].to_numpy(),sample_weight=weights)
                for t,i in idx.items():off[t]=fit.coef_[:,i];defense[t]=fit.coef_[:,count+i]
            zero=np.zeros(2)
            for g in group.itertuples():
                od=off.get(g.home_team,zero)-off.get(g.away_team,zero)
                dd=defense.get(g.away_team,zero)-defense.get(g.home_team,zero)
                row={'game_id':g.game_id}
                for j,metric in enumerate(epa.METRICS):
                    row[f'adj_off_{metric}_diff']=od[j];row[f'adj_def_{metric}_diff']=dd[j]
                rows.append(row)
    return pd.DataFrame(rows)

def score(g):
    active=g.adjustment.abs()>1e-10
    cover=(g.actual_margin-g.spread_line)*np.sign(g.adjustment)
    w=int((active&cover.gt(0)).sum());l=int((active&cover.lt(0)).sum())
    return dict(games=len(g),wins=w,losses=l,pushes=int((active&cover.eq(0)).sum()),
        no_pick=int((~active).sum()),win_pct=100*w/(w+l) if w+l else None,
        mae=float((g.actual_margin-g.predicted_margin).abs().mean()),
        rmse=float(np.sqrt(((g.actual_margin-g.predicted_margin)**2).mean())))

def run(games_path,team_path,out):
    games=base.read_games(games_path);games=games[games.season.between(2017,2025)].copy()
    prepared=epa.prepare_team_games(pd.read_csv(team_path),games)
    variants={name:games.merge(features(games,prepared,carry),on='game_id',validate='one_to_one')
        for name,carry in [('season_reset',False),('carryover_50pct',True)]}
    predictions=[];folds=[]
    for name,f in variants.items():
        if not np.isfinite(f[epa.ADJ].to_numpy()).all():raise ValueError('Nonfinite EPA features')
        f=f.dropna(subset=['actual_margin','spread_line'])
        for year in range(2020,2026):
            train=f[f.season<year];test=f[f.season.eq(year)]
            if len(train)<500 or test.empty:raise ValueError('Insufficient training fold')
            columns=base.INPUTS+epa.ADJ
            fitted=base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),Ridge(alpha=1000.))
            fitted.fit(train[columns],train.actual_margin-train.spread_line)
            g=test[['game_id','season','week','game_type','actual_margin','spread_line']].copy()
            g['adjustment']=np.clip(.25*fitted.predict(test[columns]),-2.,2.)
            g['predicted_margin']=g.spread_line+g.adjustment;g['method']=name;predictions.append(g)
            folds.append(dict(method=name,test_season=year,train_last_season=int(train.season.max()),train_games=len(train),test_games=len(test)))
    p=pd.concat(predictions,ignore_index=True);summary=[]
    for (year,name),g in p[p.season.ge(2021)].groupby(['season','method']):summary.append(dict(period=str(year),method=name,**score(g)))
    for name,g in p[p.season.ge(2021)].groupby('method'):
        summary.append(dict(period='2021-2025',method=name,**score(g)))
        early=g[g.game_type.eq('REG')&g.week.le(4)]
        summary.append(dict(period='2021-2025_REG_weeks_1-4',method=name,**score(early)))
    summary=pd.DataFrame(summary);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p.to_csv(out/'predictions.csv',index=False);summary.to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'training_folds.csv',index=False)
    for name,f in variants.items():f[['game_id']+epa.ADJ].to_csv(out/(name+'_features.csv'),index=False)
    metadata=dict(status='FIXED_CARRYOVER_DEVELOPMENT_TEST_NO_LIVE_CHANGE',prior_season_weight=.5,
        current_season_weight=1,lookback_team_games=8,oldest_allowed_season='previous season only',
        opponent_alpha=200,residual_alpha=1000,shrink=.25,cap=2,
        hashes={str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [games_path,team_path,__file__,base.__file__,epa.__file__]},
        limitations=['Previously examined seasons; not an untouched holdout.',
        'No weight grid, threshold filter or automatic model promotion.',
        'Historical EPA/spreads may be revised and lack verified pregame publication times.',
        'Reduced weight shrinks old-season evidence toward zero more strongly through the fixed ridge penalty.',
        'Even later-season predictions may differ because residual-model training features changed.',
        'No executable-price profitability established.'])
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2))
    print(summary.to_string(index=False))
    return p,summary

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--games',default='epa_games_snapshot.csv');a.add_argument('--team-games',default='carryover_team_game_epa.csv');a.add_argument('--out',default='epa_carryover_results')
    args=a.parse_args();run(args.games,args.team_games,args.out)
