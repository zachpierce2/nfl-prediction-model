#!/usr/bin/env python3
"""NFL spread-first walk-forward model. No betting claims without timestamped odds.
Usage: python model.py --games path/to/games.csv [--odds odds.csv] [--out results]
If --games omitted, fetch nflverse nfldata games.csv (requires internet).
Dependencies: pandas numpy scikit-learn scipy requests
"""
import argparse, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

URL='https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
FEATURES=['elo_diff','rest_diff','home_rest','away_rest','margin_4_diff','margin_8_diff','points_for_8_diff','points_against_8_diff','win_rate_8_diff','games_seen_diff','home_advantage']

def load_games(path):
    d=pd.read_csv(path or URL,low_memory=False)
    required=['game_id','season','week','gameday','home_team','away_team','home_score','away_score']
    missing=set(required)-set(d.columns)
    if missing: raise ValueError('Missing game columns: '+str(sorted(missing)))
    if d.game_id.isna().any() or d.game_id.duplicated().any(): raise ValueError('Null or duplicate game IDs; fix source rather than dropping games')
    d['gameday']=pd.to_datetime(d.gameday,errors='coerce',utc=True)
    if d.gameday.isna().any(): raise ValueError('Invalid game date(s)')
    for c in ['home_score','away_score','season','week']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d[d.season.between(2017,2025)].copy()
    d=d.sort_values(['gameday','game_id'])
    return d

def features_from_history(d):
    history={}; elo={}; rows=[]; last={}
    for day,games in d.groupby('gameday',sort=True):
        batch=[]
        for _,g in games.iterrows():
            def stats(team):
                h=history.get(team,[])
                def mean(k,n):return float(np.mean([x[k] for x in h[-n:]])) if h else 0.
                rest=(day-last[team]).days if team in last else 7
                return [mean('margin',4),mean('margin',8),mean('pf',8),mean('pa',8),mean('win',8),min(len(h),16),float(np.clip(rest,3,21))]
            hs,aws=stats(g.home_team),stats(g.away_team)
            row=g.to_dict()
            row.update(dict(zip(FEATURES,[elo.get(g.home_team,1500)-elo.get(g.away_team,1500),hs[6]-aws[6],hs[6],aws[6],hs[0]-aws[0],hs[1]-aws[1],hs[2]-aws[2],hs[3]-aws[3],hs[4]-aws[4],hs[5]-aws[5],1.])))
            row['actual_margin']=float(g.home_score-g.away_score) if pd.notna(g.home_score) and pd.notna(g.away_score) else np.nan
            batch.append(row)
        rows.extend(batch)
        # Update after all games on the same day, preventing same-day outcome leakage.
        for row in batch:
            if not np.isfinite(row['actual_margin']):continue
            h,a=row['home_team'],row['away_team']; m=row['actual_margin']; day=row['gameday']
            history.setdefault(h,[]).append(dict(margin=m,pf=row['home_score'],pa=row['away_score'],win=float(m>0)+.5*float(m==0)))
            history.setdefault(a,[]).append(dict(margin=-m,pf=row['away_score'],pa=row['home_score'],win=float(m<0)+.5*float(m==0)))
            eh,ea=elo.get(h,1500),elo.get(a,1500)
            exp=1/(1+10**(-(eh+55-ea)/400)); outcome=float(m>0)+.5*float(m==0)
            delta=20*(outcome-exp); elo[h]=eh+delta;elo[a]=ea-delta
            last[h]=day;last[a]=day
    return pd.DataFrame(rows)

def dec_odds(o):return 1+o/100 if o>0 else 1+100/abs(o)

def run(games,odds,out,edge):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=load_games(games); feat=features_from_history(raw)
    played=feat[feat.actual_margin.notna()].copy()
    ledger=[]
    for year in range(2020,2026):
        train=played[played.season<year]; test=played[played.season==year].copy()
        if len(train)<150 or test.empty:continue
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100))
        model.fit(train[FEATURES],train.actual_margin)
        # Residual spread estimated only from past seasons using cross-fitted earlier-year predictions.
        # Conservative fallback from past training margins until at least one prior fold exists.
        prior_errors=[r['actual_margin']-r['predicted_margin'] for r in ledger if r['season']<year]
        sigma=float(np.std(prior_errors,ddof=1)) if len(prior_errors)>=100 else float(np.std(train.actual_margin,ddof=1))
        sigma=max(sigma,7.0)
        preds=model.predict(test[FEATURES]); baseline=float(train.actual_margin.mean())
        for (_,g),pred in zip(test.iterrows(),preds):
            r={'game_id':g.game_id,'season':int(g.season),'week':int(g.week),'game_date_utc':g.gameday.isoformat(),
               'home_team':g.home_team,'away_team':g.away_team,'home_score':g.home_score,'away_score':g.away_score,
               'actual_margin':float(g.actual_margin),'predicted_margin':float(pred),'baseline_margin':baseline,
               'predicted_home_win_probability':float(norm.cdf(pred/sigma)),'sigma_pre_game':sigma,
               'error_pred_minus_actual':float(pred-g.actual_margin),'model_version':'ridge_spread_v1'}
            ledger.append(r)
    p=pd.DataFrame(ledger)
    if p.empty:raise ValueError('No test seasons; check source data')
    p.to_csv(out/'predictions.csv',index=False)
    byyear=p.groupby('season').apply(lambda x:pd.Series({'games':len(x),'model_mae':mean_absolute_error(x.actual_margin,x.predicted_margin),'baseline_mae':mean_absolute_error(x.actual_margin,x.baseline_margin),'model_rmse':np.sqrt(mean_squared_error(x.actual_margin,x.predicted_margin)),'mean_bias':(x.predicted_margin-x.actual_margin).mean()}),include_groups=False).reset_index()
    byyear.to_csv(out/'backtest_by_season.csv',index=False)
    summary={'games':len(p),'test_seasons':sorted(p.season.unique().tolist()),'model_mae':float(mean_absolute_error(p.actual_margin,p.predicted_margin)),'baseline_mae':float(mean_absolute_error(p.actual_margin,p.baseline_margin)),'model_rmse':float(np.sqrt(mean_squared_error(p.actual_margin,p.predicted_margin))),'mean_bias':float((p.predicted_margin-p.actual_margin).mean()),'betting_results':'Unavailable unless as-of odds file supplied'}
    if odds:
        o=pd.read_csv(odds)
        req={'game_id','captured_at_utc','sportsbook','market','side','line','odds_american'}
        if req-set(o):raise ValueError('Missing odds columns: '+str(sorted(req-set(o))))
        if o.duplicated(['game_id','captured_at_utc','sportsbook','market','side','line']).any():raise ValueError('Duplicate market snapshots')
        o['captured_at_utc']=pd.to_datetime(o.captured_at_utc,utc=True,errors='coerce')
        if o.captured_at_utc.isna().any():raise ValueError('Unparseable odds timestamps')
        o['market']=o.market.str.lower();o['side']=o.side.str.lower()
        if not o.market.isin(['spread','moneyline']).all() or not o.side.isin(['home','away']).all():raise ValueError('market must be spread/moneyline; side home/away')
        o['odds_american']=pd.to_numeric(o.odds_american,errors='coerce')
        o['line']=pd.to_numeric(o.line,errors='coerce')
        if o.odds_american.isna().any() or ((o.odds_american.abs()<100)).any():raise ValueError('Invalid American odds')
        p['game_date_utc']=pd.to_datetime(p.game_date_utc,utc=True)
        joined=o.merge(p,on='game_id',how='left',indicator=True)
        joined['flags']=''
        joined.loc[joined._merge!='both','flags']+='unknown_game;'
        joined.loc[joined.captured_at_utc>=joined.game_date_utc,'flags']+='not_pregame;'
        joined.loc[(joined.game_date_utc-joined.captured_at_utc)>pd.Timedelta(hours=24),'flags']+='older_than_24h;'
        # Date-only game timestamps are not kickoff timestamps: block all claims of verified executable bets.
        joined['flags']+='kickoff_time_unverified;'
        joined.to_csv(out/'odds_audit_unverified.csv',index=False)
        summary['betting_results']='Blocked: source games.csv provides game dates, not verified kickoff timestamps; odds audit exported, no simulated wagers.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'run_manifest.json').write_text(json.dumps({'generated_utc':datetime.now(timezone.utc).isoformat(),'model_version':'ridge_spread_v1','features':FEATURES,'training_policy':'Prior completed seasons only; no odds as model features','qualifying_policy':'Not enabled without verified kickoff and odds timestamps','source':games or URL},indent=2))
    print(json.dumps(summary,indent=2));print('Saved:',out.resolve())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--games');parser.add_argument('--odds');parser.add_argument('--out',default='nfl_model_results');parser.add_argument('--edge',type=float,default=.02)
    a=parser.parse_args();run(a.games,a.odds,a.out,a.edge)
