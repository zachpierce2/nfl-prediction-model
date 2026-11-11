#!/usr/bin/env python3
"""NFL V3: pregame-only walk-forward score margin model, with honest odds audit."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

FEATURES=['elo_diff','home_elo','away_elo','rest_diff','home_rest','away_rest','margin4_diff','margin8_diff','pf8_diff','pa8_diff','wins8_diff','experience_diff','qb_margin_diff','qb_experience_diff','home_field']

def load(path):
    d=pd.read_csv(path,low_memory=False)
    needed={'game_id','season','week','gameday','home_team','away_team','home_score','away_score'}
    if needed-set(d):raise ValueError(f'Missing columns: {sorted(needed-set(d))}')
    if d.game_id.isna().any() or d.game_id.duplicated().any():raise ValueError('Game IDs must be unique and nonnull')
    d['gameday']=pd.to_datetime(d.gameday,utc=True,errors='raise')
    for c in ['home_score','away_score','season','week']:d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d[d.season.between(2017,2025)].sort_values(['gameday','game_id']).copy()
    return d

def make_features(d):
    team_hist={};qb_hist={};elo={};last={};rows=[]
    for day,grp in d.groupby('gameday',sort=True):
        batch=[]
        for _,g in grp.iterrows():
            def team(t):
                h=team_hist.get(t,[])
                avg=lambda k,n:float(np.mean([r[k] for r in h[-n:]])) if h else 0.
                rest=float(np.clip((day-last[t]).days,3,21)) if t in last else 7.
                return [avg('margin',4),avg('margin',8),avg('pf',8),avg('pa',8),avg('win',8),min(len(h),16),rest]
            ht,at=team(g.home_team),team(g.away_team)
            hq=g.get('home_qb_id',np.nan);aq=g.get('away_qb_id',np.nan)
            def qb(q):
                if pd.isna(q):return (0.,0.)
                h=qb_hist.get(str(q),[])
                return (float(np.mean(h[-8:])) if h else 0.,float(min(len(h),16)))
            hqm,hqn=qb(hq);aqm,aqn=qb(aq)
            eh,ea=elo.get(g.home_team,1500.),elo.get(g.away_team,1500.)
            values=[eh-ea,eh,ea,ht[6]-at[6],ht[6],at[6],ht[0]-at[0],ht[1]-at[1],ht[2]-at[2],ht[3]-at[3],ht[4]-at[4],ht[5]-at[5],hqm-aqm,hqn-aqn,1.]
            r=g.to_dict();r.update(dict(zip(FEATURES,values)))
            r['actual_margin']=float(g.home_score-g.away_score) if pd.notna(g.home_score) and pd.notna(g.away_score) else np.nan
            batch.append(r)
        rows+=batch
        for r in batch:
            m=r['actual_margin']
            if not np.isfinite(m):continue
            h,a=r['home_team'],r['away_team']
            for t,margin,pf,pa,win in [(h,m,r['home_score'],r['away_score'],float(m>0)+.5*float(m==0)),(a,-m,r['away_score'],r['home_score'],float(m<0)+.5*float(m==0))]:
                team_hist.setdefault(t,[]).append({'margin':margin,'pf':pf,'pa':pa,'win':win})
                last[t]=day
            for q,margin in [(r.get('home_qb_id'),m),(r.get('away_qb_id'),-m)]:
                if pd.notna(q):qb_hist.setdefault(str(q),[]).append(margin)
            eh,ea=elo.get(h,1500.),elo.get(a,1500.)
            expected=1/(1+10**(-(eh+55-ea)/400))
            delta=20*(float(m>0)+.5*float(m==0)-expected)
            elo[h]=eh+delta;elo[a]=ea-delta
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--games',default='data/games.csv');ap.add_argument('--out',default='nfl_v3_results');ap.add_argument('--odds',help='Optional timestamped odds CSV for audit ONLY')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    d=make_features(load(args.games));played=d[d.actual_margin.notna()]
    ledger=[];scores=[]
    for season in range(2020,2026):
        tr=played[played.season<season];te=played[played.season==season]
        if len(tr)<150 or te.empty:continue
        models={'ridge':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100)),
                'boosted_trees':make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,l2_regularization=10,learning_rate=.04,random_state=42))}
        for name,model in models.items():
            model.fit(tr[FEATURES],tr.actual_margin)
            pred=model.predict(te[FEATURES]);mae=mean_absolute_error(te.actual_margin,pred)
            scores.append({'season':season,'model':name,'games':len(te),'mae':mae})
            for (_,g),p in zip(te.iterrows(),pred):
                ledger.append({'game_id':g.game_id,'season':season,'week':int(g.week),'game_date':g.gameday.date().isoformat(),'home_team':g.home_team,'away_team':g.away_team,'actual_margin':g.actual_margin,'predicted_margin':float(p),'model':name})
    if not ledger:raise RuntimeError('No test games found')
    pd.DataFrame(ledger).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'by_season.csv',index=False)
    summary={'method':'Walk-forward: train on all completed prior seasons; test on next season','model_mae':pd.DataFrame(scores).groupby('model').apply(lambda x:round(float(np.average(x.mae,weights=x.games)),3),include_groups=False).to_dict(),'betting_performance':'NOT VERIFIED: game-level spread_line is not timestamped; no executable wagers simulated','features':FEATURES,'note':'QB features use prior game margins by QB ID; NOT QB EPA. EPA/injuries require separate time-stamped data.'}
    if args.odds:
        odds=pd.read_csv(args.odds)
        required={'game_id','captured_at_utc','sportsbook','market','side','line','odds_american'}
        if required-set(odds):raise ValueError(f'Missing odds columns: {sorted(required-set(odds))}')
        odds['captured_at_utc']=pd.to_datetime(odds.captured_at_utc,utc=True,errors='raise')
        # Source dates do not verify actual kickoff. Preserve audit without claiming executable prices.
        odds['audit_status']='KICKOFF_TIME_UNVERIFIED'
        odds.to_csv(out/'odds_audit.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
