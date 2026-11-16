#!/usr/bin/env python3
"""Market-first exploratory NFL residual model. Home expected margin uses nflverse spread_line sign."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

FEATURES=['elo_diff','home_elo','away_elo','rest_diff','home_rest','away_rest','margin4_diff','margin8_diff','pf8_diff','pa8_diff','wins8_diff','experience_diff','qb_margin_diff','qb_experience_diff','home_field']

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
            expected=1/(1+10**(-(eh+(55 if r.get("location")=="Home" else 0)-ea)/400))
            delta=20*(float(m>0)+.5*float(m==0)-expected)
            elo[h]=eh+delta;elo[a]=ea-delta
    return pd.DataFrame(rows)

# Fixed before this experiment; do not select these using reported test outcomes.
ALPHA = 1000.0
SHRINK = 0.25
CAP = 2.0
INPUTS = [c for c in FEATURES if not c.startswith('qb_')] + ['spread_line']

def read_games(path):
    d = pd.read_csv(path, low_memory=False)
    required = {'game_id','season','week','gameday','home_team','away_team','home_score','away_score','spread_line','location'}
    if required-set(d): raise ValueError(f'Missing schedule columns: {sorted(required-set(d))}')
    if d.game_id.isna().any() or d.game_id.duplicated().any(): raise ValueError('Unique nonnull game_id required')
    d['gameday'] = pd.to_datetime(d.gameday, utc=True, errors='raise')
    for c in ['season','week','home_score','away_score','spread_line']:
        d[c] = pd.to_numeric(d[c], errors='raise')
        if np.isinf(d[c]).any(): raise ValueError(f'Infinite {c}')
    if d.season.isna().any(): raise ValueError('Missing season')
    aliases = {'LA':'LAR','STL':'LAR','SD':'LAC','OAK':'LV'}
    for c in ['home_team','away_team']: d[c] = d[c].replace(aliases)
    d = d[d.season.ge(2017)].sort_values(['gameday','game_id']).copy()
    f = make_features(d)
    # Historical actual starter identities never enter the residual estimator.
    f['home_field'] = f.location.eq('Home').astype(float)
    return f

def fit(train):
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=ALPHA))
    model.fit(train[INPUTS], train.actual_margin-train.spread_line)
    return model

def adjustment(model, data):
    return np.clip(SHRINK*model.predict(data[INPUTS]), -CAP, CAP)

def evaluate(f):
    played = f.dropna(subset=['actual_margin','spread_line'])
    rows=[]
    for year in sorted(played.season.unique()):
        if year < 2020: continue
        train=played[played.season<year]; test=played[played.season.eq(year)]
        if len(train)<500 or test.empty: continue
        m=fit(train); p=test[['game_id','season','home_team','away_team','actual_margin','spread_line']].copy()
        p['adjustment']=adjustment(m,test)
        p['predicted_margin']=p.spread_line+p.adjustment
        p['home_model_spread']=-p.predicted_margin
        p['evaluation']='EXPLORATORY_REVISED_DATA'
        rows.append(p)
    if not rows: raise ValueError('No evaluation folds with 500 prior-season games')
    return pd.concat(rows,ignore_index=True)

def summarize(p):
    rows=[]
    for period,g in [('all',p)]+list(p.groupby('season')):
        market=(g.actual_margin-g.spread_line).abs()
        error=(g.actual_margin-g.predicted_margin).abs()
        cover=(g.actual_margin-g.spread_line)*np.sign(g.adjustment)
        selected=g.adjustment.abs()>1e-10
        wins=int(((cover>0)&selected).sum()); losses=int(((cover<0)&selected).sum())
        rows.append(dict(period=str(period),games=len(g),market_mae=float(market.mean()),
            model_mae=float(error.mean()),improvement_points=float((market-error).mean()),
            hypothetical_wins=wins,hypothetical_losses=losses,
            hypothetical_pushes=int(((cover==0)&selected).sum()),
            hypothetical_win_pct=100*wins/(wins+losses) if wins+losses else None))
    return rows

def forecast(f, quotes, cutoff):
    q=pd.read_csv(quotes)
    required={'game_id','home_line','captured_at_utc','kickoff_utc','sportsbook'}
    if required-set(q): raise ValueError(f'Quotes missing {sorted(required-set(q))}')
    if q.game_id.duplicated().any(): raise ValueError('Supply one chosen quote per game')
    for c in ['captured_at_utc','kickoff_utc']:
        q[c]=pd.to_datetime(q[c],utc=True,errors='raise')
        if q[c].isna().any(): raise ValueError(f'Missing {c}')
    q['home_line']=pd.to_numeric(q.home_line,errors='raise')
    if not np.isfinite(q.home_line).all(): raise ValueError('Nonfinite home line')
    if not ((q.captured_at_utc<=cutoff)&(q.kickoff_utc>cutoff)).all():
        raise ValueError('Quotes must be captured by cutoff; games must start after cutoff')
    if not set(q.game_id)<=set(f.game_id): raise ValueError('Quote game absent from schedule')
    live=f.merge(q,on='game_id',validate='one_to_one')
    if live.actual_margin.notna().any(): raise ValueError('Forecast contains completed game')
    live['spread_line']=-live.home_line
    # Prior seasons only, matching the annual backtest; completed before cutoff.
    predictions=[]
    for season,g in live.groupby('season'):
        train=f[(f.season<season)&(f.gameday<cutoff)].dropna(subset=['actual_margin','spread_line'])
        if len(train)<500: raise ValueError('Need 500 historical training games')
        g=g.copy();g['adjustment']=adjustment(fit(train),g)
        g['predicted_margin']=g.spread_line+g.adjustment
        g['home_model_spread']=-g.predicted_margin
        g['status']='RESEARCH_ONLY_UNVALIDATED'
        predictions.append(g)
    if not predictions: raise ValueError('No forecasts')
    return pd.concat(predictions,ignore_index=True)

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--games',default='data/games.csv')
    a.add_argument('--out',default='market_results')
    a.add_argument('--quotes',help='Optional timestamped chosen home spread quotes CSV')
    a.add_argument('--cutoff',help='UTC forecast cutoff; defaults to now')
    args=a.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    f=read_games(args.games);p=evaluate(f);scores=summarize(p)
    p.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'comparison.csv',index=False)
    if args.quotes:
        cutoff=pd.Timestamp(args.cutoff) if args.cutoff else pd.Timestamp.now(tz='UTC')
        if cutoff.tzinfo is None: raise ValueError('Cutoff must include timezone')
        # Avoid later completed games entering earlier as-of feature histories.
        raw=pd.read_csv(args.games); days=pd.to_datetime(raw.gameday,utc=True)
        if ((days>=cutoff.normalize()) & raw.home_score.notna()).any():
            raise ValueError('Forecast schedule includes completed games on/after cutoff date; use an as-of schedule')
        live=forecast(f,args.quotes,cutoff)
        live[['game_id','home_team','away_team','sportsbook','captured_at_utc','kickoff_utc','home_line','adjustment','predicted_margin','home_model_spread','status']].to_csv(out/'forecasts.csv',index=False)
    report=dict(status='RESEARCH_ONLY_UNVALIDATED',method='Annual walk-forward residual ridge; training uses prior seasons only',
      target='actual home margin minus market expected home margin',alpha=ALPHA,shrink=SHRINK,cap_points=CAP,
      features=INPUTS,results=scores,
      schedule_sha256=hashlib.sha256(Path(args.games).read_bytes()).hexdigest(),
      limitations=['Historical schedule spreads have no verified quote timestamps or executable prices.',
      'Previously inspected seasons are exploratory, including later-year comparisons.',
      'No verified betting profitability or automatic promotion.',
      'Core team history and rest features only; no EPA, injury or starter-news inputs in this first version.',
      
      'Forecast timestamp validation relies on supplied quote metadata; it does not verify provider provenance.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(scores,indent=2))
if __name__=='__main__': main()
