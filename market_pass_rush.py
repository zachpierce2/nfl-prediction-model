"""One fixed pass/rush EPA ablation; development data only, no promotion."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import advanced_matchups as source
import market_model as base
import market_epa_experiment as e
import market_epa_simplification as comparison

SPLIT=[kind+'_'+c for kind in ['pass','run'] for c in e.ADJ]

def aggregate(p):
    if p.duplicated(['game_id','play_id']).any():
        raise ValueError('Duplicate plays')
    p=p[p.play_type.isin(['pass','run']) & p.epa.notna() & p.qb_kneel.fillna(0).eq(0) & p.qb_spike.fillna(0).eq(0) & p.posteam.notna() & p.defteam.notna()].copy()
    for c in ['posteam','defteam']:p[c]=p[c].replace(e.ALIASES)
    p['success']=p.epa.gt(0).astype(float)
    return p.groupby(['game_id','posteam','defteam','play_type'],as_index=False).agg(plays=('epa','size'),epa=('epa','mean'),success=('success','mean'))

def features(games,t):
    result=games[['game_id']].copy()
    # Identical eight-game, season-reset, play-weighted opponent adjustment.
    for kind in ['pass','run']:
        part=t[t.play_type.eq(kind)].drop(columns='play_type')
        prepared=e.prepare_team_games(part,games)
        f=e.build_features(games,prepared)[['game_id']+e.ADJ]
        result=result.merge(f.rename(columns={c:kind+'_'+c for c in e.ADJ}),on='game_id',validate='one_to_one')
    return result

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    games=base.read_games(args.games);games=games[games.season.between(2017,2025)].copy()
    cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
    frames=[];sources=[]
    for year in range(2017,2026):
        csv=cache/f'split_{year}.csv';meta=cache/f'split_{year}.json'
        if not csv.exists():
            parquet=cache/f'pbp_{year}.parquet'
            m=source.fetch(source.ROOT+f'pbp/play_by_play_{year}.parquet',parquet)
            aggregate(pd.read_parquet(parquet,columns=e.PBP_COLUMNS)).to_csv(csv,index=False)
            m['aggregate_sha256']=source.digest(csv);meta.write_text(json.dumps(m,indent=2))
            parquet.unlink()
        m=json.loads(meta.read_text())
        if m['aggregate_sha256']!=source.digest(csv):raise ValueError('Aggregate hash mismatch')
        sources.append(m);frames.append(pd.read_csv(csv));print('SOURCE_READY',year,flush=True)
    f=features(games,pd.concat(frames,ignore_index=True))
    f=f.merge(pd.read_csv(args.features)[['game_id']+e.ADJ],on='game_id',validate='one_to_one')
    f=games.merge(f,on='game_id',validate='one_to_one').dropna(subset=['actual_margin','spread_line'])
    if not np.isfinite(f[SPLIT+e.ADJ].to_numpy()).all():raise ValueError('Invalid features')
    arms={'epa_baseline':base.INPUTS+e.ADJ,'pass_rush_epa':base.INPUTS+SPLIT}
    predictions=[];folds=[]
    for year in range(2020,2026):
        train=f[f.season.lt(year)];test=f[f.season.eq(year)]
        if len(train)<500 or test.empty:raise ValueError('Insufficient fold')
        folds.append(dict(test_season=year,train_last_season=int(train.season.max()),train_games=len(train),test_games=len(test)))
        for name,cols in arms.items():
            model=base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),comparison.Ridge(alpha=1000.))
            model.fit(train[cols],train.actual_margin-train.spread_line)
            g=test[['game_id','season','actual_margin','spread_line']].copy()
            g['adjustment']=np.clip(.25*model.predict(test[cols]),-2,2)
            g['predicted_margin']=g.spread_line+g.adjustment;g['method']=name;predictions.append(g)
    p=pd.concat(predictions,ignore_index=True)
    reference=pd.read_csv('strength_epa_reference_predictions.csv')
    ref=reference[reference.method.eq('opponent_epa')][['game_id','predicted_margin']]
    check=p[p.method.eq('epa_baseline')].merge(ref,on='game_id',validate='one_to_one',suffixes=('','_reference'))
    if len(check)!=1693 or not np.allclose(check.predicted_margin,check.predicted_margin_reference,atol=1e-10,rtol=0):
        raise ValueError('Frozen EPA baseline does not reproduce')
    summaries=[]
    for name,g in p[p.season.ge(2021)].groupby('method'):
        summaries.append(dict(period='2021-2025',method=name,**comparison.metrics(g)))
        for year,h in g.groupby('season'):summaries.append(dict(period=str(year),method=name,**comparison.metrics(h)))
    summary=pd.DataFrame(summaries)
    wide=p[p.season.ge(2021)].pivot(index='game_id',columns='method',values='adjustment')
    residual=p[p.method.eq('epa_baseline') & p.season.ge(2021)].set_index('game_id').eval('actual_margin-spread_line')
    disagree=(np.sign(wide.epa_baseline)!=np.sign(wide.pass_rush_epa)) & residual.ne(0)
    paired=dict(nonpush_disagreements=int(disagree.sum()),baseline_wins=int(((residual*np.sign(wide.epa_baseline)>0)&disagree).sum()),candidate_wins=int(((residual*np.sign(wide.pass_rush_epa)>0)&disagree).sum()))
    p.to_csv(out/'predictions.csv',index=False);summary.to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'training_folds.csv',index=False)
    f[['game_id']+SPLIT].to_csv(out/'features.csv',index=False)
    result=dict(status='DEVELOPMENT_ONLY_NO_PROMOTION',summary=summaries,paired=paired,arms=arms,sources=sources,hashes={x:source.digest(x) for x in [args.games,args.features,__file__,base.__file__,e.__file__]},limitations=['Previously examined development seasons, not untouched validation.','Revised PBP and historical lines lack verified pregame publication timestamps.','Pass/run follows nflverse play_type; scrambles remain runs.','No executable odds, profitability claim, or live changes.'])
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(summary.to_string(index=False));print(paired)
    return p,summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--games',default='strength_games_snapshot.csv');parser.add_argument('--features',default='strength_epa_features.csv');parser.add_argument('--cache',default='pass_rush_cache');parser.add_argument('--out',default='pass_rush_results');run(parser.parse_args())
