"""Frozen exploratory EPA experiment. No changes to live forecasts."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import urllib.request
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import epa_reference_market_model as base

ALIASES={'LA':'LAR','STL':'LAR','SD':'LAC','OAK':'LV'}
METRICS=['epa','success']
RAW=['raw_off_epa_diff','raw_def_epa_diff','raw_off_success_diff','raw_def_success_diff']
ADJ=['adj_off_epa_diff','adj_def_epa_diff','adj_off_success_diff','adj_def_success_diff']
CANDIDATES={'market':None,'current':[], 'raw_epa':RAW,'opponent_epa':ADJ}
PBP_COLUMNS=['game_id','play_id','posteam','defteam','epa','play_type','qb_kneel','qb_spike']

def download_team_games(cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    frames=[];sources=[]
    for season in range(2017,2026):
        url=f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet'
        path=cache/f'play_by_play_{season}.parquet'
        if not path.exists():
            for attempt in range(3):
                try:
                    temporary=path.with_suffix('.part')
                    with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as target:
                        while chunk:=response.read(1024*1024):target.write(chunk)
                    temporary.replace(path);break
                except Exception:
                    if attempt==2:raise
                    time.sleep(2)
        sources.append({'season':season,'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        plays=pd.read_parquet(path,columns=PBP_COLUMNS)
        if plays.duplicated(['game_id','play_id']).any():raise ValueError('Duplicate play identifiers')
        plays=plays[plays.play_type.isin(['pass','run']) & plays.epa.notna()
                    & plays.qb_kneel.fillna(0).eq(0) & plays.qb_spike.fillna(0).eq(0)
                    & plays.posteam.notna() & plays.defteam.notna()].copy()
        plays['success']=plays.epa.gt(0).astype(float)
        for col in ['posteam','defteam']:plays[col]=plays[col].replace(ALIASES)
        grouped=plays.groupby(['game_id','posteam','defteam']).agg(
            plays=('epa','size'),epa=('epa','mean'),success=('success','mean')).reset_index()
        frames.append(grouped)
        print(f'Loaded {season}: {len(grouped)} offense-game rows',flush=True)
    return pd.concat(frames,ignore_index=True),sources

def prepare_team_games(team_games,games):
    teams=team_games.copy()
    for col in ['posteam','defteam']:teams[col]=teams[col].replace(ALIASES)
    if teams.duplicated(['game_id','posteam']).any():raise ValueError('Duplicate team games')
    if not np.isfinite(teams[['plays','epa','success']].to_numpy()).all():raise ValueError('Nonfinite play statistics')
    if not teams.plays.gt(0).all() or not teams.success.between(0,1).all():raise ValueError('Invalid play statistics')
    # Dates come from schedule, never from uncertain PBP completion timestamps.
    merged=teams.merge(games[['game_id','gameday','season','home_team','away_team','actual_margin']],on='game_id',validate='many_to_one')
    merged=merged[merged.actual_margin.notna() & merged.season.between(2017,2025)].copy()
    valid=((merged.posteam.eq(merged.home_team)&merged.defteam.eq(merged.away_team))|
           (merged.posteam.eq(merged.away_team)&merged.defteam.eq(merged.home_team)))
    if not valid.all():raise ValueError('PBP team names disagree with schedule')
    expected=games[games.actual_margin.notna()&games.season.between(2017,2025)]
    coverage=merged.groupby('game_id').size().reindex(expected.game_id,fill_value=0)
    if not coverage.eq(2).all():raise ValueError(f'Incomplete PBP coverage: {coverage[coverage.ne(2)].to_dict()}')
    return merged

def build_features(games,team_games):
    """Last eight games per offense within current season, reset every season.

    Each day's ridge: EPA/success = intercept + offense effect + defense allowance.
    Play-count weighting; fixed alpha 200. No same-day rows enter the rating fit.
    """
    rows=[]
    for season,season_games in games.groupby('season',sort=True):
        history=team_games[team_games.season.eq(season)]
        for day,group in season_games.groupby('gameday',sort=True):
            prior=history[history.gameday<day].sort_values(['gameday','game_id'])
            recent=prior.groupby('posteam',sort=False).tail(8)
            teams=sorted(set(group.home_team)|set(group.away_team)|set(recent.posteam)|set(recent.defteam))
            idx={t:i for i,t in enumerate(teams)};count=len(teams)
            off={};defense={};raw_off={};raw_def={}
            if not recent.empty:
                x=np.zeros((len(recent),2*count))
                x[np.arange(len(recent)),recent.posteam.map(idx).to_numpy()]=1
                x[np.arange(len(recent)),count+recent.defteam.map(idx).to_numpy()]=1
                fitted=Ridge(alpha=200.0).fit(x,recent[METRICS].to_numpy(),sample_weight=recent.plays.to_numpy())
                for t,i in idx.items():
                    off[t]=fitted.coef_[:,i];defense[t]=fitted.coef_[:,count+i]
                # Raw defense and offense use the identical recent matchup sample.
                for col,target in [('posteam',raw_off),('defteam',raw_def)]:
                    for t,g in recent.groupby(col):target[t]=np.average(g[METRICS],axis=0,weights=g.plays)
            zero=np.zeros(2)
            for g in group.itertuples():
                values={}
                for prefix,of,de in [('raw',raw_off,raw_def),('adj',off,defense)]:
                    od=of.get(g.home_team,zero)-of.get(g.away_team,zero)
                    dd=de.get(g.away_team,zero)-de.get(g.home_team,zero)
                    # Early-season missing raw team rates centered on league average.
                    if prefix=='raw' and not recent.empty:
                        avg=np.average(recent[METRICS],axis=0,weights=recent.plays)
                        od=of.get(g.home_team,avg)-of.get(g.away_team,avg)
                        dd=de.get(g.away_team,avg)-de.get(g.home_team,avg)
                    for j,metric in enumerate(METRICS):
                        values[f'{prefix}_off_{metric}_diff']=od[j]
                        values[f'{prefix}_def_{metric}_diff']=dd[j]
                rows.append({'game_id':g.game_id,**values})
    return pd.DataFrame(rows)

def score(g):
    edge=g.adjustment; selected=edge.abs()>1e-10
    cover=(g.actual_margin-g.spread_line)*np.sign(edge)
    w=int((selected & cover.gt(0)).sum());l=int((selected & cover.lt(0)).sum())
    return {'games':len(g),'mae':float((g.actual_margin-g.predicted_margin).abs().mean()),
            'improvement_vs_market':float(((g.actual_margin-g.spread_line).abs()-(g.actual_margin-g.predicted_margin).abs()).mean()),
            'wins':w,'losses':l,'pushes':int((selected&cover.eq(0)).sum()),'win_pct':100*w/(w+l) if w+l else None}

def evaluate(f):
    played=f.dropna(subset=['actual_margin','spread_line'])
    rows=[];folds=[]
    for year in range(2020,2026):
        train=played[played.season<year];test=played[played.season.eq(year)]
        assert train.season.max()<year and len(train)>=500
        folds.append({'test_season':year,'train_games':len(train),'train_last_season':int(train.season.max()),'test_games':len(test)})
        for name,extra in CANDIDATES.items():
            g=test[['game_id','season','gameday','actual_margin','spread_line']].copy()
            if extra is None:g['adjustment']=0.
            else:
                columns=base.INPUTS+extra
                fitted=base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),Ridge(alpha=1000.))
                fitted.fit(train[columns],train.actual_margin-train.spread_line)
                g['adjustment']=np.clip(.25*fitted.predict(test[columns]),-2.,2.)
            g['predicted_margin']=g.spread_line+g.adjustment;g['method']=name;rows.append(g)
    predictions=pd.concat(rows,ignore_index=True)
    summary=[];choices=[];adaptive=[]
    for year in range(2021,2026):
        history=predictions[predictions.season<year]
        # Ties favor market, then original model, then additions in declaration order.
        rank={n:i for i,n in enumerate(CANDIDATES)}
        choice=min((score(g)['mae'],rank[n],n) for n,g in history.groupby('method'))[2]
        choices.append({'test_season':year,'validation_last_season':year-1,'chosen_method':choice})
        now=predictions[predictions.season.eq(year)]
        for n,g in now.groupby('method'):summary.append({'period':str(year),'method':n,**score(g)})
        selected=now[now.method.eq(choice)].copy();selected['chosen_method']=choice;selected['method']='prior_year_selection';adaptive.append(selected)
        summary.append({'period':str(year),'method':'prior_year_selection',**score(selected)})
    adaptive=pd.concat(adaptive,ignore_index=True)
    for n,g in predictions[predictions.season.between(2021,2025)].groupby('method'):
        summary.append({'period':'2021-2025','method':n,**score(g)})
    summary.append({'period':'2021-2025','method':'prior_year_selection',**score(adaptive)})
    return predictions,pd.DataFrame(summary),pd.DataFrame(choices),pd.DataFrame(folds),adaptive

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--games',default='epa_games_snapshot.csv');parser.add_argument('--team-games');parser.add_argument('--cache',default='epa_pbp_cache');parser.add_argument('--out',default='epa_experiment_results')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    games=base.read_games(args.games);games=games[games.season.between(2017,2025)].copy()
    if args.team_games:team_games=pd.read_csv(args.team_games);sources=[]
    else:team_games,sources=download_team_games(args.cache)
    prepared=prepare_team_games(team_games,games)
    features=build_features(games,prepared)
    merged=games.merge(features,on='game_id',validate='one_to_one')
    assert np.isfinite(merged[RAW+ADJ].to_numpy()).all()
    predictions,summary,choices,folds,adaptive=evaluate(merged)
    team_games.to_csv(out/'team_game_epa.csv',index=False);features.to_csv(out/'pregame_epa_features.csv',index=False)
    predictions.to_csv(out/'candidate_predictions.csv',index=False);summary.to_csv(out/'comparison.csv',index=False)
    choices.to_csv(out/'selection_by_year.csv',index=False);folds.to_csv(out/'training_folds.csv',index=False)
    adaptive.to_csv(out/'selected_predictions.csv',index=False)
    metadata={'status':'EXPLORATORY_NO_LIVE_CHANGE','sources':sources,'games_sha256':hashlib.sha256(Path(args.games).read_bytes()).hexdigest(),
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
              'settings':{'history_games':8,'reset_each_season':True,'opponent_ridge_alpha':200,'residual_ridge_alpha':1000,'shrink':.25,'cap':2,'first_validation_season':2020},
              'limitations':['Previously inspected seasons are not an untouched holdout.','Historical PBP/EPA and spreads may have been revised; no verified pregame publication timestamps.','No verified profitability, executable historical prices, or automatic model promotion.','Success is EPA > 0 on run/pass plays, excluding kneels/spikes.','Opponent adjustment is a play-weighted additive ridge estimate, not a causal estimate.']}
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2))
    print(summary[summary.period.eq('2021-2025')].to_string(index=False),flush=True)
    print(choices.to_string(index=False),flush=True)

if __name__=='__main__':main()
