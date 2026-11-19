"""Fixed earlier-era replay. Run from repository root with python earlier_total.py CACHE_DIR."""
import sys
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
import shutil
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import market_model as base
from market_epa_experiment import ADJ, ALIASES, build_features

YEARS = range(1999,2017)


def aggregate(year, cache):
    path = cache / f'play_by_play_{year}.csv.gz'
    if not path.exists():
        with urlopen(f'https://github.com/nflverse/nflverse-data/releases/download/pbp/{path.name}',timeout=120) as stream, path.with_suffix('.part').open('wb') as target:
            shutil.copyfileobj(stream,target)
        path.with_suffix('.part').replace(path)
    cols=['game_id','posteam','defteam','play_type','epa','qb_kneel','qb_spike']
    plays=pd.read_csv(path,usecols=cols)
    plays=plays[plays.play_type.isin(['pass','run']) & plays.epa.notna() & plays.qb_kneel.fillna(0).eq(0) & plays.qb_spike.fillna(0).eq(0)].copy()
    for c in ['posteam','defteam']:
        plays[c]=plays[c].replace(ALIASES)
    plays['success']=plays.epa.gt(0)
    team=plays.groupby(['game_id','posteam','defteam']).agg(plays=('epa','size'),epa=('epa','mean'),success=('success','mean')).reset_index()
    return team,hashlib.sha256(path.read_bytes()).hexdigest()


def run(cache):
    cache.mkdir(exist_ok=True,parents=True)
    with ThreadPoolExecutor(4) as pool:
        sources=list(pool.map(lambda year:aggregate(year,cache),YEARS))
    raw=pd.read_csv('strength_games_snapshot.csv')
    raw=raw[raw.season.isin(YEARS)].copy()
    raw['gameday']=pd.to_datetime(raw.gameday,utc=True)
    for c in ['home_team','away_team']:
        raw[c]=raw[c].replace(ALIASES)
    games=base.make_features(raw.sort_values(['gameday','game_id']))
    games['home_field']=games.location.eq('Home').astype(float)
    team=pd.concat([pair[0] for pair in sources]).merge(games[['game_id','season','gameday','home_team','away_team']],on='game_id',validate='many_to_one')
    if team.duplicated(['game_id','posteam']).any() or not ((team.posteam.eq(team.home_team)&team.defteam.eq(team.away_team))|(team.posteam.eq(team.away_team)&team.defteam.eq(team.home_team))).all():
        raise ValueError('Team identity mismatch')
    counts=team.groupby('game_id').size().reindex(games.game_id,fill_value=0)
    print('Games missing full play coverage:',counts[counts.ne(2)].to_dict(),flush=True)
    # Use available prior team games only; exclude incomplete games from EPA history.
    team=team[team.game_id.isin(counts[counts.eq(2)].index)]
    games=games.merge(build_features(games,team),on='game_id',validate='one_to_one')
    eligible=games.dropna(subset=['actual_margin','spread_line','total_line'])
    records=[]
    for year in range(2003,2017):
        train=eligible[eligible.season.lt(year)]
        test=eligible[eligible.season.eq(year)].copy()
        for name,extra in [('baseline',[]),('with_total',['total_line'])]:
            cols=base.INPUTS+ADJ+extra
            model=base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),Ridge(alpha=1000))
            model.fit(train[cols],train.actual_margin-train.spread_line)
            test[name]=np.clip(.25*model.predict(test[cols]),-2,2)
        records.append(test)
    p=pd.concat(records)
    out=Path('earlier_total_results')
    out.mkdir(exist_ok=True)
    summary=[]
    for period,g in [('all',p),*p.groupby('season')]:
        row=dict(period=period,games=len(g))
        for arm in ['baseline','with_total']:
            outcome=np.sign(g[arm])*np.sign(g.actual_margin-g.spread_line)
            assert g[arm].abs().gt(1e-10).all()
            row.update({arm+'_wins':int(outcome.gt(0).sum()),arm+'_losses':int(outcome.lt(0).sum()),arm+'_pushes':int(outcome.eq(0).sum()),arm+'_mae':(g.actual_margin-g.spread_line-g[arm]).abs().mean()})
            changed=np.sign(g.baseline)!=np.sign(g.with_total)
            row[arm+'_changed_wins']=int((changed&outcome.gt(0)).sum())
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.DataFrame({'season':list(YEARS),'pbp_sha256':[pair[1] for pair in sources]}).to_csv(out/'sources.csv',index=False)
    p[['game_id','season','actual_margin','spread_line','total_line','baseline','with_total']].to_csv(out/'games.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':
    run(Path(sys.argv[1]))
