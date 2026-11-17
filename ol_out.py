"""Run with a directory containing nflverse players.csv, injuries_YEAR.csv and snap_counts_YEAR.csv (2017-25)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from advanced_matchups import canonical, GROUPS
from pregame_total import base, EPA


def feature(games, folder):
    years = range(2017, 2026)
    injuries = canonical(pd.concat([pd.read_csv(folder/f'injuries_{y}.csv') for y in years]), ['team'])
    snaps = canonical(pd.concat([pd.read_csv(folder/f'snap_counts_{y}.csv') for y in years]), ['team'])
    players = pd.read_csv(folder/'players.csv', usecols=['pfr_id','gsis_id']).dropna().drop_duplicates()
    ids = players[~players.pfr_id.duplicated(False)].set_index('pfr_id').gsis_id
    snaps['id'] = snaps.pfr_player_id.map(ids)
    shares = snaps.groupby(['game_id','team'])
    injuries['time'] = pd.to_datetime(injuries.date_modified, utc=True, errors='coerce', format='mixed')
    weeks = injuries.groupby(['season','week','game_type','team'])
    history = {}
    result = {}
    for g in games.sort_values('gameday').itertuples():
        burdens = []
        for team in [g.home_team,g.away_team]:
            prev = history.setdefault(team, [])
            key = (g.season,g.week,'REG' if g.game_type=='REG' else 'POST',team)
            value = np.nan
            if key in weeks.groups:
                reports = weeks.get_group(key)
                reports = reports[reports.time < g.gameday]
                if len(reports):
                    latest = reports[reports.time.eq(reports.groupby('gsis_id',dropna=False).time.transform('max'))]
                    out = latest[latest.position.str.upper().map(GROUPS).eq('OL') & latest.report_status.str.lower().eq('out')]
                    value = 0.
                    for pid in out.gsis_id:
                        prior = [shares.get_group((gid,team)) for gid in prev[-3:] if (gid,team) in shares.groups]
                        if len(prior)!=len(prev[-3:]) or not prior or pd.isna(pid) or (latest.gsis_id==pid).sum()!=1:
                            value=np.nan
                            break
                        sample = pd.concat(prior)
                        usage = sample.loc[sample.id.eq(pid),'offense_pct']
                        if usage.empty or not usage.between(0,1).all() or sample.loc[sample.id.eq(pid),'game_id'].duplicated().any():
                            value=np.nan
                            break
                        value += usage.sum()/len(prior)
            burdens.append(value)
            if pd.notna(g.actual_margin):
                prev.append(g.game_id)
        result[g.game_id]=burdens[0]-burdens[1]
    return pd.Series(result,name='ol_out')


def run(folder):
    games=base.read_games('strength_games_snapshot.csv')
    games=games[games.season.between(2017,2025)].merge(pd.read_csv('strength_epa_features.csv')[['game_id']+EPA],on='game_id',validate='one_to_one')
    games['ol_out']=games.game_id.map(feature(games,folder))
    games=games.dropna(subset=['actual_margin','spread_line'])
    cols=base.INPUTS+EPA+['total_line']
    results=[]
    for year in range(2021,2026):
        train=games[games.season<year]
        test=games[games.season==year].copy()
        for label,inputs in [('baseline',cols),('injury',cols+['ol_out'])]:
            estimator=base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),Ridge(alpha=1000))
            estimator.fit(train[inputs],train.actual_margin-train.spread_line)
            test[label]=np.clip(.25*estimator.predict(test[inputs]),-2,2)
        missing=test.ol_out.isna()
        test.loc[missing,'injury']=test.loc[missing,'baseline']
        results.append(test[['season','game_id','ol_out','baseline','injury','spread_line','actual_margin']])
    p=pd.concat(results)
    assert len(p)==1424 and p.game_id.is_unique
    direction=np.sign(p.actual_margin-p.spread_line)
    for label in ['baseline','injury']:
        p[label+'_grade']=np.sign(p[label])*direction
    assert (p.baseline_grade==1).sum()==725
    out=Path('ol_out_results')
    out.mkdir(exist_ok=True)
    p.to_csv(out/'games.csv',index=False)
    for period,g in [('all',p),*p.groupby('season')]:
        print(period,'games',len(g),'usable',g.ol_out.notna().sum(),'nonzero',g.ol_out.fillna(0).ne(0).sum(),'WLP',[(g.injury_grade==i).sum().item() for i in [1,-1,0]],'gained',sum(g.injury_grade>g.baseline_grade),'lost',sum(g.injury_grade<g.baseline_grade))
    print('MAE',[(p.actual_margin-p.spread_line-p[m]).abs().mean() for m in ['baseline','injury']])

if __name__=='__main__':
    run(Path(sys.argv[1]))
