"""Run from the repository root: python recovery_pace.py CACHE_DIR."""
from collections import defaultdict, deque
from pathlib import Path
import sys
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from market_epa_experiment import ADJ, ALIASES, build_features
import strength_reference_market_model as base

OUT = Path('recovery_pace_results')
OFFSETS = (-7,-3,0,3,7)


def aggregate(path):
    fields = ['game_id','posteam','defteam','fixed_drive','qtr','play_type','epa','qb_kneel','qb_spike',
              'sack','fumble','fumbled_1_team','fumbled_2_team','fumble_recovery_1_team','down']
    p = pd.read_csv(path,usecols=fields,low_memory=False)
    for col in ['posteam','defteam','fumbled_1_team','fumble_recovery_1_team']:
        p[col] = p[col].replace(ALIASES)
    snaps = p[p.play_type.isin(['run','pass']) & p.epa.notna() &
              p.qb_kneel.fillna(0).eq(0) & p.qb_spike.fillna(0).eq(0)].copy()
    snaps['success'] = snaps.epa.gt(0)
    teams = snaps.groupby(['game_id','posteam','defteam']).agg(
        plays=('epa','size'),epa=('epa','mean'),success=('success','mean')).reset_index()
    reg = p[p.qtr.between(1,4) & p.down.notna() & p.posteam.notna() &
            p.play_type.isin(['run','pass','punt','field_goal','qb_kneel','qb_spike'])]
    drives = reg.groupby('game_id').fixed_drive.nunique().div(2).rename('pace')
    teams = teams.merge(drives,on='game_id',how='left')
    fum = snaps[snaps.fumble.eq(1) & snaps.fumbled_2_team.isna() &
                snaps.fumbled_1_team.eq(snaps.posteam) &
                (snaps.fumble_recovery_1_team.eq(snaps.posteam) |
                 snaps.fumble_recovery_1_team.eq(snaps.defteam))].copy()
    fum['retained'] = fum.posteam.eq(fum.fumble_recovery_1_team)
    for kind,mask in [('sack',fum.sack.eq(1)),('other',~fum.sack.eq(1))]:
        counts = fum[mask].groupby(['game_id','posteam']).retained.agg(['size','sum'])
        counts.columns=[kind+'_n',kind+'_ret']
        teams=teams.merge(counts,on=['game_id','posteam'],how='left')
        teams[counts.columns]=teams[counts.columns].fillna(0)
    return teams


def pregame(games,stats):
    by_game={k:g for k,g in stats.groupby('game_id')}
    rows=[]
    for season,fixtures in games.groupby('season',sort=True):
        prev=stats[stats.season<season]
        rates=[(prev[k+'_ret'].sum()+1)/(prev[k+'_n'].sum()+2) for k in ['sack','other']]
        mean_pace=prev.pace.mean() if len(prev) else 11.
        histories=defaultdict(lambda:deque(maxlen=8))
        for day,slate in fixtures.groupby('gameday',sort=True):
            updates=[]
            for g in slate.itertuples():
                ts=[g.home_team,g.away_team]
                luck=[]; pace=[]
                for team in ts:
                    h=histories[team]
                    luck.append(sum(a for a,_ in h))
                    pace.append(np.mean([b for _,b in h]) if h else mean_pace)
                rows.append(dict(game_id=g.game_id,recoveries=luck[0]-luck[1],drives=np.mean(pace)))
                agg=by_game.get(g.game_id)
                if agg is None:
                    continue
                assert set(agg.posteam)==set(ts) and len(agg)==2
                net={}
                for r in agg.itertuples():
                    net[r.posteam]=r.sack_ret-rates[0]*r.sack_n+r.other_ret-rates[1]*r.other_n
                for t,opp in [ts,ts[::-1]]:
                    updates.append((t,net[t]-net[opp],agg.pace.iloc[0]))
            for t,a,b in updates:
                histories[t].append((a,b))
    return pd.DataFrame(rows)


def prepare(cache):
    files=[cache/f'play_by_play_{y}.csv.gz' for y in range(1999,2026)]
    teams=pd.concat([aggregate(p) for p in files],ignore_index=True)
    pd.DataFrame([dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]).to_csv(OUT/'sources.csv',index=False)
    return teams


def predict(teams,early):
    start,end,first=(1999,2019,2003) if early else (2017,2025,2020)
    if early:
        raw=pd.read_csv('strength_games_snapshot.csv').query('@start<=season<=@end').copy()
        raw['gameday']=pd.to_datetime(raw.gameday,utc=True)
        for col in ('home_team','away_team'):
            raw[col]=raw[col].replace(ALIASES)
        games=base.make_features(raw.sort_values(['gameday','game_id']))
        games['home_field']=games.location.eq('Home').astype(float)
    else:
        games=base.read_games('strength_games_snapshot.csv')
    games=games[games.season.between(start,end)].dropna(subset=['actual_margin','spread_line','total_line'])
    stats=teams.merge(games[['game_id','season','gameday','home_team','away_team']],on='game_id',validate='many_to_one')
    counts=stats.groupby('game_id').size().reindex(games.game_id,fill_value=0)
    missing=set(counts[counts.ne(2)].index)
    assert missing==({'1999_01_BAL_STL','2000_03_SD_KC','2000_06_BUF_MIA'} if early else set())
    stats=stats[~stats.game_id.isin(missing)]
    assert stats.pace.gt(0).all()
    features=pregame(games,stats)
    epa=build_features(games,stats) if early else pd.read_csv('strength_epa_features.csv')
    games=games.merge(features,on='game_id',validate='one_to_one').merge(epa,on='game_id',validate='one_to_one')
    folds=[]
    for y in range(first,end+1):
        train=games[games.season.lt(y)]
        test=games[games.season.eq(y)].copy()
        for name,extra in [('baseline',[]),('fumbles',['recoveries'])]:
            cols=base.INPUTS+ADJ+extra
            fit=make_pipeline(base.SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1000))
            fit.fit(train[cols],train.actual_margin-train.spread_line)
            test[name]=test.spread_line+np.clip(.25*fit.predict(test[cols]),-2,2)
        folds.append(test[['season','game_id','actual_margin','spread_line','total_line','drives','baseline','fumbles']])
    pred=pd.concat(folds)
    if not early:
        ref=pd.read_csv('strength_epa_reference_predictions.csv').query("method=='opponent_epa'").set_index('game_id')
        np.testing.assert_allclose(pred.baseline,ref.loc[pred.game_id,'predicted_margin'],rtol=0,atol=1e-10)
    elif Path('earlier_total_results/games.csv').exists():
        ref=pd.read_csv('earlier_total_results/games.csv').set_index('game_id').loc[pred.game_id]
        np.testing.assert_allclose(pred.baseline,ref.spread_line+ref.baseline,rtol=0,atol=1e-10)
    return pred


def evaluate(pred):
    ats=[]; prices=[]
    pred=pred.assign(spread_size=pred.spread_line.abs())
    for year in sorted(pred.season.unique())[1:]:
        h=pred[pred.season.lt(year)]
        t=pred[pred.season.eq(year)]
        outcome=np.sign(t.actual_margin-t.spread_line)
        baseline=outcome*np.sign(t.baseline-t.spread_line)
        candidate=outcome*np.sign(t.fumbles-t.spread_line)
        ats.append(dict(season=year,games=len(t),baseline_wins=baseline.gt(0).sum(),wins=candidate.gt(0).sum(),
                        losses=candidate.lt(0).sum(),pushes=outcome.eq(0).sum(),
                        gained=candidate.gt(baseline).sum(),lost=candidate.lt(baseline).sum()))
        for arm,margin,cols in [('baseline','baseline',[]),('fumbles','fumbles',[]),
                                ('spread_total','baseline',['spread_size','total_line']),
                                ('possessions','baseline',['spread_size','total_line','drives'])]:
            errors=(h.actual_margin-h[margin]).to_numpy()
            widths=np.ones(len(t))
            if cols:
                fit=make_pipeline(StandardScaler(),Ridge(alpha=100)).fit(h[cols],np.abs(errors))
                errors=errors/np.maximum(1,fit.predict(h[cols]))
                widths=np.maximum(1,fit.predict(t[cols]))
            residuals=np.sort(errors)
            for key in [*OFFSETS,'favorite_7.5']:
                if key=='favorite_7.5':
                    # Reverse orientation when the visitor is favored; omit pick'em.
                    strikes=np.where(t.spread_line<0,-7.5,7.5)
                    valid=t.spread_line.ne(0).to_numpy()
                else:
                    strikes=np.floor(t.spread_line)+.5+key
                    valid=np.ones(len(t),dtype=bool)
                # Integer-rounded margin exceeds half-integer k iff margin >= k+0.5.
                idx=np.searchsorted(residuals,(strikes+.5-t[margin])/widths,side='left')
                probability=1-idx/len(errors)
                loss=(probability-t.actual_margin.gt(strikes).astype(float))**2
                prices.append(dict(season=year,arm=arm,threshold=str(key),games=sum(valid),
                                   squared_error_sum=loss[valid].sum()))
    return pd.DataFrame(ats),pd.DataFrame(prices)


def run(teams):
    ats=[]; pricing=[]
    for label,early in [('recent',False),('earlier',True)]:
        a,p=evaluate(predict(teams,early))
        ats.append(a.assign(era=label));pricing.append(p.assign(era=label))
    a=pd.concat(ats); p=pd.concat(pricing)
    a.to_csv(OUT/'ats.csv',index=False);p.to_csv(OUT/'pricing.csv',index=False)
    print(a.drop(columns='season').groupby('era').sum())
    total=p.groupby(['era','threshold','arm'])[['games','squared_error_sum']].sum()
    print((total.squared_error_sum/total.games).unstack().to_string())


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    run(prepare(Path(sys.argv[1])))
