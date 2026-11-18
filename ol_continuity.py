"""Replay with nflverse snap_counts CSVs (named YEAR.csv), 2025 depth charts and player ID map."""
import argparse
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from market_epa_strength import EPA
import strength_reference_market_model as base

SLOTS = {'LT','LG','C','RG','RT'}
ALIASES = {'LA':'LAR','STL':'LAR','SD':'LAC','OAK':'LV'}


def lineup(shares):
    order = sorted(shares, key=lambda k: (-shares[k],k))
    if len(order)<5 or (len(order)>5 and shares[order[4]]==shares[order[5]]):
        return None
    return set(order[:5])


def chart_groups(depth,players):
    ids = players.dropna(subset=['gsis_id','pfr_id']).groupby('gsis_id').pfr_id.agg(set)
    mapping = {i:next(iter(v)) for i,v in ids.items() if len(v)==1}
    depth = depth.copy()
    depth['dt'] = pd.to_datetime(depth.dt,utc=True)
    depth.team = depth.team.replace(ALIASES)
    # Keep snapshot times even when the chart has missing slots.
    snapshots = depth[['team','dt']].drop_duplicates()
    rows = depth[depth.pos_abb.isin(SLOTS)&depth.pos_rank.eq(1)].copy()
    rows['id'] = rows.gsis_id.map(mapping)
    groups = {}
    for key,r in rows.groupby(['team','dt']):
        r=r[['pos_abb','id']].drop_duplicates()
        if set(r.pos_abb)==SLOTS and len(r)==5 and r.id.nunique()==5:
            groups[key]=set(r.id)
    return {t:[(s,groups.get((t,s))) for s in sorted(g.dt)] for t,g in snapshots.groupby('team')}


def chart_at(history,cut):
    for stamp,ids in reversed(history):
        if stamp<=cut:
            return ids if cut-stamp<=pd.Timedelta(hours=48) else None
    return None


def continuity(ids,hist):
    if not ids or not hist or any(row is None for row in hist):
        return None
    # Both players logging at least half the offensive snaps is a game-level proxy.
    pairs=list(combinations(sorted(ids),2))
    return np.mean([min(row.get(a,0),row.get(b,0))>=.5 for row in hist for a,b in pairs])


def features(games,snaps,charts):
    grouped = {}
    for key,g in snaps.groupby(['game_id','team']):
        if g.pfr_player_id.duplicated().any():
            raise ValueError(f'Duplicate player in {key}')
        grouped[key]=dict(zip(g.pfr_player_id,g.offense_pct))
    rows=[]
    for season,gs in games.groupby('season'):
        history=defaultdict(lambda:deque(maxlen=8))
        for day,slate in gs.groupby('gameday',sort=True):
            updates=[]
            for g in slate.itertuples():
                cut=pd.Timestamp(str(day.date())+' '+g.gametime,tz='America/New_York').tz_convert('UTC')-pd.Timedelta(hours=24)
                scores,changes=[],[]
                for team in (g.home_team,g.away_team):
                    hist=history[team]
                    last=lineup(hist[-1] or {}) if hist else None
                    if season==2025:
                        projected=chart_at(charts[team],cut)
                        comparison=last
                    else:
                        projected=last
                        comparison=lineup(hist[-2] or {}) if len(hist)>1 else None
                    scores.append(continuity(projected,hist))
                    changes.append(None if not projected or not comparison else projected!=comparison)
                    updates.append((team,grouped.get((g.game_id,team))))
                usable=None not in scores
                rows.append(dict(game_id=g.game_id,usable=usable,
                                 changed=usable and any(x is True for x in changes),
                                 continuity=scores[0]-scores[1] if usable else 0.))
            for team,stats in updates:
                history[team].append(stats)
    return pd.DataFrame(rows)


def main(snap_dir,depth_path,player_path):
    files=[snap_dir/f'{y}.csv' for y in range(2017,2026)]
    snaps=pd.concat([pd.read_csv(f) for f in files])
    snaps=snaps[snaps.position.isin(SLOTS|{'G','T','OL','OT','OG'})&snaps.offense_snaps.gt(0)].copy()
    snaps.team=snaps.team.replace(ALIASES)
    assert snaps.pfr_player_id.notna().all() and snaps.offense_pct.between(0,1).all()
    charts=chart_groups(pd.read_csv(depth_path,low_memory=False),pd.read_csv(player_path,low_memory=False))
    games=base.read_games('strength_games_snapshot.csv')
    games=games[games.season.between(2017,2025)].dropna(subset=['actual_margin','spread_line'])
    games=games.merge(features(games,snaps,charts),on='game_id',validate='one_to_one').merge(
        pd.read_csv('strength_epa_features.csv'),on='game_id',validate='one_to_one')
    folds=[]
    for y in range(2020,2026):
        train,test=games[games.season.lt(y)],games[games.season.eq(y)].copy()
        for name,extra in [('baseline',[]),('candidate',['continuity'])]:
            cols=base.INPUTS+EPA+extra
            model=base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),Ridge(alpha=1000))
            model.fit(train[cols],train.actual_margin-train.spread_line)
            test[name]=test.spread_line+np.clip(.25*model.predict(test[cols]),-2,2)
        test.loc[~test.usable,'candidate']=test.loc[~test.usable,'baseline']
        folds.append(test)
    pred=pd.concat(folds)
    ref=pd.read_csv('strength_epa_reference_predictions.csv').query("method=='opponent_epa'")
    pd.testing.assert_series_equal(pred.set_index('game_id').baseline.sort_index(),
        ref.set_index('game_id').predicted_margin.sort_index(),check_names=False,atol=1e-10,rtol=0)
    games=[]
    for y in range(2021,2026):
        prior,test=pred[pred.season.lt(y)],pred[pred.season.eq(y)].copy()
        strike=np.floor(test.spread_line)+.5
        for name in ('baseline','candidate'):
            errors=(prior.actual_margin-prior[name]).to_numpy()
            prob=[np.mean(np.floor(m+errors+.5)>k) for m,k in zip(test[name],strike)]
            test[name+'_brier']=(prob-test.actual_margin.gt(strike).astype(float))**2
            test[name+'_ats']=np.sign((test[name]-test.spread_line)*(test.actual_margin-test.spread_line))
        games.append(test)
    g=pd.concat(games)
    assert g.game_id.is_unique and len(g)==1424
    groups={'all':g,'prior_usage_2021_2024':g[g.season.lt(2025)],'changed_prior_usage':g[g.season.lt(2025)&g.changed],
            'changed_chart_2025':g[g.season.eq(2025)&g.changed]}
    groups.update({str(y):s for y,s in g.groupby('season')})
    report=[]
    for label,s in groups.items():
        b,c=s.baseline_ats,s.candidate_ats
        report.append(dict(group=label,games=len(s),usable=s.usable.sum(),base_wins=b.gt(0).sum(),
             wins=c.gt(0).sum(),losses=c.lt(0).sum(),pushes=c.eq(0).sum(),gained=c.gt(b).sum(),lost=c.lt(b).sum(),
             base_brier=s.baseline_brier.mean(),candidate_brier=s.candidate_brier.mean()))
    out=Path('ol_continuity_results')
    out.mkdir(exist_ok=True)
    pd.DataFrame(report).to_csv(out/'summary.csv',index=False)
    pd.DataFrame([dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files+[depth_path,player_path]]).to_csv(out/'sources.csv',index=False)
    g[['game_id','season','spread_line','actual_margin','baseline','candidate','continuity','usable','changed',
       'candidate_brier','baseline_brier']].to_csv(out/'games.csv',index=False)
    print(pd.DataFrame(report).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('snap_dir','depth_path','player_path'):
        parser.add_argument(name,type=Path)
    main(**vars(parser.parse_args()))
