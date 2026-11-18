"""Usage: python qb_mix_replay.py PBP_CACHE depth_charts_2025.csv
nflverse PBP CSV.GZ seasons 2017–2025 and the nflverse 2025 depth chart CSV.
"""
from pathlib import Path
import sys
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from market_epa_strength import EPA
import strength_reference_market_model as base

ALIASES = {'LA':'LAR','STL':'LAR','SD':'LAC','OAK':'LV'}
OUT = Path('qb_mix_results')


def qb_plays(path):
    cols = ['game_id','posteam','defteam','qb_dropback','qb_kneel','qb_spike','play_type',
            'passer_player_id','passer_id','qb_scramble','rusher_player_id','qb_epa']
    p = pd.read_csv(path,usecols=cols,low_memory=False)
    p = p[p.qb_dropback.eq(1) & p.play_type.isin(['pass','run']) &
          p.qb_kneel.fillna(0).eq(0) & p.qb_spike.fillna(0).eq(0)].copy()
    p['qb'] = p.passer_player_id.fillna(p.passer_id).fillna(p.rusher_player_id.where(p.qb_scramble.eq(1)))
    p = p.dropna(subset=['qb','qb_epa','posteam','defteam'])
    for col in ['posteam','defteam']:
        p[col] = p[col].replace(ALIASES)
    return p.groupby(['game_id','posteam','defteam','qb']).agg(
        n=('qb_epa','size'),value=('qb_epa','mean')).reset_index()


def ratings(p):
    if p.empty:
        return {},0.
    backups = p[p.qb.ne(p.starter) & p.starter.notna()]
    prior = np.average(backups.value,weights=backups.n) if len(backups) else 0.
    q, players = pd.factorize(p.qb)
    d, defenses = pd.factorize(p.defteam)
    x = np.zeros((len(p),len(players)+len(defenses)))
    idx = np.arange(len(p))
    x[idx,q] = 1
    x[idx,len(players)+d] = 1
    # Center the defense indicators under exposure weights to anchor the QB prior.
    x[:,len(players):] -= np.average(x[:,len(players):],axis=0,weights=p.n)
    fit = Ridge(alpha=100,fit_intercept=False).fit(x,p.value-prior,sample_weight=p.n)
    return dict(zip(players,prior+fit.coef_[:len(players)])),prior


def depth_groups(path):
    cols = ['dt','team','pos_abb','pos_rank','gsis_id']
    d = pd.read_csv(path,usecols=cols)
    d = d[d.pos_abb.eq('QB')].copy()
    d['dt'] = pd.to_datetime(d.dt,utc=True)
    d.team = d.team.replace(ALIASES)
    return {team:rows for team,rows in d.groupby('team')}


def chart_id(rows,cutoff):
    recent = rows[rows.dt.le(cutoff) & rows.dt.ge(cutoff-pd.Timedelta(hours=48))]
    if recent.empty:
        return None
    first = recent[recent.dt.eq(recent.dt.max()) & recent.pos_rank.eq(1)].gsis_id
    return first.iloc[0] if len(first) and first.notna().all() and first.nunique()==1 else None


def features(games,plays,charts):
    rows = []
    for day,group in games.groupby('gameday',sort=True):
        # Exclude the preceding calendar day's games to respect the 24-hour chart cutoff.
        past = plays[plays.gameday.lt(day-pd.Timedelta(days=1)) & plays.gameday.ge(day-pd.Timedelta(days=730))]
        values,prior = ratings(past)
        for g in group.itertuples():
            cutoff = pd.Timestamp(str(day.date())+' '+g.gametime,tz='America/New_York').tz_convert('UTC')-pd.Timedelta(hours=24)
            gaps, switches, previous, selected = [], [], [], []
            for side in ['home','away']:
                team = getattr(g,side+'_team')
                history = past[past.season.eq(g.season) & past.posteam.eq(team)]
                days = sorted(history.gameday.unique())[-8:]
                history = history[history.gameday.isin(days)]
                last = None if history.empty else history.loc[history.gameday.eq(days[-1]),'starter'].iloc[0]
                qb = chart_id(charts[team],cutoff) if g.season==2025 else last
                previous.append(last)
                selected.append(qb)
                switches.append(pd.notna(qb) and pd.notna(last) and qb!=last)
                if history.empty or pd.isna(qb):
                    gaps.append(None)
                else:
                    mix = np.average(history.qb.map(values).fillna(prior),weights=history.n)
                    gaps.append(values.get(qb,prior)-mix)
            usable = all(x is not None for x in gaps)
            actual = [g.home_qb_id,g.away_qb_id]
            # Actual current-game IDs are used solely for postgame evaluation labels.
            changed = any(pd.notna(a) and pd.notna(b) and a!=b for a,b in zip(actual,previous))
            rows.append(dict(game_id=g.game_id,gap=gaps[0]-gaps[1] if usable else 0.,
                             usable=usable,chart_pair=g.season==2025 and all(pd.notna(selected)),
                             flagged=any(switches),actual_change=changed,
                             home_expected=selected[0],away_expected=selected[1],
                             home_previous=previous[0],away_previous=previous[1]))
    return pd.DataFrame(rows)


def main(cache,chart):
    OUT.mkdir(exist_ok=True)
    games = base.read_games('strength_games_snapshot.csv')
    games = games[games.season.between(2017,2025)].dropna(subset=['actual_margin','spread_line'])
    files = [cache/f'play_by_play_{year}.csv.gz' for year in range(2017,2026)]
    p = pd.concat([qb_plays(path) for path in files]).merge(games[['game_id','season','gameday',
        'home_team','away_team','home_qb_id','away_qb_id']],on='game_id',validate='many_to_one')
    assert p.groupby('game_id').posteam.nunique().reindex(games.game_id).eq(2).all()
    assert ((p.posteam.eq(p.home_team)&p.defteam.eq(p.away_team)) |
            (p.posteam.eq(p.away_team)&p.defteam.eq(p.home_team))).all()
    p['starter'] = np.where(p.posteam.eq(p.home_team),p.home_qb_id,p.away_qb_id)
    games = games.merge(features(games,p,depth_groups(chart)),on='game_id',validate='one_to_one')
    games = games.merge(pd.read_csv('strength_epa_features.csv'),on='game_id',validate='one_to_one')
    frames = []
    for year in range(2020,2026):
        train,test = games[games.season.lt(year)],games[games.season.eq(year)].copy()
        for arm,extra in [('baseline',[]),('candidate',['gap'])]:
            cols = base.INPUTS+EPA+extra
            model = make_pipeline(base.SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1000))
            model.fit(train[cols],train.actual_margin-train.spread_line)
            test[arm] = test.spread_line+np.clip(.25*model.predict(test[cols]),-2,2)
        test.loc[~test.usable,'candidate'] = test.loc[~test.usable,'baseline']
        frames.append(test)
    pred = pd.concat(frames)
    ref = pd.read_csv('strength_epa_reference_predictions.csv').query("method=='opponent_epa'")
    pd.testing.assert_series_equal(pred.set_index('game_id').baseline.sort_index(),
        ref.set_index('game_id').predicted_margin.sort_index(),check_names=False,atol=1e-10,rtol=0)
    scored = []
    for year in range(2021,2026):
        past,test = pred[pred.season.lt(year)],pred[pred.season.eq(year)].copy()
        for arm in ['baseline','candidate']:
            errors = (past.actual_margin-past[arm]).to_numpy()
            strike = np.floor(test.spread_line)+.5
            prob = [np.mean(np.floor(mu+errors+.5)>k) for mu,k in zip(test[arm],strike)]
            test[arm+'_brier'] = (prob-(test.actual_margin>strike).astype(float))**2
            test[arm+'_result'] = np.sign((test[arm]-test.spread_line)*(test.actual_margin-test.spread_line))
        scored.append(test)
    s = pd.concat(scored)
    assert s.game_id.is_unique and len(s)==1424
    cols = ['season','game_id','gap','usable','flagged','actual_change','chart_pair','spread_line',
            'actual_margin','home_previous','away_previous','home_expected','away_expected',
            'home_qb_id','away_qb_id','baseline','candidate','baseline_brier','candidate_brier',
            'baseline_result','candidate_result']
    s[cols].to_csv(OUT/'games.csv',index=False)
    counts = []
    chart_games = s[s.season.eq(2025) & s.chart_pair]
    for side in ['home','away']:
        actual, expected, previous = [chart_games[c] for c in
            [side+'_qb_id',side+'_expected',side+'_previous']]
        valid = actual.notna() & expected.notna() & previous.notna()
        flag = valid & expected.ne(previous)
        change = valid & actual.ne(previous)
        counts.append(dict(eligible=int(valid.sum()),expected_changes=int(flag.sum()),
                           actual_changes=int(change.sum()),correct_new_qb=int((flag & expected.eq(actual)).sum()),
                           missed_changes=int((change & ~flag).sum())))
    pd.DataFrame(counts).sum().to_frame().T.to_csv(OUT/'starter_audit.csv',index=False)
    groups = {'all':s,'proxy_2021_2024':s[s.season.lt(2025)],'charts_2025':s[s.season.eq(2025)],
              '2025_chart_change':s[s.flagged & s.chart_pair & s.usable],
              '2025_actual_change':s[s.season.eq(2025) & s.actual_change],
              'all_actual_change':s[s.actual_change]}
    groups.update({str(y):g for y,g in s.groupby('season')})
    rows = []
    for label,g in groups.items():
        a,b = g.candidate_result,g.baseline_result
        rows.append(dict(group=label,games=len(g),usable=int(g.usable.sum()),chart_pairs=int(g.chart_pair.sum()),
                         wins=int(a.gt(0).sum()),losses=int(a.lt(0).sum()),pushes=int(a.eq(0).sum()),
                         base_wins=int(b.gt(0).sum()),gained=int(a.gt(b).sum()),lost=int(a.lt(b).sum()),
                         candidate_brier=g.candidate_brier.mean(),base_brier=g.baseline_brier.mean()))
    report = pd.DataFrame(rows)
    report.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame([dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files+[chart]]).to_csv(OUT/'sources.csv',index=False)
    print(report.to_string(index=False))


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]))
