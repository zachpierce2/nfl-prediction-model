"""Independent prior-season NFL spread replay. Run from repo root with --help."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from market_epa_experiment import ADJ, ALIASES, build_features
import strength_reference_market_model as base

FOOTBALL = [c for c in base.FEATURES if not c.startswith('qb_')] + ADJ
OUT = Path('football_spread_results')


def older_ratings(schedule, cache):
    frames = []
    cols = ['game_id','posteam','defteam','epa','play_type','qb_kneel','qb_spike']
    for year in range(1999,2020):
        p = pd.read_csv(cache/f'play_by_play_{year}.csv.gz',usecols=cols,low_memory=False)
        p = p[p.play_type.isin(['run','pass']) & p.epa.notna() &
              p.qb_kneel.fillna(0).eq(0) & p.qb_spike.fillna(0).eq(0)].copy()
        for col in ['posteam','defteam']:
            p[col] = p[col].replace(ALIASES)
        p['success'] = p.epa.gt(0)
        frames.append(p.groupby(['game_id','posteam','defteam']).agg(
            plays=('epa','size'),epa=('epa','mean'),success=('success','mean')).reset_index())
    teams = pd.concat(frames).merge(schedule[['game_id','season','gameday']],on='game_id',validate='many_to_one')
    teams = teams[teams.game_id.map(teams.groupby('game_id').size()).eq(2)]
    return build_features(schedule,teams)[['game_id']+ADJ]


def replay(era,schedule,features,first,last):
    data = schedule.merge(features[['game_id']+ADJ],on='game_id',validate='one_to_one')
    frames = []
    for year in range(first,last+1):
        train = data[data.season.lt(year) & data.actual_margin.notna()]
        current = data[data.season.eq(year) & data.actual_margin.notna() & data.spread_line.notna()].copy()
        assert train.season.max()<year and current[FOOTBALL].notna().all().all()
        model = base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),
                                   base.Ridge(alpha=1000))
        model.fit(train[FOOTBALL],train.actual_margin)
        current['football'] = model.predict(current[FOOTBALL])
        market_train = train.dropna(subset=['spread_line'])
        cols = base.INPUTS+ADJ
        residual = base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),
                                      base.Ridge(alpha=1000))
        residual.fit(market_train[cols],market_train.actual_margin-market_train.spread_line)
        current['current'] = current.spread_line+np.clip(.25*residual.predict(current[cols]),-2,2)
        frames.append(current[['game_id','season','week','spread_line','current','football','actual_margin']])
    return pd.concat(frames).assign(era=era)


def summarize(data):
    rows=[]
    outcomes=np.sign(data.actual_margin-data.spread_line)
    gap=(data.football-data.spread_line).abs()
    opposite=(data.football-data.spread_line)*(data.current-data.spread_line)<0
    slices={'all':data.index==data.index,'opposite_picks':opposite,'gap_under_1':gap.lt(1),
            'gap_1_to_under_3':gap.ge(1)&gap.lt(3),'gap_3_to_under_7':gap.ge(3)&gap.lt(7),
            'gap_7_plus':gap.ge(7),'gap_3_plus':gap.ge(3)}
    for label,mask in slices.items():
        g=data[mask]
        for name in ['spread_line','current','football']:
            lean=np.sign(g[name]-g.spread_line)
            graded=lean*outcomes[mask]
            rows.append(dict(group=label,model=name,games=len(g),
                             mae=(g[name]-g.actual_margin).abs().mean(),
                             wins=graded.gt(0).sum(),losses=graded.lt(0).sum(),
                             pushes=(lean.ne(0)&outcomes[mask].eq(0)).sum(),no_pick=lean.eq(0).sum()))
    return pd.DataFrame(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--earlier-cache',type=Path,help='Optional 1999–2019 nflverse PBP CSV.GZ folder')
    parser.add_argument('--earlier-features',type=Path,help='Optional precomputed earlier EPA feature table')
    parser.add_argument('--current-schedule',type=Path)
    parser.add_argument('--current-features',type=Path)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    recent=replay('2021-2025',base.read_games('strength_games_snapshot.csv'),
                  pd.read_csv('strength_epa_features.csv'),2021,2025)
    ref=pd.read_csv('strength_epa_reference_predictions.csv').query("method=='opponent_epa'").set_index('game_id')
    np.testing.assert_allclose(recent.current,ref.loc[recent.game_id,'predicted_margin'],atol=1e-10,rtol=0)
    frames=[recent]
    if args.earlier_features or args.earlier_cache:
        old=pd.read_csv('strength_games_snapshot.csv').query('1999<=season<=2019').copy()
        old.gameday=pd.to_datetime(old.gameday,utc=True)
        for c in ['home_team','away_team']:
            old[c]=old[c].replace(ALIASES)
        old=base.make_features(old.sort_values(['gameday','game_id']))
        old['home_field']=old.location.eq('Home').astype(float)
        rates=pd.read_csv(args.earlier_features) if args.earlier_features else older_ratings(old,args.earlier_cache)
        frames.append(replay('2004-2019',old,rates,2004,2019))
    if args.current_schedule and args.current_features:
        games=base.read_games(args.current_schedule)
        year=int(games.season.max())
        frames.append(replay(str(year),games,pd.read_csv(args.current_features),year,year))
    result=pd.concat(frames,ignore_index=True)
    assert result.game_id.is_unique
    result.to_csv(OUT/'games.csv',index=False)
    summary=pd.concat([summarize(g).assign(era=e) for e,g in result.groupby('era')])
    summary.to_csv(OUT/'summary.csv',index=False)
    print(summary.query("group in ['all','opposite_picks','gap_3_plus']").to_string(index=False))
