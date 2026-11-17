"""Fixed advanced-feature ablations for market/EPA. Research only; no promotion."""
import argparse
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import market_model as base
from market_epa_experiment import ADJ, ALIASES

ROOT='https://github.com/nflverse/nflverse-data/releases/download/'
PBP=['game_id','play_id','home_team','away_team','posteam','defteam','play_type',
     'qb_dropback','qb_kneel','qb_spike','yards_gained','sack','total_home_epa']
GROUPS={'T':'OL','OT':'OL','LT':'OL','RT':'OL','G':'OL','OG':'OL','LG':'OL','RG':'OL','C':'OL','OL':'OL',
        'WR':'SKILL','TE':'SKILL','RB':'SKILL','FB':'SKILL','DE':'FRONT','DT':'FRONT','NT':'FRONT',
        'DL':'FRONT','EDGE':'FRONT','LB':'FRONT','ILB':'FRONT','OLB':'FRONT','CB':'DB','DB':'DB',
        'S':'DB','SS':'DB','FS':'DB','K':'ST','P':'ST','LS':'ST'}
INJ_METRICS=[f'{s}_{g}' for s in ['out','doubtful','questionable'] for g in ['OL','SKILL','FRONT','DB','ST']]
PRESSURE=['pressure_allowed_diff','pressure_created_diff','pressure_matchup_diff',
          'sack_allowed_diff','sack_created_diff','home_pressure_missing','away_pressure_missing']
EXPLOSIVE=['explosive_pass_off_diff','explosive_pass_def_diff','explosive_rush_off_diff',
           'explosive_rush_def_diff','explosive_pass_matchup_diff','explosive_rush_matchup_diff',
           'home_explosive_missing','away_explosive_missing']
SPECIAL=['special_net_epa_diff','home_special_missing','away_special_missing']
INJURY=[m+'_diff' for m in INJ_METRICS]+['home_injury_missing','away_injury_missing','injury_unknown_usage_diff']
ARMS={'epa_baseline':[], 'epa_pressure':PRESSURE, 'epa_explosive':EXPLOSIVE,
      'epa_special':SPECIAL, 'epa_injury':INJURY,
      'epa_all':PRESSURE+EXPLOSIVE+SPECIAL+INJURY}
VERSION='advanced_matchups_fixed_v1'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fetch(url,path,refresh=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    meta=path.with_suffix(path.suffix+'.json')
    if not refresh and path.exists() and meta.exists():
        m=json.loads(meta.read_text())
        if m['sha256']!=digest(path):raise ValueError('Source cache hash mismatch')
        return m
    tmp=path.with_suffix(path.suffix+'.partial')
    req=urllib.request.Request(url,headers={'User-Agent':'NFLAdvancedResearch/1.0'})
    with urllib.request.urlopen(req,timeout=120) as response,tmp.open('wb') as out:
        while chunk:=response.read(1024*1024):out.write(chunk)
        modified=response.headers.get('Last-Modified')
    tmp.replace(path)
    m=dict(url=url,sha256=digest(path),captured_at_utc=datetime.now(timezone.utc).isoformat(),
           last_modified=modified)
    meta.write_text(json.dumps(m,indent=2))
    return m

def canonical(f,columns):
    f=f.copy()
    for c in columns:
        if c in f:f[c]=f[c].replace(ALIASES)
    return f

def aggregate_plays(frame):
    if set(PBP)-set(frame):raise ValueError('Missing PBP columns')
    if frame.duplicated(['game_id','play_id']).any():raise ValueError('Duplicate PBP play')
    p=canonical(frame,['home_team','away_team','posteam','defteam']).sort_values(['game_id','play_id']).copy()
    # Cumulative home EPA differences avoid ambiguity in kickoff possession attribution.
    total=p.groupby('game_id').total_home_epa.ffill().fillna(0)
    p['home_epa_change']=total-total.groupby(p.game_id).shift(fill_value=0)
    normal=p[p.play_type.isin(['pass','run']) & p.qb_kneel.fillna(0).eq(0) &
             p.qb_spike.fillna(0).eq(0) & p.posteam.notna() & p.defteam.notna()].copy()
    normal['dropback']=normal.qb_dropback.eq(1).astype(int)
    normal['rush']=((normal.play_type=='run') & normal.qb_dropback.ne(1)).astype(int)
    normal['pass_explosive']=(normal.dropback.eq(1)&normal.yards_gained.ge(20)).astype(int)
    normal['rush_explosive']=(normal.rush.eq(1)&normal.yards_gained.ge(10)).astype(int)
    normal['sacks']=normal.sack.fillna(0)
    t=normal.groupby(['game_id','posteam','defteam'],as_index=False).agg(
        dropbacks=('dropback','sum'),rushes=('rush','sum'),pass_explosives=('pass_explosive','sum'),
        rush_explosives=('rush_explosive','sum'),sacks=('sacks','sum'))
    special=p[p.play_type.isin(['field_goal','extra_point','punt','kickoff'])].copy()
    st=special.groupby('game_id',as_index=False).agg(st_home_epa=('home_epa_change','sum'),st_plays=('play_id','size'))
    t=t.merge(p[['game_id','home_team']].drop_duplicates(),on='game_id',validate='many_to_one')
    t=t.merge(st,on='game_id',how='left',validate='many_to_one')
    t['st_epa']=np.where(t.posteam.eq(t.home_team),t.st_home_epa,-t.st_home_epa)
    t[['st_epa','st_plays']]=t[['st_epa','st_plays']].fillna(0)
    return t.drop(columns=['home_team','st_home_epa'])

def sources(games,cache):
    cache=Path(cache);teams=[];injuries=[];snaps=[];provenance=[]
    code_hash=digest(__file__)
    for season in sorted(games.season.astype(int).unique()):
        live=season>2025
        aggregate=cache/f'team_{season}.csv'
        manifest=cache/f'team_{season}.json'
        if aggregate.exists() and manifest.exists() and not live:
            meta=json.loads(manifest.read_text())
            if meta['code_sha256']!=code_hash or meta['aggregate_sha256']!=digest(aggregate):
                aggregate.unlink()
        if not aggregate.exists() or live:
            path=cache/f'pbp_{season}.parquet'
            pm=fetch(ROOT+f'pbp/play_by_play_{season}.parquet',path,live)
            t=aggregate_plays(pd.read_parquet(path,columns=PBP))
            source_list=[pm]
            if season>=2018:
                pp=cache/f'pressure_{season}.csv'
                source_list.append(fetch(ROOT+f'pfr_advstats/advstats_week_pass_{season}.csv',pp,live))
                pressure=canonical(pd.read_csv(pp),['team','opponent'])
                if pressure.duplicated(['game_id','team','pfr_player_id']).any():
                    raise ValueError('Duplicate pressure player')
                pressure=pressure.groupby(['game_id','team'],as_index=False).times_pressured.sum(min_count=1)
                t=t.merge(pressure,left_on=['game_id','posteam'],right_on=['game_id','team'],how='left',validate='one_to_one').drop(columns='team')
                # Inconsistent denominators remain unavailable, never clipped into validity.
                t.loc[~t.times_pressured.between(0,t.dropbacks),'times_pressured']=np.nan
            else:t['times_pressured']=np.nan
            t.to_csv(aggregate,index=False)
            meta=dict(code_sha256=code_hash,aggregate_sha256=digest(aggregate),sources=source_list)
            manifest.write_text(json.dumps(meta,indent=2))
            path.unlink(missing_ok=True)
        teams.append(pd.read_csv(aggregate));provenance.append(json.loads(manifest.read_text()))
        for tag,name,collection in [('injuries',f'injuries_{season}.csv',injuries),
                                    ('snap_counts',f'snap_counts_{season}.csv',snaps)]:
            path=cache/name
            m=fetch(ROOT+tag+'/'+name,path,live);provenance.append(m)
            data=pd.read_csv(path)
            # Preserve observation metadata; these are NOT historical capture times.
            data['_source_captured']=m['captured_at_utc']
            collection.append(data)
        print('ADVANCED_SOURCE_READY',season,flush=True)
    path=cache/'players.csv';provenance.append(fetch(ROOT+'players/players.csv',path))
    return pd.concat(teams,ignore_index=True),pd.concat(injuries,ignore_index=True),pd.concat(snaps,ignore_index=True),pd.read_csv(path,low_memory=False),provenance

def matchup_features(games,stats):
    if stats.duplicated(['game_id','posteam']).any():raise ValueError('Duplicate team aggregate')
    h=stats.merge(games[['game_id','gameday','season','actual_margin','home_team','away_team']],
                  on='game_id',validate='many_to_one')
    valid=(h.posteam.eq(h.home_team)&h.defteam.eq(h.away_team))|(h.posteam.eq(h.away_team)&h.defteam.eq(h.home_team))
    if not valid.all():raise ValueError('Team aggregate disagrees with schedule')
    h=h[h.actual_margin.notna()].copy()
    rows=[]
    for g in games.itertuples():
        prior=h[h.season.eq(g.season)&h.gameday.lt(g.gameday)]
        def profile(team):
            own=prior[prior.posteam.eq(team)].sort_values(['gameday','game_id']).tail(8)
            defense=prior[prior.defteam.eq(team)].sort_values(['gameday','game_id']).tail(8)
            def ratio(f,num,den):
                available=f[f[num].notna()&f[den].gt(0)]
                return float(available[num].sum()/available[den].sum()) if len(available) else np.nan
            return dict(pa=ratio(own,'times_pressured','dropbacks'),pc=ratio(defense,'times_pressured','dropbacks'),
                        sa=ratio(own,'sacks','dropbacks'),sc=ratio(defense,'sacks','dropbacks'),
                        po=ratio(own,'pass_explosives','dropbacks'),pd=ratio(defense,'pass_explosives','dropbacks'),
                        ro=ratio(own,'rush_explosives','rushes'),rd=ratio(defense,'rush_explosives','rushes'),
                        st=ratio(own,'st_epa','st_plays'))
        a,b=profile(g.home_team),profile(g.away_team)
        r=dict(game_id=g.game_id,pressure_allowed_diff=a['pa']-b['pa'],pressure_created_diff=a['pc']-b['pc'],
               pressure_matchup_diff=a['pa']*b['pc']-b['pa']*a['pc'],
               sack_allowed_diff=a['sa']-b['sa'],sack_created_diff=a['sc']-b['sc'],
               explosive_pass_off_diff=a['po']-b['po'],explosive_pass_def_diff=a['pd']-b['pd'],
               explosive_rush_off_diff=a['ro']-b['ro'],explosive_rush_def_diff=a['rd']-b['rd'],
               explosive_pass_matchup_diff=a['po']*b['pd']-b['po']*a['pd'],
               explosive_rush_matchup_diff=a['ro']*b['rd']-b['ro']*a['rd'],special_net_epa_diff=a['st']-b['st'])
        for side,p in [('home',a),('away',b)]:
            r[side+'_pressure_missing']=float(any(pd.isna(p[k]) for k in ['pa','pc']))
            r[side+'_explosive_missing']=float(any(pd.isna(p[k]) for k in ['po','pd','ro','rd']))
            r[side+'_special_missing']=float(pd.isna(p['st']))
        rows.append(r)
    return pd.DataFrame(rows)

def injury_features(games,injuries,snaps,players,live_cutoff=None):
    inj=canonical(injuries,['team']).copy()
    for c in ['season','week']:inj[c]=pd.to_numeric(inj[c],errors='raise')
    if 'date_modified' not in inj:inj['date_modified']=None
    inj['_published']=pd.to_datetime(inj.date_modified,utc=True,errors='coerce',format='mixed')
    inj['_observed']=pd.to_datetime(inj.get('_source_captured',pd.Series(index=inj.index,dtype=str)),utc=True,errors='coerce',format='mixed')
    ids=players[['pfr_id','gsis_id']].dropna().drop_duplicates()
    ambiguous=ids.groupby('pfr_id').gsis_id.nunique()
    ids=ids[~ids.pfr_id.isin(ambiguous[ambiguous.gt(1)].index)]
    sn=canonical(snaps,['team'])
    if sn.duplicated(['game_id','team','pfr_player_id']).any():raise ValueError('Duplicate snap player')
    sn=sn.merge(ids,left_on='pfr_player_id',right_on='pfr_id',how='left',validate='many_to_one')
    sn=sn.merge(games[['game_id','gameday','actual_margin']],on='game_id',validate='many_to_one')
    for c in ['offense_pct','defense_pct','st_pct']:
        sn[c]=pd.to_numeric(sn[c],errors='coerce')
        sn.loc[~sn[c].between(0,1),c]=np.nan
    rows=[]
    for g in games.itertuples():
        is_live=live_cutoff is not None and pd.isna(g.actual_margin) and g.gameday>=pd.Timestamp(live_cutoff).normalize()
        cutoff=pd.Timestamp(live_cutoff) if is_live else g.gameday
        kind='REG' if g.game_type=='REG' else 'POST'
        ti=inj[inj.season.eq(g.season)&inj.week.eq(g.week)&inj.game_type.eq(kind)]
        # Historical values are a diagnostic only. Missing publication times cannot be backdated.
        if is_live:
            ti=ti[ti._observed.le(cutoff)&ti._observed.ge(cutoff-pd.Timedelta(hours=48))&
                  (ti._published.isna()|ti._published.le(ti._observed))].copy()
        else:ti=ti[ti._published.notna()&ti._published.lt(cutoff)].copy()
        values=[]
        for team in [g.home_team,g.away_team]:
            team_reports=ti[ti.team.eq(team)].copy()
            available=not team_reports.empty
            v={m:0. if available else np.nan for m in INJ_METRICS}
            v['missing']=float(not available);v['unknown']=0. if available else np.nan
            if available:
                team_reports['_time']=team_reports._published.fillna(team_reports._observed)
                latest=team_reports.groupby('gsis_id')._time.transform('max')
                team_reports=team_reports[team_reports._time.eq(latest)]
                dup=set(team_reports.loc[team_reports.duplicated('gsis_id',keep=False),'gsis_id'])
                history=sn[sn.team.eq(team)&sn.gameday.lt(g.gameday)&sn.actual_margin.notna()]
                last=history[['game_id','gameday']].drop_duplicates().sort_values('gameday').tail(3).game_id
                history=history[history.game_id.isin(last)]
                shares=history.groupby('gsis_id')[['offense_pct','defense_pct','st_pct']].mean()
                for item in team_reports.itertuples():
                    group=GROUPS.get(str(item.position).upper())
                    status=str(item.report_status).lower()
                    if group is None or status not in ['out','doubtful','questionable']:continue
                    if pd.isna(item.gsis_id) or item.gsis_id in dup or item.gsis_id not in shares.index:
                        v['unknown']+=1;continue
                    col='offense_pct' if group in ['OL','SKILL'] else 'st_pct' if group=='ST' else 'defense_pct'
                    share=shares.loc[item.gsis_id,col]
                    if pd.isna(share):v['unknown']+=1
                    elif share>=.5 or (group=='ST' and share>0):v[status+'_'+group]+=float(share)
            values.append(v)
        a,b=values
        r={m+'_diff':a[m]-b[m] for m in INJ_METRICS}
        r.update(game_id=g.game_id,home_injury_missing=a['missing'],away_injury_missing=b['missing'],
                 injury_unknown_usage_diff=a['unknown']-b['unknown'])
        rows.append(r)
    return pd.DataFrame(rows)

def fit(train,columns):
    m=base.make_pipeline(base.SimpleImputer(strategy='median',keep_empty_features=True),
                         base.StandardScaler(),Ridge(alpha=1000.))
    m.fit(train[columns],train.actual_margin-train.spread_line)
    return m

def evaluate(frame):
    predictions=[];folds=[]
    played=frame.dropna(subset=['actual_margin','spread_line'])
    for year in range(2020,2026):
        tr=played[played.season.lt(year)];te=played[played.season.eq(year)]
        if len(tr)<500 or te.empty:raise ValueError('Incomplete annual fold')
        for name,extra in ARMS.items():
            columns=base.INPUTS+ADJ+extra
            m=fit(tr,columns)
            p=te[['game_id','season','actual_margin','spread_line']].copy()
            p['adjustment']=np.clip(.25*m.predict(te[columns]),-2,2)
            p['predicted_margin']=p.spread_line+p.adjustment;p['method']=name;predictions.append(p)
            folds.append(dict(method=name,test_season=year,train_last_season=int(tr.season.max()),train_games=len(tr),test_games=len(te)))
            print('ADVANCED_FOLD',year,name,flush=True)
    return pd.concat(predictions,ignore_index=True),pd.DataFrame(folds)

def report(predictions):
    rows=[]
    for name,g in predictions[predictions.season.ge(2021)].groupby('method'):
        for period,b in [('2021-2025',g)]+[(str(y),v) for y,v in g.groupby('season')]:
            cover=(b.actual_margin-b.spread_line)*np.sign(b.adjustment)
            active=b.adjustment.abs().gt(1e-10);w=int((active&cover.gt(0)).sum());l=int((active&cover.lt(0)).sum())
            rows.append(dict(method=name,period=period,games=len(b),wins=w,losses=l,
                pushes=int((active&cover.eq(0)).sum()),no_pick=int((~active).sum()),
                win_pct=100*w/(w+l) if w+l else None,
                mae=float((b.actual_margin-b.predicted_margin).abs().mean()),
                rmse=float(np.sqrt(((b.actual_margin-b.predicted_margin)**2).mean()))))
    return pd.DataFrame(rows)

def build(games,cache,out,live_cutoff=None):
    team,inj,sn,players,provenance=sources(games,cache)
    extra=matchup_features(games,team)
    extra=extra.merge(injury_features(games,inj,sn,players,live_cutoff),on='game_id',validate='one_to_one')
    extra.to_csv(out/'advanced_features.csv.gz',index=False)
    team.to_csv(out/'team_aggregates.csv.gz',index=False)
    # Preserve exact injury and usage inputs, including actual observation times.
    inj.to_csv(out/'injury_source_rows.csv.gz',index=False)
    sn.to_csv(out/'snap_source_rows.csv.gz',index=False)
    players[['gsis_id','pfr_id']].to_csv(out/'player_ids.csv.gz',index=False)
    (out/'source_provenance.json').write_text(json.dumps(provenance,indent=2))
    return extra

def main():
    a=argparse.ArgumentParser();a.add_argument('--games',default='strength_games_snapshot.csv')
    a.add_argument('--cache',default='advanced_matchup_cache');a.add_argument('--out',default='advanced_matchup_results')
    args=a.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    games=base.read_games(args.games);games=games[games.season.between(2017,2025)].copy()
    extra=build(games,args.cache,out)
    epa=pd.read_csv('strength_epa_features.csv')
    f=games.merge(epa[['game_id']+ADJ],on='game_id',validate='one_to_one').merge(extra,on='game_id',validate='one_to_one')
    if np.isinf(f[PRESSURE+EXPLOSIVE+SPECIAL+INJURY].to_numpy()).any():raise ValueError('Infinite features')
    p,folds=evaluate(f);summary=report(p)
    p.to_csv(out/'predictions.csv',index=False);folds.to_csv(out/'training_folds.csv',index=False)
    summary.to_csv(out/'comparison.csv',index=False)
    ref=pd.read_csv('strength_epa_reference_predictions.csv')
    replay=p[p.method.eq('epa_baseline')].merge(ref[ref.method.eq('opponent_epa')],on='game_id',validate='one_to_one',suffixes=('_new','_ref'))
    if len(replay)!=1693 or not np.allclose(replay.predicted_margin_new,replay.predicted_margin_ref,atol=1e-10,rtol=0):
        raise ValueError('Baseline replay mismatch')
    coverage=extra.groupby(games.set_index('game_id').loc[extra.game_id,'season'].to_numpy())[
        ['home_pressure_missing','away_pressure_missing','home_injury_missing','away_injury_missing']].mean()
    coverage.to_csv(out/'missing_fraction_by_season.csv')
    result=dict(status='DEVELOPMENT_ONLY_NO_LIVE_PROMOTION',version=VERSION,arms=ARMS,
        results=summary[summary.period.eq('2021-2025')].to_dict('records'),
        settings=dict(window_games=8,reset_each_season=True,pass_explosive_yards=20,rush_explosive_yards=10,
                      snap_history_team_games=3,key_player_snap_share=.5,specialist_minimum='positive prior special-teams usage',ridge_alpha=1000,shrink=.25,cap=2),
        baseline_reproduced_games=len(replay),
        hashes={p:digest(p) for p in [__file__,base.__file__,args.games,'strength_epa_features.csv','strength_epa_reference_predictions.csv']},
        limits=['Previously examined historical seasons are development data, not untouched validation.',
                'Pressure faced is a team/QB outcome, not an isolated offensive-line grade or ESPN pass-block win rate.',
                'Historical sources can be revised; game-date lagging does not prove original publication availability.',
                'Injury effects use partial reports, prior usage proxies and categorical designations, not medical probabilities.',
                'Missing injury publication timestamps are withheld from historical features, notably 2025.',
                'Injury report absence does not establish a healthy roster; IR/reserve coverage is incomplete.',
                'No odds threshold filtering, parameter search, automatic promotion or certified betting profit.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print('ADVANCED_RESULT_JSON='+json.dumps(result),flush=True)

if __name__=='__main__':main()
