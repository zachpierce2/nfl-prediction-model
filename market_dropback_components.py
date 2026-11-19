"""Fixed, development-only disjoint play component ablation. See DROPBACK_PROTOCOL.md."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import advanced_matchups as source
import market_model as base
import market_epa_experiment as original
from market_epa_simplification import metrics

VERSION='dropback_components_v1'
CATEGORIES=['pass_attempt','sack','scramble','designed_run']
METRICS=['epa','success','rate']
EXTRA=[f'{kind}_{side}_{metric}_diff' for kind in CATEGORIES for side in ['off','def'] for metric in METRICS]
REFRESH=['refreshed_'+c for c in original.ADJ]
COLUMNS=original.PBP_COLUMNS+['qb_dropback','qb_scramble','sack']

def aggregate(p):
    if set(COLUMNS)-set(p):raise ValueError('Missing component PBP columns')
    if p.duplicated(['game_id','play_id']).any():raise ValueError('Duplicate play identifiers')
    p=p[p.play_type.isin(['pass','run'])&p.epa.notna()&p.posteam.notna()&p.defteam.notna()&p.qb_kneel.fillna(0).eq(0)&p.qb_spike.fillna(0).eq(0)].copy()
    for flag in ['qb_dropback','qb_scramble','sack']:
        if not p[flag].isin([0,1]).all():raise ValueError('Missing or invalid play flag: '+flag)
    if not np.isfinite(p.epa.to_numpy()).all():raise ValueError('Nonfinite EPA')
    invalid=((p.sack.eq(1)|p.qb_scramble.eq(1))&p.qb_dropback.ne(1))|(p.play_type.eq('pass')&p.qb_dropback.ne(1))|(p.play_type.eq('run')&p.qb_dropback.eq(1)&p.qb_scramble.ne(1))
    if invalid.any():raise ValueError('Inconsistent play classification flags')
    for c in ['posteam','defteam']:p[c]=p[c].replace(original.ALIASES)
    # Replay reversals can leave both sack/scramble flags; the final sack flag wins.
    p['category']=np.select([p.sack.eq(1),p.qb_scramble.eq(1),p.qb_dropback.eq(1)],['sack','scramble','pass_attempt'],default='designed_run')
    p['success']=p.epa.gt(0).astype(float)
    keys=['game_id','posteam','defteam']
    total=p.groupby(keys,as_index=False).agg(plays=('epa','size'))
    grouped=p.groupby(keys+['category'],as_index=False).agg(epa_sum=('epa','sum'),success_sum=('success','sum'),category_plays=('epa','size'))
    dense=total.merge(pd.DataFrame({'category':CATEGORIES}),how='cross').merge(grouped,on=keys+['category'],how='left',validate='one_to_one').fillna({'epa_sum':0.,'success_sum':0.,'category_plays':0})
    dense['epa']=dense.epa_sum/dense.plays;dense['success']=dense.success_sum/dense.plays;dense['rate']=dense.category_plays/dense.plays
    return dense

def prepare(games,t):
    if t.duplicated(['game_id','posteam','category']).any():raise ValueError('Duplicate team/category games')
    if set(t.category)!=set(CATEGORIES):raise ValueError('Missing play categories')
    if not np.isfinite(t[['plays','epa','success','rate']].to_numpy()).all() or not t.plays.gt(0).all():raise ValueError('Invalid component statistics')
    if not t.success.between(0,1).all() or not t.rate.between(0,1).all() or not t.success.le(t.rate+1e-12).all():raise ValueError('Invalid component contributions')
    h=t.merge(games[['game_id','gameday','season','home_team','away_team','actual_margin']],on='game_id',validate='many_to_one')
    valid=(h.posteam.eq(h.home_team)&h.defteam.eq(h.away_team))|(h.posteam.eq(h.away_team)&h.defteam.eq(h.home_team))
    if not valid.all():raise ValueError('Component team disagrees with schedule')
    played=games[games.actual_margin.notna()]
    counts=h.groupby('game_id').size().reindex(played.game_id,fill_value=0)
    if not counts.eq(8).all():raise ValueError('Incomplete category coverage')
    for _,g in h.groupby(['game_id','posteam']):
        if set(g.category)!=set(CATEGORIES) or g.plays.nunique()!=1 or not np.isclose(g.rate.sum(),1,atol=1e-12):raise ValueError('Component conservation failed')
    return h[h.actual_margin.notna()].copy()

def build_features(games,t):
    history=prepare(games,t);rows=[]
    for year,season in games.groupby('season',sort=True):
        h=history[history.season.eq(year)]
        for day,group in season.groupby('gameday',sort=True):
            prior=h[h.gameday.lt(day)]
            # Select eight GAMES, including zero-count categories, never eight category appearances.
            game_rows=prior[['game_id','posteam','gameday']].drop_duplicates().sort_values(['gameday','game_id']).groupby('posteam',sort=False).tail(8)
            recent=prior.merge(game_rows[['game_id','posteam']],on=['game_id','posteam'],validate='many_to_one')
            teams=sorted(set(group.home_team)|set(group.away_team)|set(recent.posteam)|set(recent.defteam));idx={team:i for i,team in enumerate(teams)};n=len(teams)
            profiles={}
            for category in CATEGORIES:
                r=recent[recent.category.eq(category)];off={};defense={}
                if len(r):
                    x=np.zeros((len(r),2*n));x[np.arange(len(r)),r.posteam.map(idx).to_numpy()]=1;x[np.arange(len(r)),n+r.defteam.map(idx).to_numpy()]=1
                    fit=Ridge(alpha=200.).fit(x,r[METRICS].to_numpy(),sample_weight=r.plays.to_numpy())
                    for team,i in idx.items():off[team]=fit.coef_[:,i];defense[team]=fit.coef_[:,n+i]
                profiles[category]=(off,defense)
            zero=np.zeros(3)
            for game in group.itertuples():
                row=dict(game_id=game.game_id)
                for category,(off,defense) in profiles.items():
                    od=off.get(game.home_team,zero)-off.get(game.away_team,zero)
                    dd=defense.get(game.away_team,zero)-defense.get(game.home_team,zero)
                    for j,m in enumerate(METRICS):row[f'{category}_off_{m}_diff']=od[j];row[f'{category}_def_{m}_diff']=dd[j]
                rows.append(row)
    return pd.DataFrame(rows)

def evaluate(frame):
    arms={'epa_baseline':base.INPUTS+original.ADJ,'refreshed_epa_control':base.INPUTS+REFRESH,'dropback_components':base.INPUTS+EXTRA};predictions=[];folds=[]
    frame=frame.dropna(subset=['actual_margin','spread_line'])
    for year in range(2020,2026):
        train=frame[frame.season.lt(year)];test=frame[frame.season.eq(year)]
        if len(train)<500 or test.empty:raise ValueError('Insufficient fold')
        folds.append(dict(test_season=year,train_last_season=int(train.season.max()),train_games=len(train),test_games=len(test)))
        for name,cols in arms.items():
            model=base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),Ridge(alpha=1000.))
            model.fit(train[cols],train.actual_margin-train.spread_line)
            g=test[['game_id','season','actual_margin','spread_line']].copy();g['adjustment']=np.clip(.25*model.predict(test[cols]),-2,2)
            g['predicted_margin']=g.spread_line+g.adjustment;g['method']=name;predictions.append(g)
    return pd.concat(predictions,ignore_index=True),pd.DataFrame(folds)

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
    games=base.read_games(args.games);games=games[games.season.between(2017,2025)].copy()
    snapshot=Path(args.team_components)
    if snapshot.exists():
        manifest=snapshot.with_suffix('').with_suffix('.json')
        known=json.loads(manifest.read_text())
        if known['version']!=VERSION or known['sha256']!=source.digest(snapshot):raise ValueError('Frozen component snapshot mismatch')
        t=pd.read_csv(snapshot);sources=known['sources']
    else:
        frames=[];sources=[]
        for year in range(2017,2026):
            csv=cache/f'components_{year}.csv';manifest=cache/f'components_{year}.json'
            if not csv.exists():
                parquet=cache/f'pbp_{year}.parquet';m=source.fetch(source.ROOT+f'pbp/play_by_play_{year}.parquet',parquet)
                aggregate(pd.read_parquet(parquet,columns=COLUMNS)).to_csv(csv,index=False)
                m.update(aggregate_sha256=source.digest(csv),version=VERSION);manifest.write_text(json.dumps(m,indent=2));parquet.unlink()
            m=json.loads(manifest.read_text())
            if m['version']!=VERSION or m['aggregate_sha256']!=source.digest(csv):raise ValueError('Component cache mismatch')
            sources.append(m);frames.append(pd.read_csv(csv));print('COMPONENT_SOURCE_READY',year,flush=True)
        t=pd.concat(frames,ignore_index=True)
    totals=t.groupby(['game_id','posteam','defteam'],as_index=False).agg(plays=('plays','first'),epa=('epa','sum'),success=('success','sum'))
    frozen=pd.read_csv('carryover_team_game_epa.csv')
    matched=totals.merge(frozen,on=['game_id','posteam','defteam'],suffixes=('','_frozen'),validate='one_to_one')
    if len(matched)!=len(frozen) or len(totals)!=len(frozen):raise ValueError('Original aggregate source coverage differs')
    source_revision={metric:dict(changed_offense_game_rows=int((~np.isclose(matched[metric],matched[metric+'_frozen'],atol=1e-10,rtol=0)).sum()),max_absolute_change=float((matched[metric]-matched[metric+'_frozen']).abs().max())) for metric in ['plays','epa','success']}
    (out/'source_revision.json').write_text(json.dumps(source_revision,indent=2))
    f=build_features(games,t)
    rebuilt=original.build_features(games,original.prepare_team_games(totals,games))[['game_id']+original.ADJ]
    # Additivity: disjoint components must recover the original rating definition on the SAME source.
    check=f.merge(rebuilt,on='game_id',validate='one_to_one')
    for side in ['off','def']:
        for metric in ['epa','success']:
            summed=check[[f'{kind}_{side}_{metric}_diff' for kind in CATEGORIES]].sum(axis=1)
            if not np.allclose(summed,check[f'adj_{side}_{metric}_diff'],atol=1e-12,rtol=0):raise ValueError('Component rating conservation failed')
    f=f.merge(rebuilt.rename(columns={c:'refreshed_'+c for c in original.ADJ}),on='game_id',validate='one_to_one')
    f=games.merge(f,on='game_id',validate='one_to_one').merge(pd.read_csv(args.features)[['game_id']+original.ADJ],on='game_id',validate='one_to_one')
    if not np.isfinite(f[EXTRA+REFRESH+original.ADJ].to_numpy()).all():raise ValueError('Incomplete model features')
    p,folds=evaluate(f)
    ref=pd.read_csv('strength_epa_reference_predictions.csv').query("method == 'opponent_epa'")
    check=p[p.method.eq('epa_baseline')].merge(ref[['game_id','predicted_margin']],on='game_id',validate='one_to_one',suffixes=('','_reference'))
    if len(check)!=1693 or not np.allclose(check.predicted_margin,check.predicted_margin_reference,atol=1e-10,rtol=0):raise ValueError('Baseline reproduction failed')
    summaries=[]
    for name,g in p[p.season.ge(2021)].groupby('method'):
        summaries.append(dict(period='2021-2025',method=name,**metrics(g)))
        for year,h in g.groupby('season'):summaries.append(dict(period=str(year),method=name,**metrics(h)))
    summary=pd.DataFrame(summaries)
    b=p[p.method.eq('dropback_components')&p.season.ge(2021)].set_index('game_id')
    paired=[];changes=b[['season','actual_margin','spread_line']].copy()
    changes['candidate_margin']=b.predicted_margin;changes['candidate_adjustment']=b.adjustment
    for opponent in ['epa_baseline','refreshed_epa_control']:
        a=p[p.method.eq(opponent)&p.season.ge(2021)].set_index('game_id').reindex(b.index)
        residual=a.actual_margin-a.spread_line;changed=np.sign(a.adjustment)!=np.sign(b.adjustment);eligible=changed&residual.ne(0)
        paired.append(dict(comparison='dropback_components vs '+opponent,nonpush_disagreements=int(eligible.sum()),control_wins=int(((residual*np.sign(a.adjustment)>0)&eligible).sum()),candidate_wins=int(((residual*np.sign(b.adjustment)>0)&eligible).sum())))
        changes[opponent+'_margin']=a.predicted_margin;changes[opponent+'_adjustment']=a.adjustment;changes[opponent+'_pick_changed']=changed
    changes=changes.reset_index()
    p.to_csv(out/'predictions.csv',index=False);summary.to_csv(out/'comparison.csv',index=False);folds.to_csv(out/'training_folds.csv',index=False)
    f[['game_id']+EXTRA+REFRESH].to_csv(out/'features.csv',index=False);t.to_csv(out/'team_components.csv.gz',index=False);changes.to_csv(out/'changed_picks.csv',index=False)
    classification=t.groupby('category').category_plays.sum().astype(int).to_dict()
    report=dict(status='DEVELOPMENT_ONLY_NO_LIVE_CHANGE',version=VERSION,summary=summaries,paired=paired,source_revision=source_revision,classification_counts=classification,features=EXTRA,sources=sources,
        hashes={x:source.digest(x) for x in [args.games,args.features,__file__,base.__file__,original.__file__,'DROPBACK_PROTOCOL.md','strength_epa_reference_predictions.csv','carryover_team_game_epa.csv']},
        limitations=['Previously examined development seasons, not untouched validation.','Revised PBP/lines lack verified pregame publication timestamps.','Fixed representation expansion; category contributions use all eligible plays as denominator.','No filtering, parameter search, automatic promotion or executable-price profit claim.'])
    if snapshot.exists():
        report['hashes'].update({str(x):source.digest(x) for x in [snapshot,manifest]})
    (out/'results.json').write_text(json.dumps(report,indent=2))
    print(summary.to_string(index=False));print(paired);return p,summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--games',default='strength_games_snapshot.csv');parser.add_argument('--features',default='strength_epa_features.csv');parser.add_argument('--cache',default='dropback_cache');parser.add_argument('--team-components',default='dropback_team_components.csv.gz');parser.add_argument('--out',default='dropback_results');run(parser.parse_args())
