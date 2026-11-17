"""Market/EPA challenger reusing V3 QB EPA, success, and dropback definitions."""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import auto_market as auto
import market_model as base
import model_v3_epa as old
import qb_market as confirmations
from market_epa_experiment import ADJ

QB=['qb_epa_diff','qb_success_diff','qb_dropbacks_diff','home_qb_missing','away_qb_missing']
VERSION='market_epa_qb_passing_v1'
ROOT=Path('data/auto_market/qb_passing_shadow')
CACHE=Path('qb_passing_cache')
PBP_COLUMNS=['game_id','play_id','posteam','defteam','epa','play_type','qb_dropback',
    'rush_attempt','qb_kneel','qb_spike','passer_player_id','passer_id','qb_epa']
FIELDS=auto.FORECAST_FIELDS+['model_version','core_forecast_sha256','qb_inputs_sha256',
    'home_confirmed_qb_id','away_confirmed_qb_id']+QB


def load_stats(seasons,cache):
    """Reuse V3 aggregation; keep compact aggregates instead of raw PBP in cache."""
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    frames=[];sources=[]
    for season in sorted(set(int(s) for s in seasons)):
        path=cache/('qb_'+str(season)+'.csv')
        provenance=path.with_suffix('.json')
        # Completed historical seasons are cacheable; the live season must refresh.
        if not (season<=2025 and path.exists() and provenance.exists()):
            raw=cache/('play_by_play_'+str(season)+'.parquet')
            if season>2025:raw.unlink(missing_ok=True)
            url=old.PBP_URL.format(season=season)
            old.download(url,raw)
            p=pd.read_parquet(raw,columns=PBP_COLUMNS)
            if p.duplicated(['game_id','play_id']).any():
                raise ValueError('Duplicate QB play identifiers')
            _,q=old.aggregate_pbp(p)
            q.to_csv(path,index=False)
            metadata=dict(season=season,url=url,raw_sha256=auto.digest(raw.read_bytes()),
                aggregate_sha256=auto.digest(path.read_bytes()),
                captured_at_utc=auto.stamp(datetime.now(timezone.utc)),
                aggregation_code_sha256=auto.digest(Path(old.__file__).read_bytes()))
            provenance.write_text(json.dumps(metadata,indent=2))
            raw.unlink()
        metadata=json.loads(provenance.read_text())
        if metadata['aggregate_sha256']!=auto.digest(path.read_bytes()) or metadata['aggregation_code_sha256']!=auto.digest(Path(old.__file__).read_bytes()):
            raise ValueError('QB cache hash or aggregation code mismatch')
        q=pd.read_csv(path,dtype={'passer_player_id':str})
        if q.duplicated(['game_id','posteam','passer_player_id']).any():
            raise ValueError('Duplicate QB aggregate')
        numeric=q[['epa_sum','success_sum','dropbacks']]
        if not np.isfinite(numeric.to_numpy()).all() or not q.dropbacks.gt(0).all() or not q.success_sum.between(0,q.dropbacks).all():
            raise ValueError('Invalid QB aggregates')
        frames.append(q);sources.append(metadata)
        print('QB passing source ready:',season,flush=True)
    return pd.concat(frames,ignore_index=True),sources


def profile(history,qid):
    samples=history.get(str(qid),[])[-8:] if qid is not None and pd.notna(qid) else []
    n=sum(r['dropbacks'] for r in samples)
    return dict(epa=sum(r['epa_sum'] for r in samples)/(n+100),
        success=(sum(r['success_sum'] for r in samples)+50)/(n+100),
        dropbacks=min(n,400),missing=float(not samples))


def difference(history,home,away):
    h,a=profile(history,home),profile(history,away)
    return dict(qb_epa_diff=h['epa']-a['epa'],qb_success_diff=h['success']-a['success'],
        qb_dropbacks_diff=h['dropbacks']-a['dropbacks'],
        home_qb_missing=h['missing'],away_qb_missing=a['missing'])


def features(games,stats,starter_ids=None):
    """Batch by date before history updates. Dict overrides are confirmed live IDs."""
    history=defaultdict(list);last={};rows=[]
    by_game={gid:g for gid,g in stats.groupby('game_id')}
    for day,group in games.sort_values(['gameday','game_id']).groupby('gameday',sort=True):
        for g in group.itertuples():
            if starter_ids is None:
                ids=(last.get(g.home_team),last.get(g.away_team))
            else:
                ids=starter_ids.get(g.game_id,(None,None))
            rows.append(dict(game_id=g.game_id,**difference(history,*ids)))
        for g in group.itertuples():
            if pd.isna(g.actual_margin):continue
            q=by_game.get(g.game_id)
            if q is None:
                raise ValueError('Missing prior-game QB PBP coverage: '+str(g.game_id))
            if not set(q.posteam)<={g.home_team,g.away_team}:
                raise ValueError('QB PBP team disagrees with schedule')
            for team,sub in q.groupby('posteam'):
                leader=sub.sort_values(['dropbacks','passer_player_id'],ascending=[False,True]).iloc[0]
                last[team]=leader.passer_player_id
            for r in q.to_dict('records'):history[str(r['passer_player_id'])].append(r)
    return pd.DataFrame(rows)


def backtest(out,cache=CACHE):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    f=confirmations.load_frame('strength_games_snapshot.csv','strength_epa_features.csv')
    f=f[f.season.between(2017,2025)].copy()
    stats,sources=load_stats(range(2017,2026),cache)
    lagged=features(f,stats)
    actual={r.game_id:(r.home_qb_id,r.away_qb_id) for r in f.itertuples()}
    identified=features(f,stats,actual)
    arms={'epa_baseline':None,'epa_qb_last_observed':lagged,
          'epa_qb_starter_diagnostic':identified}
    predictions=[];folds=[]
    for name,extra in arms.items():
        frame=f if extra is None else f.merge(extra,on='game_id',validate='one_to_one')
        columns=base.INPUTS+ADJ+([] if extra is None else QB)
        for year in range(2020,2026):
            train=frame[frame.season.lt(year)].dropna(subset=['actual_margin','spread_line'])
            test=frame[frame.season.eq(year)].dropna(subset=['actual_margin','spread_line'])
            if len(train)<500 or test.empty:raise ValueError('Insufficient QB fold')
            adj=np.clip(.25*confirmations.fit(train,columns).predict(test[columns]),-2.,2.)
            p=test[['game_id','season','actual_margin','spread_line']].copy()
            p['adjustment']=adj;p['predicted_margin']=p.spread_line+adj;p['method']=name
            predictions.append(p)
            folds.append(dict(method=name,test_season=year,train_last_season=int(train.season.max()),
                training_games=len(train),test_games=len(test)))
    p=pd.concat(predictions,ignore_index=True);summary=[]
    for name,g in p[p.season.ge(2021)].groupby('method'):
        summary.extend(dict(method=name,**r) for r in base.summarize(g))
    p.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'training_folds.csv',index=False)
    lagged.to_csv(out/'last_observed_features.csv',index=False)
    identified.to_csv(out/'starter_diagnostic_features.csv',index=False)
    stats.to_csv(out/'qb_game_stats.csv.gz',index=False)
    report=dict(status='DEVELOPMENT_ONLY_NO_LIVE_PROMOTION',model_version=VERSION,
        results=summary,sources=sources,features=base.INPUTS+ADJ+QB,
        settings=dict(history_games=8,epa_prior_dropbacks=100,success_prior_successes=50,
            dropback_cap=400,ridge_alpha=1000,shrink=.25,adjustment_cap=2),
        limitations=['Historical revised PBP and spreads lack verified publication times.',
        'Actual starter arm is a retrospective identity diagnostic, not a pregame backtest.',
        'Last-observed arm uses only prior-game identities and can miss replacements.',
        'All evaluated seasons have previously been inspected. No profit established.',
        'No parameter search, thresholds, or automatic model promotion.'],
        hashes={str(p):auto.digest(Path(p).read_bytes()) for p in [
            __file__,old.__file__,'market_model.py','strength_games_snapshot.csv','strength_epa_features.csv']})
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print('QB_PASSING_RESULT_JSON='+json.dumps(report),flush=True)
    return report


def record(schedule_path,core_rows,folder):
    ROOT.mkdir(parents=True,exist_ok=True)
    if not core_rows:return dict(status='NO_NEW_CORE_FORECASTS',new_qb_forecasts=0)
    reports=pd.read_csv(confirmations.CONFIRMATIONS,dtype=str,keep_default_na=False)
    (folder/'qb_passing_confirmations_asof.csv').write_bytes(confirmations.CONFIRMATIONS.read_bytes())
    cutoff=datetime.now(timezone.utc);eligible=[];missing=[]
    for core in core_rows:
        ids=tuple(confirmations.confirmed_ids(reports,core['game_id'],core[s+'_team'],
            auto.stamp(cutoff),core['kickoff_utc']) for s in ['home','away'])
        if None in ids:missing.append(core['game_id'])
        else:eligible.append((core,ids))
    if not eligible:return dict(status='MISSING_CONFIRMED_STARTERS',new_qb_forecasts=0,
        confirmation_rows=len(reports),missing_confirmation_games=missing)
    frame=confirmations.load_frame(schedule_path,folder/'epa_features_asof.csv.gz')
    stats,sources=load_stats(frame.loc[frame.actual_margin.notna(),'season'].unique(),CACHE)
    historical_ids={r.game_id:(r.home_qb_id,r.away_qb_id) for r in frame.itertuples()}
    # Only historical training uses retrospectively recorded IDs. Live uses evidence.
    historical_ids.update({c['game_id']:ids for c,ids in eligible})
    qb=features(frame,stats,historical_ids)
    frame=frame.merge(qb,on='game_id',validate='one_to_one')
    qb.to_csv(folder/'qb_passing_features_asof.csv.gz',index=False)
    stats.to_csv(folder/'qb_passing_stats_asof.csv.gz',index=False)
    (folder/'qb_passing_sources.json').write_text(json.dumps(sources,indent=2))
    input_hash=auto.digest((folder/'qb_passing_features_asof.csv.gz').read_bytes()+
        (folder/'qb_passing_confirmations_asof.csv').read_bytes()+
        (folder/'epa_features_asof.csv.gz').read_bytes()+Path(schedule_path).read_bytes())
    from epa_shadow import core_hash
    rows=[]
    for core,ids in eligible:
        test=frame[frame.game_id.eq(core['game_id'])].copy()
        if len(test)!=1 or test.actual_margin.notna().any():raise ValueError('Invalid live QB game')
        train=frame[frame.season.lt(test.iloc[0].season)].dropna(subset=['actual_margin','spread_line'])
        columns=base.INPUTS+ADJ+QB
        if len(train)<500 or not np.isfinite(train[ADJ+QB].to_numpy()).all() or not np.isfinite(test[ADJ+QB].to_numpy()).all():
            raise ValueError('Incomplete QB passing inputs')
        test['spread_line']=-float(core['home_line'])
        adj=float(np.clip(.25*confirmations.fit(train,columns).predict(test[columns])[0],-2.,2.))
        row=dict(core);margin=-float(core['home_line'])+adj
        row.update(adjustment=adj,predicted_home_margin=margin,home_model_spread=-margin,
            model_version=VERSION,core_forecast_sha256=core_hash(core),qb_inputs_sha256=input_hash,
            home_confirmed_qb_id=ids[0],away_confirmed_qb_id=ids[1],
            model_sha256=auto.digest(Path(__file__).read_bytes()),
            pipeline_sha256=auto.digest(Path(old.__file__).read_bytes()),
            **{c:float(test.iloc[0][c]) for c in QB})
        rows.append(row)
    finished=datetime.now(timezone.utc)
    if any(not auto.utc(r['forecast_created_at_utc'])<=cutoff<=finished<auto.utc(r['kickoff_utc']) for r in rows):
        raise ValueError('QB passing forecast crossed kickoff')
    for r in rows:r['forecast_created_at_utc']=auto.stamp(finished)
    saved=auto.archive_rows(ROOT/'forecasts.csv',FIELDS,rows,'game_id')
    return dict(status='QB_PASSING_SHADOW_SAVED',new_qb_forecasts=len(saved),
        missing_confirmation_games=missing,confirmation_rows=len(reports))


def grade_shadow(grader,core_rows,finals,quotes,now):
    import epa_shadow
    from epa_shadow import core_hash
    ROOT.mkdir(parents=True,exist_ok=True)
    candidates=grader.read_rows(ROOT/'forecasts.csv')
    cores={r['game_id']:r for r in core_rows}
    pairs=[]
    for r in candidates:
        c=cores.get(r['game_id'])
        if c is None or r['core_forecast_sha256']!=core_hash(c):
            raise ValueError('QB passing/core archive mismatch')
        for k in ['quote_id','sportsbook','home_team','away_team','home_line','captured_at_utc','kickoff_utc']:
            if str(r[k])!=str(c[k]):raise ValueError('QB passing/core quote mismatch')
        if not auto.utc(r['captured_at_utc'])<=auto.utc(c['forecast_created_at_utc'])<=auto.utc(r['forecast_created_at_utc'])<auto.utc(r['kickoff_utc']):
            raise ValueError('QB passing archive timing invalid')
        pairs.append((c,r))
    paired_core=[];paired_qb=[]
    for c,r in pairs:
        final=grader.match(r,finals,quotes)
        if final:
            paired_core.append(grader.grade(c,final,now))
            paired_qb.append(grader.grade(r,final,now))
    pd.DataFrame(paired_core).to_csv(ROOT/'paired_core_grades.csv',index=False)
    pd.DataFrame(paired_qb).to_csv(ROOT/'graded_predictions.csv',index=False)
    # Compare the QB challenger directly with the same-run EPA baseline, too.
    epa_path=epa_shadow.ROOT/'forecasts.csv'
    baseline_rows=grader.read_rows(epa_path) if epa_path.exists() else []
    if len({r['game_id'] for r in baseline_rows})!=len(baseline_rows):
        raise ValueError('Duplicate EPA forecast game IDs')
    baselines={r['game_id']:r for r in baseline_rows}
    epa_pairs=[];epa_grades=[];qb_epa_grades=[]
    for c,r in pairs:
        e=baselines.get(r['game_id'])
        if e is None:continue
        if e.get('model_version')!=epa_shadow.VERSION or e.get('core_forecast_sha256')!=core_hash(c):
            raise ValueError('QB/EPA baseline identity mismatch')
        for k in ['quote_id','sportsbook','home_team','away_team','home_line','captured_at_utc','kickoff_utc']:
            if str(e[k])!=str(r[k]):raise ValueError('QB/EPA quote mismatch')
        if not auto.utc(c['forecast_created_at_utc'])<=auto.utc(e['forecast_created_at_utc'])<=auto.utc(r['forecast_created_at_utc'])<auto.utc(r['kickoff_utc']):
            raise ValueError('QB/EPA baseline timing invalid')
        epa_pairs.append((e,r))
        final=grader.match(r,finals,quotes)
        if final:
            epa_grades.append(grader.grade(e,final,now))
            qb_epa_grades.append(grader.grade(r,final,now))
    pd.DataFrame(epa_grades).to_csv(ROOT/'paired_epa_baseline_grades.csv',index=False)
    pd.DataFrame(qb_epa_grades).to_csv(ROOT/'paired_epa_qb_grades.csv',index=False)
    report=dict(model_version=VERSION,paired_forecasts=len(pairs),
        core=grader.summary(paired_core,[c for c,r in pairs],now),
        qb_passing=grader.summary(paired_qb,[r for c,r in pairs],now),
        epa_comparison=dict(paired_forecasts=len(epa_pairs),missing_epa_forecasts=len(pairs)-len(epa_pairs),
            epa_baseline=grader.summary(epa_grades,[e for e,r in epa_pairs],now),
            qb_passing=grader.summary(qb_epa_grades,[r for e,r in epa_pairs],now)))
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='qb_passing_results')
    p.add_argument('--cache',default=str(CACHE));args=p.parse_args()
    backtest(args.out,args.cache)
