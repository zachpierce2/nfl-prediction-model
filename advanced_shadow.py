"""Prospective advanced challengers; original core/EPA archives are preserved."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import advanced_matchups as advanced
import auto_market as auto
import market_model as base
import epa_shadow

ROOT=Path('data/auto_market/advanced_shadow')
EXTRA=['model_version','arm','core_forecast_sha256','advanced_inputs_sha256']
VALUES=advanced.ADJ+advanced.PRESSURE+advanced.EXPLOSIVE+advanced.SPECIAL+advanced.INJURY
FIELDS=auto.FORECAST_FIELDS+EXTRA+VALUES

def eligible(arm,row):
    required={
        'epa_baseline':[],
        'epa_pressure':['pressure'],
        'epa_explosive':['explosive'],
        'epa_special':['special'],
        'epa_injury':['injury'],
        'epa_all':['pressure','explosive','special','injury']}[arm]
    return all(float(row[side+'_'+group+'_missing'])==0 for side in ['home','away'] for group in required)

def verified_pairs(candidates,cores):
    # Reuse exact-quote and immutable core-hash checks; enforce core-to-candidate time too.
    pairs=epa_shadow.verified_pairs(candidates,cores)
    for core,candidate in pairs:
        if auto.utc(candidate['forecast_created_at_utc'])<auto.utc(core['forecast_created_at_utc']):
            raise ValueError('Advanced forecast precedes its core pair')
    return pairs

def record(schedule_path,core_rows,folder):
    if not core_rows:return dict(status='NO_NEW_CORE_FORECASTS',arms={})
    folder=Path(folder)/'advanced';folder.mkdir(parents=True,exist_ok=True)
    games=base.read_games(schedule_path)
    team,inj,sn,players,sources=advanced.sources(games,'advanced_matchup_cache')
    cutoff=datetime.now(timezone.utc)
    extra=advanced.matchup_features(games,team).merge(
        advanced.injury_features(games,inj,sn,players,pd.Timestamp(cutoff)),
        on='game_id',validate='one_to_one')
    extra.to_csv(folder/'advanced_features_asof.csv.gz',index=False)
    team.to_csv(folder/'team_aggregates_asof.csv.gz',index=False)
    inj.to_csv(folder/'injuries_asof.csv.gz',index=False)
    sn.to_csv(folder/'snaps_asof.csv.gz',index=False)
    players[['gsis_id','pfr_id']].to_csv(folder/'player_ids_asof.csv.gz',index=False)
    (folder/'sources.json').write_text(json.dumps(sources,indent=2))
    epa_path=folder.parent/'epa_features_asof.csv.gz'
    epa=pd.read_csv(epa_path)
    frame=games.merge(epa[['game_id']+advanced.ADJ],on='game_id',how='left',validate='one_to_one').merge(
        extra,on='game_id',validate='one_to_one')
    frame.to_csv(folder/'model_frame_asof.csv.gz',index=False)
    input_hash=auto.digest((folder/'model_frame_asof.csv.gz').read_bytes()+
        (folder/'sources.json').read_bytes()+Path(schedule_path).read_bytes())
    rows={name:[] for name in advanced.ARMS};missing={name:[] for name in advanced.ARMS}
    for core in core_rows:
        test=frame[frame.game_id.eq(core['game_id'])].copy()
        if len(test)!=1 or test.actual_margin.notna().any():raise ValueError('Invalid advanced forecast game')
        if not np.isfinite(test[advanced.ADJ].to_numpy()).all():raise ValueError('EPA inputs missing')
        train=frame[frame.season.lt(test.iloc[0].season)].dropna(subset=['actual_margin','spread_line'])
        if len(train)<500 or not np.isfinite(train[advanced.ADJ].to_numpy()).all():
            raise ValueError('Incomplete advanced training data')
        test['spread_line']=-float(core['home_line'])
        for arm,extra_columns in advanced.ARMS.items():
            if not eligible(arm,test.iloc[0]):
                missing[arm].append(core['game_id']);continue
            columns=base.INPUTS+advanced.ADJ+extra_columns
            model=advanced.fit(train,columns)
            adj=float(np.clip(.25*model.predict(test[columns])[0],-2,2))
            row=dict(core);margin=-float(core['home_line'])+adj
            row.update(adjustment=adj,predicted_home_margin=margin,home_model_spread=-margin,
                model_version=advanced.VERSION,arm=arm,core_forecast_sha256=epa_shadow.core_hash(core),
                advanced_inputs_sha256=input_hash,model_sha256=auto.digest(Path(advanced.__file__).read_bytes()),
                pipeline_sha256=auto.digest(Path(__file__).read_bytes()),
                **{c:float(test.iloc[0][c]) if pd.notna(test.iloc[0][c]) else '' for c in VALUES})
            rows[arm].append(row)
    finished=datetime.now(timezone.utc)
    if any(not auto.utc(c['forecast_created_at_utc'])<=cutoff<=finished<auto.utc(c['kickoff_utc']) for c in core_rows):
        raise ValueError('Advanced calculation crossed kickoff')
    result={}
    for arm,candidates in rows.items():
        for row in candidates:row['forecast_created_at_utc']=auto.stamp(finished)
        saved=auto.archive_rows(ROOT/arm/'forecasts.csv',FIELDS,candidates,'game_id')
        result[arm]=dict(new_forecasts=len(saved),missing_feature_games=missing[arm])
    return dict(status='ADVANCED_SHADOW_RECORDED',feature_cutoff_utc=auto.stamp(cutoff),arms=result)

def grade_shadow(grader,core_rows,finals,quotes,now):
    ROOT.mkdir(parents=True,exist_ok=True);report={}
    for arm in advanced.ARMS:
        folder=ROOT/arm;folder.mkdir(parents=True,exist_ok=True)
        rows=grader.read_rows(folder/'forecasts.csv')
        pairs=verified_pairs(rows,core_rows)
        # Compare every candidate with the contemporaneously generated EPA-only arm.
        baseline=grader.read_rows(ROOT/'epa_baseline'/'forecasts.csv')
        epa_by_id={r['game_id']:r for r in baseline}
        graded=[];core_grades=[];epa_grades=[]
        for core,row in pairs:
            final=grader.match(row,finals,quotes)
            if not final:continue
            graded.append(grader.grade(row,final,now));core_grades.append(grader.grade(core,final,now))
            epa=epa_by_id.get(row['game_id'])
            if epa is None or epa['core_forecast_sha256']!=row['core_forecast_sha256']:
                raise ValueError('Advanced candidate missing identical EPA baseline pair')
            epa_grades.append(grader.grade(epa,final,now))
        for name,data in [('graded_predictions',graded),('paired_core_grades',core_grades),('paired_epa_grades',epa_grades)]:
            pd.DataFrame(data).to_csv(folder/(name+'.csv'),index=False)
        report[arm]=dict(paired_forecasts=len(pairs),candidate=grader.summary(graded,rows,now),
            core=grader.summary(core_grades,[c for c,r in pairs],now),
            epa=grader.summary(epa_grades,[epa_by_id[r['game_id']] for c,r in pairs],now))
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2))
    return report
