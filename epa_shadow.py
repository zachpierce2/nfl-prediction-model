"""Prospective paired core/EPA tracking. Never backfill predictions after kickoff."""
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import auto_market as auto
import market_model as model
from market_epa_experiment import ADJ, PBP_COLUMNS, ALIASES, build_features

VERSION='opponent_epa_alpha1000_shrink025_cap2_v1'
ROOT=Path('data/auto_market/epa_shadow')
EXTRA=['model_version','core_forecast_sha256','epa_inputs_sha256']+ADJ
FIELDS=auto.FORECAST_FIELDS+EXTRA

def core_hash(row):
    return auto.digest(json.dumps({k:str(row[k]) for k in auto.FORECAST_FIELDS},sort_keys=True).encode())

def live_features(games,folder):
    historical_path=Path('strength_epa_features.csv')
    historical=pd.read_csv(historical_path)
    if historical.game_id.duplicated().any():raise ValueError('Duplicate frozen EPA game IDs')
    current=games[games.season.gt(2025)].copy();frames=[];sources=[]
    for season,season_games in current.groupby('season'):
        expected=season_games[season_games.actual_margin.notna()]
        if expected.empty:continue
        url=f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{int(season)}.parquet'
        raw,_=auto.download(url,'Current-season EPA refresh')
        plays=pd.read_parquet(io.BytesIO(raw),columns=PBP_COLUMNS)
        plays=plays[plays.game_id.isin(expected.game_id)].copy()
        if plays.duplicated(['game_id','play_id']).any():raise ValueError('Duplicate current EPA plays')
        plays=plays[plays.play_type.isin(['pass','run']) & plays.epa.notna()
            & plays.qb_kneel.fillna(0).eq(0) & plays.qb_spike.fillna(0).eq(0)
            & plays.posteam.notna() & plays.defteam.notna()].copy()
        plays['success']=plays.epa.gt(0).astype(float)
        for c in ['posteam','defteam']:plays[c]=plays[c].replace(ALIASES)
        team=plays.groupby(['game_id','posteam','defteam']).agg(plays=('epa','size'),epa=('epa','mean'),success=('success','mean')).reset_index()
        team=team.merge(expected[['game_id','gameday','season','home_team','away_team']],on='game_id',validate='many_to_one')
        coverage=team.groupby('game_id').size().reindex(expected.game_id,fill_value=0)
        if not coverage.eq(2).all() or team.duplicated(['game_id','posteam']).any():
            raise ValueError('EPA feed not complete for prior scored games; no EPA forecasts saved')
        valid=((team.posteam.eq(team.home_team)&team.defteam.eq(team.away_team))|
            (team.posteam.eq(team.away_team)&team.defteam.eq(team.home_team)))
        if not valid.all() or not np.isfinite(team[['epa','success','plays']].to_numpy()).all():
            raise ValueError('Invalid current EPA team statistics')
        frames.append(team)
        sources.append(dict(season=int(season),url=url,pbp_sha256=auto.digest(raw),prior_scored_games=len(expected)))
    cols=['game_id','gameday','season','posteam','defteam','plays','epa','success']
    team=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=cols)
    live=build_features(current,team) if not current.empty else pd.DataFrame(columns=['game_id']+ADJ)
    features=pd.concat([historical[['game_id']+ADJ],live[['game_id']+ADJ]],ignore_index=True)
    if features.game_id.duplicated().any():raise ValueError('Duplicate merged EPA features')
    # Archive the exact derived inputs and raw-source hashes available in this run.
    team.to_csv(folder/'epa_team_games_asof.csv.gz',index=False)
    features.to_csv(folder/'epa_features_asof.csv.gz',index=False)
    metadata=dict(model_version=VERSION,captured_at_utc=auto.stamp(datetime.now(timezone.utc)),
        historical_features_sha256=auto.digest(historical_path.read_bytes()),current_sources=sources,
        feature_builder_sha256=auto.digest(Path('market_epa_experiment.py').read_bytes()),
        limitations=['Historical features are revised development data through 2025.',
        'Current EPA uses only prior scored game dates from this run schedule; same-day scores excluded.',
        'Exact derived feature and team-game inputs archived; source PBP hashes retained.'])
    (folder/'epa_provenance.json').write_text(json.dumps(metadata,indent=2))
    return features,auto.digest((folder/'epa_features_asof.csv.gz').read_bytes())

def record(schedule_path,core_rows,folder):
    ROOT.mkdir(parents=True,exist_ok=True)
    if not core_rows:
        return dict(status='NO_NEW_CORE_FORECASTS',new_epa_forecasts=0,model_version=VERSION)
    f=model.read_games(schedule_path)
    features,input_hash=live_features(f,folder)
    f=f.merge(features,on='game_id',how='left',validate='one_to_one')
    pending=pd.DataFrame(core_rows)
    live=f.merge(pending,on='game_id',validate='one_to_one',suffixes=('','_quote'))
    live['spread_line']=-live.home_line
    predictions={}
    for season,g in live.groupby('season'):
        train=f[(f.season<season)&f.actual_margin.notna()&f.spread_line.notna()]
        if len(train)<500 or not np.isfinite(train[ADJ].to_numpy()).all() or not np.isfinite(g[ADJ].to_numpy()).all():
            raise ValueError('Incomplete EPA training or forecast features')
        columns=model.INPUTS+ADJ
        fitted=model.make_pipeline(model.SimpleImputer(strategy='median'),model.StandardScaler(),Ridge(alpha=1000.))
        fitted.fit(train[columns],train.actual_margin-train.spread_line)
        adjustment=np.clip(.25*fitted.predict(g[columns]),-2.,2.)
        for (_,r),adj in zip(g.iterrows(),adjustment):predictions[r.game_id]=(float(adj),{c:float(r[c]) for c in ADJ})
    finished=datetime.now(timezone.utc)
    if any(not auto.utc(r['forecast_created_at_utc'])<=finished<auto.utc(r['kickoff_utc']) for r in core_rows):
        raise ValueError('Invalid EPA calculation time relative to core forecast and kickoff')
    rows=[]
    for core in core_rows:
        adj,values=predictions[core['game_id']]
        row=dict(core);margin=-float(core['home_line'])+adj
        row.update(adjustment=adj,predicted_home_margin=margin,home_model_spread=-margin,
            forecast_created_at_utc=auto.stamp(finished),model_version=VERSION,
            core_forecast_sha256=core_hash(core),epa_inputs_sha256=input_hash,
            model_sha256=auto.digest(Path(__file__).read_bytes()),
            pipeline_sha256=auto.digest(Path('market_epa_experiment.py').read_bytes()),**values)
        rows.append(row)
    saved=auto.archive_rows(ROOT/'forecasts.csv',FIELDS,rows,'game_id')
    return dict(status='PAIRED_PREGAME_FORECASTS_SAVED',new_epa_forecasts=len(saved),model_version=VERSION)

def verified_pairs(epa_rows,core_rows):
    by_id={r['game_id']:r for r in core_rows};pairs=[]
    for r in epa_rows:
        core=by_id.get(r['game_id'])
        if core is None or r['core_forecast_sha256']!=core_hash(core):raise ValueError('EPA/core archive mismatch')
        for key in ['quote_id','sportsbook','home_team','away_team','captured_at_utc','kickoff_utc','home_line','home_price_american','away_price_american']:
            if str(r[key])!=str(core[key]):raise ValueError('EPA/core quote mismatch')
        if not auto.utc(r['captured_at_utc'])<=auto.utc(r['forecast_created_at_utc'])<auto.utc(r['kickoff_utc']):
            raise ValueError('EPA forecast timing invalid')
        r=dict(r,forecast_archive=str(ROOT/'forecasts.csv'))
        pairs.append((core,r))
    return pairs

def grade_shadow(grader,core_rows,finals,quotes,now):
    ROOT.mkdir(parents=True,exist_ok=True)
    epa_rows=grader.read_rows(ROOT/'forecasts.csv')
    pairs=verified_pairs(epa_rows,core_rows)
    epa_grades=[];core_grades=[];epa_lines=[];core_lines=[]
    for core,r in pairs:
        final=grader.match(r,finals,quotes)
        if final:
            core_grades.append(grader.grade(core,final,now));epa_grades.append(grader.grade(r,final,now))
        epa_lines.append(grader.line_comparison(r,quotes,now,final))
        core_lines.append(grader.line_comparison(core,quotes,now,final))
    for name,rows in [('graded_predictions',epa_grades),('paired_core_grades',core_grades),('line_comparison',epa_lines),('paired_core_line_comparison',core_lines)]:
        with (ROOT/(name+'.csv')).open('w',newline='') as f:
            fields=list(rows[0]) if rows else ['game_id','status']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    report=dict(updated_at_utc=auto.stamp(now),model_version=VERSION,paired_forecasts=len(pairs),
        core=grader.summary(core_grades,[c for c,r in pairs],now),
        epa=grader.summary(epa_grades,epa_rows,now))
    # Use identical later quotes for both models: EPA is created slightly after core.
    paired_lines=[(c,e) for c,e in zip(core_lines,epa_lines) if c['status']=='NEAR_KICKOFF_PROXY' and e['status']=='NEAR_KICKOFF_PROXY' and c['latest_quote_id']==e['latest_quote_id']]
    report['matched_near_kickoff_comparisons']=len(paired_lines)
    for name,index in [('core',0),('epa',1)]:
        report[name]['mean_matched_line_value_points']=sum(p[index]['line_value_points'] for p in paired_lines)/len(paired_lines) if paired_lines else None
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2))
    text='\n## Prospective core versus EPA (paired games only)\n\n'
    text+=f'Paired pregame forecasts: {len(pairs)}. Historical predictions are not backfilled.\n\n'
    text+='| Model | Graded | Wins | Losses | Pushes | ATS win rate | MAE |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name in ['core','epa']:
        s=report[name];pct=s['hypothetical_win_pct'];mae=s['model_mae']
        pct_text=f'{pct:.2f}%' if pct is not None else 'N/A';mae_text=f'{mae:.4f}' if mae is not None else 'N/A'
        text+=f'| {name} | {s["graded_games"]} | {s["hypothetical_wins"]} | {s["hypothetical_losses"]} | {s["pushes"]} | {pct_text} | {mae_text} |\n'
    text+=f'\nMatched same-book near-kickoff comparisons: {len(paired_lines)}. '
    text+='Separate line-value means are in epa_shadow/summary.json. These are point-spread proxies, not certified closing prices or profit.\n'
    (ROOT/'report.md').write_text(text)
    return report,text

def main():
    import sys
    if '--self-test' in sys.argv:
        import unittest
        import test_epa_shadow
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_epa_shadow))
        if not result.wasSuccessful():raise SystemExit(1)
    elif '--validate-live' in sys.argv:
        import tempfile
        latest=json.loads(Path('data/auto_market/latest_run.json').read_text())
        snapshot=Path('data/auto_market/snapshots')/latest['snapshot_id']/'schedule_asof.csv.gz'
        games=model.read_games(snapshot)
        with tempfile.TemporaryDirectory() as d:
            features,_=live_features(games,Path(d))
            merged=games.merge(features,on='game_id',how='left',validate='one_to_one')
            scored=merged[merged.actual_margin.notna()]
            if not np.isfinite(scored[ADJ].to_numpy()).all():raise ValueError('Live EPA validation has missing features')
            print(json.dumps(dict(status='LIVE_EPA_INPUTS_VALIDATED_NO_FORECASTS_SAVED',scored_games=len(scored),feature_rows=len(features))))
    else:raise SystemExit('This module is called by auto_market.py and grade_market.py')

if __name__=='__main__':main()
