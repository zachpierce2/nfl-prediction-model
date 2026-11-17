"""Same-game prospective EPA/last-observed/expected-QB comparison."""
import json
from pathlib import Path
import auto_market as auto
import epa_shadow
from qb_passing import QB

ROOT=Path('data/auto_market/qb_starter_comparison')
VERSION='market_epa_last_observed_same_fit_v1'
FIELDS=auto.FORECAST_FIELDS+['model_version','core_forecast_sha256','qb_inputs_sha256',
    'home_last_qb_id','away_last_qb_id','expected_forecast_sha256']+QB
QUOTE=['quote_id','home_team','away_team','sportsbook','home_line','home_price_american',
    'away_line','away_price_american','captured_at_utc','kickoff_utc','snapshot_id']
CHANGE_FIELDS=['game_id','quote_id','starter_changed','home_expected_qb_id','home_last_qb_id',
    'away_expected_qb_id','away_last_qb_id','epa_margin','last_observed_margin','expected_margin',
    'epa_side','last_observed_side','expected_side','expected_vs_last_pick_changed',
    'actual_margin','epa_result','last_observed_result','expected_result',
    'epa_absolute_error','last_observed_absolute_error','expected_absolute_error']

def row_hash(row):
    return auto.digest(json.dumps({k:str(v) for k,v in row.items() if k!='forecast_archive'},sort_keys=True).encode())

def indexed(rows,label):
    if len({r['game_id'] for r in rows})!=len(rows):raise ValueError('Duplicate '+label+' game IDs')
    return {r['game_id']:r for r in rows}

def side(row):
    edge=float(row['predicted_home_margin'])+float(row['home_line'])
    return 'NONE' if abs(edge)<=1e-10 else 'HOME' if edge>0 else 'AWAY'

def validate(core,expected,control,epa=None):
    import expected_qb_shadow
    if control.get('model_version')!=VERSION or expected.get('model_version')!=expected_qb_shadow.VERSION:
        raise ValueError('Starter comparison model version mismatch')
    if control.get('expected_forecast_sha256')!=row_hash(expected):raise ValueError('Expected forecast hash mismatch')
    if control.get('qb_inputs_sha256')!=expected.get('qb_inputs_sha256'):raise ValueError('QB inputs differ')
    for row in [expected,control]+([epa] if epa else []):
        if row.get('core_forecast_sha256')!=epa_shadow.core_hash(core):raise ValueError('Starter comparison core mismatch')
        for k in QUOTE:
            if str(row[k])!=str(core[k]):raise ValueError('Starter comparison quote mismatch')
        if not auto.utc(row['captured_at_utc'])<=auto.utc(core['forecast_created_at_utc'])<=auto.utc(row['forecast_created_at_utc'])<auto.utc(row['kickoff_utc']):
            raise ValueError('Starter comparison timing invalid')
    if epa and epa.get('model_version')!=epa_shadow.VERSION:raise ValueError('EPA version mismatch')
    if auto.utc(expected['forecast_created_at_utc'])!=auto.utc(control['forecast_created_at_utc']):
        raise ValueError('QB arms were not created together')
    if epa and auto.utc(epa['forecast_created_at_utc'])>auto.utc(expected['forecast_created_at_utc']):
        raise ValueError('EPA pair was created after QB arms')

def write_csv(path,fields,rows):
    import csv
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def pair_stats(grades,a,b):
    aa={r['game_id']:r for r in grades[a]};bb={r['game_id']:r for r in grades[b]}
    changed=[gid for gid in aa if aa[gid]['selected_side']!=bb[gid]['selected_side']]
    return dict(comparison=a+' vs '+b,graded_games=len(aa),pick_disagreements=len(changed),
        first_wins_other_loses=sum(aa[i]['ats_result']=='WIN' and bb[i]['ats_result']=='LOSS' for i in changed),
        other_wins_first_loses=sum(bb[i]['ats_result']=='WIN' and aa[i]['ats_result']=='LOSS' for i in changed),
        mean_absolute_error_improvement=sum(float(bb[i]['model_absolute_error'])-float(aa[i]['model_absolute_error']) for i in aa)/len(aa) if aa else None)

def grade_shadow(grader,core_rows,finals,quotes,now):
    import expected_qb_shadow
    ROOT.mkdir(parents=True,exist_ok=True)
    cores=indexed(core_rows,'core')
    expected=indexed(grader.read_rows(expected_qb_shadow.ROOT/'forecasts.csv'),'expected QB')
    controls=indexed(grader.read_rows(ROOT/'forecasts.csv'),'last observed QB')
    epas=indexed(grader.read_rows(epa_shadow.ROOT/'forecasts.csv'),'EPA')
    triples=[]
    for gid,control in controls.items():
        if gid not in expected or gid not in cores:raise ValueError('Control is missing expected/core forecast')
        validate(cores[gid],expected[gid],control,epas.get(gid))
        if gid in epas:triples.append((epas[gid],control,expected[gid]))
    arms=['epa','last_observed','expected'];forecasts={k:[] for k in arms};grades={k:[] for k in arms}
    changes=[];changed_ids=set()
    for e,c,x in triples:
        gid=x['game_id']
        changed=any(str(x[s+'_expected_qb_id'])!=str(c[s+'_last_qb_id']) for s in ['home','away'])
        if changed:changed_ids.add(gid)
        final=grader.match(x,finals,quotes)
        row=dict(game_id=gid,quote_id=x['quote_id'],starter_changed=changed,
            **{s+'_expected_qb_id':x[s+'_expected_qb_id'] for s in ['home','away']},
            **{s+'_last_qb_id':c[s+'_last_qb_id'] for s in ['home','away']},
            epa_margin=e['predicted_home_margin'],last_observed_margin=c['predicted_home_margin'],expected_margin=x['predicted_home_margin'],
            epa_side=side(e),last_observed_side=side(c),expected_side=side(x),
            expected_vs_last_pick_changed=side(x)!=side(c),actual_margin='')
        for arm,r in zip(arms,[e,c,x]):
            forecasts[arm].append(r);row[arm+'_result']='';row[arm+'_absolute_error']=''
            if final:
                g=grader.grade(r,final,now);grades[arm].append(g)
                row[arm+'_result']=g['ats_result'];row[arm+'_absolute_error']=g['model_absolute_error'];row['actual_margin']=g['actual_home_margin']
        changes.append(row)
    report=dict(version=VERSION,updated_at_utc=auto.stamp(now),complete_triples=len(triples),
        expected_forecasts=len(expected),last_observed_forecasts=len(controls),
        expected_missing_control=len(set(expected)-set(controls)),
        controls_missing_epa=len(set(controls)-set(epas)),
        starter_changed_forecasts=len(changed_ids),
        same_game_results={a:grader.summary(grades[a],forecasts[a],now) for a in arms},
        changed_starter_results={a:grader.summary([g for g in grades[a] if g['game_id'] in changed_ids],[r for r in forecasts[a] if r['game_id'] in changed_ids],now) for a in arms},
        disagreements=[pair_stats(grades,a,b) for a,b in [('expected','last_observed'),('expected','epa'),('last_observed','epa')]],
        interpretation='Prospective identical-game comparison. QB arms share training, fitted coefficients, source snapshots and quote; only live starter IDs differ. No backfill or automatic promotion. Zero-game rates are null; no profitability established.')
    for arm in arms:write_csv(ROOT/(arm+'_paired_grades.csv'),grader.GRADE_FIELDS,grades[arm])
    write_csv(ROOT/'changed_picks.csv',CHANGE_FIELDS,changes)
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2))
    text='# Starting QB comparison\n\nAll three models use the same games and sportsbook quotes.\n\n'
    text+='| Model | Graded | W–L–P | ATS | MAE |\n|---|---:|---|---:|---:|\n'
    for arm in arms:
        r=report['same_game_results'][arm];pct=r['hypothetical_win_pct'];mae=r['model_mae']
        pct_text='N/A' if pct is None else f'{pct:.2f}%'
        mae_text='N/A' if mae is None else f'{mae:.4f}'
        text+=f"| {arm} | {r['graded_games']} | {r['hypothetical_wins']}–{r['hypothetical_losses']}–{r['pushes']} | {pct_text} | {mae_text} |\n"
    text+=f"\nComplete forecast triples: {len(triples)}. Starter identity changes: {len(changed_ids)}. Expected forecasts missing a control: {report['expected_missing_control']}. Controls missing EPA: {report['controls_missing_epa']}.\n\n"
    text+='The changed-starter subgroup and pairwise disagreements are in summary.json; per-game IDs, picks, results and errors are in changed_picks.csv. A small early sample does not establish improvement.\n'
    (ROOT/'report.md').write_text(text)
    return report,text
