"""Daily prospective grading using explicitly completed Odds API results."""
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlencode
import auto_market as auto

ROOT=Path('data/auto_market/grading')
FINAL_FIELDS=['provider_event_id','home_team','away_team','kickoff_utc','home_score','away_score','provider_last_update','retrieved_at_utc','snapshot_id']
GRADE_FIELDS=['game_id','forecast_archive','forecast_row_sha256','quote_id','provider_event_id','home_team','away_team','sportsbook','home_line','home_price_american','away_price_american','predicted_home_margin','actual_home_margin','home_score','away_score','selected_side','ats_result','market_absolute_error','model_absolute_error','improvement_points','forecast_created_at_utc','kickoff_utc','graded_at_utc','score_snapshot_id']

def read_rows(path):
    if not path.exists():return []
    with path.open(newline='') as f:return list(csv.DictReader(f))

def number(value):
    n=float(value)
    if not math.isfinite(n):raise ValueError('Nonfinite numeric forecast or score')
    return n

def local_day(value):
    return auto.pd.Timestamp(auto.utc(value)).tz_convert('America/New_York').date()

def forecasts():
    rows=[];seen=set()
    for path in [Path('data/market_forecasts.csv'),Path('data/auto_market/forecasts.csv')]:
        for row in read_rows(path):
            if row['game_id'] in seen:raise ValueError('Duplicate game across forecast archives; resolve before grading')
            seen.add(row['game_id'])
            row['forecast_archive']=str(path)
            if auto.utc(row['forecast_created_at_utc'])>=auto.utc(row['kickoff_utc']):
                raise ValueError('Forecast was not created before its recorded kickoff')
            for field in ['home_line','predicted_home_margin']:number(row[field])
            rows.append(row)
    return rows

def completed(events,now,snapshot):
    rows=[]
    for e in events:
        if e.get('completed') is not True:continue
        if auto.utc(e['commence_time'])>=now:raise ValueError('Completed result has a future kickoff')
        home=auto.NAMES.get(e['home_team']);away=auto.NAMES.get(e['away_team'])
        if not home or not away:continue
        scores=e.get('scores') or []
        hs=[s for s in scores if s.get('name')==e['home_team']]
        aws=[s for s in scores if s.get('name')==e['away_team']]
        if len(hs)!=1 or len(aws)!=1:raise ValueError('Completed event lacks two unique team scores')
        h,a=number(hs[0]['score']),number(aws[0]['score'])
        if any(x<0 or int(x)!=x for x in [h,a]):raise ValueError('Invalid final score')
        rows.append(dict(provider_event_id=e['id'],home_team=home,away_team=away,kickoff_utc=e['commence_time'],
            home_score=int(h),away_score=int(a),provider_last_update=e.get('last_update') or '',
            retrieved_at_utc=auto.stamp(now),snapshot_id=snapshot))
    if len({r['provider_event_id'] for r in rows})!=len(rows):raise ValueError('Duplicate provider event IDs')
    return rows

def match(row,finals,quotes):
    ids={q['provider_event_id'] for q in quotes if q['game_id']==row['game_id']}
    candidates=[f for f in finals if f['home_team']==row['home_team'] and f['away_team']==row['away_team']
        and local_day(f['kickoff_utc'])==local_day(row['kickoff_utc'])
        and (not ids or f['provider_event_id'] in ids)]
    if len(candidates)>1:raise ValueError('Ambiguous final-score match')
    return candidates[0] if candidates else None

def grade(row,final,now):
    if auto.utc(row['forecast_created_at_utc'])>=auto.utc(final['kickoff_utc']):
        raise ValueError('Forecast was created after actual provider kickoff')
    home_line=number(row['home_line']);pred=number(row['predicted_home_margin'])
    actual=int(final['home_score'])-int(final['away_score'])
    adjustment=pred+home_line
    if abs(adjustment)<=1e-10:side='NONE';result='NO_LEAN'
    else:
        side='HOME' if adjustment>0 else 'AWAY'
        cover=(actual+home_line)*(1 if side=='HOME' else -1)
        result='PUSH' if abs(cover)<=1e-10 else ('WIN' if cover>0 else 'LOSS')
    market_error=abs(actual+home_line);model_error=abs(actual-pred)
    original={k:v for k,v in row.items() if k!='forecast_archive'}
    return dict(game_id=row['game_id'],forecast_archive=row.get('forecast_archive','fixture'),
        forecast_row_sha256=hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest(),
        quote_id=row['quote_id'],provider_event_id=final['provider_event_id'],home_team=row['home_team'],away_team=row['away_team'],
        sportsbook=row['sportsbook'],home_line=home_line,home_price_american=row.get('home_price_american',''),
        away_price_american=row.get('away_price_american',''),predicted_home_margin=pred,actual_home_margin=actual,
        home_score=final['home_score'],away_score=final['away_score'],selected_side=side,ats_result=result,
        market_absolute_error=market_error,model_absolute_error=model_error,improvement_points=market_error-model_error,
        forecast_created_at_utc=row['forecast_created_at_utc'],kickoff_utc=final['kickoff_utc'],graded_at_utc=auto.stamp(now),
        score_snapshot_id=final['snapshot_id'])

def summary(grades,all_forecasts,now):
    wins=sum(r['ats_result']=='WIN' for r in grades);losses=sum(r['ats_result']=='LOSS' for r in grades)
    pushes=sum(r['ats_result']=='PUSH' for r in grades);n=len(grades)
    seen={r['game_id'] for r in grades};pending=[r for r in all_forecasts if r['game_id'] not in seen]
    overdue=[r['game_id'] for r in pending if auto.utc(r['kickoff_utc'])<now-timedelta(days=3)]
    return dict(updated_at_utc=auto.stamp(now),forecasts=len(all_forecasts),graded_games=n,pending_games=len(pending),
        hypothetical_wins=wins,hypothetical_losses=losses,pushes=pushes,no_lean=sum(r['ats_result']=='NO_LEAN' for r in grades),
        hypothetical_win_pct=100*wins/(wins+losses) if wins+losses else None,
        market_mae=sum(number(r['market_absolute_error']) for r in grades)/n if n else None,
        model_mae=sum(number(r['model_absolute_error']) for r in grades)/n if n else None,
        missing_result_over_3_days=overdue,
        interpretation='Diagnostic direction of every nonzero model adjustment; excludes pushes from win rate. Not an executed betting record; no profitability claim.')

def main():
    now=datetime.now(timezone.utc);sid=now.strftime('%Y%m%dT%H%M%S%fZ')
    rows=forecasts();ROOT.mkdir(parents=True,exist_ok=True)
    key=os.environ.get('ODDS_API_KEY','').strip()
    if not key:raise ValueError('Missing ODDS_API_KEY repository secret')
    params=urlencode(dict(apiKey=key,daysFrom=3,dateFormat='iso'))
    raw,headers=auto.download('https://api.the-odds-api.com/v4/sports/americanfootball_nfl/scores/?'+params,'Final score refresh')
    events=json.loads(raw)
    if not isinstance(events,list):raise ValueError('Scores response is not an event list')
    final_rows=completed(events,now,sid)
    folder=ROOT/'score_snapshots';folder.mkdir(exist_ok=True)
    (folder/(sid+'.json.gz')).write_bytes(gzip.compress(raw))
    finals_path=ROOT/'final_scores.csv';cached=read_rows(finals_path)
    old={f['provider_event_id']:f for f in cached}
    corrections=[]
    for f in final_rows:
        prev=old.get(f['provider_event_id'])
        if prev and any(str(prev[c])!=str(f[c]) for c in ['home_score','away_score','home_team','away_team']):
            corrections.append(f['provider_event_id'])
    if corrections:raise ValueError('Provider revised a cached final score; inspect score snapshots before changing grades')
    auto.archive_rows(finals_path,FINAL_FIELDS,final_rows,'provider_event_id')
    finals=read_rows(finals_path)
    grade_path=ROOT/'graded_predictions.csv';existing=read_rows(grade_path);graded_ids={r['game_id'] for r in existing}
    quotes=read_rows(Path('data/auto_market/quotes.csv'));fresh=[]
    for row in rows:
        if row['game_id'] in graded_ids:continue
        final=match(row,finals,quotes)
        if final:fresh.append(grade(row,final,now))
    auto.archive_rows(grade_path,GRADE_FIELDS,fresh,'game_id')
    grades=read_rows(grade_path);report=summary(grades,rows,now)
    h={k.lower():v for k,v in headers.items()}
    report.update(newly_graded=len(fresh),remaining_api_credits=h.get('x-requests-remaining','unknown'),last_request_credits=h.get('x-requests-last','unknown'))
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2))
    pct=report['hypothetical_win_pct'];win_text=f'{pct:.2f}%' if pct is not None else 'N/A'
    mae=lambda x:f'{x:.4f}' if x is not None else 'N/A'
    text=(f'# NFL prospective grading\n\nUpdated: {auto.stamp(now)}\n\n'
        f'| Metric | Result |\n|---|---:|\n| Forecasts | {len(rows)} |\n| Graded games | {len(grades)} |\n'
        f'| Pending | {report["pending_games"]} |\n| Hypothetical wins | {report["hypothetical_wins"]} |\n'
        f'| Hypothetical losses | {report["hypothetical_losses"]} |\n| Pushes | {report["pushes"]} |\n'
        f'| No lean | {report["no_lean"]} |\n| Win rate excluding pushes | {win_text} |\n'
        f'| Sportsbook MAE | {mae(report["market_mae"])} |\n| Model MAE | {mae(report["model_mae"])} |\n\n'
        'Grades use the frozen forecast and recorded spread. Any nonzero adjustment counts as a hypothetical lean, however tiny. This is not an executed betting record. No profit or ROI is claimed.\n\n'
        'Only provider-confirmed completed games are graded. Pending games are not losses. Scores are fetched daily with a three-day lookback; outages longer than that may leave missing results.\n')
    (ROOT/'report.md').write_text(text)
    print(json.dumps(report,indent=2))
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:f.write(text)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if isinstance(exc,(ValueError,RuntimeError,FileNotFoundError)):print(f'ERROR: {exc}',file=sys.stderr)
        else:print(f'ERROR: {type(exc).__name__}; inspect archive schema or provider availability',file=sys.stderr)
        sys.exit(1)
