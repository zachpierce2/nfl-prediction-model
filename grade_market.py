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


# Point spread movement only: archived observations are not certified closing odds.
def line_comparison(row, quotes, now, final=None):
    kickoff=auto.utc(final['kickoff_utc'] if final else row['kickoff_utc'])
    created=auto.utc(row['forecast_created_at_utc'])
    captured=auto.utc(row['captured_at_utc'])
    initial=number(row['home_line'])
    lean=number(row['predicted_home_margin'])+initial
    side='NONE' if abs(lean)<=1e-10 else ('HOME' if lean>0 else 'AWAY')
    result=dict(game_id=row['game_id'],sportsbook=row['sportsbook'],selected_side=side,
        forecast_quote_id=row['quote_id'],initial_home_line=initial,
        initial_selected_line=initial if side=='HOME' else (-initial if side=='AWAY' else ''),
        initial_selected_price=row.get('home_price_american' if side=='HOME' else 'away_price_american',''),
        forecast_created_at_utc=row['forecast_created_at_utc'],kickoff_utc=auto.stamp(kickoff),
        kickoff_source='FINAL_PROVIDER' if final else 'FORECAST_SCHEDULE',
        status='',latest_quote_id='',latest_captured_at_utc='',latest_provider_updated_at_utc='',
        latest_selected_line='',latest_selected_price='',minutes_before_kickoff='',line_value_points='')
    if not captured<=created<kickoff:
        result['status']='INVALID_FORECAST_TIMING';return result
    if side=='NONE':result['status']='NO_LEAN';return result
    if now<kickoff:result['status']='PENDING_KICKOFF';return result
    anchors=[q for q in quotes if q.get('quote_id')==row['quote_id'] and q.get('game_id')==row['game_id']]
    event_ids={q.get('provider_event_id') for q in anchors if q.get('provider_event_id')}
    candidates=[]
    for q in quotes:
        if (q.get('game_id')!=row['game_id'] or q.get('sportsbook')!=row['sportsbook']
            or q.get('home_team')!=row['home_team'] or q.get('away_team')!=row['away_team']):continue
        if event_ids and q.get('provider_event_id') not in event_ids:continue
        if final and q.get('provider_event_id')!=final['provider_event_id']:continue
        try:
            at=auto.utc(q['captured_at_utc']);updated=auto.utc(q['provider_updated_at_utc'])
            # Exclude initial quote, after-kickoff data, stale odds, and rescheduled-event observations.
            if q['quote_id']==row['quote_id'] or not created<=at<kickoff or at>now:continue
            quote_kickoff=auto.utc(q['kickoff_utc'])
            if at>=min(quote_kickoff,kickoff,auto.utc(row['kickoff_utc'])):continue
            if quote_kickoff!=kickoff:
                if not final or not q.get('provider_event_id') or event_ids!={q['provider_event_id']}:continue
                if abs(quote_kickoff-kickoff)>timedelta(minutes=5):continue
            if not timedelta(0)<=at-updated<=timedelta(minutes=30):continue
            hl=number(q['home_line']);al=number(q['away_line'])
            if abs(hl+al)>1e-6:continue
        except (KeyError,ValueError,TypeError):continue
        candidates.append((at,updated,q['quote_id'],q))
    if not candidates:result['status']='NO_LATER_SAME_BOOK_QUOTE';return result
    at,updated,_,q=max(candidates,key=lambda item:item[:3])
    latest=number(q['home_line'] if side=='HOME' else q['away_line'])
    gap=(kickoff-at).total_seconds()/60
    result.update(status='NEAR_KICKOFF_PROXY' if gap<=60 else 'EARLY_LAST_OBSERVATION',
        latest_quote_id=q['quote_id'],latest_captured_at_utc=q['captured_at_utc'],
        latest_provider_updated_at_utc=q['provider_updated_at_utc'],latest_selected_line=latest,
        latest_selected_price=q.get('home_price_american' if side=='HOME' else 'away_price_american',''),
        minutes_before_kickoff=round(gap,3),line_value_points=result['initial_selected_line']-latest)
    return result

def line_report(rows, quotes, finals, now):
    comparisons=[line_comparison(r,quotes,now,match(r,finals,quotes)) for r in rows]
    # Rebuild this derived report; original forecasts and quotes remain append-only.
    if comparisons:
        with (ROOT/'line_comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    else:
        (ROOT/'line_comparison.csv').write_text('game_id,status,line_value_points\n')
    near=[r for r in comparisons if r['status']=='NEAR_KICKOFF_PROXY']
    early=[r for r in comparisons if r['status']=='EARLY_LAST_OBSERVATION']
    counts={status:sum(r['status']==status for r in comparisons) for status in sorted({r['status'] for r in comparisons})}
    report=dict(updated_at_utc=auto.stamp(now),forecasts=len(rows),status_counts=counts,
        near_kickoff_games=len(near),early_observations=len(early),
        mean_near_kickoff_line_value_points=sum(r['line_value_points'] for r in near)/len(near) if near else None,
        positive_near_kickoff_games=sum(r['line_value_points']>0 for r in near),
        negative_near_kickoff_games=sum(r['line_value_points']<0 for r in near),
        unchanged_near_kickoff_games=sum(r['line_value_points']==0 for r in near),
        certified_closing_lines=0,
        interpretation='Same sportsbook, later pre-kickoff archived quote. Positive points mean a better original spread for the selected side. Near-kickoff means within 60 minutes; earlier observations excluded from primary average. Not certified closing odds, price-adjusted CLV, or profit.')
    (ROOT/'line_comparison_summary.json').write_text(json.dumps(report,indent=2))
    avg=report['mean_near_kickoff_line_value_points']
    avg_text=f'{avg:+.3f}' if avg is not None else 'N/A'
    text=('\n## Line movement against later sportsbook prices\n\n'
        f'Near-kickoff comparisons (within 60 minutes): {len(near)}. Average point advantage: {avg_text}.\n\n'
        f'Earlier last observations: {len(early)} (excluded from that average).\n\n'
        'Positive means the saved pick had a better spread than the later same-book quote. '
        'Each CSV row shows the quote time and minutes before kickoff. Three-hour polling may miss the closing price. '
        'These are proxies, not certified closing lines; odds/juice are shown separately and are not included in point value.\n\n'
        f'Status counts: {json.dumps(counts,sort_keys=True)}\n')
    return report,text

def test_line_comparison():
    import unittest
    class LineTests(unittest.TestCase):
        def setUp(self):
            self.row=dict(game_id='fixture',quote_id='first',sportsbook='Book',home_team='H',away_team='A',
                home_line='-3',predicted_home_margin='4',captured_at_utc='2026-11-15T10:00:00Z',
                forecast_created_at_utc='2026-11-15T10:01:00Z',kickoff_utc='2026-11-15T20:00:00Z')
            self.q=dict(game_id='fixture',quote_id='later',sportsbook='Book',home_team='H',away_team='A',
                home_line='-4',away_line='4',captured_at_utc='2026-11-15T19:30:00Z',
                provider_updated_at_utc='2026-11-15T19:29:00Z',kickoff_utc='2026-11-15T20:00:00Z')
            self.now=auto.utc('2026-11-16T00:00:00Z')
        def test_home_and_away_signs(self):
            self.assertEqual(line_comparison(self.row,[self.q],self.now)['line_value_points'],1)
            self.row['predicted_home_margin']='2'
            self.assertEqual(line_comparison(self.row,[self.q],self.now)['line_value_points'],-1)
        def test_postkickoff_and_wrong_book_excluded(self):
            for change in [dict(captured_at_utc='2026-11-15T20:01:00Z'),dict(sportsbook='Other'),dict(kickoff_utc='2026-11-15T21:00:00Z')]:
                self.assertEqual(line_comparison(self.row,[dict(self.q,**change)],self.now)['status'],'NO_LATER_SAME_BOOK_QUOTE')
        def test_pending_and_missing_not_zero(self):
            r=line_comparison(self.row,[],auto.utc('2026-11-15T19:00:00Z'))
            self.assertEqual(r['status'],'PENDING_KICKOFF');self.assertEqual(r['line_value_points'],'')
            self.assertEqual(line_comparison(self.row,[],self.now)['line_value_points'],'')
        def test_stale_and_early_quotes(self):
            stale=dict(self.q,provider_updated_at_utc='2026-11-15T18:00:00Z')
            self.assertEqual(line_comparison(self.row,[stale],self.now)['status'],'NO_LATER_SAME_BOOK_QUOTE')
            early=dict(self.q,captured_at_utc='2026-11-15T17:00:00Z',provider_updated_at_utc='2026-11-15T16:59:00Z')
            self.assertEqual(line_comparison(self.row,[early],self.now)['status'],'EARLY_LAST_OBSERVATION')
        def test_latest_valid_quote_selected(self):
            older=dict(self.q,quote_id='older',captured_at_utc='2026-11-15T19:00:00Z',provider_updated_at_utc='2026-11-15T18:59:00Z',home_line='-5',away_line='5')
            r=line_comparison(self.row,[self.q,older],self.now)
            self.assertEqual(r['latest_quote_id'],'later');self.assertEqual(r['minutes_before_kickoff'],30)
        def test_verified_kickoff_drift(self):
            anchor=dict(self.q,quote_id='first',provider_event_id='event')
            later=dict(self.q,provider_event_id='event')
            final=dict(kickoff_utc='2026-11-15T20:05:00Z',provider_event_id='event')
            def status(q=later, f=final, a=anchor):
                return line_comparison(self.row,[a,q],self.now,f)['status']
            self.assertEqual(status(),'NEAR_KICKOFF_PROXY')
            for q in [dict(later,provider_event_id='other'),dict(later,captured_at_utc=later['kickoff_utc'],provider_updated_at_utc=later['kickoff_utc'])]:
                self.assertEqual(status(q),'NO_LATER_SAME_BOOK_QUOTE')
            self.assertEqual(status(f=dict(final,kickoff_utc='2026-11-15T20:05:01Z')),'NO_LATER_SAME_BOOK_QUOTE')
            self.assertEqual(status(a=dict(anchor,provider_event_id='')),'NO_LATER_SAME_BOOK_QUOTE')
        def test_no_lean(self):
            self.row['predicted_home_margin']='3'
            self.assertEqual(line_comparison(self.row,[self.q],self.now)['status'],'NO_LEAN')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LineTests))
    if not result.wasSuccessful():raise ValueError('Line comparison tests failed')

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
    quotes=read_rows(Path('data/auto_market/quotes.csv')) + read_rows(Path('data/auto_market/near_kickoff/quotes.csv'));fresh=[]
    for row in rows:
        if row['game_id'] in graded_ids:continue
        final=match(row,finals,quotes)
        if final:fresh.append(grade(row,final,now))
    auto.archive_rows(grade_path,GRADE_FIELDS,fresh,'game_id')
    grades=read_rows(grade_path);report=summary(grades,rows,now)
    h={k.lower():v for k,v in headers.items()}
    report.update(newly_graded=len(fresh),remaining_api_credits=h.get('x-requests-remaining','unknown'),last_request_credits=h.get('x-requests-last','unknown'))
    movement,movement_text=line_report(rows,quotes,finals,now)
    import epa_shadow
    paired,paired_text=epa_shadow.grade_shadow(sys.modules[__name__],rows,finals,quotes,now)
    import advanced_shadow
    report['advanced_paired_comparison']=advanced_shadow.grade_shadow(sys.modules[__name__],rows,finals,quotes,now)
    report['epa_paired_comparison']=paired
    import qb_passing
    report['qb_passing_paired_comparison']=qb_passing.grade_shadow(sys.modules[__name__],rows,finals,quotes,now)
    import expected_qb_shadow
    report['expected_qb_paired_comparison']=expected_qb_shadow.grade_shadow(sys.modules[__name__],rows,finals,quotes,now)
    import qb_starter_comparison
    report['qb_starter_comparison'],starter_text=qb_starter_comparison.grade_shadow(sys.modules[__name__],rows,finals,quotes,now)
    import total_shadow
    report['pregame_total']=total_shadow.grade(sys.modules[__name__],finals,now)
    import consensus_shadow
    report['consensus_comparison']=consensus_shadow.grade(finals,now)
    report['line_comparison']=movement
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
    text+=movement_text+paired_text+starter_text
    (ROOT/'report.md').write_text(text)
    print(json.dumps(report,indent=2))
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:f.write(text)

if __name__=='__main__':
    try:
        if '--self-test' in sys.argv:test_line_comparison()
        else:main()
    except Exception as exc:
        if isinstance(exc,(ValueError,RuntimeError,FileNotFoundError)):print(f'ERROR: {exc}',file=sys.stderr)
        else:print(f'ERROR: {type(exc).__name__}; inspect archive schema or provider availability',file=sys.stderr)
        sys.exit(1)