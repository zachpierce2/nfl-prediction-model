"""Automatically archive current NFL spreads and first pregame forecasts."""
import csv
import hashlib
import gzip
import io
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd
import market_model as model

ROOT = Path('data/auto_market')
SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
BOOK_ORDER = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'williamhill_us', 'betrivers', 'betonlineag', 'bovada']
NAMES = {
'Arizona Cardinals':'ARI','Atlanta Falcons':'ATL','Baltimore Ravens':'BAL','Buffalo Bills':'BUF',
'Carolina Panthers':'CAR','Chicago Bears':'CHI','Cincinnati Bengals':'CIN','Cleveland Browns':'CLE',
'Dallas Cowboys':'DAL','Denver Broncos':'DEN','Detroit Lions':'DET','Green Bay Packers':'GB',
'Houston Texans':'HOU','Indianapolis Colts':'IND','Jacksonville Jaguars':'JAX','Kansas City Chiefs':'KC',
'Las Vegas Raiders':'LV','Los Angeles Chargers':'LAC','Los Angeles Rams':'LAR','Miami Dolphins':'MIA',
'Minnesota Vikings':'MIN','New England Patriots':'NE','New Orleans Saints':'NO','New York Giants':'NYG',
'New York Jets':'NYJ','Philadelphia Eagles':'PHI','Pittsburgh Steelers':'PIT','San Francisco 49ers':'SF',
'Seattle Seahawks':'SEA','Tampa Bay Buccaneers':'TB','Tennessee Titans':'TEN','Washington Commanders':'WAS'}
QUOTE_FIELDS = ['quote_id','provider_event_id','game_id','home_team','away_team','bookmaker_key','sportsbook',
'home_line','home_price_american','away_line','away_price_american','provider_updated_at_utc',
'captured_at_utc','kickoff_utc','snapshot_id','source']
FORECAST_FIELDS = ['game_id','quote_id','home_team','away_team','sportsbook','home_line','home_price_american',
'away_line','away_price_american','adjustment','predicted_home_margin','home_model_spread',
'captured_at_utc','kickoff_utc','forecast_created_at_utc','history_last_scored_date',
'snapshot_id','model_sha256','pipeline_sha256','github_run_id','status']

def stamp(dt):
    return dt.astimezone(timezone.utc).isoformat()

def utc(value):
    result = datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timestamp missing timezone')
    return result.astimezone(timezone.utc)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def download(url, label):
    # Never print URL/response bodies: the odds URL contains the secret key.
    try:
        with urlopen(Request(url, headers={'User-Agent':'NFLMarketResearch/1.0'}), timeout=45) as response:
            return response.read(), dict(response.headers)
    except HTTPError as exc:
        raise RuntimeError(f'{label}: HTTP {exc.code}; check service availability, key and quota') from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError(f'{label}: network request failed') from None

def schedule_frame(data, now):
    d = pd.read_csv(io.BytesIO(data))
    needed = {'game_id','season','week','gameday','home_team','away_team','home_score','away_score','spread_line','location'}
    if needed-set(d):
        raise ValueError('Schedule is missing required model columns')
    if d.game_id.isna().any() or d.game_id.duplicated().any():
        raise ValueError('Schedule game IDs must be unique')
    for c in ['home_team','away_team']:
        d[c] = d[c].replace({'LA':'LAR','STL':'LAR','SD':'LAC','OAK':'LV'})
    days = pd.to_datetime(d.gameday, errors='raise')
    today = pd.Timestamp(pd.Timestamp(now).tz_convert('America/New_York').date())
    # Daily dates cannot prove intraday completion: exclude all same-day results.
    d.loc[days >= today, ['home_score','away_score']] = float('nan')
    # Partial/in-progress scores are not eligible team history.
    partial = d.home_score.isna() | d.away_score.isna()
    d.loc[partial, ['home_score','away_score']] = float('nan')
    return d

def archive_rows(path, fields, rows, unique_key):
    old = set()
    if path.exists():
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fields:
                raise ValueError(f'Unexpected archive header: {path.name}')
            old = {row[unique_key] for row in reader}
    fresh = []
    for row in rows:
        if str(row[unique_key]) not in old:
            fresh.append(row)
            old.add(str(row[unique_key]))
    if fresh:
        path.parent.mkdir(parents=True,exist_ok=True)
        exists=path.exists()
        with path.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            if not exists: writer.writeheader()
            writer.writerows(fresh)
    return fresh

def match_quotes(events, schedule, captured, snapshot_id, max_days=7):
    rows=[]; skipped=[]
    for e in events:
        kickoff=utc(e['commence_time'])
        if kickoff <= captured or (max_days is not None and kickoff > captured+timedelta(days=max_days)): continue
        home,away=NAMES.get(e.get('home_team')),NAMES.get(e.get('away_team'))
        if not home or not away:
            skipped.append('Unknown provider team name');continue
        date=pd.Timestamp(kickoff).tz_convert('America/New_York').date().isoformat()
        matches=schedule[(schedule.home_team==home)&(schedule.away_team==away)&(schedule.gameday==date)]
        if len(matches)!=1:
            skipped.append(f'No unique schedule match: {away} at {home} on {date}');continue
        candidates=[]
        for book in e.get('bookmakers',[]):
            if book['key'] not in BOOK_ORDER: continue
            for market in book.get('markets',[]):
                if market.get('key')!='spreads': continue
                # Market-level time preferred; older responses may supply book time.
                updated_raw=market.get('last_update') or book.get('last_update')
                if not updated_raw:continue
                updated=utc(updated_raw)
                if not timedelta(0)<=captured-updated<=timedelta(minutes=30):continue
                outcomes=market.get('outcomes',[])
                hs=[o for o in outcomes if o.get('name')==e['home_team']]
                aws=[o for o in outcomes if o.get('name')==e['away_team']]
                if len(hs)!=1 or len(aws)!=1:continue
                hl,al=float(hs[0]['point']),float(aws[0]['point'])
                hp,ap=float(hs[0]['price']),float(aws[0]['price'])
                if not all(math.isfinite(x) for x in [hl,al,hp,ap]):continue
                if abs(hl+al)>1e-6 or abs(hl)>100:continue
                if any(abs(p)<100 or p!=int(p) for p in [hp,ap]):continue
                row=dict(provider_event_id=e['id'],game_id=matches.iloc[0].game_id,home_team=home,away_team=away,
                    bookmaker_key=book['key'],sportsbook=book['title'],home_line=hl,home_price_american=int(hp),
                    away_line=al,away_price_american=int(ap),provider_updated_at_utc=stamp(updated),
                    captured_at_utc=stamp(captured),kickoff_utc=stamp(kickoff),snapshot_id=snapshot_id,
                    source='THE_ODDS_API_CURRENT_SPREADS')
                identity={k:v for k,v in row.items() if k not in ['snapshot_id','captured_at_utc']}
                row['quote_id']=digest(json.dumps(identity,sort_keys=True).encode())[:24]
                candidates.append(row)
        if candidates:
            rows.append(min(candidates,key=lambda r:BOOK_ORDER.index(r['bookmaker_key'])))
        else: skipped.append(f'No fresh two-sided spread at supported books: {away} at {home}')
    return rows, skipped

def predict(schedule_path, quotes, snapshot_id, captured):
    f=model.read_games(schedule_path)
    cutoff=pd.Timestamp(captured)
    pending=[q for q in quotes if utc(q['kickoff_utc'])<=captured+timedelta(hours=24)]
    if not pending:return []
    q=pd.DataFrame(pending)
    live=f.merge(q,on='game_id',validate='one_to_one',suffixes=('','_quote'))
    if live.actual_margin.notna().any():raise ValueError('Selected game already has a score')
    live['spread_line']=-live.home_line
    results=[]
    for season,group in live.groupby('season'):
        train=f[(f.season<season)&(f.gameday<cutoff)].dropna(subset=['actual_margin','spread_line'])
        if len(train)<500:raise ValueError('Insufficient prior-season training history')
        group=group.copy();group['adjustment']=model.adjustment(model.fit(train),group)
        results.append(group)
    out=pd.concat(results,ignore_index=True)
    finished=datetime.now(timezone.utc)
    if any(utc(k)<=finished for k in out.kickoff_utc):
        raise ValueError('A game started during calculation; no forecasts saved this run')
    last=str(pd.read_csv(schedule_path).dropna(subset=['home_score','away_score']).gameday.max())
    rows=[]
    for _,r in out.iterrows():
        margin=float(r.home_line)*-1+float(r.adjustment)
        row={k:r[k] for k in ['game_id','quote_id','home_team','away_team','sportsbook','home_line',
             'home_price_american','away_line','away_price_american','captured_at_utc','kickoff_utc']}
        row.update(adjustment=float(r.adjustment),predicted_home_margin=margin,home_model_spread=-margin,
            forecast_created_at_utc=stamp(finished),history_last_scored_date=last,snapshot_id=snapshot_id,
            model_sha256=digest(Path(model.__file__).read_bytes()),pipeline_sha256=digest(Path(__file__).read_bytes()),
            github_run_id=os.environ.get('GITHUB_RUN_ID','local'),status='RESEARCH_ONLY_UNVALIDATED')
        rows.append(row)
    return rows

def main():
    key=os.environ.get('ODDS_API_KEY','').strip()
    if not key:raise ValueError('Add the ODDS_API_KEY repository Actions secret first')
    ROOT.mkdir(parents=True,exist_ok=True)
    raw_schedule,_=download(SCHEDULE_URL,'Schedule refresh')
    markets = 'spreads,totals'
    params=urlencode(dict(apiKey=key,regions='us',markets=markets,oddsFormat='american',dateFormat='iso'))
    raw_odds,headers=download('https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?'+params,'Odds refresh')
    captured=datetime.now(timezone.utc)
    events=json.loads(raw_odds)
    if not isinstance(events,list):raise ValueError('Odds response was not an event list')
    sid=captured.strftime('%Y%m%dT%H%M%S%fZ')
    folder=ROOT/'snapshots'/sid;folder.mkdir(parents=True)
    # Provider response has no API key; request URL is never archived.
    (folder/'odds.json.gz').write_bytes(gzip.compress(raw_odds))
    (folder/'schedule_raw.csv.gz').write_bytes(gzip.compress(raw_schedule))
    schedule=schedule_frame(raw_schedule,captured)
    schedule_path=folder/'schedule_asof.csv.gz';schedule.to_csv(schedule_path,index=False)
    quotes,skipped=match_quotes(events,schedule,captured,sid)
    archive_rows(ROOT/'quotes.csv',QUOTE_FIELDS,quotes,'quote_id')
    seen=set()
    forecast_path=ROOT/'forecasts.csv'
    if forecast_path.exists():
        with forecast_path.open(newline='') as f:seen={r['game_id'] for r in csv.DictReader(f)}
    # Preserve original manual forecasts as the first predictions for those games.
    legacy=Path('data/market_forecasts.csv')
    if legacy.exists():
        with legacy.open(newline='') as f:seen.update(r['game_id'] for r in csv.DictReader(f))
    forecasts=predict(schedule_path,[q for q in quotes if q['game_id'] not in seen],sid,captured)
    saved=archive_rows(forecast_path,FORECAST_FIELDS,forecasts,'game_id')
    # EPA is an independent shadow forecast from exactly this run's new core quotes.
    import epa_shadow
    try:
        epa_status=epa_shadow.record(schedule_path,saved,folder)
    except Exception as exc:
        # Preserve core forecasts and make any missing EPA pair explicit. Never backfill.
        epa_status=dict(status='EPA_UNAVAILABLE_AT_FORECAST',new_epa_forecasts=0,
            error_type=type(exc).__name__,model_version=epa_shadow.VERSION)
        if isinstance(exc,(ValueError,RuntimeError)):epa_status['reason']=str(exc)
    (folder/'epa_status.json').write_text(json.dumps(epa_status,indent=2))
    import consensus_shadow
    try:
        consensus_status = consensus_shadow.record(events, quotes, sid)
    except Exception as exc:
        consensus_status = {'status':'CONSENSUS_UNAVAILABLE','error_type':type(exc).__name__}
    import total_shadow
    try:
        total_status = total_shadow.record(schedule_path, quotes, events, folder)
    except Exception as exc:
        total_status = {'status':'TOTAL_UNAVAILABLE','error':type(exc).__name__}
    (folder/'total_status.json').write_text(json.dumps(total_status,indent=2))
    import opening_tracker
    try:
        opening_status = opening_tracker.capture(events, schedule, schedule_path, captured, folder)
    except Exception as exc:
        opening_status = {'status': 'ERROR', 'error': type(exc).__name__}
        print('::warning::Opening tracker failed: ' + type(exc).__name__)
    (folder/'opening_status.json').write_text(json.dumps(opening_status, indent=2))
    # Reuse the existing timestamped starter-confirmation file in an independent candidate.
    import qb_market
    try:
        qb_status=qb_market.record(schedule_path,saved,folder)
    except Exception as exc:
        qb_status=dict(status='QB_UNAVAILABLE_AT_FORECAST',new_qb_forecasts=0,
            error_type=type(exc).__name__)
        if isinstance(exc,(ValueError,RuntimeError)):qb_status['reason']=str(exc)
    (folder/'qb_status.json').write_text(json.dumps(qb_status,indent=2))
    import qb_passing
    try:
        passing_status=qb_passing.record(schedule_path,saved,folder)
    except Exception as exc:
        passing_status=dict(status='QB_PASSING_UNAVAILABLE_AT_FORECAST',new_qb_forecasts=0,
            error_type=type(exc).__name__)
        if isinstance(exc,(ValueError,RuntimeError)):passing_status['reason']=str(exc)
    (folder/'qb_passing_status.json').write_text(json.dumps(passing_status,indent=2))


    import expected_qb_shadow
    try:
        expected_status=expected_qb_shadow.record(schedule_path,saved,folder)
    except Exception as exc:
        expected_status=dict(status='EXPECTED_QB_UNAVAILABLE_AT_FORECAST',new_qb_forecasts=0,error_type=type(exc).__name__)
        if isinstance(exc,(ValueError,RuntimeError)):expected_status['reason']=str(exc)
    (folder/'expected_qb_status.json').write_text(json.dumps(expected_status,indent=2))

    import advanced_shadow
    try:
        advanced_status=advanced_shadow.record(schedule_path,saved,folder)
    except Exception as exc:
        advanced_status=dict(status='ADVANCED_UNAVAILABLE_AT_FORECAST',error_type=type(exc).__name__)
        if isinstance(exc,(ValueError,RuntimeError)):advanced_status['reason']=str(exc)
    (folder/'advanced_status.json').write_text(json.dumps(advanced_status,indent=2))
    normalized_headers={k.lower():v for k,v in headers.items()}
    report=dict(captured_at_utc=stamp(captured),snapshot_id=sid,provider_events=len(events),
        opening_comparison=opening_status,consensus_shadow=consensus_status,total_shadow=total_status,eligible_quotes=len(quotes),new_forecasts=len(saved),epa_shadow=epa_status,advanced_shadow=advanced_status,qb_shadow=qb_status,qb_passing_shadow=passing_status,expected_qb_shadow=expected_status,skipped=skipped,
        remaining_credits=normalized_headers.get('x-requests-remaining','unknown'),
        last_request_credits=normalized_headers.get('x-requests-last','unknown'),
        schedule_sha256=digest(raw_schedule),odds_sha256=digest(raw_odds),
        policy='First automatic quote within 24h of kickoff; fixed bookmaker priority; same-day scores excluded')
    (folder/'run.json').write_text(json.dumps(report,indent=2))
    (ROOT/'latest_run.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    summary=os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary,'a') as f:
            f.write('## EPA shadow capture\n\n'+json.dumps(epa_status,sort_keys=True)+'\n\n')
    if summary:
        with open(summary,'a') as f:
            f.write(f'## Automatic NFL market run\n\nArchived {len(quotes)} eligible quotes; saved {len(saved)} new forecasts.\n\nRemaining provider credits: {report["remaining_credits"]}.\n\nDownload nfl-auto-market-results for CSVs and source snapshots. No bets are placed. Research model remains unvalidated.\n')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        # Sanitized failure only; no request URL, API key or response body in logs.
        if isinstance(exc,(ValueError,RuntimeError,FileNotFoundError)):
            print(f'ERROR: {exc}',file=sys.stderr)
        else:print(f'ERROR: {type(exc).__name__}; check data schema or service availability',file=sys.stderr)
        sys.exit(1)
