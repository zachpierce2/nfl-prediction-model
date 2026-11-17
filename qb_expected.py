"""Prospective starter expectations; never relabel depth charts as confirmations."""
import csv
import gzip
import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd
import auto_market as auto
import qb_announcements as announcements
import qb_market
import qb_news

ROOT=Path('data/auto_market/qb_expected')
VERSION='expected_qb_v1'
BASE='https://github.com/nflverse/nflverse-data/releases/download/'
FIELDS=['game_id','team','qb_id','qb_name','status','reason','source_url',
        'published_at_utc','observed_at_utc','kickoff_utc','injury_status','practice_status','evidence_json']
ALIASES={'LA':'LAR','STL':'LAR','SD':'LAC','OAK':'LV'}

def fetch(url,path):
    with urlopen(Request(url,headers={'User-Agent':'NFL-QB-Expectations/1.0'}),timeout=90) as response:
        raw=response.read();modified=response.headers.get('Last-Modified')
    observed=datetime.now(timezone.utc)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(raw))
    meta=dict(url=url,observed_at_utc=auto.stamp(observed),sha256=auto.digest(raw),
              last_modified_utc=auto.stamp(parsedate_to_datetime(modified)) if modified else None)
    return pd.read_csv(io.BytesIO(raw),dtype=str,keep_default_na=False),meta

def fresh(meta,now,hours=48):
    try:
        modified=auto.utc(meta['last_modified_utc']);observed=auto.utc(meta['observed_at_utc'])
        return modified<=observed<=now and now-modified<=timedelta(hours=hours)
    except (KeyError,TypeError,ValueError):return False

def canonical(f):
    f=f.copy()
    if 'team' in f:f['team']=f.team.replace(ALIASES)
    return f

def derive(games,schedule,depth,injuries,players,reports,metadata,now):
    """Use only sources actually observed by cutoff; expectations are game-specific."""
    for key in ['depth','injuries','players','schedule']:
        meta=metadata.get(key,{})
        if meta.get('observed_at_utc') and auto.utc(meta['observed_at_utc'])>now:
            raise ValueError('Source observation follows cutoff: '+key)
    depth=canonical(depth.rename(columns={'player_gsis_id':'gsis_id'}))
    injuries=canonical(injuries)
    if len(depth):
        need={'dt','team','gsis_id','player_name','pos_abb','pos_rank'}
        if need-set(depth):raise ValueError('Depth schema missing columns')
        depth['_dt']=pd.to_datetime(depth.dt,utc=True,errors='coerce')
        depth=depth[depth._dt.notna() & depth._dt.le(now)]
        depth=depth[depth._dt.eq(depth.groupby('team')._dt.transform('max'))]
        depth['rank']=pd.to_numeric(depth.pos_rank,errors='coerce')
    if len(injuries) and {'season','week','team','gsis_id','report_status','practice_status'}-set(injuries):
        raise ValueError('Injury schema missing columns')
    output=[]
    for game in games:
        kickoff=auto.utc(game['kickoff_utc'])
        if kickoff<=now:continue
        for team in [game['home_team'],game['away_team']]:
            evidence=[]
            row=dict(game_id=game['game_id'],team=team,qb_id='',qb_name='',status='unknown',reason='NO_CURRENT_STARTER_EVIDENCE',source_url='',published_at_utc='',observed_at_utc=auto.stamp(now),kickoff_utc=game['kickoff_utc'],injury_status='unknown',practice_status='unknown')
            confirmed=qb_market.confirmed_ids(reports,game['game_id'],team,auto.stamp(now),game['kickoff_utc'])
            team_reports=reports[(reports.game_id==game['game_id']) & (reports.team==team)]
            valid_reports=[]
            for r in team_reports.to_dict('records'):
                if auto.utc(r['published_at_utc'])<=auto.utc(r['observed_at_utc'])<=now:
                    valid_reports.append(r)
            latest=max(valid_reports,key=lambda r:(auto.utc(r['observed_at_utc']),auto.utc(r['published_at_utc']))) if valid_reports else None
            if confirmed:
                row.update(qb_id=confirmed,qb_name=latest.get('qb_name',''),status='confirmed',reason='EXPLICIT_STARTER_REPORT',source_url=latest['source_url'],published_at_utc=latest['published_at_utc'])
                evidence.append(dict(kind='confirmation',**latest))
            else:
                candidates=[];unresolved_depth=False;depth_meta=metadata.get('depth',{})
                observed_depth=depth_meta.get('observed_at_utc')
                if len(depth) and observed_depth and auto.utc(observed_depth)<=now:
                    td=depth[depth.team.eq(team)]
                    if len(td) and now-td._dt.iloc[0].to_pydatetime()<=timedelta(hours=48):
                        tq=td[td.pos_abb.eq('QB') & td['rank'].ge(1)]
                        candidates=[dict(qb_id=r.gsis_id,qb_name=r.player_name,rank=float(r.rank),kind='depth_chart',published_at_utc=r.dt,source_url=depth_meta['url']) for r in tq.itertuples() if re.fullmatch(r'00-\d{7}',str(r.gsis_id))]
                        if len(tq) and any(not re.fullmatch(r'00-\d{7}',str(r.gsis_id)) for r in tq[tq['rank'].eq(tq['rank'].min())].itertuples()):
                            candidates=[];unresolved_depth=True;row['reason']='UNRESOLVED_DEPTH_ID'
                        evidence.extend(candidates)
                inj=pd.DataFrame()
                im=metadata.get('injuries',{})
                if len(injuries) and fresh(im,now):
                    typecol='season_type' if 'season_type' in injuries else 'game_type'
                    if typecol in injuries:
                        inj=injuries[injuries.team.eq(team)&injuries.season.eq(str(game['season']))&injuries.week.eq(str(game['week']))&injuries[typecol].eq(game.get('game_type','REG'))]
                        evidence.extend(dict(kind='injury_report',source_url=im['url'],observed_at_utc=im['observed_at_utc'],**r) for r in inj.to_dict('records'))
                excluded=set()
                if len(inj):
                    for qid,g in inj.groupby('gsis_id'):
                        statuses=set(g.report_status.str.strip().str.lower())
                        if statuses=={'out'}:excluded.add(qid)
                # No source-confirmed player can silently remain the expected starter after revocation.
                revoked=latest['qb_id'] if latest and latest['status'].lower()!='confirmed' else None
                if not candidates and not unresolved_depth:
                    # Earlier completed starts, current season only, with current team identity.
                    ids={r['gsis_id']:r['display_name'] for r in players if r.get('position')=='QB' and r.get('latest_team')==team and r.get('last_season')==str(game['season'])}
                    recent=[]
                    for old in schedule:
                        try:day=datetime.fromisoformat(old['gameday']).replace(tzinfo=timezone.utc)
                        except (ValueError,KeyError):continue
                        if str(old.get('season'))!=str(game['season']) or not (timedelta(0)<now-day<=timedelta(days=35)) or day.date()>=now.date():continue
                        if old.get('home_score','')=='' or old.get('away_score','')=='':continue
                        for side in ['home','away']:
                            qid=old.get(side+'_qb_id','')
                            if ALIASES.get(old.get(side+'_team'),old.get(side+'_team'))==team and qid in ids:
                                recent.append((day,dict(qb_id=qid,qb_name=ids[qid],rank=1,kind='recent_start',published_at_utc='',source_url=metadata['schedule']['url'],prior_game_id=old['game_id'])))
                    if recent:
                        candidate=max(recent,key=lambda x:x[0])[1];evidence.append(candidate);candidates=[candidate]
                usable=[c for c in candidates if c['qb_id'] not in excluded]
                if usable:
                    rank=min(c['rank'] for c in usable);best={c['qb_id']:c for c in usable if c['rank']==rank}
                    if len(best)==1:
                        b=next(iter(best.values()))
                        row.update(qb_id=b['qb_id'],qb_name=b['qb_name'],status='expected',reason=b['kind'].upper(),source_url=b['source_url'],published_at_utc=b['published_at_utc'])
                        qi=inj[inj.gsis_id.eq(b['qb_id'])] if len(inj) else pd.DataFrame()
                        if len(qi):
                            statuses=set(qi.report_status.str.strip().str.lower())
                            practices=set(qi.practice_status.str.strip())
                            row.update(injury_status='|'.join(sorted(statuses)) or 'no_game_designation',practice_status='|'.join(sorted(practices)))
                            if len(statuses)>1 or statuses & {'questionable','doubtful','out'} or any(p in {'Did Not Participate In Practice','Limited Participation in Practice'} for p in practices):
                                row.update(status='uncertain',reason='INJURY_OR_PRACTICE_UNCERTAINTY')
                        if revoked==b['qb_id']:row.update(status='uncertain',reason='STARTER_REPORT_REVOKED_OR_NEGATIVE')
                        if excluded and b['kind']=='depth_chart' and b['rank']>1:row['reason']='DEPTH_REPLACEMENT_FOR_OUT' if row['status']=='expected' else row['reason']
                    else:row['reason']='AMBIGUOUS_DEPTH_RANK'
                elif candidates:row['reason']='NO_AVAILABLE_RANKED_QB'
            if row['status']=='confirmed' and len(injuries) and fresh(metadata.get('injuries',{}),now):
                typecol='season_type' if 'season_type' in injuries else 'game_type'
                if typecol in injuries:
                    qi=injuries[injuries.team.eq(team)&injuries.season.eq(str(game['season']))&injuries.week.eq(str(game['week']))&injuries[typecol].eq(game.get('game_type','REG'))&injuries.gsis_id.eq(row['qb_id'])]
                    if len(qi):
                        statuses=set(qi.report_status.str.strip().str.lower())
                        row['injury_status']='|'.join(sorted(statuses)) or 'no_game_designation'
                        row['practice_status']='|'.join(sorted(set(qi.practice_status)))
                        evidence.extend(dict(kind='injury_report',source_url=metadata['injuries']['url'],**r) for r in qi.to_dict('records'))
                        if statuses & {'out','doubtful'} and auto.utc(metadata['injuries']['last_modified_utc'])>auto.utc(row['published_at_utc']):
                            row.update(status='uncertain',reason='CONFIRMATION_CONFLICTS_WITH_NEWER_INJURY_FEED')
            row['evidence_json']=json.dumps(evidence,sort_keys=True)
            output.append(row)
    return output

def resolve(rows,game_id,team,cutoff,kickoff):
    now=auto.utc(cutoff);kick=auto.utc(kickoff)
    eligible=[r for r in rows.to_dict('records') if r['game_id']==game_id and r['team']==team and auto.utc(r['observed_at_utc'])<=now<kick]
    if not eligible:return None
    latest=max(auto.utc(r['observed_at_utc']) for r in eligible)
    chosen=[r for r in eligible if auto.utc(r['observed_at_utc'])==latest]
    if len(chosen)!=1:raise ValueError('Ambiguous expected starter observations')
    r=chosen[0]
    if auto.utc(r['kickoff_utc'])!=kick:raise ValueError('Expected starter kickoff changed')
    if now-latest>timedelta(hours=6):return None
    if r['published_at_utc'] and auto.utc(r['published_at_utc'])>latest:raise ValueError('Future starter evidence')
    if r['status'] not in {'expected','confirmed'}:return None
    if not re.fullmatch(r'00-\d{7}',r['qb_id']) or not r['source_url'].startswith('https://'):raise ValueError('Invalid expected starter identity/source')
    return r['qb_id']

def collect(out,local=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    metadata={};frames={}
    def obtain(key,url):
        try:
            if local:
                frame=pd.read_csv(Path(local)/(key+'.csv'),dtype=str,keep_default_na=False)
                meta=json.loads((Path(local)/'metadata.json').read_text())[key]
            else:frame,meta=fetch(url,out/(key+'.csv.gz'))
            metadata[key]=meta;frames[key]=frame
        except Exception as exc:
            metadata[key]=dict(url=url,status='UNAVAILABLE',error_type=type(exc).__name__);frames[key]=pd.DataFrame()
    obtain('schedule',announcements.SCHEDULE_URL)
    now=datetime.now(timezone.utc)
    if frames['schedule'].empty:raise ValueError('Expected QB schedule unavailable')
    schedule=frames['schedule'].to_dict('records');games=announcements.upcoming(schedule,now)
    seasons={g['season'] for g in games}
    if len(seasons)>1:raise ValueError('Multiple upcoming seasons')
    if seasons:
        year=next(iter(seasons))
        obtain('depth',BASE+f'depth_charts/depth_charts_{year}.csv')
        obtain('injuries',BASE+f'injuries/injuries_{year}.csv')
        obtain('players',announcements.PLAYERS_URL)
    else:
        for k in ['depth','injuries','players']:frames[k]=pd.DataFrame()
    manual=qb_news.read_rows(Path('nfl_data_inputs/qb_starters.csv'))
    path=Path(os.environ.get('QB_CONFIRMATIONS_PATH',''))
    if path.is_file():reports=qb_news.read_rows(path)
    else:reports=announcements.combine_confirmations(manual,qb_news.read_rows(Path('data/auto_market/qb_announcements/automatic_ledger.csv')))
    now=datetime.now(timezone.utc)
    rows=derive(games,schedule,frames['depth'],frames['injuries'],frames['players'].to_dict('records'),pd.DataFrame(reports,columns=announcements.FIELDS),metadata,now)
    path=out/'starters_asof.csv';pd.DataFrame(rows,columns=FIELDS).to_csv(path,index=False)
    counts={s:sum(r['status']==s for r in rows) for s in ['expected','confirmed','uncertain','unknown']}
    report=dict(version=VERSION,code_sha256=auto.digest(Path(__file__).read_bytes()),snapshot_sha256=auto.digest(path.read_bytes()),observed_at_utc=auto.stamp(now),upcoming_games=len(games),team_rows=len(rows),status_counts=counts,
        usable_both_sides=sum(all(any(r['game_id']==g['game_id'] and r['team']==t and r['status'] in {'expected','confirmed'} for r in rows) for t in [g['home_team'],g['away_team']]) for g in games),sources=metadata,
        limitations=['Expected is not confirmed; no probabilities are invented.','Missing injury rows do not establish health; reserve-list coverage is incomplete.','Questionable/doubtful or limited/no practice remain uncertain and skip the expected-QB forecast.','Recent starts are a current-season fallback, not an announcement.','No backfill, historical accuracy claim or primary-model promotion.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'latest.csv').write_bytes(path.read_bytes());(ROOT/'latest.json').write_text(json.dumps(report,indent=2))
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'],'a') as env:env.write('QB_EXPECTATIONS_PATH='+str(path.resolve())+'\n')
    print(json.dumps(report,indent=2));return path,report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',default='qb_expected_evidence/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));p.add_argument('--local');a=p.parse_args();collect(a.out,a.local)
