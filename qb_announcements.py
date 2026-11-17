"""Conservative official-club announcement adapter; unmatched text means abstention."""
import argparse
import csv
import hashlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
import qb_news as news

PLAYERS_URL = 'https://github.com/nflverse/nflverse-data/releases/download/players/players.csv'
SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
FIELDS = ['game_id','team','qb_id','qb_name','status','source_url','published_at_utc','observed_at_utc']
NICKNAMES = dict(zip(news.DOMAINS, 'Cardinals Falcons Ravens Bills Panthers Bears Bengals Browns Cowboys Broncos Lions Packers Texans Colts Jaguars Chiefs Raiders Chargers Rams Dolphins Vikings Patriots Saints Giants Jets Eagles Steelers 49ers Seahawks Buccaneers Titans Commanders'.split()))
VERSION = 'official_explicit_game_v2'


def norm(text):
    return ' '.join(str(text).replace('\u2019', "'").split())


class SchemaParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.buffer = []
        self.items = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('type') == 'application/ld+json':
            self.inside = True
            self.buffer = []

    def handle_data(self, data):
        if self.inside:
            self.buffer.append(data)

    def handle_endtag(self, tag):
        if tag == 'script' and self.inside:
            self.inside = False
            try:
                self.items.append(json.loads(''.join(self.buffer)))
            except json.JSONDecodeError:
                pass


def article(raw, url, team, observed):
    parser = SchemaParser()
    parser.feed(raw.decode('utf-8'))
    def walk(obj):
        if isinstance(obj, list):
            for child in obj:
                yield from walk(child)
        elif isinstance(obj, dict):
            yield obj
            if '@graph' in obj:
                yield from walk(obj['@graph'])
    matches = [x for obj in parser.items for x in walk(obj)
        if x.get('@type') == 'NewsArticle' and x.get('url') == url]
    if len(matches) != 1 or not news.official(url, news.DOMAINS[team]):
        raise ValueError('No unique canonical official NewsArticle')
    a = matches[0]
    published = news.utc(a['datePublished'])
    modified = news.utc(a.get('dateModified', a['datePublished']))
    if not published <= modified <= observed or published < observed - timedelta(days=8):
        raise ValueError('Invalid, future, or stale article timestamp')
    if not isinstance(a.get('articleBody'), str) or not a['articleBody'].strip():
        raise ValueError('Missing article body')
    return dict(title=norm(a['headline']), body=a['articleBody'],
        published_at_utc=modified.isoformat(), original_published_at_utc=published.isoformat(),
        observed_at_utc=observed.isoformat(), source_url=url, team=team)


def qb_players(rows, season):
    return [r for r in rows if r.get('position')=='QB' and r.get('gsis_id')
        and re.fullmatch(r'00-\d{7}', r['gsis_id'])
        and r.get('last_season')==str(season) and r.get('latest_team') in news.DOMAINS]


def upcoming(rows, now):
    result = []
    for r in rows:
        if r.get('game_type') != 'REG' or not r.get('gametime'):
            continue
        kickoff = datetime.fromisoformat(r['gameday']+'T'+r['gametime']).replace(tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc)
        if now < kickoff <= now + timedelta(days=8):
            r = dict(r)
            r['kickoff_utc'] = kickoff.isoformat()
            for side in ['home_team','away_team']:
                r[side] = {'LA':'LAR','OAK':'LV','SD':'LAC'}.get(r[side],r[side])
            result.append(r)
    return result


def interpret(a, games, players):
    """Full-name + explicit decision + exact game context in ONE statement."""
    output = []
    observed, published = news.utc(a['observed_at_utc']), news.utc(a['published_at_utc'])
    # Only headline and first three body paragraphs: exclude sidebars and historical recaps.
    passages = [a['title']] + [norm(p) for p in a['body'].split('\n') if p.strip()][:3]
    # Reuse the exact source prose in the audit, not as a model feature.
    decisions = []
    for g in games:
        if a['team'] not in [g['home_team'], g['away_team']]:
            continue
        if not published <= observed < news.utc(g['kickoff_utc']):
            continue
        opponent = g['away_team'] if a['team']==g['home_team'] else g['home_team']
        week = r'\bWeek\s+'+str(int(g['week']))+r'\b'
        opp = r'\b(?:against|versus|vs\.?)\s+(?:the\s+)?(?:[A-Za-z]+\s+){0,3}'+re.escape(NICKNAMES[opponent])+r'\b'
        for p in players:
            if p['latest_team'] != a['team'] or p['last_season'] != str(g['season']):
                continue
            name = norm(p['display_name'])
            for passage in passages:
                # Reject speculation, conditions, quotations and questions in decision text.
                if len(passage)>450 or re.search(r'\b(if|unless|barring|could|may|might|expected|likely|optimistic|hopes?|would|should|last year|but|however|pending|questionable|doubtful|undecided|uncertain|not cleared|not confirmed)\b|[?"“”]', passage, re.I):
                    continue
                if any(int(y)!=int(g['season']) for y in re.findall(r'\b20\d{2}\b',passage)):
                    continue
                # Verified club phrasing can specify team + week + season rather
                # than repeat the opponent. Accept only a unique scheduled game.
                unique = [x for x in games if str(x['season'])==str(g['season'])
                    and int(x['week'])==int(g['week']) and a['team'] in [x['home_team'],x['away_team']]]
                designation = (r'\b'+re.escape(name)+r'\s+will be the\s+'
                    +re.escape(NICKNAMES[a['team']])+r"'?\s+starting quarterback\s+heading into\s+Week\s+"
                    +str(int(g['week']))+r'\s+of the\s+'+str(g['season'])+r'\s+season\b')
                if len(unique)==1 and re.search(designation,passage,re.I):
                    decisions.append((g,p,'confirmed',passage))
                    continue
                start = re.search(r'\b'+re.escape(name)+r'\s+(?:(?:is|has been)\s+)?(?:will start|is starting|set to start|to start|named (?:the )?starting quarterback|will not start|won\x27t start|ruled out)\b', passage, re.I)
                if not start:
                    continue
                # The match's tail cannot cross into a second sentence.
                tail = passage[start.end():]
                if not re.search(week, tail, re.I) or not re.search(opp, tail, re.I):
                    continue
                if re.search(r'\.(?:\s|$)', tail[:max(re.search(week,tail,re.I).end(),re.search(opp,tail,re.I).end())]):
                    continue
                status = 'unknown' if re.search(r'will not|won\x27t|ruled out', start.group(), re.I) else 'confirmed'
                decisions.append((g, p, status, passage))
    for g, p, status, passage in decisions:
        row = dict(game_id=g['game_id'],team=a['team'],qb_id=p['gsis_id'],qb_name=p['display_name'],
            status=status,source_url=a['source_url'],published_at_utc=a['published_at_utc'],
            observed_at_utc=a['observed_at_utc'])
        if row not in [r['confirmation'] for r in output]:
            output.append(dict(confirmation=row,statement=passage,adapter=VERSION))
    return output


def reconcile(old, decisions, checked, observed):
    """Append decisions once; re-observation cannot resurrect an older announcement.

    Recheck every previously accepted source. A failed fetch, removed decision or
    ambiguous multi-player decision revokes its confirmation conservatively.
    """
    fresh = []
    keys = {(r['source_url'],r['published_at_utc'],r['game_id'],r['team'],r['qb_id'],r['status']) for r in old}
    latest = {}
    for r in old:
        key = (r['game_id'],r['team'])
        if key not in latest or news.utc(r['observed_at_utc']) >= news.utc(latest[key]['observed_at_utc']):
            latest[key] = r
    grouped = {}
    for d in decisions:
        r = dict(d['confirmation'])
        grouped.setdefault((r['game_id'],r['team']),[]).append(r)
    for key, rows in grouped.items():
        newest = max(news.utc(r['published_at_utc']) for r in rows)
        rows = [r for r in rows if news.utc(r['published_at_utc'])==newest]
        r = dict(rows[0])
        if len({(r['qb_id'],r['status']) for r in rows}) != 1:
            r['status'] = 'unknown'
        prev = latest.get(key)
        if prev and newest < news.utc(prev['published_at_utc']):
            continue
        ident = (r['source_url'],r['published_at_utc'],r['game_id'],r['team'],r['qb_id'],r['status'])
        if ident not in keys or (prev and prev['status']!='confirmed' and r['status']=='confirmed'):
            fresh.append(r)
    for key, prev in latest.items():
        if prev['status'] != 'confirmed' or prev['source_url'] not in checked:
            continue
        matching = [d['confirmation'] for d in decisions if d['confirmation']['source_url']==prev['source_url'] and (d['confirmation']['game_id'],d['confirmation']['team'])==key and d['confirmation']['qb_id']==prev['qb_id'] and d['confirmation']['status']=='confirmed']
        if not matching and not any((r['game_id'],r['team'])==key for r in fresh):
            r = dict(prev, status='unknown', observed_at_utc=observed.isoformat())
            fresh.append(r)
    return fresh


def fetch(url, limit=4_000_000):
    with urlopen(Request(url,headers={'User-Agent':'NFLModelResearch/1.0'}),timeout=20) as response:
        raw = response.read(limit+1)
        final_url = response.url
    if len(raw)>limit:
        raise ValueError('Response too large')
    return raw, final_url


def combine_confirmations(manual, automatic):
    """Any manually managed game/team keeps exclusive manual authority."""
    managed = {(r['game_id'],r['team']) for r in manual}
    return manual + [r for r in automatic if (r['game_id'],r['team']) not in managed]


def run(snapshot, out, destination, ledger_root=Path('data/auto_market/qb_announcements')):
    now = datetime.now(timezone.utc)
    out.mkdir(parents=True,exist_ok=False)
    # Snapshot exact player identity and schedule inputs; no scores are used.
    raw_players, _ = fetch(PLAYERS_URL,20_000_000)
    raw_games, _ = fetch(SCHEDULE_URL)
    (out/'players.csv').write_bytes(raw_players)
    (out/'schedule.csv').write_bytes(raw_games)
    games = upcoming(list(csv.DictReader(io.StringIO(raw_games.decode()))),now)
    seasons = {int(g['season']) for g in games}
    player_rows = list(csv.DictReader(io.StringIO(raw_players.decode())))
    players = [r for season in seasons for r in qb_players(player_rows,season)]
    old = news.read_rows(destination)
    collected = json.loads((snapshot/'news.json').read_text())
    # Match current player names too: a player's name may appear without "QB".
    selected = {(r['source_url'],r['team']) for r in collected if r['qb_headline'] or any(
        p['latest_team']==r['team'] and norm(p['display_name']).lower() in norm(r['title']).lower() for p in players)}
    pending_ids = {g['game_id'] for g in games}
    selected.update((r['source_url'],r['team']) for r in old if r['game_id'] in pending_ids)
    checked = set()
    sources = []
    decisions = []
    def one(pair):
        url, team = pair
        try:
            if not news.official(url,news.DOMAINS[team]):
                raise ValueError('Non-official source')
            raw, final = fetch(url)
            if final != url:
                raise ValueError('Canonical URL changed')
            observed = datetime.now(timezone.utc)
            digest = hashlib.sha256(raw).hexdigest()
            (out/(digest+'.html')).write_bytes(raw)
            a = article(raw,url,team,observed)
            return dict(source_url=url,team=team,status='CHECKED',sha256=digest,observed_at_utc=observed.isoformat()),interpret(a,games,players)
        except Exception as exc:
            return dict(source_url=url,team=team,status='UNAVAILABLE_OR_UNSUPPORTED',error_type=type(exc).__name__),[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for source, found in pool.map(one,sorted(selected)):
            checked.add(source['source_url'])
            sources.append(source)
            decisions.extend(found)
    observed = datetime.now(timezone.utc)
    # A run crossing kickoff cannot write those rows.
    live_ids = {g['game_id'] for g in games if news.utc(g['kickoff_utc'])>observed}
    decisions = [d for d in decisions if d['confirmation']['game_id'] in live_ids]
    automatic_old = news.read_rows(ledger_root/'automatic_ledger.csv')
    additions = reconcile(automatic_old,decisions,checked,observed)
    additions = [r for r in additions if r['game_id'] in live_ids]
    ledger = automatic_old + additions
    # Atomic write of a separate automatic ledger; manual input is never modified.
    write_csv(ledger_root/'automatic_ledger.csv',ledger)
    # Automatic observations never override a manually managed game/team.
    combined = combine_confirmations(old,ledger)
    write_csv(out/'confirmations_asof.csv',combined)
    coverage = news.coverage(news.read_rows(Path('data/auto_market/quotes.csv')),combined,observed)
    # Use the same eligibility gate on the full upcoming schedule, separately from odds coverage.
    schedule_coverage = news.coverage([dict(g,captured_at_utc=now.isoformat()) for g in games],combined,observed)
    schedule_coverage['scheduled_upcoming_games'] = schedule_coverage.pop('quoted_upcoming_games')
    schedule_coverage['scope'] = 'Every upcoming regular-season game in the current schedule over eight days'
    for g in schedule_coverage['games']:
        g['schedule_captured_at_utc'] = g.pop('latest_quote_at_utc')
    summary = dict(adapter=VERSION,observed_at_utc=observed.isoformat(),schedule_upcoming_games=len(games),
        player_identity_rows=len(players),sources_attempted=len(sources),sources_checked=sum(s['status']=='CHECKED' for s in sources),
        matched_decisions=len(decisions),new_ledger_rows=len(additions),
        new_confirmations=sum(r['status']=='confirmed' for r in additions),coverage=coverage,schedule_coverage=schedule_coverage,
        limitations=['Requires explicit regular-season week/opponent or team/week/season with a unique scheduled game, in headline or first three paragraphs.',
            'Other wording, missing identity, speculative and unsupported articles abstain.',
            'This is a conservative partial source adapter, not comprehensive starter coverage.',
            'Previously archived forecasts are never backfilled.'])
    (out/'sources.json').write_text(json.dumps(sources,indent=2))
    (out/'decisions.json').write_text(json.dumps(decisions,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (ledger_root/'latest.json').write_text(json.dumps(summary,indent=2))
    return out/'confirmations_asof.csv', summary


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k:r[k] for k in FIELDS} for r in rows)
    temp.replace(path)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('qb_announcement_evidence')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = p.parse_args()
    _, report = run(args.snapshot,args.out,Path('nfl_data_inputs/qb_starters.csv'))
    print(json.dumps(report,indent=2))
