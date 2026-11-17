"""Archive official-team news as evidence, never infer confirmed starters from text."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

# Official club websites. A URL's inclusion is not a claim that its RSS is available.
DOMAINS = dict(zip(
    'ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LV LAC LAR MIA MIN NE NO NYG NYJ PHI PIT SF SEA TB TEN WAS'.split(),
    'azcardinals.com atlantafalcons.com baltimoreravens.com buffalobills.com panthers.com chicagobears.com bengals.com clevelandbrowns.com dallascowboys.com denverbroncos.com detroitlions.com packers.com houstontexans.com colts.com jaguars.com chiefs.com raiders.com chargers.com therams.com miamidolphins.com vikings.com patriots.com neworleanssaints.com giants.com newyorkjets.com philadelphiaeagles.com steelers.com 49ers.com seahawks.com buccaneers.com tennesseetitans.com commanders.com'.split()))


def utc(value):
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('Timezone required')
    return d.astimezone(timezone.utc)


def official(url, domain):
    p = urlparse(url)
    return p.scheme == 'https' and p.hostname in {domain, 'www.' + domain}


def parse_feed(raw, team, observed):
    """Keep dated official links only; headline relevance does not confer confirmation."""
    root = ET.fromstring(raw)
    if root.tag != 'rss':
        raise ValueError('Expected RSS')
    rows = []
    for item in root.findall('./channel/item'):
        link, title = item.findtext('link', '').strip(), item.findtext('title', '').strip()
        try:
            published = parsedate_to_datetime(item.findtext('pubDate', ''))
            if published.tzinfo is None:
                continue
            published = published.astimezone(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            continue
        if not official(link, DOMAINS[team]) or not title or not observed - timedelta(days=8) <= published <= observed:
            continue
        rows.append(dict(team=team, title=title, source_url=link,
            published_at_utc=published.isoformat(), observed_at_utc=observed.isoformat(),
            status='UNREVIEWED_NEWS_NOT_A_STARTER_CONFIRMATION',
            qb_headline=any(w in title.lower() for w in ['quarterback', ' qb', 'starter', 'will start', 'to start'])))
    return rows


def collect_one(team, folder):
    url = 'https://www.' + DOMAINS[team] + '/rss/news'
    try:
        with urlopen(Request(url, headers={'User-Agent': 'NFLModelResearch/1.0'}), timeout=20) as response:
            if not official(response.url, DOMAINS[team]):
                raise ValueError('Non-club redirect')
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError('Oversized feed')
        observed = datetime.now(timezone.utc)
        # Save exact response even when parsing fails.
        (folder / (team + '.xml')).write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        rows = parse_feed(raw, team, observed)
        return dict(team=team, source_url=url, status='FETCHED',
            observed_at_utc=observed.isoformat(), sha256=digest, recent_items=len(rows)), rows
    except Exception as exc:
        return dict(team=team, source_url=url, status='UNAVAILABLE', error_type=type(exc).__name__), []


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def coverage(quotes, confirmations, now):
    """Count exact game/team confirmations using the candidate's own eligibility gate."""
    import pandas as pd
    from qb_market import confirmed_ids
    games = {}
    for q in quotes:
        kickoff = utc(q['kickoff_utc'])
        if now < kickoff <= now + timedelta(days=8):
            old = games.get(q['game_id'])
            if old is None or utc(q['captured_at_utc']) > utc(old['captured_at_utc']):
                games[q['game_id']] = q
    columns = ['game_id','team','qb_id','status','source_url','published_at_utc','observed_at_utc']
    reports = pd.DataFrame(confirmations, columns=columns)
    rows = []
    for gid, q in sorted(games.items()):
        sides = {}
        for side in ['home', 'away']:
            try:
                qid = confirmed_ids(reports, gid, q[side + '_team'], now.isoformat(), q['kickoff_utc'])
                sides[side] = dict(team=q[side + '_team'], qb_id=qid, status='CONFIRMED' if qid else 'MISSING')
            except (ValueError, TypeError):
                sides[side] = dict(team=q[side + '_team'], qb_id=None, status='INVALID_REPORT')
        rows.append(dict(game_id=gid, kickoff_utc=q['kickoff_utc'],
            latest_quote_at_utc=q['captured_at_utc'],
            both_confirmed=all(s['status']=='CONFIRMED' for s in sides.values()), **sides))
    return dict(quoted_upcoming_games=len(rows), both_confirmed_games=sum(r['both_confirmed'] for r in rows),
        games=rows, scope='Upcoming eight days in existing odds archive; not full schedule coverage')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/qb_news')
    p.add_argument('--teams', nargs='+', choices=sorted(DOMAINS), default=sorted(DOMAINS))
    args = p.parse_args()
    now = datetime.now(timezone.utc)
    root = Path(args.out)
    folder = root / now.strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=8) as pool:
        collected = list(pool.map(lambda team: collect_one(team, folder), args.teams))
    sources = [s for s, _ in collected]
    news = [r for _, rows in collected for r in rows]
    (folder / 'sources.json').write_text(json.dumps(sources, indent=2))
    (folder / 'news.json').write_text(json.dumps(news, indent=2))
    report = coverage(read_rows(Path('data/auto_market/quotes.csv')),
        read_rows(Path('nfl_data_inputs/qb_starters.csv')), datetime.now(timezone.utc))
    report.update(captured_at_utc=now.isoformat(), source_snapshot=str(folder),
        sources_fetched=sum(s['status']=='FETCHED' for s in sources), sources_attempted=len(sources),
        recent_news_items=len(news), qb_headlines=sum(r['qb_headline'] for r in news),
        automatic_confirmations_created=0,
        blocker='No validated automatic confirmed-starter source is connected. News is evidence only.')
    (folder / 'coverage.json').write_text(json.dumps(report, indent=2))
    (root / 'latest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
