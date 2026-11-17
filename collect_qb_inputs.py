"""Workflow entry point: collect evidence and provide an as-of confirmation input."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import qb_news
import qb_announcements


def main():
    root=Path('qb_announcement_evidence')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    snapshot, sources, _ = qb_news.collect(root/'feeds', sorted(qb_news.DOMAINS))
    path, report = qb_announcements.run(snapshot,root/'articles',Path('nfl_data_inputs/qb_starters.csv'))
    report.update(club_feeds_attempted=len(sources),club_feeds_fetched=sum(s['status']=='FETCHED' for s in sources))
    Path('data/auto_market/qb_announcements/latest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'],'a') as f:
            f.write('QB_CONFIRMATIONS_PATH='+str(path.resolve())+'\n')
    return path, report


if __name__=='__main__':
    main()
