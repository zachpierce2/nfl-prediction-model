"""Usage: python qb_cutoff_replay.py depth_charts_2025.csv strength_games_snapshot.csv"""
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

CUTS = (1440,360,90)


def replay(charts,schedule):
    charts = charts[charts.pos_abb.eq('QB')].copy()
    charts['dt'] = pd.to_datetime(charts.dt,utc=True)
    games = schedule[schedule.season.eq(2025)].sort_values(['gameday','game_id'])
    past,rows = {},[]
    for g in games.itertuples():
        kickoff = pd.Timestamp(g.gameday+' '+g.gametime,tz='America/New_York').tz_convert('UTC')
        for side in ['home','away']:
            team = getattr(g,side+'_team')
            actual = getattr(g,side+'_qb_id')
            q = charts[charts.team.replace({'LAR':'LA'}).eq({'LAR':'LA'}.get(team,team))]
            for minutes in CUTS:
                cutoff = kickoff-pd.Timedelta(minutes=minutes)
                fresh = q[q.dt.between(cutoff-pd.Timedelta(hours=48),cutoff)]
                top = fresh[fresh.dt.eq(fresh.dt.max()) & fresh.pos_rank.eq(1)]
                named = top.gsis_id.iloc[0] if len(top) and top.gsis_id.notna().all() and top.gsis_id.nunique()==1 else None
                rows.append(dict(game=g.game_id,team=team,minutes=minutes,expected=named,actual=actual,
                                 previous=past.get(team),source_at=top.dt.max()))
            past[team]=actual
    df=pd.DataFrame(rows)
    for m,r in df.groupby('minutes',sort=False):
        valid=r.dropna(subset=['actual','expected'])
        changed=valid[valid.previous.notna() & valid.actual.ne(valid.previous)]
        flagged=valid[valid.previous.notna() & valid.expected.ne(valid.previous)]
        print(m,len(valid),sum(valid.expected==valid.actual),len(changed),sum(changed.expected==changed.actual),
              len(flagged),sum(flagged.expected==flagged.actual))
    return df


if __name__=='__main__':
    rows=replay(pd.read_csv(sys.argv[1],low_memory=False),pd.read_csv(sys.argv[2],low_memory=False))
    rows.to_csv('qb_cutoff_2025.csv',index=False)
