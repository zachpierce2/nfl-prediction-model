#!/usr/bin/env python3
"""Exploratory V3 EPA comparison. Requires the original model_v3.py alongside it."""
import argparse
import hashlib
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import model_v3 as baseline

PBP_URL = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet'
ALIASES = {'LA':'LAR', 'STL':'LAR', 'SD':'LAC', 'OAK':'LV'}
METRICS = ['off_epa','def_epa','pass_epa','def_pass_epa','rush_epa','def_rush_epa','success','def_success','adj_off','adj_def']
EXTRA = [f'{m}_diff' for m in METRICS] + ['qb_epa_diff','qb_success_diff','qb_dropbacks_diff','team_history_diff','home_epa_missing','away_epa_missing','home_qb_missing','away_qb_missing']

def canonical(t):
    return ALIASES.get(t, t)

def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, temp)
            temp.replace(path)
            return
        except Exception:
            temp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)

def aggregate_pbp(frame):
    required = {'game_id','posteam','defteam','epa','play_type','qb_dropback','rush_attempt','qb_kneel','qb_spike','passer_player_id','passer_id','qb_epa'}
    if required - set(frame):
        raise ValueError(f'Missing PBP columns: {sorted(required-set(frame))}')
    p = frame.copy()
    p['epa'] = pd.to_numeric(p['epa'], errors='coerce')
    p = p[p['play_type'].isin(['pass','run']) & p['epa'].notna()
          & p['posteam'].notna() & p['defteam'].notna()
          & p['qb_kneel'].fillna(0).ne(1) & p['qb_spike'].fillna(0).ne(1)].copy()
    p['posteam'] = p['posteam'].map(canonical)
    p['defteam'] = p['defteam'].map(canonical)
    p['success'] = (p['epa'] > 0).astype(float)
    p['pass_epa'] = p['epa'].where(p['qb_dropback'].eq(1))
    p['rush_epa'] = p['epa'].where(p['rush_attempt'].eq(1) & p['qb_dropback'].ne(1))
    teams = p.groupby(['game_id','posteam','defteam'], as_index=False).agg(
        off_epa=('epa','mean'), pass_epa=('pass_epa','mean'),
        rush_epa=('rush_epa','mean'), success=('success','mean'), plays=('epa','size'))
    p['passer_player_id'] = p['passer_id'].fillna(p['passer_player_id'])
    p['qb_epa'] = pd.to_numeric(p['qb_epa'],errors='coerce').fillna(p['epa'])
    q = p[p['qb_dropback'].eq(1) & p['passer_player_id'].notna()]
    qbs = q.groupby(['game_id','posteam','passer_player_id'], as_index=False).agg(
        epa_sum=('qb_epa','sum'), success_sum=('success','sum'), dropbacks=('epa','size'))
    return teams, qbs

def load_pbp(seasons, cache):
    teams, qbs, sources = [], [], []
    wanted = ['game_id','posteam','defteam','epa','play_type','qb_dropback','rush_attempt','qb_kneel','qb_spike','passer_player_id','passer_id','qb_epa']
    for season in seasons:
        path = cache / f'play_by_play_{season}.parquet'
        url = PBP_URL.format(season=season)
        print(f'Loading PBP {season}', flush=True)
        download(url, path)
        frame = pd.read_parquet(path, columns=wanted)
        t, q = aggregate_pbp(frame)
        teams.append(t); qbs.append(q)
        sources.append({'season':int(season),'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.concat(teams, ignore_index=True), pd.concat(qbs, ignore_index=True), sources

def epa_features(games, team_stats, qb_stats):
    # All feature rows for a date are constructed before any updates for that date.
    stats = {(r.game_id,r.posteam):r._asdict() for r in team_stats.itertuples(index=False)}
    qb_games = {gid:g for gid,g in qb_stats.groupby('game_id')}
    histories, qb_history, last_qb = defaultdict(list), defaultdict(list), {}
    rows = []
    def profile(team):
        hist = histories[team][-8:]
        values = {}
        for metric in METRICS:
            samples = [r.get(metric, np.nan) for r in hist]
            valid = np.isfinite(samples)
            weights = .85 ** np.arange(len(samples)-1, -1, -1)
            values[metric] = float(np.average(np.array(samples)[valid], weights=weights[valid])) if valid.any() else np.nan
        qid = last_qb.get(team)
        qhist = qb_history[qid][-8:] if qid else []
        n = sum(x['dropbacks'] for x in qhist)
        # Shrink to zero using 100 prior dropbacks to limit small-sample noise.
        values.update(qb_epa=sum(x['epa_sum'] for x in qhist)/(n+100),
                      qb_success=(sum(x['success_sum'] for x in qhist)+50)/(n+100),
                      qb_dropbacks=min(n,400), team_history=len(hist),
                      epa_missing=float(not hist), qb_missing=float(not qhist))
        return values
    for day, group in games.sort_values(['gameday','game_id']).groupby('gameday', sort=True):
        profiles = {}
        for g in group.itertuples(index=False):
            h,a = canonical(g.home_team),canonical(g.away_team)
            hp,ap = profile(h),profile(a)
            profiles[g.game_id] = (hp,ap)
            r = {'game_id':g.game_id}
            for key in METRICS + ['qb_epa','qb_success','qb_dropbacks','team_history']:
                r[key+'_diff'] = hp[key]-ap[key]
            for side,p in [('home',hp),('away',ap)]:
                r[side+'_epa_missing'] = p['epa_missing']
                r[side+'_qb_missing'] = p['qb_missing']
            rows.append(r)
        for g in group.itertuples(index=False):
            if pd.isna(g.home_score) or pd.isna(g.away_score):
                continue
            h,a = canonical(g.home_team),canonical(g.away_team)
            hp,ap = profiles[g.game_id]
            for team,opp,op in [(h,a,ap),(a,h,hp)]:
                own,other = stats.get((g.game_id,team)),stats.get((g.game_id,opp))
                if own is None or other is None:
                    raise ValueError(f'Missing PBP coverage for completed game {g.game_id}: {team}')
                r = {k:own[k] for k in ['off_epa','pass_epa','rush_epa','success']}
                r.update(def_epa=other['off_epa'],def_pass_epa=other['pass_epa'],
                         def_rush_epa=other['rush_epa'],def_success=other['success'])
                # Adjustment uses opponent's profile BEFORE this historical game.
                # Positive adj_off is good; negative adj_def is good.
                od = op['def_epa'] if np.isfinite(op['def_epa']) else 0.
                oo = op['off_epa'] if np.isfinite(op['off_epa']) else 0.
                r.update(adj_off=r['off_epa']-od,adj_def=r['def_epa']-oo)
                histories[team].append(r)
            q = qb_games.get(g.game_id)
            if q is not None:
                for team, sub in q.groupby('posteam'):
                    dominant = sub.sort_values(['dropbacks','passer_player_id'],ascending=[False,True]).iloc[0]
                    last_qb[team] = dominant['passer_player_id']
                for r in q.to_dict('records'):
                    qb_history[r['passer_player_id']].append(r)
    return pd.DataFrame(rows)

def spread_report(predictions):
    rows = []
    for name,group in predictions.groupby('model'):
        for edge in [0,1,2,3,4,5,6,7]:
            valid = group.dropna(subset=['spread_line']).copy()
            valid['edge'] = valid.predicted_margin-valid.spread_line
            bets = valid[(valid.edge.abs()>=edge) & (valid.edge!=0)].copy()
            bets['cover'] = (bets.actual_margin-bets.spread_line)*np.sign(bets.edge)
            for period in ['all'] + sorted(group.season.unique().tolist()):
                b = bets if period == 'all' else bets[bets.season==period]
                wins,losses,pushes = int((b.cover>0).sum()),int((b.cover<0).sum()),int((b.cover==0).sum())
                n = len(b); profit = wins*100-losses*110
                years = group.season.nunique() if period=='all' else 1
                rows.append(dict(model=name,minimum_edge=edge,period=period,bets=n,
                                 bets_per_season=round(n/years,1),wins=wins,losses=losses,pushes=pushes,
                                 win_pct=round(100*wins/(wins+losses),2) if wins+losses else None,
                                 hypothetical_profit=profit,roi_pct=round(100*profit/(110*n),2) if n else None))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games',default='data/games.csv')
    ap.add_argument('--cache',default='data/epa_cache')
    ap.add_argument('--out',default='nfl_v3_epa_results')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    games = baseline.load(args.games)
    teams,qbs,sources = load_pbp(sorted(games.season.astype(int).unique()),Path(args.cache))
    extra = epa_features(games,teams,qbs)
    features = baseline.make_features(games).merge(extra,on='game_id',validate='one_to_one')
    features = features[features.actual_margin.notna()].copy()
    # Actual historical starter IDs are deliberately withheld from BOTH arms.
    # They may not have been announced at the intended betting time.
    basic = [f for f in baseline.FEATURES if f not in ['qb_margin_diff','qb_experience_diff']]
    features.to_csv(out/'pregame_features.csv',index=False)
    all_predictions, scores = [], []
    for season in range(2020,2026):
        train = features[features.season<season]
        test = features[features.season==season]
        if len(train)<150 or test.empty:
            continue
        for arm,columns in [('baseline',basic),('epa',basic+EXTRA)]:
            models = {
                'ridge':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100)),
                'boosted_trees':make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,l2_regularization=10,learning_rate=.04,random_state=42))}
            for kind,model in models.items():
                name = arm+'_'+kind
                model.fit(train[columns],train.actual_margin)
                predicted = model.predict(test[columns])
                mae = float(np.mean(np.abs(predicted-test.actual_margin)))
                scores.append(dict(season=season,model=name,games=len(test),mae=mae))
                cols = ['game_id','season','week','home_team','away_team','actual_margin']
                p = test[cols].copy()
                p['spread_line'] = pd.to_numeric(test['spread_line'],errors='coerce') if 'spread_line' in test else np.nan
                p['predicted_margin'] = predicted; p['model'] = name
                all_predictions.append(p)
                print(f'{season} {name}: MAE {mae:.3f}',flush=True)
    if not all_predictions:
        raise ValueError('No evaluation games found')
    predictions = pd.concat(all_predictions,ignore_index=True)
    predictions.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'by_season.csv',index=False)
    spread_report(predictions).to_csv(out/'spread_backtest.csv',index=False)
    overall = {name:float(np.mean(abs(g.predicted_margin-g.actual_margin))) for name,g in predictions.groupby('model')}
    summary = {'model_mae':overall,'method':'Fixed models, train prior seasons only, evaluate 2020-2025',
               'qb_method':'Last observed team QB from an earlier game; never use actual current-game starter. Prior 8 games, shrink EPA by 100 dropbacks.',
               'adjustment':'Simple opponent adjustment using opponent profile before each historical game; not a jointly fitted strength rating.',
               'history':'Team previous 8 games, recency decay .85 per game; history carries across seasons.',
               'baseline_change':'Original QB-margin inputs removed in both arms to avoid relying on retrospectively recorded starter IDs. Recomputed baseline differs from original V3.',
               'limitations':['All 2020-2025 seasons are exploratory; their outcomes have already been inspected.',
                              'Historical EPA values are current revised estimates, not archived point-in-time data; upstream EPA models may use later information.',
                              'Historical spreads have no verified quote timestamps or actual prices. Profit assumes -110.',
                              'Last observed QB may be wrong after injury, trade or offseason. No injury/news feed.',
                              'No injury or weather features; no guarantee of 57% or 100 bets per season.'],
               'missing_spread_prediction_rows':int(predictions.spread_line.isna().sum()),
               'features':{'baseline':basic,'epa':basic+EXTRA},'sources':sources,
               'schedule_sha256':hashlib.sha256(Path(args.games).read_bytes()).hexdigest(),
               'baseline_code_sha256':hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(overall,indent=2))

if __name__ == '__main__':
    main()
