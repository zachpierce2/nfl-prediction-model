"""Reuse confirmed starter inputs in a separate market/EPA QB research candidate."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import auto_market as auto
import market_model as base
from market_epa_experiment import ADJ

QB = ['qb_margin_diff', 'qb_experience_diff']
VERSION = 'market_epa_confirmed_qb_margin_v1'
ROOT = Path('data/auto_market/qb_shadow')
CONFIRMATIONS = Path(os.environ.get('QB_CONFIRMATIONS_PATH','nfl_data_inputs/qb_starters.csv'))
FIELDS = auto.FORECAST_FIELDS + ['model_version','core_forecast_sha256','qb_inputs_sha256',
    'home_confirmed_qb_id','away_confirmed_qb_id'] + QB


def confirmed_ids(rows, game_id, team, cutoff, kickoff):
    """Latest observed report wins; questionable/unknown reports revoke confirmation."""
    required = {'game_id','team','qb_id','status','source_url','published_at_utc','observed_at_utc'}
    if required-set(rows):
        raise ValueError('QB confirmations have missing columns')
    cutoff, kickoff = auto.utc(str(cutoff)), auto.utc(str(kickoff))
    eligible = []
    for r in rows.to_dict('records'):
        if str(r['game_id']) != str(game_id) or str(r['team']) != str(team):
            continue
        published = auto.utc(str(r['published_at_utc']))
        observed = auto.utc(str(r['observed_at_utc']))
        if published > observed:
            raise ValueError('QB publication follows observation')
        if published <= observed <= cutoff < kickoff:
            eligible.append((observed, published, r))
    if not eligible:
        return None
    latest = max((o,p) for o,p,r in eligible)
    reports = [r for o,p,r in eligible if (o,p)==latest]
    identities = {(str(r['qb_id']).strip(), str(r['status']).strip().lower()) for r in reports}
    if len(identities) != 1:
        raise ValueError('Conflicting latest QB reports')
    r = reports[0]
    if str(r['status']).strip().lower() != 'confirmed':
        return None
    qid = str(r['qb_id']).strip()
    if not qid or qid.lower() == 'nan' or not str(r['source_url']).startswith(('https://','http://')):
        raise ValueError('Confirmed QB needs an ID and source URL')
    return qid


def features_for_ids(history, day, home_qb, away_qb):
    """Existing V3 proxy: prior eight starts' margins, not individual QB EPA."""
    prior = history[(history.gameday < day) & history.actual_margin.notna()].sort_values(['gameday','game_id'])
    def rating(qid):
        margins = []
        if qid is None or pd.isna(qid):
            return 0., 0.
        for r in prior.to_dict('records'):
            if str(r.get('home_qb_id')) == str(qid):
                margins.append(float(r['actual_margin']))
            elif str(r.get('away_qb_id')) == str(qid):
                margins.append(-float(r['actual_margin']))
        return (float(np.mean(margins[-8:])) if margins else 0., float(min(len(margins),16)))
    hm, hn = rating(home_qb)
    am, an = rating(away_qb)
    return dict(qb_margin_diff=hm-am, qb_experience_diff=hn-an)


def fit(train, columns):
    estimator = base.make_pipeline(base.SimpleImputer(strategy='median'),
        base.StandardScaler(), Ridge(alpha=1000.))
    estimator.fit(train[columns], train.actual_margin-train.spread_line)
    return estimator


def load_frame(games, features):
    frame = base.read_games(games)
    if not {'home_qb_id','away_qb_id'} <= set(frame):
        raise ValueError('Historical QB IDs unavailable')
    frame = frame.merge(pd.read_csv(features)[['game_id']+ADJ],
        on='game_id', how='left', validate='one_to_one')
    scored = frame.actual_margin.notna() & frame.season.le(2025)
    if not np.isfinite(frame.loc[scored, ADJ].to_numpy()).all():
        raise ValueError('Incomplete historical EPA features')
    return frame


def backtest(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    frame = load_frame('strength_games_snapshot.csv','strength_epa_features.csv')
    frame = frame[frame.season.between(2017,2025)].dropna(subset=['actual_margin','spread_line'])
    predictions = []; folds = []
    for year in range(2020,2026):
        train, test = frame[frame.season.lt(year)], frame[frame.season.eq(year)]
        if len(train)<500 or test.empty:
            raise ValueError('Insufficient fold')
        folds.append(dict(test_season=year,train_last_season=int(train.season.max()),train_games=len(train),test_games=len(test)))
        for name, columns in [('epa_baseline',base.INPUTS+ADJ),
                              ('epa_qb_retrospective',base.INPUTS+ADJ+QB)]:
            adj = np.clip(.25*fit(train,columns).predict(test[columns]),-2.,2.)
            p = test[['game_id','season','actual_margin','spread_line']].copy()
            p['method']=name; p['adjustment']=adj
            p['predicted_margin']=p.spread_line+adj
            predictions.append(p)
    p = pd.concat(predictions,ignore_index=True)
    rows=[]
    for name,g in p[p.season.ge(2021)].groupby('method'):
        rows.extend(dict(method=name,**r) for r in base.summarize(g))
    p.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'training_folds.csv',index=False)
    report = dict(status='RETROSPECTIVE_STARTER_DIAGNOSTIC_ONLY',features=base.INPUTS+ADJ+QB,
        results=rows,confirmation_rows=len(pd.read_csv(CONFIRMATIONS)),
        limitations=['Historical starter IDs lack pregame publication timestamps.',
        'QB margin is a team-result proxy, not individual quarterback efficiency.',
        'Historical data and seasons are previously examined development evidence.',
        'No validated win-rate improvement, automatic promotion, or profit established.'],
        hashes={str(f):auto.digest(Path(f).read_bytes()) for f in [
            __file__,'market_model.py','strength_games_snapshot.csv','strength_epa_features.csv',str(CONFIRMATIONS)]})
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report


def record(schedule_path, core_rows, folder):
    """Called only with fresh core forecasts; never reconstruct old predictions."""
    ROOT.mkdir(parents=True,exist_ok=True)
    if not core_rows:
        return dict(status='NO_NEW_CORE_FORECASTS',new_qb_forecasts=0)
    reports = pd.read_csv(CONFIRMATIONS,keep_default_na=False,dtype=str)
    (folder/'qb_confirmations_asof.csv').write_bytes(CONFIRMATIONS.read_bytes())
    capture = datetime.now(timezone.utc)
    eligible=[]; missing=[]
    for core in core_rows:
        ids = [confirmed_ids(reports,core['game_id'],core[side+'_team'],
            auto.stamp(capture),core['kickoff_utc']) for side in ['home','away']]
        if None in ids:
            missing.append(core['game_id'])
        else:
            eligible.append((core,ids))
    if not eligible:
        return dict(status='MISSING_CONFIRMED_STARTERS',new_qb_forecasts=0,
                    missing_confirmation_games=missing,confirmation_rows=len(reports))
    frame = load_frame(schedule_path,folder/'epa_features_asof.csv.gz')
    rows=[]
    from epa_shadow import core_hash
    input_hash=auto.digest((folder/'qb_confirmations_asof.csv').read_bytes()+
        (folder/'epa_features_asof.csv.gz').read_bytes()+Path(schedule_path).read_bytes())
    for core,ids in eligible:
        test=frame[frame.game_id.eq(core['game_id'])].copy()
        if len(test)!=1 or test.actual_margin.notna().any():
            raise ValueError('Invalid QB forecast schedule')
        day=test.iloc[0].gameday
        values=features_for_ids(frame,day,*ids)
        for c,v in values.items(): test[c]=v
        test['spread_line']=-float(core['home_line'])
        train=frame[frame.season.lt(test.iloc[0].season)].dropna(subset=['actual_margin','spread_line'])
        columns=base.INPUTS+ADJ+QB
        if len(train)<500 or not np.isfinite(train[ADJ+QB].to_numpy()).all() or not np.isfinite(test[ADJ+QB].to_numpy()).all():
            raise ValueError('Incomplete QB candidate inputs')
        adj=float(np.clip(.25*fit(train,columns).predict(test[columns])[0],-2.,2.))
        row=dict(core)
        margin=-float(core['home_line'])+adj
        row.update(adjustment=adj,predicted_home_margin=margin,home_model_spread=-margin,
            model_version=VERSION,core_forecast_sha256=core_hash(core),qb_inputs_sha256=input_hash,
            home_confirmed_qb_id=ids[0],away_confirmed_qb_id=ids[1],
            model_sha256=auto.digest(Path(__file__).read_bytes()),
            pipeline_sha256=auto.digest(Path('market_model.py').read_bytes()),**values)
        rows.append(row)
    finished=datetime.now(timezone.utc)
    if any(not auto.utc(r['forecast_created_at_utc'])<=capture<=finished<auto.utc(r['kickoff_utc']) for r in rows):
        raise ValueError('QB forecast crossed kickoff')
    for r in rows:r['forecast_created_at_utc']=auto.stamp(finished)
    saved=auto.archive_rows(ROOT/'forecasts.csv',FIELDS,rows,'game_id')
    return dict(status='QB_SHADOW_SAVED',new_qb_forecasts=len(saved),
                missing_confirmation_games=missing,confirmation_rows=len(reports))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='qb_market_results')
    args=parser.parse_args()
    backtest(args.out)
