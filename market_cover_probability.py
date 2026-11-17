"""Fixed regularized cover-probability experiment; historical development data only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import brier_score_loss,log_loss
import market_model as base

EPA=['adj_off_epa_diff','adj_def_epa_diff','adj_off_success_diff','adj_def_success_diff']
C=0.01 # Fixed before evaluation. No grid or threshold search.

def score(g):
    residual=g.actual_margin-g.spread_line
    active=g.direction.ne(0)
    cover=residual*g.direction
    w=int((active&cover.gt(0)).sum());l=int((active&cover.lt(0)).sum())
    result=dict(games=len(g),wins=w,losses=l,pushes=int((active&residual.eq(0)).sum()),
        no_pick=int((~active).sum()),win_pct=100*w/(w+l) if w+l else None)
    binary=g[residual.ne(0)&g.home_cover_probability.notna()]
    result['probability_games']=len(binary)
    result['brier']=None;result['log_loss']=None;result['calibration_error_10_bins']=None
    if len(binary):
        y=(binary.actual_margin-binary.spread_line).gt(0).astype(int)
        prob=binary.home_cover_probability
        result['brier']=float(brier_score_loss(y,prob))
        result['log_loss']=float(log_loss(y,prob,labels=[0,1]))
        bins=np.minimum((prob.to_numpy()*10).astype(int),9);ece=0.
        for b in np.unique(bins):
            mask=bins==b;ece+=mask.mean()*abs(y.to_numpy()[mask].mean()-prob.to_numpy()[mask].mean())
        result['calibration_error_10_bins']=float(ece)
    return result

def run(games_path,features_path,out):
    f=base.read_games(games_path);f=f[f.season.between(2017,2025)].copy()
    e=pd.read_csv(features_path)
    if e.game_id.isna().any() or e.game_id.duplicated().any():
        raise ValueError('Unique nonnull EPA game_id required')
    f=f.merge(e[['game_id']+EPA],on='game_id',how='left',validate='one_to_one')
    if not np.isfinite(f[EPA].to_numpy()).all():raise ValueError('Missing EPA features')
    f=f.dropna(subset=['actual_margin','spread_line']);columns=base.INPUTS+EPA
    predictions=[];folds=[]
    for year in range(2020,2026):
        train=f[f.season<year];test=f[f.season.eq(year)]
        binary=train[(train.actual_margin-train.spread_line).ne(0)]
        if len(binary)<500 or test.empty:raise ValueError('Insufficient fold')
        if train.gameday.max() >= test.gameday.min():
            raise ValueError('Training dates overlap evaluation dates')
        y=(binary.actual_margin-binary.spread_line).gt(0).astype(int)
        logistic=base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),
            LogisticRegression(C=C,solver='lbfgs',max_iter=2000,tol=1e-8))
        logistic.fit(binary[columns],y)
        ridge=base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),Ridge(alpha=1000.))
        ridge.fit(train[columns],train.actual_margin-train.spread_line)
        adjustment=np.clip(.25*ridge.predict(test[columns]),-2.,2.)
        folds.append(dict(test_season=year,train_last_season=int(train.season.max()),
            train_last_date=str(train.gameday.max()),test_first_date=str(test.gameday.min()),
            train_games=len(train),classifier_training_nonpush_games=len(binary),test_games=len(test)))
        for name,prob,direction in [
            ('margin_core_epa',np.full(len(test),np.nan),np.where(abs(adjustment)>1e-10,np.sign(adjustment),0)),
            ('cover_logistic',logistic.predict_proba(test[columns])[:,1],None),
            ('constant_half',np.full(len(test),.5),np.zeros(len(test))),
            ('prior_home_cover_rate',np.full(len(test),float(y.mean())),np.zeros(len(test)))]:
            g=test[['game_id','season','gameday','home_team','away_team','actual_margin','spread_line']].copy()
            g['method']=name;g['home_cover_probability']=prob
            # A fixed 0.5 tie has no lean, matching the margin baseline's zero adjustment.
            g['direction']=np.sign(prob-.5) if direction is None else direction
            if name=='margin_core_epa':g['predicted_margin']=g.spread_line+adjustment
            else:g['predicted_margin']=np.nan
            predictions.append(g)
    p=pd.concat(predictions,ignore_index=True)
    # Every method must cover the same games, with finite classifier probabilities.
    expected=set(f[f.season.between(2020,2025)].game_id)
    for name,g in p.groupby('method'):
        if g.game_id.duplicated().any() or set(g.game_id)!=expected:
            raise ValueError(f'Invalid game coverage: {name}')
        if name!='margin_core_epa' and not (np.isfinite(g.home_cover_probability).all() and g.home_cover_probability.between(0,1).all()):
            raise ValueError(f'Invalid probabilities: {name}')
    summary=[]
    for (year,name),g in p[p.season.ge(2021)].groupby(['season','method']):summary.append(dict(period=str(year),method=name,**score(g)))
    for name,g in p[p.season.ge(2021)].groupby('method'):summary.append(dict(period='2021-2025',method=name,**score(g)))
    summary=pd.DataFrame(summary);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p.to_csv(out/'predictions.csv',index=False);summary.to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'training_folds.csv',index=False)
    g=p[p.method.eq('cover_logistic')&p.season.ge(2021)].copy()
    g=g[(g.actual_margin-g.spread_line).ne(0)]
    g['observed_home_cover']=(g.actual_margin-g.spread_line).gt(0).astype(float)
    g['bin']=pd.cut(g.home_cover_probability,[0,.4,.45,.5,.55,.6,1],include_lowest=True)
    calibration=g.groupby('bin',observed=False).agg(games=('game_id','size'),mean_probability=('home_cover_probability','mean'),observed_home_cover_rate=('observed_home_cover','mean')).reset_index()
    calibration.to_csv(out/'calibration.csv',index=False)
    margin=p[p.method.eq('margin_core_epa')&p.season.ge(2021)].set_index('game_id')
    classification=p[p.method.eq('cover_logistic')&p.season.ge(2021)].set_index('game_id')
    residual=margin.actual_margin-margin.spread_line
    a=classification.direction;b=margin.direction
    nonpush=residual.ne(0)&a.ne(0)&b.ne(0)
    paired=dict(disagreements=int((nonpush&a.ne(b)).sum()),
        classifier_wins_margin_loses=int((nonpush&(residual*a).gt(0)&(residual*b).lt(0)).sum()),
        margin_wins_classifier_loses=int((nonpush&(residual*b).gt(0)&(residual*a).lt(0)).sum()))
    meta=dict(status='DEVELOPMENT_ONLY_NO_LIVE_CHANGE',C=C,solver='lbfgs',features=columns,
        training='Prior seasons only; pushes excluded only from classifier training and probability scoring',
        probability='P(home covers | no push); not an unconditional cover or push model',
        pick_threshold=.5,ties='No lean',paired=paired,
        hashes={str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [games_path,features_path,__file__,base.__file__]},
        limitations=['All evaluated seasons previously examined; not an untouched holdout.',
        'No hyperparameter search, probability calibration fitting or automatic promotion.',
        'Margin model has no validated cover-probability output; probability benchmarks are 0.5 and prior training home-cover rate.',
        'Historical EPA/spreads may be revised and lack verified pregame publication timestamps.',
        'Calibration error is descriptive and bin-dependent; probability scores exclude pushes.',
        'No executable-price profit or betting EV established.'])
    (out/'provenance.json').write_text(json.dumps(meta,indent=2))
    pooled=summary[summary.period.eq('2021-2025')]
    report=['# Direct cover-probability backtest', '',
        'Annual prior-season-only training; all games included. Probabilities condition on no push.',
        '', '```', pooled.to_string(index=False), '```', '',
        'Development data only. Historical spreads/EPA lack verified publication timestamps.',
        'No live model change or verified betting profitability.']
    (out/'summary.md').write_text('\n'.join(report)+'\n')
    print(summary.to_string(index=False));print(calibration.to_string(index=False));print(json.dumps(paired))
    return p,summary

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--games',default='strength_games_snapshot.csv');a.add_argument('--features',default='strength_epa_features.csv');a.add_argument('--out',default='cover_probability_results')
    args=a.parse_args();run(args.games,args.features,args.out)
