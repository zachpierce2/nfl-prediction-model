"""Single audit-motivated wide-spread feature; development test, no live change."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import market_model as base
from market_epa_experiment import ADJ
import epa_error_audit as audit

FEATURE='wide_spread_game'
SPEC=dict(feature='absolute market spread strictly greater than seven points',
    threshold_points=7,ridge_alpha=1000,shrink=.25,cap=2,
    decision='One feature chosen after inspecting the development error audit; no threshold or hyperparameter search.')

def add_feature(frame):
    f=frame.copy()
    f[FEATURE]=f.spread_line.abs().gt(7).astype(float)
    return f

def fit(train,columns):
    m=base.make_pipeline(base.SimpleImputer(strategy='median'),base.StandardScaler(),Ridge(alpha=1000.))
    m.fit(train[columns],train.actual_margin-train.spread_line)
    return m

def evaluate(frame):
    f=add_feature(frame).dropna(subset=['actual_margin','spread_line'])
    rows=[];folds=[]
    for year in range(2020,2026):
        train=f[f.season.lt(year)];test=f[f.season.eq(year)]
        if len(train)<500 or test.empty:raise ValueError('Incomplete annual fold')
        for name,extra in [('epa_baseline',[]),('epa_wide_spread',[FEATURE])]:
            columns=base.INPUTS+ADJ+extra
            m=fit(train,columns)
            adj=np.clip(.25*m.predict(test[columns]),-2.,2.)
            p=test[['game_id','season','actual_margin','spread_line']].copy()
            p['adjustment']=adj;p['predicted_margin']=p.spread_line+p.adjustment;p['method']=name
            rows.append(p)
            coefficient=float(m.named_steps['ridge'].coef_[-1]) if extra else None
            folds.append(dict(method=name,test_season=year,train_last_season=int(train.season.max()),
                train_games=len(train),test_games=len(test),wide_feature_standardized_coefficient=coefficient))
    return pd.concat(rows,ignore_index=True),pd.DataFrame(folds)

def paired(predictions,games):
    a=predictions[predictions.method.eq('epa_wide_spread')&predictions.season.ge(2021)]
    b=predictions[predictions.method.eq('epa_baseline')&predictions.season.ge(2021)]
    p=a.merge(b,on='game_id',validate='one_to_one',suffixes=('_new','_base'))
    p=p.merge(games[['game_id','week']],on='game_id',validate='one_to_one')
    p['season']=p.season_new
    if not np.allclose(p.spread_line_new,p.spread_line_base,atol=0,rtol=0):raise ValueError('Different paired lines')
    residual=p.actual_margin_new-p.spread_line_new
    a_w=residual*np.sign(p.adjustment_new)>0;b_w=residual*np.sign(p.adjustment_base)>0
    eligible=residual.ne(0)&p.adjustment_new.abs().gt(1e-10)&p.adjustment_base.abs().gt(1e-10)
    p['mae_gain']=(p.actual_margin_base-p.predicted_margin_base).abs()-(p.actual_margin_new-p.predicted_margin_new).abs()
    p['win_gain']=(a_w.astype(float)-b_w.astype(float)).where(eligible,np.nan)
    return dict(games=len(p),disagreements=int((eligible&(np.sign(p.adjustment_new)!=np.sign(p.adjustment_base))).sum()),
        candidate_only_wins=int((eligible&a_w&~b_w).sum()),baseline_only_wins=int((eligible&b_w&~a_w).sum()),
        mae_gain=float(p.mae_gain.mean()),mae_gain_interval=audit.interval(p,'mae_gain'),
        ats_win_rate_gain_percentage_points=float(100*p.win_gain.mean()),
        ats_win_rate_gain_interval_percentage_points=[100*x for x in audit.interval(p[p.win_gain.notna()],'win_gain')])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='epa_error_audit_results/targeted');args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    games=base.read_games('strength_games_snapshot.csv')
    games=games[games.season.between(2017,2025)].copy()
    features=pd.read_csv('strength_epa_features.csv')
    f=games.merge(features[['game_id']+ADJ],on='game_id',validate='one_to_one')
    if not np.isfinite(f[ADJ].to_numpy()).all():raise ValueError('Missing EPA features')
    p,folds=evaluate(f)
    reference=pd.read_csv('strength_epa_reference_predictions.csv')
    b=p[p.method.eq('epa_baseline')].merge(reference[reference.method.eq('opponent_epa')],on='game_id',validate='one_to_one',suffixes=('_new','_ref'))
    if len(b)!=1693 or not np.allclose(b.predicted_margin_new,b.predicted_margin_ref,atol=1e-10,rtol=0):
        raise ValueError('Frozen baseline not reproduced')
    summaries=[]
    for name,g in p[p.season.ge(2021)].groupby('method'):
        d=audit.enrich(g,games)
        for period,part in [('2021-2025',d)]+[(str(y),x) for y,x in d.groupby('season')]:
            summaries.append(dict(method=name,period=period,scope='all_games',**audit.metrics(part)))
        wide=d[d.spread_line.abs().gt(7)]
        summaries.append(dict(method=name,period='2021-2025',scope='spread_over_7',**audit.metrics(wide)))
    summary=pd.DataFrame(summaries);comparison=paired(p,games)
    p.to_csv(out/'predictions.csv',index=False);summary.to_csv(out/'comparison.csv',index=False)
    folds.to_csv(out/'training_folds.csv',index=False)
    result=dict(status='AUDIT_MOTIVATED_DEVELOPMENT_ONLY',specification=SPEC,
        results=summaries,paired=comparison,baseline_reproduced_games=len(b),
        limitations=audit.LIMITS+['Feature choice used the reported audit seasons; annual folds do not turn this into independent validation.',
            'The learned coefficient is not the observed 2.27-point audit bias. No fixed home bonus is inserted.',
            'Every game is evaluated; the wide-spread indicator is not a betting filter.',
            'No live promotion or prospective forecasts are created by this experiment.'],
        hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
            __file__,audit.__file__,base.__file__,'strength_games_snapshot.csv','strength_epa_features.csv','strength_epa_reference_predictions.csv']})
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print('TARGETED_RESULT_JSON='+json.dumps(result),flush=True)

if __name__=='__main__':main()
