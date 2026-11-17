"""Reproducible error audit of frozen EPA development predictions. No live changes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import market_model as model

PREDICTIONS='strength_epa_reference_predictions.csv'
GAMES='strength_games_snapshot.csv'
AXES=['line_size','favorite_location','pick_type','season_phase','edge_size','rest_bucket','site']
LIMITS=[
 '2021-2025 seasons have already been examined; every subgroup is exploratory.',
 'Subgroups overlap and many comparisons are reported; intervals are descriptive, not multiplicity-adjusted proof.',
 'Historical spreads and features may be revised and lack verified pregame publication times.',
 'Largest realized errors are outcome diagnostics, not eligibility rules for future bets.',
 'Historical diagnostics never count as timestamped prospective performance.']

def enrich(p,games):
    if p.game_id.duplicated().any() or games.game_id.duplicated().any():
        raise ValueError('Duplicate game IDs')
    columns=['game_id','week','game_type','home_team','away_team','rest_diff','location','actual_margin','spread_line']
    d=p.merge(games[columns],on='game_id',how='left',validate='one_to_one',suffixes=('','_schedule'))
    if len(d)!=len(p) or d.week.isna().any():raise ValueError('Missing schedule match')
    for c in ['actual_margin','spread_line']:
        if not np.allclose(d[c],d[c+'_schedule'],rtol=0,atol=1e-10):
            raise ValueError('Frozen prediction disagrees with schedule '+c)
    if not np.allclose(d.predicted_margin,d.spread_line+d.adjustment,rtol=0,atol=1e-10):
        raise ValueError('Adjustment sign or prediction mismatch')
    d['signed_error']=d.actual_margin-d.predicted_margin
    d['absolute_error']=d.signed_error.abs()
    d['market_absolute_error']=(d.actual_margin-d.spread_line).abs()
    d['mae_gain']=d.market_absolute_error-d.absolute_error
    d['cover']=(d.actual_margin-d.spread_line)*np.sign(d.adjustment)
    d['active']=d.adjustment.abs().gt(1e-10)
    size=d.spread_line.abs()
    d['line_size']=np.select([size.le(3),size.le(7)],['0_to_3','over_3_to_7'],default='over_7')
    d['favorite_location']=np.select([d.spread_line.gt(0),d.spread_line.lt(0)],['home_favorite','away_favorite'],default='pickem')
    d['pick_type']=np.select([d.adjustment.mul(d.spread_line).gt(0),d.adjustment.mul(d.spread_line).lt(0)],['favorite','underdog'],default='pickem_or_no_lean')
    d['season_phase']=np.select([d.game_type.ne('REG'),d.week.le(4),d.week.le(12)],['postseason','weeks_1_4','weeks_5_12'],default='weeks_13_plus')
    e=d.adjustment.abs()
    d['edge_size']=np.select([e.lt(.1),e.lt(.25),e.lt(.5)],['under_0.10','0.10_to_0.25','0.25_to_0.50'],default='0.50_plus')
    d['rest_bucket']=np.select([d.rest_diff.gt(0),d.rest_diff.lt(0)],['home_more_rest','away_more_rest'],default='equal_rest')
    d['site']=np.where(d.location.eq('Home'),'home_site','neutral_or_other')
    return d

def interval(g,column,draws=2000):
    blocks=g.groupby(['season','week'])[column].agg(['sum','count'])
    rng=np.random.default_rng(7321)
    picks=rng.integers(0,len(blocks),size=(draws,len(blocks)))
    values=blocks['sum'].to_numpy()[picks].sum(axis=1)/blocks['count'].to_numpy()[picks].sum(axis=1)
    return [float(x) for x in np.quantile(values,[.025,.975])]

def metrics(g,bootstrap=False):
    w=int((g.active&g.cover.gt(0)).sum());l=int((g.active&g.cover.lt(0)).sum())
    result=dict(games=len(g),wins=w,losses=l,pushes=int((g.active&g.cover.eq(0)).sum()),
        no_pick=int((~g.active).sum()),win_pct=100*w/(w+l) if w+l else None,
        signed_error=float(g.signed_error.mean()),mae=float(g.absolute_error.mean()),
        market_mae=float(g.market_absolute_error.mean()),mae_gain=float(g.mae_gain.mean()),
        mean_absolute_adjustment=float(g.adjustment.abs().mean()))
    if bootstrap:
        result['signed_error_interval']=interval(g,'signed_error')
        result['mae_gain_interval']=interval(g,'mae_gain')
    return result

def audit(d):
    summary=[];yearly=[]
    for axis in AXES:
        for value,g in d.groupby(axis):
            summary.append(dict(axis=axis,bucket=str(value),**metrics(g,True)))
            for year,b in g.groupby('season'):
                yearly.append(dict(axis=axis,bucket=str(value),season=int(year),**metrics(b)))
    groups=pd.DataFrame(summary);years=pd.DataFrame(yearly)
    stability=[]
    for r in summary:
        y=years[(years.axis==r['axis'])&years.bucket.eq(r['bucket'])]
        direction=np.sign(r['signed_error'])
        discovery=d[(d.season.le(2023))&d[r['axis']].eq(r['bucket'])]
        later=d[(d.season.ge(2024))&d[r['axis']].eq(r['bucket'])]
        stability.append(dict(axis=r['axis'],bucket=r['bucket'],games=r['games'],
            same_direction_years=int((np.sign(y.signed_error)==direction).sum()),
            years=len(y),signed_error=r['signed_error'],mae_gain=r['mae_gain'],
            early_games=len(discovery),early_signed_error=float(discovery.signed_error.mean()) if len(discovery) else None,
            later_games=len(later),later_signed_error=float(later.signed_error.mean()) if len(later) else None))
    return groups,years,pd.DataFrame(stability)

def prospective(root):
    root=Path(root);forecasts=root/'epa_shadow/forecasts.csv'
    grades=root/'epa_shadow/graded_predictions.csv'
    def read(path):
        if not path.exists() or path.stat().st_size==0:return pd.DataFrame()
        try:return pd.read_csv(path)
        except pd.errors.EmptyDataError:return pd.DataFrame()
    f,g=read(forecasts),read(grades)
    if not f.empty:
        if f.game_id.duplicated().any():raise ValueError('Duplicate prospective game')
        for c in ['captured_at_utc','forecast_created_at_utc','kickoff_utc']:
            f[c]=pd.to_datetime(f[c],utc=True,errors='raise')
        if not ((f.captured_at_utc<=f.forecast_created_at_utc)&(f.forecast_created_at_utc<f.kickoff_utc)).all():
            raise ValueError('Invalid prospective timing')
    if not g.empty and (f.empty or not set(g.game_id)<=set(f.game_id)):
        raise ValueError('Prospective grades without forecasts')
    return dict(timestamped_epa_forecasts=len(f),graded_epa_games=len(g),
        status='NO_GRADED_PROSPECTIVE_EPA_SAMPLE' if g.empty else 'READ_EXISTING_PROSPECTIVE_ARCHIVE',
        note='No historical predictions or reconstructed 2026 forecasts are added to this count.')

def main():
    a=argparse.ArgumentParser();a.add_argument('--out',default='epa_error_audit_results');args=a.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(PREDICTIONS);p=p[p.method.eq('opponent_epa')&p.season.between(2021,2025)].copy()
    if len(p)!=1424:raise ValueError('Expected all 1,424 frozen evaluation games')
    d=enrich(p,model.read_games(GAMES));groups,years,stability=audit(d)
    for name,f in [('game_errors',d),('subgroups',groups),('subgroups_by_year',years),('stability',stability)]:
        f.to_csv(out/(name+'.csv'),index=False)
    d.nlargest(25,'absolute_error').to_csv(out/'largest_errors.csv',index=False)
    result=dict(status='EXPLORATORY_ERROR_AUDIT_NO_LIVE_CHANGE',overall=metrics(d,True),
        subgroups=groups.to_dict('records'),stability=stability.to_dict('records'),
        adjustment_quantiles={str(k):float(v) for k,v in d.adjustment.abs().quantile([0,.25,.5,.75,.9,1]).items()},
        prospective=prospective('data/auto_market'),limits=LIMITS,
        hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,PREDICTIONS,GAMES,model.__file__]})
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print('ERROR_AUDIT_RESULT_JSON='+json.dumps(result),flush=True)

if __name__=='__main__':main()
