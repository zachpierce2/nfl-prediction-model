"""Offline chronological audit. Never promotes a candidate into live forecasts."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import audit_reference_market_model as m

# Freeze these before inspecting this experiment's results.
WEIGHTS = (0.0, 0.10, 0.25, 0.50, 1.0)
CAP = 2.0

def metrics(g):
    error = g.actual_margin - g.predicted_margin
    edge = g.adjustment
    selected = edge.abs() > 1e-10
    cover = (g.actual_margin - g.spread_line) * np.sign(edge)
    w = int((selected & (cover > 0)).sum())
    l = int((selected & (cover < 0)).sum())
    p = int((selected & (cover == 0)).sum())
    return dict(games=len(g), mae=float(error.abs().mean()),
                rmse=float(np.sqrt((error**2).mean())),
                improvement_vs_market=float(((g.actual_margin-g.spread_line).abs()-error.abs()).mean()),
                wins=w, losses=l, pushes=p, win_pct=100*w/(w+l) if w+l else None)

def run(games, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(games, low_memory=False)
    f = m.read_games(games)
    played = f.dropna(subset=['actual_margin','spread_line']).copy()
    checks = {}
    checks['unique_ids'] = bool(raw.game_id.notna().all() and not raw.game_id.duplicated().any())
    checks['valid_locations'] = bool(raw.location.dropna().isin(['Home','Neutral']).all())
    checks['neutral_field_encoded_zero'] = bool(f.loc[f.location.eq('Neutral'),'home_field'].eq(0).all())
    checks['no_postgame_fields_in_estimator'] = not bool(set(m.INPUTS) & {'result','total','home_score','away_score','actual_margin'})
    checks['all_played_inputs_finite'] = bool(np.isfinite(played[m.INPUTS].to_numpy()).all())
    # Mutate same-day and future outcomes; earlier and same-day features must be invariant.
    with tempfile.TemporaryDirectory() as td:
        changed = raw.copy()
        changed.loc[pd.to_datetime(changed.gameday) >= pd.Timestamp('2024-09-05'), ['home_score','away_score']] = [99, 0]
        altered = Path(td)/'altered.csv'; changed.to_csv(altered,index=False)
        other = m.read_games(altered)
        keep = f.gameday <= pd.Timestamp('2024-09-05',tz='UTC')
        pd.testing.assert_frame_equal(f.loc[keep,m.INPUTS],other.loc[keep,m.INPUTS])
    checks['same_day_and_future_score_invariance'] = True
    class Extreme:
        def predict(self, x): return np.array([100.,-100.,0.])
    np.testing.assert_allclose(m.adjustment(Extreme(),pd.DataFrame({c:[0,0,0] for c in m.INPUTS})),[2,-2,0])
    checks['original_adjustment_cap'] = True
    checks['home_favorite_minus_2_5_maps_to_margin_plus_2_5'] = -(-2.5) == 2.5
    audit = {'checks': checks,
             'missing_values': raw.isna().sum().astype(int).to_dict(),
             'games_with_scores_but_missing_spread_2021_2025': int((f.season.between(2021,2025)&f.actual_margin.notna()&f.spread_line.isna()).sum()),
             'estimator_inputs': m.INPUTS,
             'limitations': ['Historical spreads and features lack verified as-of timestamps.',
                            'All 2021-2025 seasons have already been inspected; none is an untouched holdout.',
                            'Historical input revisions and executable odds cannot be audited from this snapshot.',
                            'Same-day postgame result/total fields can remain in archived schedules, but are excluded from estimator inputs.',
                            'Elo carries across seasons without preseason regression; histories cross season boundaries.',
                            'The existing rest feature measures calendar days, including long offseasons clipped to 21 days.',
                            'No EPA, injury news, or QB starter effects enter this version.']}
    predictions=[]; folds=[]
    for year in range(2020,2026):
        train=played[played.season<year]; test=played[played.season.eq(year)]
        assert train.season.max()<year and len(train)>=500
        fitted=m.fit(train)
        residual=fitted.predict(test[m.INPUTS])
        folds.append(dict(test_season=year,train_games=len(train),train_last_season=int(train.season.max()),test_games=len(test)))
        for weight in WEIGHTS:
            g=test[['game_id','season','gameday','home_team','away_team','actual_margin','spread_line']].copy()
            g['weight']=weight
            g['adjustment']=np.clip(weight*residual,-CAP,CAP)
            g['predicted_margin']=g.spread_line+g.adjustment
            predictions.append(g)
    all_p=pd.concat(predictions,ignore_index=True)
    rows=[]; choices=[]; adaptive=[]
    for year in range(2021,2026):
        # Select on strictly earlier annual out-of-sample predictions, never current season.
        validation=all_p[all_p.season<year]
        ranked=sorted((metrics(g)['mae'],float(w)) for w,g in validation.groupby('weight'))
        selected=ranked[0][1]
        choices.append(dict(test_season=year,validation_first_season=2020,
                            validation_last_season=year-1,chosen_weight=selected,
                            prior_validation_mae=ranked[0][0]))
        test=all_p[all_p.season.eq(year)]
        for weight,g in test.groupby('weight'):
            rows.append(dict(period=str(year),method=f'fixed_{weight:g}',**metrics(g)))
        chosen=test[test.weight.eq(selected)].copy();chosen['chosen_weight']=selected
        adaptive.append(chosen)
        rows.append(dict(period=str(year),method='prior_year_selection',**metrics(chosen)))
    adaptive=pd.concat(adaptive,ignore_index=True)
    five=all_p[all_p.season.between(2021,2025)]
    for weight,g in five.groupby('weight'):
        rows.append(dict(period='2021-2025',method=f'fixed_{weight:g}',**metrics(g)))
    rows.append(dict(period='2021-2025',method='prior_year_selection',**metrics(adaptive)))
    comparison=pd.DataFrame(rows)
    comparison.to_csv(out/'comparison.csv',index=False)
    all_p.to_csv(out/'candidate_predictions.csv',index=False)
    adaptive.to_csv(out/'chronologically_selected_predictions.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'selection_by_year.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'training_folds.csv',index=False)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    (out/'provenance.json').write_text(json.dumps({'model_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
        'games_sha256':hashlib.sha256(Path(games).read_bytes()).hexdigest(),
        'weights':WEIGHTS,'cap':CAP,'selection_metric':'Prior annual out-of-sample mean absolute margin error; ties favor lower weight',
        'status':'EXPLORATORY_ONLY_NO_LIVE_MODEL_CHANGE'},indent=2))
    print(comparison[comparison.period.eq('2021-2025')].to_string(index=False))
    print(pd.DataFrame(choices).to_string(index=False))
    assert all(checks.values()), checks
    return comparison

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--games',required=True);parser.add_argument('--out',default='audit_results')
    args=parser.parse_args();run(args.games,args.out)
