"""Research-only discrete margin probabilities and same-snapshot sportsbook pricing.

Never fits on the evaluation season. Historical sources are revised development
snapshots, not proof of executable profitability or calibrated probabilities.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import auto_market as auto

def markdown_table(frame):
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines += ['| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines)


VERSION = 'prior_season_empirical_integer_margin_v1'
SOURCE = Path('strength_epa_reference_predictions.csv')
ROOT = Path('data/auto_market/pricing_shadow')
MIN_HISTORY = 200


def source_frame(path=SOURCE):
    p = pd.read_csv(path)
    required = {'game_id', 'season', 'gameday', 'actual_margin', 'predicted_margin', 'spread_line', 'method'}
    if required - set(p):
        raise ValueError('Missing probability-source columns')
    p = p[p.method.isin(['opponent_epa', 'market'])].copy()
    if p.empty or p.duplicated(['game_id', 'method']).any():
        raise ValueError('Missing or duplicate probability history')
    for c in ['season', 'actual_margin', 'predicted_margin', 'spread_line']:
        p[c] = pd.to_numeric(p[c], errors='raise')
    if not np.isfinite(p[['season', 'actual_margin', 'predicted_margin', 'spread_line']]).all().all():
        raise ValueError('Nonfinite probability history')
    if (p.actual_margin % 1).ne(0).any() or (p.season % 1).ne(0).any():
        raise ValueError('Scores and seasons must be integers')
    p['gameday'] = pd.to_datetime(p.gameday, utc=True, errors='raise')
    if p.gameday.isna().any():
        raise ValueError('Missing history dates')
    a = p[p.method.eq('opponent_epa')].set_index('game_id').sort_index()
    b = p[p.method.eq('market')].set_index('game_id').sort_index()
    if not a[['season', 'gameday', 'actual_margin', 'spread_line']].equals(b[['season', 'gameday', 'actual_margin', 'spread_line']]):
        raise ValueError('Probability controls must have identical game coverage and outcomes')
    return p


def residual_history(p, season, cutoff, method):
    h = p[p.method.eq(method) & p.season.lt(season) & p.gameday.lt(pd.Timestamp(cutoff))]
    if len(h) < MIN_HISTORY:
        raise ValueError('Insufficient prior-season out-of-fold residuals')
    return (h.actual_margin - h.predicted_margin).to_numpy(), h


def probabilities(margin, home_line, residuals):
    """Project empirical errors onto integer margins; ties round toward +infinity."""
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1 or not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError('Finite nonempty residual vector required')
    if not np.isfinite([margin, home_line]).all() or abs(home_line) > 100 or home_line * 2 % 1:
        raise ValueError('Finite margin and integer/half-point line required')
    results = np.floor(float(margin) + residuals + .5) + float(home_line)
    return dict(home_win_probability=float(np.mean(results > 0)),
                push_probability=float(np.mean(results == 0)),
                away_win_probability=float(np.mean(results < 0)))


def payout(price):
    price = float(price)
    if not np.isfinite(price) or abs(price) < 100 or price % 1:
        raise ValueError('Invalid American odds')
    return price / 100 if price > 0 else 100 / -price


def price_side(win, push, price):
    win, push = float(win), float(push)
    if not np.isfinite([win, push]).all() or min(win, push) < 0 or win + push > 1 + 1e-12:
        raise ValueError('Invalid win/push probabilities')
    profit = payout(price)
    lose = max(0., 1 - win - push)
    return dict(conditional_break_even=1 / (1 + profit),
                conditional_model_cover=win / (1 - push) if push < 1 else None,
                expected_net_per_unit=win * profit - lose)


def evaluate(path=SOURCE, out=Path('pricing_results')):
    p = source_frame(path)
    records, folds = [], []
    for season in sorted(p.season.unique()):
        test = p[p.season.eq(season) & p.method.eq('opponent_epa')]
        cutoff = test.gameday.min()
        if int((p.method.eq('opponent_epa') & p.season.lt(season)).sum()) < MIN_HISTORY:
            continue
        for method in ['opponent_epa', 'market']:
            residuals, h = residual_history(p, season, cutoff, method)
            folds.append(dict(season=int(season), method=method, history_games=len(h),
                              history_last_season=int(h.season.max()), history_last_date=str(h.gameday.max())))
            for r in p[p.season.eq(season) & p.method.eq(method)].itertuples():
                prob = probabilities(r.predicted_margin, -r.spread_line, residuals)
                outcome = r.actual_margin - r.spread_line
                observed = np.array([outcome > 0, outcome == 0, outcome < 0], dtype=float)
                estimates = np.array(list(prob.values()))
                records.append(dict(game_id=r.game_id, season=int(season), method=method, **prob,
                    observed_home=int(outcome > 0), observed_push=int(outcome == 0),
                    observed_away=int(outcome < 0), brier_three_outcome=float(np.sum((estimates-observed)**2)),
                    log_loss_three_outcome=float(-np.log(max(float(estimates[observed.astype(bool)][0]),1e-15)))))
    results = pd.DataFrame(records)
    if results.empty:
        raise ValueError('No eligible probability evaluation folds')
    summary = []
    for period, g in [('all', results)] + [(str(y), g) for y, g in results.groupby('season')]:
        for method, m in g.groupby('method'):
            summary.append(dict(period=period, method=method, games=len(m),
                brier_three_outcome=float(m.brier_three_outcome.mean()),
                log_loss_three_outcome=float(m.log_loss_three_outcome.mean()),
                mean_push_probability=float(m.push_probability.mean()), observed_push_rate=float(m.observed_push.mean())))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out/'probabilities.csv', index=False)
    pd.DataFrame(summary).to_csv(out/'comparison.csv', index=False)
    pd.DataFrame(folds).to_csv(out/'folds.csv', index=False)
    meta = dict(version=VERSION, status='RESEARCH_ONLY_NO_PROMOTION',
        source_sha256=auto.digest(Path(path).read_bytes()), code_sha256=auto.digest(Path(__file__).read_bytes()),
        method='Prior-season out-of-fold residuals; round simulated final margins to nearest integer, half up',
        limitations=['Previously examined revised historical development data; not untouched validation.',
            'Probabilities are experimental and are not claimed to be calibrated.',
            'No historical book prices available: no historical EV, ROI or profitable filter claim.',
            'Unconditional home/push/away scoring includes every evaluation game.',
            'Empirical distribution can assign zero mass to unseen outcomes; log loss clipped at 1e-15.',
            'Rounding and pooled residuals can misrepresent key-number frequencies; needs prospective evaluation.'])
    (out/'provenance.json').write_text(json.dumps(meta, indent=2)+'\n')
    table = pd.DataFrame(summary)
    text = '# Sportsbook pricing probability experiment\n\nResearch only; no live promotion or profitability claim.\n\n'
    text += 'Prior-season empirical errors produce home-cover, push and away-cover probabilities. Lower scores are better.\n\n'
    text += markdown_table(table)+'\n\n'
    text += '\n'.join('- '+s for s in meta['limitations'])+'\n'
    (out/'report.md').write_text(text)
    return results, table


def build_prices(quotes, forecasts, p, now):
    """Only same-snapshot quotes available before a fresh EPA forecast are paired."""
    by_game = {r['game_id']: r for r in forecasts}
    if len(by_game) != len(forecasts):
        raise ValueError('Duplicate EPA forecasts')
    rows = []
    seen = set()
    for q in quotes:
        if q['quote_id'] in seen:
            raise ValueError('Duplicate pricing quote')
        seen.add(q['quote_id'])
        f = by_game.get(q['game_id'])
        if f is None or q['snapshot_id'] != f['snapshot_id']:
            continue
        if any(q[k] != f[k] for k in ['home_team', 'away_team', 'kickoff_utc']):
            raise ValueError('Pricing quote/forecast identity mismatch')
        if f.get('model_version') != 'opponent_epa_alpha1000_shrink025_cap2_v1':
            raise ValueError('Unsupported EPA model version')
        captured = auto.utc(q['captured_at_utc']); updated = auto.utc(q['provider_updated_at_utc'])
        created = auto.utc(f['forecast_created_at_utc']); kickoff = auto.utc(q['kickoff_utc'])
        if not updated <= captured <= created <= now < kickoff or (captured-updated).total_seconds() > 1800:
            raise ValueError('Pricing quote/forecast timing invalid')
        hl, al = float(q['home_line']), float(q['away_line'])
        if not np.isfinite([hl, al]).all() or abs(hl+al) > 1e-9:
            raise ValueError('Pricing lines do not oppose')
        season = int(str(q['game_id']).split('_')[0])
        residuals, h = residual_history(p, season, created, 'opponent_epa')
        prob = probabilities(float(f['predicted_home_margin']), hl, residuals)
        implied_home = 1/(1+payout(q['home_price_american']))
        implied_away = 1/(1+payout(q['away_price_american']))
        for side in ['home', 'away']:
            price = q[side+'_price_american']
            row = dict(game_id=q['game_id'], quote_id=q['quote_id'], snapshot_id=q['snapshot_id'],
                forecast_quote_id=f['quote_id'], forecast_sha256=auto.digest(json.dumps(f,sort_keys=True).encode()),
                home_team=q['home_team'], away_team=q['away_team'],
                bookmaker_key=q['bookmaker_key'], sportsbook=q['sportsbook'], side=side,
                line=q[side+'_line'], price_american=price, **prob,
                **price_side(prob[side+'_win_probability'], prob['push_probability'], price),
                market_no_vig_conditional=(implied_home if side=='home' else implied_away)/(implied_home+implied_away),
                captured_at_utc=q['captured_at_utc'], forecast_created_at_utc=f['forecast_created_at_utc'],
                priced_at_utc=auto.stamp(now), kickoff_utc=q['kickoff_utc'],
                residual_games=len(h), residual_last_season=int(h.season.max()), model_version=VERSION,
                status='RESEARCH_ONLY_UNCALIBRATED_NO_BET_RECOMMENDATION')
            rows.append(row)
    # Rank alternatives only within identical game/side/line: no hidden bet selection.
    if rows:
        d = pd.DataFrame(rows)
        d['same_line_price_rank'] = d.groupby(['game_id','side','line']).price_american.transform(
            lambda prices: prices.map(payout).rank(ascending=False,method='min')).astype(int)
        rows = d.to_dict('records')
    return rows


def record(quotes, folder, captured):
    import epa_shadow
    forecast_path = epa_shadow.ROOT/'forecasts.csv'
    forecasts = []
    if forecast_path.exists():
        with forecast_path.open(newline='') as f:
            forecasts = list(csv.DictReader(f))
    now = datetime.now(timezone.utc)
    rows = build_prices(quotes, forecasts, source_frame(), now)
    ROOT.mkdir(parents=True, exist_ok=True)
    source_hash = auto.digest(SOURCE.read_bytes())
    for r in rows:
        r['residual_source_sha256'] = source_hash
        r['pricing_code_sha256'] = auto.digest(Path(__file__).read_bytes())
        r['pricing_id'] = auto.digest((r['quote_id']+r['side']+r['forecast_sha256']+VERSION).encode())
    if rows:
        auto.archive_rows(ROOT/'prices.csv',list(rows[0]),rows,'pricing_id')
        pd.DataFrame(rows).to_csv(folder/'sportsbook_prices.csv',index=False)
    status = dict(status='RESEARCH_PRICES_SAVED' if rows else 'NO_FRESH_SAME_SNAPSHOT_EPA_FORECASTS',
        quote_sides=len(rows), games=len({r['game_id'] for r in rows}), model_version=VERSION,
        residual_source_sha256=source_hash, policy='Same-snapshot fresh EPA only; no backfill or primary-model changes')
    (ROOT/'latest_status.json').write_text(json.dumps(status,indent=2)+'\n')
    text = '# Sportsbook pricing research\n\n'+json.dumps(status,indent=2)+'\n\n'
    text += 'Experimental probabilities are uncalibrated. EV is a model estimate, not demonstrated profit. All game/side alternatives remain visible; no bet is selected.\n\n'
    if rows:
        text += markdown_table(pd.DataFrame(rows)[['game_id','sportsbook','side','line','price_american','push_probability',
            'conditional_break_even','conditional_model_cover','expected_net_per_unit','same_line_price_rank']])+'\n'
    (ROOT/'report.md').write_text(text)
    return status


def grade_prices(grader, finals, now):
    """Score frozen quote probabilities, once per quote; sides are not independent trials."""
    import epa_shadow
    prices = grader.read_rows(ROOT/'prices.csv')
    quotes = {r['quote_id']:r for r in grader.read_rows(auto.ROOT/'all_book_quotes.csv')}
    forecasts = {r['game_id']:r for r in grader.read_rows(epa_shadow.ROOT/'forecasts.csv')}
    graded = []; pending = set(); seen = set()
    for r in prices:
        # One probability vector per two-sided quote; away sides have the same vector.
        if r['side'] != 'home':
            continue
        if r['quote_id'] in seen:
            raise ValueError('Duplicate priced quote in grading')
        seen.add(r['quote_id'])
        q = quotes.get(r['quote_id']); f = forecasts.get(r['game_id'])
        if q is None or f is None:
            raise ValueError('Missing original pricing quote/forecast')
        if auto.digest(json.dumps(f,sort_keys=True).encode()) != r['forecast_sha256']:
            raise ValueError('Pricing forecast hash mismatch')
        for c in ['game_id','snapshot_id','sportsbook','home_team','away_team','kickoff_utc','captured_at_utc']:
            if str(q[c]) != str(r[c]):
                raise ValueError('Pricing archive identity mismatch')
        if float(r['line']) != float(q['home_line']) or float(r['price_american']) != float(q['home_price_american']):
            raise ValueError('Pricing archive line/price mismatch')
        if not auto.utc(r['captured_at_utc']) <= auto.utc(r['forecast_created_at_utc']) <= auto.utc(r['priced_at_utc']) < auto.utc(r['kickoff_utc']):
            raise ValueError('Pricing archive timing invalid')
        estimates = np.array([float(r[c]) for c in ['home_win_probability','push_probability','away_win_probability']])
        if not np.isfinite(estimates).all() or (estimates<0).any() or not np.isclose(estimates.sum(),1,atol=1e-12,rtol=0):
            raise ValueError('Pricing archive probabilities invalid')
        final = grader.match(q,finals,[q])
        if final is None:
            pending.add(r['game_id']); continue
        if auto.utc(r['priced_at_utc']) >= auto.utc(final['kickoff_utc']):
            raise ValueError('Pricing occurred after actual kickoff')
        result = int(final['home_score'])-int(final['away_score'])+float(q['home_line'])
        observed = np.array([result>0,result==0,result<0],dtype=float)
        conditional = estimates[0]/(1-estimates[1]) if estimates[1]<1 else None
        market_conditional = (1/(1+payout(q['home_price_american'])))/(1/(1+payout(q['home_price_american']))+1/(1+payout(q['away_price_american'])))
        graded.append(dict(game_id=r['game_id'],quote_id=r['quote_id'],
            conditional_brier=(conditional-float(result>0))**2 if result!=0 and conditional is not None else None,
            market_no_vig_conditional_brier=(market_conditional-float(result>0))**2 if result!=0 and conditional is not None else None,
            brier_three_outcome=float(np.sum((estimates-observed)**2)),
            log_loss_three_outcome=float(-np.log(max(float(estimates[observed.astype(bool)][0]),1e-15))),
            observed_outcome='home' if result>0 else ('push' if result==0 else 'away'),
            graded_at_utc=auto.stamp(now)))
    ROOT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(graded,columns=['game_id','quote_id','brier_three_outcome','log_loss_three_outcome','observed_outcome','graded_at_utc','conditional_brier','market_no_vig_conditional_brier']).to_csv(ROOT/'graded_probabilities.csv',index=False)
    binary = [r for r in graded if r['conditional_brier'] is not None]
    report = dict(updated_at_utc=auto.stamp(now),
        paired_nonpush_quotes=len(binary),
        conditional_brier=float(np.mean([r['conditional_brier'] for r in binary])) if binary else None,
        market_no_vig_conditional_brier=float(np.mean([r['market_no_vig_conditional_brier'] for r in binary])) if binary else None,
        priced_quotes=len(seen), graded_quotes=len(graded),
        graded_games=len({r['game_id'] for r in graded}), pending_games=len(pending),
        brier_three_outcome=float(np.mean([r['brier_three_outcome'] for r in graded])) if graded else None,
        log_loss_three_outcome=float(np.mean([r['log_loss_three_outcome'] for r in graded])) if graded else None,
        interpretation='Quote-weighted probability scores. Multiple books for a game are correlated, not independent evidence. No bet selection or ROI claim.')
    (ROOT/'grading_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=str(SOURCE))
    parser.add_argument('--out', default='pricing_results')
    args = parser.parse_args()
    _, summary = evaluate(args.source, args.out)
    print(summary.to_string(index=False))
