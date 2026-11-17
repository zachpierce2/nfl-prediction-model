"""Publish the designated market + EPA model from its immutable forecast archive."""
import argparse
import csv
import json
import math
from pathlib import Path

import epa_shadow
import grade_market

ROOT = Path('data/auto_market/primary')


def same_price_timing(rows):
    """Use the grader's latest eligible quote for the selected side of each game."""
    counts = dict(better=0, equal=0, worse=0, changed_price=0, missing_price=0)
    points = []
    for row in rows:
        if row['status'] != 'NEAR_KICKOFF_PROXY':
            continue
        try:
            prices = [float(row[k]) for k in ('initial_selected_price','latest_selected_price')]
            if not all(p.is_integer() and abs(p) >= 100 for p in prices):
                raise ValueError
        except (TypeError, ValueError):
            counts['missing_price'] += 1
            continue
        # +100 and -100 are equivalent even-money payouts.
        if prices[0] != prices[1] and set(prices) != {100, -100}:
            counts['changed_price'] += 1
            continue
        edge = float(row['line_value_points'])
        if not math.isfinite(edge):
            raise ValueError('Nonfinite spread comparison')
        counts['better' if edge > 0 else 'worse' if edge < 0 else 'equal'] += 1
        points.append(edge)
    return dict(**counts, comparable=len(points), primary_forecasts=len(rows),
                mean_points=sum(points)/len(points) if points else None)


def publish(mode):
    ROOT.mkdir(parents=True, exist_ok=True)
    pairs = epa_shadow.verified_pairs(
        grade_market.read_rows(epa_shadow.ROOT/'forecasts.csv'), grade_market.forecasts())
    with (ROOT/'forecasts.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=epa_shadow.FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(row for _, row in pairs)
    if mode == 'grade':
        summary = json.loads((epa_shadow.ROOT/'summary.json').read_text())['epa']
        comparison = json.loads(Path('data/auto_market/total_shadow/summary.json').read_text())
        lines = grade_market.read_rows(epa_shadow.ROOT/'line_comparison.csv')
        if len(lines) != len(pairs) or {r['game_id'] for r in lines} != {r['game_id'] for _,r in pairs}:
            raise ValueError('EPA timing rows and verified forecasts disagree')
        timing = same_price_timing(lines)
        report = dict(same_price_timing=timing, primary_model='market_plus_epa', primary=summary, total_comparison=comparison)
        (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        text = (f"# Primary NFL model — market + EPA\n\n"
                f"Forecasts: {summary['forecasts']}. Graded: {summary['graded_games']}. "
                f"Record: {summary['hypothetical_wins']} wins, {summary['hypothetical_losses']} losses, "
                f"{summary['pushes']} pushes.\n\n"
                f"Pregame-total comparison: {comparison['paired']} pairs, {comparison['graded']} graded, "
                f"{comparison['disagreements']} graded disagreements. "
                f"Market + EPA won {comparison['baseline']['wins_on_disagreements']} disagreements "
                f"and the total model won {comparison['with_total']['wins_on_disagreements']}.\n")
        text += (f"\nSame-book spread within 60 minutes of kickoff, at the original price: "
                 f"{timing['better']} better, {timing['equal']} equal, {timing['worse']} worse. "
                 f"{timing['comparable']} comparable games across {len(pairs)} primary forecasts. "
                 f"Different price: {timing['changed_price']}. Unavailable price: {timing['missing_price']}.\n")
        if timing['comparable']:
            text += f"Mean original spread advantage: {timing['mean_points']:+.2f} points.\n"
        (ROOT/'report.md').write_text(text)
    print(f'Published {len(pairs)} verified primary forecasts.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['capture','grade'])
    publish(parser.parse_args().mode)
