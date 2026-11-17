"""Publish the designated market + EPA model from its immutable forecast archive."""
import argparse
import csv
import json
from pathlib import Path

import epa_shadow
import grade_market

ROOT = Path('data/auto_market/primary')


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
        report = dict(primary_model='market_plus_epa', primary=summary, total_comparison=comparison)
        (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        text = (f"# Primary NFL model — market + EPA\n\n"
                f"Forecasts: {summary['forecasts']}. Graded: {summary['graded_games']}. "
                f"Record: {summary['hypothetical_wins']} wins, {summary['hypothetical_losses']} losses, "
                f"{summary['pushes']} pushes.\n\n"
                f"Pregame-total comparison: {comparison['paired']} pairs, {comparison['graded']} graded, "
                f"{comparison['disagreements']} graded disagreements. "
                f"Market + EPA won {comparison['baseline']['wins_on_disagreements']} disagreements "
                f"and the total model won {comparison['with_total']['wins_on_disagreements']}.\n")
        (ROOT/'report.md').write_text(text)
    print(f'Published {len(pairs)} verified primary forecasts.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['capture','grade'])
    publish(parser.parse_args().mode)
