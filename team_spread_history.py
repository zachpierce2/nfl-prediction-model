"""Walk-forward test of recent team performance relative to the historical spread.

Run from the repo root with python team_spread_history.py.
Season boundaries reset the eight-game team histories. The prior-season fit
allows either persistence or reversal. No games are filtered by edge size.
"""
import argparse
from collections import defaultdict, deque

import numpy as np
import pandas as pd


def add_history(games):
    frames = []
    for _, season in games.groupby("season", sort=True):
        history = defaultdict(lambda: deque(maxlen=8))
        for _, day in season.groupby("gameday", sort=True):
            signals = []
            updates = []
            for game in day.itertuples():
                _, _, away, home = game.game_id.split("_")
                # A partial window starts with zero (market expectation).
                signals.append((sum(history[home]) - sum(history[away])) / 8)
                residual = game.actual_margin - game.spread_line
                updates.extend([(home, residual), (away, -residual)])
            frames.append(day.assign(signal=signals))
            for team, residual in updates:
                history[team].append(residual)
    return pd.concat(frames, ignore_index=True)


def evaluate(path):
    archived = pd.read_csv(path)
    games = archived.loc[archived.method.eq("opponent_epa")].copy()
    if games.empty or games.game_id.duplicated().any():
        raise ValueError("Expected unique archived opponent EPA predictions")
    if not np.isfinite(games[["adjustment", "spread_line", "actual_margin"]]).all().all():
        raise ValueError("Missing or invalid archived game")
    games = add_history(games)
    results = []
    for year in range(2021, 2026):
        train = games.loc[games.season.lt(year)]
        test = games.loc[games.season.eq(year)].copy()
        # Fit one additional coefficient to the unshrunk baseline residual.
        # Publish at the same quarter strength and two-point cap as primary.
        target = train.actual_margin - train.spread_line - train.adjustment / 0.25
        denominator = train.signal.dot(train.signal)
        if denominator <= 0 or test.empty:
            raise ValueError(f"Incomplete season fold: {year}")
        coefficient = train.signal.dot(target) / denominator
        candidate = (test.adjustment + 0.25 * coefficient * test.signal).clip(-2, 2)
        side = np.sign(candidate).where(candidate.ne(0), np.sign(test.adjustment))
        if side.eq(0).any() or test.adjustment.eq(0).any():
            raise ValueError("Every game must have a pick")
        margin = test.actual_margin - test.spread_line
        test["baseline"] = np.sign(test.adjustment) * np.sign(margin)
        test["candidate"] = side * np.sign(margin)
        results.append(test)
    scores = pd.concat(results)
    for period, group in [*scores.groupby("season"), ("All", scores)]:
        def record(column):
            return tuple(int(group[column].eq(outcome).sum()) for outcome in (1, -1, 0))
        gains = ((group.baseline == -1) & (group.candidate == 1)).sum()
        costs = ((group.baseline == 1) & (group.candidate == -1)).sum()
        print(period, "baseline", record("baseline"), "candidate", record("candidate"),
              "losses to wins", gains, "wins to losses", costs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="?", default="strength_epa_reference_predictions.csv")
    evaluate(parser.parse_args().predictions)
