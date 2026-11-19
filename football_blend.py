"""Walk forward over the frozen independent-spread replay. Run from repository root."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("football_blend_results")
WEIGHTS = (0, .05, .10, .15, .20)


def blend(rows, weight):
    return rows.current + weight * (rows.football - rows.current)


def grade(rows, prediction):
    side = np.sign(prediction - rows.spread_line)
    outcome = np.sign(rows.actual_margin - rows.spread_line)
    result = side * outcome
    return dict(games=len(rows), wins=(result > 0).sum(), losses=(result < 0).sum(),
                pushes=((side != 0) & (outcome == 0)).sum(), passes=(side == 0).sum(),
                mae=(prediction - rows.actual_margin).abs().mean())


def choose_weight(prior):
    # Penalize a loss symmetrically to a win. Break ties on margin error, then weight.
    choices = []
    for weight in WEIGHTS:
        score = grade(prior, blend(prior, weight))
        choices.append((score["losses"] - score["wins"], score["mae"], weight))
    return min(choices)[2]


def main():
    games = pd.read_csv("football_spread_results/games.csv")
    history = pd.concat([pd.read_csv(OUT / "warmup.csv"), games], ignore_index=True)
    fields = ["season", "current", "football", "actual_margin", "spread_line"]
    assert history.game_id.is_unique and np.isfinite(history[fields]).all().all()
    seasons = []
    for year, test in games.groupby("season"):
        start = 2020 if year >= 2021 else 2003
        prior = history[history.season.between(start, year - 1)]
        assert len(prior) and prior.season.max() < year
        weight = choose_weight(prior)
        seasons.append(test.assign(weight=weight, blended=blend(test, weight),
                                   training_games=len(prior), last_training_year=prior.season.max()))
    predictions = pd.concat(seasons, ignore_index=True)
    predictions.to_csv(OUT / "games.csv", index=False)
    summaries = []
    for period, rows in [*predictions.groupby("era"), *predictions.groupby("season")]:
        changed = np.sign(rows.current - rows.spread_line) != np.sign(rows.blended - rows.spread_line)
        for group, subset in [("all", rows), ("changed", rows[changed])]:
            for model in ("current", "blended"):
                summaries.append(dict(period=str(period), group=group, model=model,
                                      **grade(subset, subset[model])))
    pd.DataFrame(summaries).drop_duplicates().to_csv(OUT / "summary.csv", index=False)
    predictions[["season", "weight", "training_games", "last_training_year"]].drop_duplicates().to_csv(
        OUT / "weights.csv", index=False)


if __name__ == "__main__":
    main()
