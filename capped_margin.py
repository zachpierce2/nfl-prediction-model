"""Compare clipping the training residual at three touchdowns. Run from repo root."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import game_state_epa as state
import market_model as model
from market_epa_experiment import ADJ, ALIASES, build_features


def earlier_games(cache):
    schedule = pd.read_csv("strength_games_snapshot.csv").query("1999 <= season <= 2016").copy()
    schedule["gameday"] = pd.to_datetime(schedule.gameday, utc=True)
    schedule[["home_team", "away_team"]] = schedule[["home_team", "away_team"]].replace(ALIASES)
    games = model.make_features(schedule.sort_values(["gameday", "game_id"]))
    games["home_field"] = games.location.eq("Home").astype(float)
    cache.mkdir(parents=True, exist_ok=True)
    years = [state.aggregate(year, cache)[0].query("arm == 'benchmark'") for year in range(1999, 2017)]
    history = pd.concat(years).merge(games[["game_id","season","gameday","actual_margin"]], on="game_id",
                                     validate="many_to_one").dropna(subset=["actual_margin"])
    complete = history.groupby("game_id").size()
    history = history[history.game_id.isin(complete[complete.eq(2)].index)]
    return games.merge(build_features(games, history), on="game_id", validate="one_to_one")


def evaluate(games, first, last):
    columns = model.INPUTS + ADJ + ["total_line"]
    played = games.dropna(subset=["spread_line", "actual_margin", "total_line"])
    if not np.isfinite(played[columns]).all().all():
        raise ValueError("Missing training or test inputs")
    rows = []
    for year in range(first, last + 1):
        train = played[played.season < year]
        test = played[played.season == year]
        residual = train.actual_margin - train.spread_line
        pair = test[["game_id", "season", "spread_line", "actual_margin"]].copy()
        for arm, target in [("benchmark", residual), ("capped", residual.clip(-21, 21))]:
            estimator = model.make_pipeline(model.SimpleImputer(strategy="median"),
                                            model.StandardScaler(), model.Ridge(alpha=model.ALPHA))
            estimator.fit(train[columns], target)
            pair[arm] = np.clip(model.SHRINK * estimator.predict(test[columns]), -model.CAP, model.CAP)
        rows.append(pair)
    pairs = pd.concat(rows)
    if pairs.game_id.duplicated().any() or (pairs[["benchmark", "capped"]].abs() <= 1e-10).any().any():
        raise ValueError("Duplicate game or no pick")
    return pairs


def summarize(pairs):
    rows = []
    for period, group in [(f"{pairs.season.min()}-{pairs.season.max()}", pairs), *pairs.groupby("season")]:
        changed = np.sign(group.capped) != np.sign(group.benchmark)
        for arm in ["benchmark", "capped"]:
            outcomes = np.sign(group.actual_margin - group.spread_line) * np.sign(group[arm])
            rows.append(dict(period=period, arm=arm, wins=int(outcomes.gt(0).sum()),
                             losses=int(outcomes.lt(0).sum()), pushes=int(outcomes.eq(0).sum()),
                             changed_wins=int((changed & outcomes.gt(0)).sum()),
                             changed_pushes=int((changed & outcomes.eq(0)).sum())))
    return pd.DataFrame(rows)


def run(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    recent = model.read_games("strength_games_snapshot.csv").query("season <= 2025")
    recent = recent.merge(pd.read_csv("strength_epa_features.csv"), on="game_id", how="left", validate="one_to_one")
    recent_pairs = evaluate(recent, 2021, 2025)
    summary = summarize(recent_pairs)
    archived = pd.read_csv("pregame_total_results/summary.csv").query("model == 'with_total'")
    verify = summary.query("arm == 'benchmark'").copy()
    verify["period"] = verify.period.replace({"2021-2025": "all"}).astype(str)
    stats = ["wins", "losses", "pushes"]
    pd.testing.assert_frame_equal(verify.set_index("period")[stats], archived.set_index("period")[stats])
    early_pairs = evaluate(earlier_games(cache), 2003, 2016)
    combined = pd.concat([summary, summarize(early_pairs)])
    combined.to_csv(out/"summary.csv", index=False)
    pd.concat([recent_pairs, early_pairs]).to_csv(out/"games.csv", index=False)
    print(combined[combined.period.isin(["2021-2025","2003-2016"])].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.cache, args.out)
