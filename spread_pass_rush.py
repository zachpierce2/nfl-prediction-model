"""Compare combined and split opponent-adjusted ratings on the archived spread replay."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from football_spread import ADJ, ALIASES, base
from sklearn.linear_model import Ridge

KEY = ["game_id", "posteam", "defteam"]
OUT = Path("spread_pass_rush_results")


def aggregate(path):
    columns = KEY + ["play_type", "qb_kneel", "qb_spike", "epa"]
    p = pd.read_csv(path, usecols=columns, low_memory=False)
    p = p[p.play_type.isin(["pass", "run"]) & p.qb_kneel.fillna(0).eq(0)
          & p.qb_spike.fillna(0).eq(0)].dropna(subset=KEY + ["epa"]).copy()
    for col in ["posteam", "defteam"]:
        p[col] = p[col].replace(ALIASES)
    p["success"] = p.epa.gt(0)
    return p.groupby(KEY + ["play_type"]).agg(
        plays=("epa", "size"), epa=("epa", "sum"), success=("success", "sum")).reset_index()


def ratings(schedule, plays):
    # Identical eight-game window, ridge penalty, play-count weights and date cutoff
    # for the pooled control and each play type.
    schedules = schedule[["game_id", "gameday", "home_team", "away_team"]]
    joined = plays.merge(schedules, on="game_id", validate="many_to_one")
    valid = ((joined.posteam == joined.home_team) & (joined.defteam == joined.away_team)
             | (joined.posteam == joined.away_team) & (joined.defteam == joined.home_team))
    assert valid.all()
    pairs = joined[KEY].drop_duplicates()
    complete = pairs.groupby("game_id").size().eq(2)
    joined = joined[joined.game_id.isin(complete[complete].index)]
    keys = KEY + ["gameday"]
    pooled = joined.groupby(keys)[["plays", "epa", "success"]].sum().reset_index()
    teams = sorted(set(schedule.home_team) | set(schedule.away_team))
    ids = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    features = schedules[["game_id"]].copy().set_index("game_id")
    for label, group in [("pooled", pooled), *joined.groupby("play_type")]:
        group = group.sort_values(["gameday", "game_id"])
        cols = [label + "_" + c for c in ADJ]
        for day, games in schedules.groupby("gameday"):
            recent = group[group.gameday < day].groupby("posteam", sort=False).tail(8)
            effects = np.zeros((2, 2*n))
            if len(recent):
                x = np.zeros((len(recent), 2*n))
                idx = np.arange(len(recent))
                x[idx, recent.posteam.map(ids)] = 1
                x[idx, n + recent.defteam.map(ids)] = 1
                targets = recent[["epa", "success"]].div(recent.plays, axis=0)
                effects = Ridge(alpha=200).fit(x, targets, sample_weight=recent.plays).coef_
            h, a = games.home_team.map(ids), games.away_team.map(ids)
            off = effects[:, h] - effects[:, a]
            defense = effects[:, n+a] - effects[:, n+h]
            features.loc[games.game_id, cols] = np.array([off[0], defense[0], off[1], defense[1]]).T
    return features.reset_index()


def replay(schedule, rates, years):
    data = schedule.merge(rates, on="game_id", validate="one_to_one").dropna(subset=["actual_margin", "spread_line"])
    rows = []
    for year in years:
        train, test = data[data.season < year], data[data.season == year]
        g = test[["game_id", "season", "actual_margin", "spread_line"]].copy()
        for name, labels in [("pooled", ["pooled"]), ("split", ["pass", "run"])]:
            cols = base.INPUTS + [label + "_" + c for label in labels for c in ADJ]
            assert np.isfinite(data[cols]).all().all()
            fit = base.make_pipeline(base.SimpleImputer(strategy="median"), base.StandardScaler(),
                                     Ridge(alpha=base.ALPHA))
            fit.fit(train[cols], train.actual_margin - train.spread_line)
            g[name] = test.spread_line + np.clip(base.SHRINK * fit.predict(test[cols]), -base.CAP, base.CAP)
        rows.append(g)
    return pd.concat(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--current-plays", required=True)
    parser.add_argument("--current-schedule", required=True)
    args = parser.parse_args()
    old = pd.read_csv("strength_games_snapshot.csv").query("1999 <= season <= 2019").copy()
    old.gameday = pd.to_datetime(old.gameday, utc=True)
    for c in ["home_team", "away_team"]:
        old[c] = old[c].replace(ALIASES)
    old = base.make_features(old.sort_values(["gameday", "game_id"]))
    old["home_field"] = old.location.eq("Home").astype(float)
    modern = base.read_games("strength_games_snapshot.csv")
    current = base.read_games(args.current_schedule)
    schedule = pd.concat([old[old.season < 2017], modern, current[current.season == 2026]])
    frames = []
    for year, games in schedule.groupby("season"):
        path = args.current_plays if year == 2026 else args.cache / f"play_by_play_{year}.csv.gz"
        frames.append(ratings(games, aggregate(path)))
    rates = pd.concat(frames)
    pred = pd.concat([replay(old, rates, range(2004, 2020)), replay(modern, rates, range(2021, 2026)),
                      replay(current, rates, [2026])]).set_index("game_id").sort_index()
    ref = pd.read_csv("football_spread_results/games.csv").set_index("game_id").sort_index()
    pd.testing.assert_index_equal(pred.index, ref.index)
    np.testing.assert_allclose(pred[["spread_line", "actual_margin"]], ref[["spread_line", "actual_margin"]])
    pred["current"] = ref.current
    pred["era"] = ref.era
    rows = []
    for label, g in [*pred.groupby("era"), *pred.groupby("season")]:
        for name in ["current", "pooled", "split"]:
            side = np.sign(g[name] - g.spread_line)
            outcome = np.sign(g.actual_margin - g.spread_line)
            grades = side * outcome
            original = np.sign(g.current - g.spread_line) * outcome
            rows.append(dict(period=str(label), model=name, games=len(g), wins=(grades > 0).sum(),
                             losses=(grades < 0).sum(), pushes=((side != 0) & (outcome == 0)).sum(),
                             passes=(side == 0).sum(), fixed=((grades > 0) & (original < 0)).sum(),
                             spoiled=((grades < 0) & (original > 0)).sum(),
                             mae=(g[name] - g.actual_margin).abs().mean(),
                             max_diff_from_current=(g[name] - g.current).abs().max()))
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).drop_duplicates().to_csv(OUT / "summary.csv", index=False)
    pred.to_csv(OUT / "games.csv")


if __name__ == "__main__":
    main()
