"""Run from repository root: python game_state_epa.py CACHE OUTPUT."""
import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

import market_model as base
import market_epa_experiment as epa


def aggregate(year, cache):
    path = cache / f"{year}.parquet"
    if not path.exists():
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"
        temporary = path.with_suffix(".part")
        urlretrieve(url, temporary)
        temporary.replace(path)
    plays = pd.read_parquet(path, columns=epa.PBP_COLUMNS + ["wp"])
    frames, counts = aggregate_plays(plays)
    return frames, dict(year=year, **counts, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def aggregate_plays(plays):
    plays = plays.loc[plays.play_type.isin(["pass", "run"]) & plays.epa.notna()
                      & plays.qb_kneel.fillna(0).eq(0) & plays.qb_spike.fillna(0).eq(0)
                      & plays.posteam.notna() & plays.defteam.notna()].copy()
    plays[["posteam", "defteam"]] = plays[["posteam", "defteam"]].replace(epa.ALIASES)
    if not plays.wp.dropna().between(0, 1).all():
        raise ValueError("Invalid pre-play win probability")
    discounted = plays.wp.lt(.05) | plays.wp.gt(.95)
    keys = [plays.game_id, plays.posteam, plays.defteam]
    frames = []
    for arm in ["benchmark", "weighted"]:
        weight = np.where(discounted & (arm == "weighted"), .5, 1.)
        totals = pd.DataFrame(dict(plays=weight, epa=plays.epa * weight,
                                   success=plays.epa.gt(0) * weight)).groupby(keys).sum()
        totals[["epa", "success"]] = totals[["epa", "success"]].div(totals.plays, axis=0)
        frames.append(totals.reset_index().assign(arm=arm))
    return pd.concat(frames), dict(plays=len(plays), discounted=int(discounted.sum()),
                                  missing_wp=int(plays.wp.isna().sum()))


def replay(schedule, team_games, first_year, output):
    games = base.make_features(schedule.sort_values(["gameday", "game_id"]))
    games["home_field"] = games.location.eq("Home").astype(float)
    arms = []
    for arm, history in team_games.groupby("arm"):
        history = history.merge(games[["game_id", "season", "gameday", "actual_margin", "home_team", "away_team"]],
                                on="game_id", validate="many_to_one").dropna(subset=["actual_margin"])
        if history.duplicated(["game_id", "posteam"]).any() or not (
                (history.posteam.eq(history.home_team) & history.defteam.eq(history.away_team)) |
                (history.posteam.eq(history.away_team) & history.defteam.eq(history.home_team))).all():
            raise ValueError("Invalid team identifiers")
        complete = history.groupby("game_id").size()
        history = history[history.game_id.isin(complete[complete.eq(2)].index)]
        features = epa.build_features(games, history)
        if first_year == 2021:
            features[["game_id"] + epa.ADJ].to_csv(output / f"training_{arm}.csv", index=False)
        frame = games.merge(features, on="game_id", validate="one_to_one")
        frame = frame.dropna(subset=["actual_margin", "spread_line", "total_line"])
        columns = base.INPUTS + epa.ADJ + ["total_line"]
        picks = []
        for year in sorted(frame.season.unique()):
            if year < first_year:
                continue
            train, test = frame[frame.season < year], frame[frame.season == year]
            estimator = base.make_pipeline(base.SimpleImputer(strategy="median"), base.StandardScaler(),
                                           base.Ridge(alpha=1000))
            estimator.fit(train[columns], train.actual_margin - train.spread_line)
            adjustment = np.clip(.25 * estimator.predict(test[columns]), -2, 2)
            if (abs(adjustment) <= 1e-10).any():
                raise ValueError("Unexpected no-pick")
            picks.append(test[["game_id", "season", "actual_margin", "spread_line"]].assign(**{arm: adjustment}))
        arms.append(pd.concat(picks))
    return arms[0].merge(arms[1], validate="one_to_one", on=["game_id", "season", "actual_margin", "spread_line"])


def run(cache, output):
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    schedule = pd.read_csv("strength_games_snapshot.csv")
    schedule["gameday"] = pd.to_datetime(schedule.gameday, utc=True)
    schedule[["home_team", "away_team"]] = schedule[["home_team", "away_team"]].replace(epa.ALIASES)
    records, sources = [], []
    for start, end, first in [(2017, 2025, 2021), (1999, 2016, 2003)]:
        teams = []
        for year in range(start, end + 1):
            frame, source = aggregate(year, cache)
            teams.append(frame)
            sources.append(source)
        pairs = replay(schedule[schedule.season.between(start, end)], pd.concat(teams), first, output)
        pairs.to_csv(output / f"picks_{first}_{end}.csv", index=False)
        for period, group in [(f"{first}-{end}", pairs), *pairs.groupby("season")]:
            changed = np.sign(group.benchmark) != np.sign(group.weighted)
            for arm in ["benchmark", "weighted"]:
                result = np.sign(group[arm]) * np.sign(group.actual_margin - group.spread_line)
                records.append(dict(period=period, arm=arm, wins=int((result == 1).sum()),
                                    losses=int((result == -1).sum()), pushes=int((result == 0).sum()),
                                    changed_wins=int((changed & result.eq(1)).sum()),
                                    changed_pushes=int((changed & result.eq(0)).sum())))
        print(records[-2*(end-first+2):][:2], flush=True)
        pd.DataFrame(records).to_csv(output/"summary.csv", index=False)
        pd.DataFrame(sources).to_csv(output/"sources.csv", index=False)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
