"""Refit the spread adjustment after dropping one group. Run from the repository root."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from football_spread import ADJ, ALIASES, base

GROUPS = {
    "full": [],
    "elo": ["home_elo", "away_elo", "elo_diff"],
    "rest": ["home_rest", "away_rest", "rest_diff"],
    "recent_results": ["margin4_diff", "margin8_diff", "pf8_diff", "pa8_diff", "wins8_diff"],
    "experience": ["experience_diff"],
    "home_field": ["home_field"],
    "line_as_predictor": ["spread_line"],
    "opponent_efficiency": ADJ,
}


def replay(schedule, features, years):
    data = schedule.merge(features[["game_id", *ADJ]], on="game_id", validate="one_to_one")
    played = data.dropna(subset=["actual_margin", "spread_line"])
    columns = base.INPUTS + ADJ
    assert sorted(c for group in GROUPS.values() for c in group) == sorted(columns)
    folds = []
    for year in years:
        train = played[played.season < year]
        test = played[played.season == year]
        result = test[["game_id", "season", "spread_line", "actual_margin"]].copy()
        for group, removed in GROUPS.items():
            inputs = [col for col in columns if col not in removed]
            fit = base.make_pipeline(base.SimpleImputer(strategy="median"), base.StandardScaler(),
                                     base.Ridge(alpha=base.ALPHA))
            fit.fit(train[inputs], train.actual_margin - train.spread_line)
            adjustment = np.clip(base.SHRINK * fit.predict(test[inputs]), -base.CAP, base.CAP)
            result[group] = test.spread_line + adjustment
        folds.append(result)
    return pd.concat(folds)


def summarize(period):
    output = []
    gap = (period.full - period.spread_line).abs()
    for label, games in [("all", period), ("at_least_0.10", period[gap >= .1]),
                         ("under_0.10", period[gap < .1])]:
        outcome = np.sign(games.actual_margin - games.spread_line)
        original = np.sign(games.full - games.spread_line) * outcome
        for name in GROUPS:
            side = np.sign(games[name] - games.spread_line)
            grades = side * outcome
            output.append(dict(subset=label, removed=name, games=len(games), wins=(grades > 0).sum(),
                               losses=(grades < 0).sum(), pushes=((side != 0) & (outcome == 0)).sum(),
                               passes=(side == 0).sum(), gained=((grades > 0) & (original < 0)).sum(),
                               lost=((grades < 0) & (original > 0)).sum(),
                               mae=(games[name] - games.actual_margin).abs().mean()))
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earlier-features", required=True)
    parser.add_argument("--current-schedule", required=True)
    parser.add_argument("--current-features", required=True)
    args = parser.parse_args()
    old = pd.read_csv("strength_games_snapshot.csv").query("1999 <= season <= 2019").copy()
    old.gameday = pd.to_datetime(old.gameday, utc=True)
    for col in ["home_team", "away_team"]:
        old[col] = old[col].replace(ALIASES)
    old = base.make_features(old.sort_values(["gameday", "game_id"]))
    old["home_field"] = old.location.eq("Home").astype(float)
    runs = [
        replay(old, pd.read_csv(args.earlier_features), range(2004, 2020)),
        replay(base.read_games("strength_games_snapshot.csv"), pd.read_csv("strength_epa_features.csv"),
               range(2021, 2026)),
        replay(base.read_games(args.current_schedule), pd.read_csv(args.current_features), [2026]),
    ]
    games = pd.concat(runs).set_index("game_id", verify_integrity=True).sort_index()
    archive = pd.read_csv("football_spread_results/games.csv").set_index("game_id").sort_index()
    pd.testing.assert_index_equal(games.index, archive.index)
    for col in ["spread_line", "actual_margin", "full"]:
        np.testing.assert_allclose(games[col], archive["current" if col == "full" else col], atol=1e-10, rtol=0)
    games["era"] = archive.era
    periods = [*games.groupby("era"), *games.groupby("season")]
    report = pd.concat([summarize(g).assign(period=str(p)) for p, g in periods]).drop_duplicates()
    out = Path("spread_input_ablation_results")
    out.mkdir(exist_ok=True)
    report.to_csv(out / "summary.csv", index=False)
    games.to_csv(out / "games.csv")


if __name__ == "__main__":
    main()
