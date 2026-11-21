"""Paired historical test of recent overtime exposure. Run from the repo root."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import strength_reference_market_model as base
from market_epa_strength import EPA, score


def recent_overtime(games):
    """Home minus away exposure in the preceding seven calendar days."""
    sides = []
    for side in ("home", "away"):
        team = games[["game_id", "gameday", "overtime", "actual_margin", side + "_team"]].copy()
        team = team.rename(columns={side + "_team": "team"})
        team["side"] = side
        sides.append(team)
    appearances = pd.concat(sides).sort_values("gameday")
    played = appearances.loc[appearances.actual_margin.notna()].copy()
    if played.duplicated(["team", "gameday"]).any() or not played.overtime.isin([0, 1]).all():
        raise ValueError("Ambiguous team history or missing overtime flag")
    prior = played[["team", "gameday", "overtime"]].rename(columns={"overtime": "prior_ot"})
    paired = pd.merge_asof(appearances, prior, on="gameday", by="team",
                           allow_exact_matches=False, tolerance=pd.Timedelta(days=7))
    flags = paired.pivot(index="game_id", columns="side", values="prior_ot").fillna(0)
    return flags.home - flags.away


def run():
    games = base.read_games("strength_games_snapshot.csv")
    games = games.loc[games.season.between(2017, 2025)].copy()
    games["recent_ot"] = games.game_id.map(recent_overtime(games))
    games = games.merge(pd.read_csv("strength_epa_features.csv"), on="game_id",
                        how="left", validate="one_to_one")
    if games[EPA + ["recent_ot"]].isna().any().any():
        raise ValueError("Missing features")
    played = games.dropna(subset=["actual_margin", "spread_line"])
    reference = pd.read_csv("strength_epa_reference_predictions.csv")
    reference = reference[reference.method.eq("opponent_epa")].set_index("game_id")
    records = []
    for year in range(2021, 2026):
        train = played[played.season.lt(year)]
        test = played[played.season.eq(year)]
        for name, extra in [("primary", []), ("overtime", ["recent_ot"])]:
            inputs = base.INPUTS + EPA + extra
            model = base.make_pipeline(base.SimpleImputer(strategy="median"),
                                       base.StandardScaler(), Ridge(alpha=base.ALPHA))
            model.fit(train[inputs], train.actual_margin - train.spread_line)
            result = test[["season", "game_id", "spread_line", "actual_margin"]].copy()
            result["adjustment"] = np.clip(base.SHRINK * model.predict(test[inputs]),
                                           -base.CAP, base.CAP)
            result["predicted_margin"] = result.spread_line + result.adjustment
            result["model"] = name
            if name == "primary":
                np.testing.assert_allclose(result.predicted_margin,
                                           reference.loc[result.game_id, "predicted_margin"],
                                           rtol=0, atol=1e-10)
            records.append(result)
    results = pd.concat(records)
    for name, group in results.groupby("model"):
        for period, rows in [*group.groupby("season"), ("pooled", group)]:
            stats = score(rows)
            print(name, period, *(stats[k] for k in ("wins", "losses", "pushes", "win_pct", "mae")))
    picks = results.pivot(index="game_id", columns="model", values="adjustment")
    print("Changed sides:", np.sign(picks.primary).ne(np.sign(picks.overtime)).sum())


if __name__ == "__main__":
    run()
