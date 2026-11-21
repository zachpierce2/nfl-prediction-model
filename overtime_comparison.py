"""Paired historical test of recent overtime exposure. Run from the repo root."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import strength_reference_market_model as base
from market_epa_strength import EPA


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


def replay():
    games = base.read_games("strength_games_snapshot.csv")
    games["recent_ot"] = games.game_id.map(recent_overtime(games))
    games = games.merge(pd.read_csv("strength_epa_features.csv")[["game_id"] + EPA],
                        on="game_id", how="left", validate="one_to_one")
    games = games[games.season.between(2017, 2025)].dropna(subset=["actual_margin", "spread_line"])
    assert np.isfinite(games[EPA + ["total_line", "recent_ot"]]).all().all()
    reference = pd.read_csv("pregame_total_results/summary.csv", dtype={"period": str})
    for year in range(2021, 2026):
        train = games[games.season < year]
        test = games[games.season == year]
        results = []
        for extra in [[], ["recent_ot"]]:
            columns = base.INPUTS + EPA + ["total_line"] + extra
            estimator = base.make_pipeline(base.SimpleImputer(strategy="median"), base.StandardScaler(),
                                           Ridge(alpha=base.ALPHA))
            estimator.fit(train[columns], train.actual_margin - train.spread_line)
            adjustment = np.clip(base.SHRINK * estimator.predict(test[columns]), -base.CAP, base.CAP)
            assert np.all(abs(adjustment) > 1e-10)
            results.append(np.sign(adjustment) * np.sign(test.actual_margin - test.spread_line))
        benchmark, candidate = results
        def record(grades):
            return tuple(int((grades == value).sum()) for value in (1, -1, 0))
        archived = reference[(reference.period == str(year)) & (reference.model == "with_total")]
        assert record(benchmark) == tuple(archived[["wins", "losses", "pushes"]].iloc[0])
        changed = benchmark != candidate
        print(year, record(benchmark), record(candidate), "changed decisions", record(candidate[changed]))


if __name__ == "__main__":
    replay()
