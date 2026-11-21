"""Record the fixed 2026 game-state experiment at the pregame-total quote."""
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import auto_market as auto
from game_state_epa import aggregate_plays
import market_epa_experiment as epa
import market_model as base


def freeze_model(folder):
    games = base.read_games("strength_games_snapshot.csv")
    train = games[games.season.between(2017, 2025)].dropna(subset=["spread_line", "total_line", "actual_margin"])
    frozen = {}
    for arm in ("benchmark", "weighted"):
        inputs = base.INPUTS + epa.ADJ + ["total_line"]
        frame = train.merge(pd.read_csv(folder/f"training_{arm}.csv"), on="game_id", validate="one_to_one", how="left")
        if not np.isfinite(frame[inputs]).all().all():
            raise ValueError("Missing historical inputs")
        model = base.make_pipeline(base.StandardScaler(), base.Ridge(alpha=1000))
        model.fit(frame[inputs], frame.actual_margin - frame.spread_line)
        scale, ridge = model.steps[0][1], model.steps[1][1]
        coef = ridge.coef_ / scale.scale_
        intercept = float(ridge.intercept_ - coef @ scale.mean_)
        np.testing.assert_allclose(frame[inputs] @ coef + intercept, model.predict(frame[inputs]), atol=1e-10)
        frozen[arm] = dict(coefficients=dict(zip(inputs, coef)), intercept=intercept)
    (folder/"season_2026.json").write_text(json.dumps(frozen, indent=2)+"\n")


def record(games, rows, folder, target):
    games = games[games.season.eq(2026)].copy()
    rows = [r for r in rows if r["game_id"] in set(games.game_id)]
    if not rows:
        return 0
    source = json.loads((folder/"epa_provenance.json").read_text())
    expected = games.loc[games.actual_margin.notna(), ["game_id", "season", "gameday", "home_team", "away_team"]]
    history = pd.DataFrame(columns=["arm","season","gameday","game_id","posteam","defteam","plays","epa","success"])
    pbp_hash = ""
    if not expected.empty:
        ref, = [r for r in source["current_sources"] if r["season"] == 2026]
        raw, _ = auto.download(ref["url"], "Game-state play data")
        pbp_hash = auto.digest(raw)
        if pbp_hash != ref["pbp_sha256"]:
            raise ValueError("Play data changed since matched capture")
        plays = pd.read_parquet(io.BytesIO(raw), columns=epa.PBP_COLUMNS+["wp"])
        stats, _ = aggregate_plays(plays[plays.game_id.isin(expected.game_id)])
        history = stats.merge(expected, on="game_id", validate="many_to_one")
        counts = history.groupby(["game_id","arm"]).size()
        if len(counts) != len(expected)*2 or not counts.eq(2).all():
            raise ValueError("Incomplete play history")
    calibration = Path("game_state_results/season_2026.json")
    models = json.loads(calibration.read_text())
    targets = games[games.game_id.isin([r["game_id"] for r in rows])]
    ratings = {arm: epa.build_features(targets, history[history.arm.eq(arm)]).set_index("game_id")
               for arm in models}
    core = targets.set_index("game_id")
    saved = []
    for row in rows:
        key = row["game_id"]
        spread = -float(row["home_line"])
        margins = {}
        for arm, fitted in models.items():
            inputs = core.loc[key].to_dict()
            inputs.update(ratings[arm].loc[key, epa.ADJ].to_dict())
            inputs.update(spread_line=spread, total_line=float(row["total_line"]))
            adjustment = fitted["intercept"] + sum(inputs[k]*w for k,w in fitted["coefficients"].items())
            if not np.isfinite(adjustment):
                raise ValueError("Missing forecast input")
            margins[arm] = spread + np.clip(.25*adjustment, -2, 2)
        now = datetime.now(timezone.utc)
        if auto.utc(row["forecast_created_at_utc"]) <= now < auto.utc(row["kickoff_utc"]):
            saved.append(dict(row, frozen_total_margin=row["predicted_home_margin"],
                              baseline_margin=margins["benchmark"], predicted_home_margin=margins["weighted"],
                              forecast_created_at_utc=auto.stamp(now), pbp_sha256=pbp_hash,
                              fit_sha256=auto.digest(calibration.read_bytes())))
    return len(auto.archive_rows(target/"pairs.csv",list(saved[0]),saved,"game_id")) if saved else 0


if __name__ == "__main__":
    freeze_model(Path("game_state_results"))
