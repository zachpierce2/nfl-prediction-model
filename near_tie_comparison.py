"""Replay the fixed 0.10-point tie-break on two all-game forecast archives."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compare(benchmark_path, football_path, output):
    benchmark = pd.read_csv(benchmark_path)
    football = pd.read_csv(football_path)
    fields = ["game_id", "season", "spread_line", "actual_margin"]
    years = [*range(2004, 2017), *range(2021, 2026)]
    benchmark = benchmark[benchmark.season.isin(years)][fields + ["benchmark"]]
    football = football[football.season.isin(years)][fields + ["football"]]
    pairs = benchmark.merge(football, on=fields, how="outer", validate="one_to_one", indicator=True)
    if not pairs._merge.eq("both").all() or pairs.game_id.duplicated().any():
        raise ValueError("Forecast archives disagree on game, season, line, or outcome")
    pairs = pairs.drop(columns="_merge")
    if not np.isfinite(pairs[["benchmark","football","spread_line","actual_margin"]]).all().all():
        raise ValueError("Nonfinite forecast or outcome")
    if pairs.benchmark.abs().le(1e-10).any():
        raise ValueError("Missing benchmark side")
    pairs["eligible"] = pairs.benchmark.abs().le(.10)
    pairs["original"] = np.sign(pairs.benchmark)
    football_edge = pairs.football - pairs.spread_line
    pairs["tiebreak"] = np.where(pairs.eligible & football_edge.abs().gt(1e-10),
                                  np.sign(football_edge), pairs.original)
    pairs["switched"] = pairs.original.ne(pairs.tiebreak)
    rows = []
    for label, group in [("2004-2016", pairs[pairs.season.lt(2017)]),
                         ("2021-2025", pairs[pairs.season.ge(2021)]), *pairs.groupby("season")]:
        for arm in ["original", "tiebreak"]:
            result = np.sign(group.actual_margin - group.spread_line) * group[arm]
            rows.append(dict(period=label, arm=arm, games=len(group), eligible=int(group.eligible.sum()),
                             switched=int(group.switched.sum()), wins=int(result.gt(0).sum()),
                             losses=int(result.lt(0).sum()), pushes=int(result.eq(0).sum()),
                             switched_wins=int((group.switched & result.gt(0)).sum()),
                             switched_pushes=int((group.switched & result.eq(0)).sum())))
    summary = pd.DataFrame(rows)
    recent = summary[(summary.period == "2021-2025") & (summary.arm == "original")].iloc[0]
    assert (recent.wins, recent.losses, recent.pushes) == (725, 666, 33)
    output.mkdir(exist_ok=True, parents=True)
    pairs.to_csv(output/"games.csv", index=False)
    summary.to_csv(output/"summary.csv", index=False)
    print(summary.head(4).to_string(index=False))


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
