# NFL spread-first baseline v1

This is a **new explicitly defined baseline**, not a recovered trained model. It does not modify previous prediction formulas. Primary target: home margin of victory (spread). A normal residual distribution supplies provisional home-win probabilities for eventual moneyline comparison. Both market odds can be audited, but executable EV selections are deliberately disabled until verified kickoff times, as-of odds, and point-in-time data are supplied.

## Run

```bash
pip install -r requirements.txt
python model.py --out results
# Or download https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv and run offline:
python model.py --games games.csv --out results
```

## Outputs

- `predictions.csv`: every out-of-sample game projection (2020–2025)
- `backtest_by_season.csv`: MAE, RMSE and bias against historical-average baseline
- `summary.json`: pooled performance, not betting ROI
- `run_manifest.json`: pinned feature names and model identity
- `odds_audit_unverified.csv` when `--odds` supplied: retains odds and flags unverifiable kickoff timestamps

## Leakage boundaries

Feature engineering uses prior game dates only, with same-day updates deferred. Walk-forward folds train on prior **seasons** only. Ridge regularization is fixed (alpha=100), not optimized against test seasons. NFLverse source may contain retroactive statistical corrections and cannot establish historical publication times; for strict as-of validation use archived point-in-time feeds. Model probabilities assume normal residuals and should be calibrated on untouched later seasons before betting use.

## Odds CSV contract

`game_id,captured_at_utc,sportsbook,market,side,line,odds_american` where market is `spread` or `moneyline`, side is `home` or `away`, and timestamp includes timezone. Records are retained for audit; not interpreted as verified pre-kickoff prices without actual kickoff timestamps.

## Future advanced upgrades (approval required)

Point-in-time team EPA and QB availability, injury report archive, opponent adjustments, nested rolling time-series validation, calibrated cover probabilities with push mass, market line/odds snapshots, strict selection lock and append-only bet ledger. None of these is silently incorporated into v1.
