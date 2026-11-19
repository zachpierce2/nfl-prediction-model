"""Rebuild prior-season calibration seeds using football_spread.py and matching earlier PBP."""
import argparse
from pathlib import Path
import pandas as pd
import football_spread as f

def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--earlier-cache',type=Path)
    source.add_argument('--earlier-features',type=Path)
    args=parser.parse_args()
    raw=pd.read_csv('strength_games_snapshot.csv').query('1999<=season<=2003').copy()
    raw.gameday=pd.to_datetime(raw.gameday,utc=True)
    for c in ['home_team','away_team']:
        raw[c]=raw[c].replace(f.ALIASES)
    old=f.base.make_features(raw.sort_values(['gameday','game_id']))
    old['home_field']=old.location.eq('Home').astype(float)
    features=pd.read_csv(args.earlier_features) if args.earlier_features else f.older_ratings(old,args.earlier_cache)
    seeds=[f.replay('early seed',old,features,2003,2003),
           f.replay('recent seed',f.base.read_games('strength_games_snapshot.csv'),
                    pd.read_csv('strength_epa_features.csv'),2020,2020)]
    dest=Path('football_calibration_results')
    dest.mkdir(exist_ok=True)
    pd.concat(seeds)[['game_id','season','football','actual_margin']].to_csv(dest/'warmup.csv',index=False)

if __name__=='__main__':
    main()
