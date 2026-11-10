from pathlib import Path
import pandas as pd

URL = 'https://github.com/nflverse/nfldata/raw/refs/heads/master/data/games.csv'

Path('data').mkdir(exist_ok=True)

print('Downloading NFL historical games...')
df = pd.read_csv(URL)

required = {'game_id', 'season', 'home_team', 'away_team', 'home_score', 'away_score'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f'Missing required columns: {sorted(missing)}')

df = df[df['season'].between(2017, 2025)].copy()
if df.empty:
    raise ValueError('No historical games found')

output = Path('data/games.csv')
df.to_csv(output, index=False)
print(f'Saved {len(df)} games to {output}')
