"""Inspect published source schemas before adding new matchup inputs."""
import io, json, zipfile, urllib.request
import pandas as pd

def main():
    for filename in ['nfl_advanced_data_auto.zip','NFL_V3_Advanced_Data_Upgrade.zip','nfl_v3_shadow_tools.zip']:
        try:
            with zipfile.ZipFile(filename) as z:
                print('ARCHIVE', filename, z.namelist(), flush=True)
                for name in z.namelist():
                    if name.endswith('.py') and any(t in name.lower() for t in ['feature','model','report','data_layer']):
                        body=z.read(name).decode()
                        print('EXISTING',name,body[:45000],flush=True)
        except FileNotFoundError:
            pass
    for year in [2018,2025]:
        for kind,url in [
            ('pressure',f'https://github.com/nflverse/nflverse-data/releases/download/pfr_advstats/advstats_week_pass_{year}.csv'),
            ('injuries',f'https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.csv'),
            ('snaps',f'https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{year}.csv')]:
            try:
                frame=pd.read_csv(url)
                print('SCHEMA',kind,year,len(frame),frame.columns.tolist(),flush=True)
                print(frame.head(2).to_json(orient='records'),flush=True)
            except Exception as e:
                print('UNAVAILABLE',kind,year,type(e).__name__,str(e)[:200],flush=True)

if __name__=='__main__':
    main()
