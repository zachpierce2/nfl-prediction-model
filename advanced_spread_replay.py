"""From the repository root run python advanced_spread_replay.py PBP_CACHE."""
import sys
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import strength_reference_market_model as base
from market_epa_strength import EPA

KEY = ['game_id', 'posteam', 'defteam']
ARM = {'baseline': [], 'series': ['series'], 'early': ['early'], 'both': ['series', 'early']}
RATINGS = ['series', 'early']


def aggregate(path):
    cols = KEY + ['down', 'qtr', 'score_differential', 'series_result', 'series_success',
                  'series', 'play_type', 'qb_kneel', 'qb_spike', 'epa']
    p = pd.read_csv(path, usecols=cols, low_memory=False).dropna(subset=KEY)
    for col in KEY[1:]:
        p[col] = p[col].replace({'LA':'LAR', 'STL':'LAR', 'SD':'LAC', 'OAK':'LV'})
    series = p.dropna(subset=['down','series_result','series_success','series'])
    series = series[series.series_result.ne('QB kneel')].drop_duplicates(KEY+['series'])
    plays = p[p.play_type.isin(['pass','run']) & p.qb_kneel.fillna(0).eq(0)
              & p.qb_spike.fillna(0).eq(0)].dropna(subset=['epa']).copy()
    plays['success'] = plays.epa.gt(0)
    overall = plays.groupby(KEY).agg(plays=('epa','size'),epa=('epa','mean'),success=('success','mean'))
    early = plays[plays.down.isin([1,2]) & plays.qtr.between(1,3) & plays.score_differential.abs().le(8)]
    counts = series.groupby(KEY).series_success.agg(series='mean', series_n='size')
    return overall.join(counts,how='outer').join(early.groupby(KEY).epa.agg(early='mean',early_n='size'), how='outer').reset_index()


def pregame(games, teams):
    features = []
    for season, schedule in games.groupby('season'):
        season_history = teams[teams.season.eq(season)]
        for day, slate in schedule.groupby('gameday'):
            recent = season_history[season_history.gameday.lt(day)].sort_values(['gameday','game_id'])
            recent = recent.groupby('posteam').tail(8)
            names = sorted(set(slate.home_team) | set(slate.away_team) |
                           set(recent.posteam) | set(recent.defteam))
            index = {team:i for i,team in enumerate(names)}
            n = len(names)
            values = slate[['game_id']].copy()
            for metric in RATINGS:
                h = recent.dropna(subset=[metric])
                coefs = np.zeros(2*n)
                if len(h):
                    x = np.zeros((len(h), 2*n))
                    x[np.arange(len(h)), h.posteam.map(index)] = 1
                    x[np.arange(len(h)), n+h.defteam.map(index)] = 1
                    fit = Ridge(alpha=200).fit(x,h[metric],sample_weight=h[metric+'_n'])
                    coefs = fit.coef_
                home = slate.home_team.map(index)
                away = slate.away_team.map(index)
                values[metric+'_off'] = coefs[home] - coefs[away]
                values[metric+'_def'] = coefs[n+away] - coefs[n+home]
            features.append(values)
    return pd.concat(features)


def forecast(games, first_test):
    forecasts = []
    for season in range(first_test, int(games.season.max())+1):
        train = games[games.season.lt(season)]
        test = games[games.season.eq(season)].copy()
        for name, metrics in ARM.items():
            cols = base.INPUTS + EPA + [m+s for m in metrics for s in ('_off','_def')]
            model = make_pipeline(base.SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1000))
            model.fit(train[cols], train.actual_margin-train.spread_line)
            test[name] = test.spread_line + np.clip(.25*model.predict(test[cols]),-2,2)
        forecasts.append(test[['game_id','season','actual_margin','spread_line','total_line']+list(ARM)])
    return pd.concat(forecasts,ignore_index=True)


def pricing_inputs(games):
    return pd.DataFrame({'total':games.total_line,'spread_size':games.spread_line.abs()})


def score(pred):
    rows = []
    for season in sorted(pred.season.unique())[1:]:
        prior = pred[pred.season.lt(season)]
        test = pred[pred.season.eq(season)]
        for name, margin in [('baseline','baseline'),('series','series'),('early','early'),
                             ('uncertainty','baseline'),('all_three','both')]:
            errors = (prior.actual_margin-prior[margin]).to_numpy()
            scales = np.ones(len(test))
            if name in ('uncertainty','all_three'):
                # Model typical absolute error, standardize historical errors, then rescale for this game.
                scale = make_pipeline(StandardScaler(),Ridge(alpha=100))
                scale.fit(pricing_inputs(prior),np.abs(errors))
                scales = np.maximum(1,scale.predict(pricing_inputs(test)))
                errors = errors / np.maximum(1,scale.predict(pricing_inputs(prior)))
            for g,width in zip(test.itertuples(),scales):
                strike = np.floor(g.spread_line)+.5
                mu = getattr(g,margin)
                probability = np.mean(np.floor(mu+width*errors+.5)>strike)
                actual = g.actual_margin-g.spread_line
                result = int(np.sign((mu-g.spread_line)*actual))
                baseline = int(np.sign((g.baseline-g.spread_line)*actual))
                rows.append(dict(game_id=g.game_id,season=season,arm=name,margin=mu,strike=strike,
                                 probability=probability,brier=(probability-int(g.actual_margin>strike))**2,
                                 outcome=result,gained=int(result>baseline),lost=int(result<baseline)))
    return pd.DataFrame(rows)


def main(cache, earlier=False):
    out = Path('advanced_spread_earlier' if earlier else 'advanced_spread_results')
    out.mkdir(exist_ok=True)
    first, last, first_test = (1999,2019,2003) if earlier else (2017,2025,2020)
    if earlier:
        raw = pd.read_csv('strength_games_snapshot.csv').query('@first <= season <= @last').copy()
        raw['gameday'] = pd.to_datetime(raw.gameday,utc=True)
        for col in ['home_team','away_team']:
            raw[col] = raw[col].replace({'LA':'LAR','STL':'LAR','SD':'LAC','OAK':'LV'})
        games = base.make_features(raw.sort_values(['gameday','game_id']))
        games['home_field'] = games.location.eq('Home').astype(float)
    else:
        games = base.read_games('strength_games_snapshot.csv')
    games = games[games.season.between(first,last)].dropna(subset=['actual_margin','spread_line'])
    assert games.total_line.notna().all()
    seasons, hashes = [], []
    for year in range(first,last+1):
        path = cache / f'play_by_play_{year}.csv.gz'
        seasons.append(aggregate(path))
        hashes.append({'season':year, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    teams = pd.concat(seasons).merge(games[['game_id','season','gameday','home_team','away_team']],on='game_id')
    counts = teams.groupby('game_id').size().reindex(games.game_id,fill_value=0)
    missing = set(counts[counts.ne(2)].index)
    expected_missing = {'1999_01_BAL_STL','2000_03_SD_KC','2000_06_BUF_MIA'} if earlier else set()
    assert missing == expected_missing, missing
    teams = teams[~teams.game_id.isin(missing)]
    assert ((teams.posteam.eq(teams.home_team)&teams.defteam.eq(teams.away_team)) |
            (teams.posteam.eq(teams.away_team)&teams.defteam.eq(teams.home_team))).all()
    features = pregame(games,teams)
    if earlier:
        # Reuse the earlier-era reference generator published in PR 18.
        from market_epa_experiment import build_features
        frozen = build_features(games, teams.dropna(subset=['epa','success','plays']))
    else:
        frozen = pd.read_csv('strength_epa_features.csv')
    games = games.merge(features,on='game_id',validate='one_to_one').merge(
        frozen,on='game_id',validate='one_to_one')
    pred = forecast(games, first_test)
    if earlier and Path('earlier_total_results/games.csv').exists():
        ref = pd.read_csv('earlier_total_results/games.csv')
        reference = (ref.set_index('game_id').eval('spread_line + baseline')).sort_index()
    elif not earlier:
        ref = pd.read_csv('strength_epa_reference_predictions.csv').query("method=='opponent_epa'")
        reference = ref.set_index('game_id').predicted_margin.sort_index()
    if not earlier or Path('earlier_total_results/games.csv').exists():
        pd.testing.assert_series_equal(pred.set_index('game_id').baseline.sort_index(),reference,
                                       check_names=False,atol=1e-10,rtol=0)
    scores = score(pred)
    assert scores.groupby('arm').game_id.nunique().eq(4272 if earlier else 1424).all()
    def summary(frame):
        return frame.groupby('arm').agg(games=('outcome','size'),wins=('outcome',lambda s:s.gt(0).sum()),
            losses=('outcome',lambda s:s.lt(0).sum()),pushes=('outcome',lambda s:s.eq(0).sum()),
            gained=('gained','sum'),lost=('lost','sum'),brier=('brier','mean')).reset_index()
    report = pd.concat([summary(scores).assign(period='all')] +
                       [summary(g).assign(period=y) for y,g in scores.groupby('season')])
    report.to_csv(out/'summary.csv',index=False)
    scores.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(hashes).to_csv(out/'sources.csv',index=False)
    print(report.to_string(index=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]), '--earlier' in sys.argv[2:])
