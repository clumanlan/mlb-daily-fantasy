import pandas as pd
import awswrangler as wr
import logging
from datetime import datetime
import numpy as np
from statsapi import lookup_team
import statsapi


gamebox_summary_paths = ['s3://mlbdk-model/gamebox_summary/historical/game_boxscore_summary_info_2001_to_2012.parquet',
                         's3://mlbdk-model/gamebox_summary/historical/game_boxscore_summary_info_2013_to_2022.parquet',
                         "s3://mlbdk-model/gamebox_summary/historical/gamebox_summary_missing.parquet"]
gamebox_summary =  [wr.s3.read_parquet(path) for path in gamebox_summary_paths]
gamebox_summary = pd.concat(gamebox_summary, ignore_index=True)

season_playoff_game_paths = ['s3://mlbdk-model/season_playoff_game_details/historical/season_n_playoff_game_data_2001_to_2012.parquet',
                             's3://mlbdk-model/season_playoff_game_details/historical/season_n_playoff_game_data_2013_to_2022.parquet',
                             's3://mlbdk-model/season_playoff_game_details/historical/season_n_playoff_game_data_missing.parquet']

season_playoff_game = [wr.s3.read_parquet(path) for path in season_playoff_game_paths]
season_playoff_game = pd.concat(season_playoff_game, ignore_index=True)
season_playoff_game = season_playoff_game.drop_duplicates()

unique_team_ids = season_playoff_game.away_id.unique()

team_lookup_df = []
for team in unique_team_ids:
    df = lookup_team(team)[0]
    team_lookup_df.append(df)

team_lookup_df = pd.DataFrame(team_lookup_df)

batter_boxscore_paths = ['s3://mlbdk-model/batter_boxscore_stats/historical/batter_boxscore_stats_2001_to_2012.parquet',
                         's3://mlbdk-model/batter_boxscore_stats/historical/batter_boxscore_stats_2013_to_2022.parquet',
                         "s3://mlbdk-model/batter_boxscore_stats/historical/batter_boxscore_stats_missing.parquet"]
batter_boxscore = [wr.s3.read_parquet(path) for path in batter_boxscore_paths]
batter_boxscore = pd.concat(batter_boxscore, ignore_index=True)
batter_boxscore = batter_boxscore.drop_duplicates()


pitcher_boxscore_paths = ['s3://mlbdk-model/pitcher_boxscore_stats/historical/pitcher_boxscore_stats_2001_to_2012.parquet',
                          's3://mlbdk-model/pitcher_boxscore_stats/historical/pitcher_boxscore_stats_2013_to_2022.parquet',
                          "s3://mlbdk-model/pitcher_boxscore_stats/historical/pitcher_boxscore_stats_missing.parquet"]

pitcher_boxscore = [wr.s3.read_parquet(path) for path in pitcher_boxscore_paths]
pitcher_boxscore = pd.concat(pitcher_boxscore, ignore_index=True)
pitcher_boxscore = pitcher_boxscore.drop_duplicates()


def process_game_df(game_df):

    game_rel_cols_base = ['pk', 'type', 'doubleHeader', 'gamedayType', 'gameNumber',
        'season', 'dateTime','officialDate', 'time', 'ampm', 'dayNight', 'detailedState', 'statusCode', 
        'attendance', 'venue_id', 'venue_name', 'venue_tz', 'capacity', 'turfType', 'roofType', 
        'leftLine', 'leftCenter', 'center', 'rightCenter', 'rightLine', 'condition', 'temp', 'wind'
    ]

    game_away_rel_cols = ['pk', 'away_id', 'away_name', 'away_leaguename', 'away_divisionname', 'away_gamesplayed',
        'away_wins', 'away_losses', 'away_ties', 'away_pct', 'away_flyOuts', 'away_groundOuts', 'away_runs', 'away_doubles', 'away_triples',
        'away_homeRuns', 'away_strikeOuts', 'away_baseOnBalls', 'away_intentionalWalks', 'away_hits',
        'away_hitByPitch', 'away_atBats', 'away_obp', 'away_caughtStealing', 'away_stolenBases','away_sacBunts',
        'away_sacFlies',  'away_avg', 'away_slg', 'away_ops',
        'away_groundIntoDoublePlay', 'away_plateAppearances', 'away_totalBases', 'away_leftOnBase', 'away_atBatsPerHomeRun']

    game_home_rel_cols = ['pk', 'home_id', 'home_name', 'home_leaguename', 
        'home_divisionname', 'home_gamesplayed', 'home_wins', 'home_losses', 'home_ties', 'home_pct',
        'home_flyOuts', 'home_groundOuts', 'home_runs', 'home_doubles', 'home_triples',
        'home_homeRuns', 'home_strikeOuts', 'home_baseOnBalls', 'home_intentionalWalks', 'home_hits',
        'home_hitByPitch', 'home_atBats', 'home_obp', 'home_caughtStealing', 'home_stolenBases', 'home_sacBunts',
        'home_sacFlies',   'home_avg', 'home_slg', 'home_ops',
        'home_groundIntoDoublePlay', 'home_plateAppearances', 'home_totalBases', 'home_leftOnBase', 'home_atBatsPerHomeRun']


    cols_missing_from_home = ['away_airOuts',  'home_airOuts', 'away_numberOfPitches','home_numberOfPitches',
        'away_era', 'home_era',  'away_inningsPitched', 'home_inningsPitched',  'away_saveOpportunities', 'home_saveOpportunities',
        'away_earnedRuns', 'away_whip', 'away_battersFaced', 'away_outs', 'away_completeGames', 'away_shutouts',
        'away_pitchesThrown', 'away_balls', 'away_strikes', 'away_strikePercentage', 'away_rbi', 
        'away_pitchesPerInning', 'away_runsScoredPer9', 'away_homeRunsPer9',
        'home_earnedRuns', 'home_whip', 'home_battersFaced', 'home_outs', 'home_completeGames', 'home_shutouts',
        'home_pitchesThrown', 'home_balls', 'home_strikes', 'home_strikePercentage', 'home_rbi', 
        'home_pitchesPerInning', 'home_runsScoredPer9',   'home_homeRunsPer9', 'away_passedBall','home_passedBall']


    game_base = game_df[game_rel_cols_base]
    game_base_wind_values = game_base['wind'].str.split(',', expand=True)
    game_base_wind_values.columns = ['wind_speed', 'wind_direction']
    game_base_wind_values['wind_speed'] = game_base_wind_values['wind_speed'].str.replace(' mph', '')
    game_base_processed = game_base.drop(['wind'], axis=1)
    game_base_processed = game_base_processed.join(game_base_wind_values)

    game_home_filtered = game_df[game_home_rel_cols]
    game_home_filtered.columns = game_home_filtered.columns.str.replace('home_', '')
    game_home_processed = game_base_processed.merge(game_home_filtered, on=['pk']).reset_index(drop=True)

    game_away_filtered = game_df[game_away_rel_cols]
    game_away_filtered.columns = game_away_filtered.columns.str.replace('away_', '')
    game_away_processed = game_base_processed.merge(game_away_filtered, on=['pk']).reset_index(drop=True)

    game_combined_processed = pd.concat([game_home_processed, game_away_processed], ignore_index=True)
    game_combined_processed = game_combined_processed.rename({'pk': 'gamepk', 'type': 'game_type', 'id':'team_id', 'name': 'team_name'}, axis=1)

    return game_combined_processed



# PROCESS GAMEBOX DF -----------------------------------------------------------

def process_pitcher_gamebox(gamebox_df):
    """
    Creates a pitcher table of game summary stats: batters_faced, winning_pitcher
    """
    gamebox_winning_pitcher = gamebox_df[gamebox_df['label'] == 'WP'].reset_index(drop=True)
    gamebox_batters_faced = gamebox_df[gamebox_df['label'] == 'Batters faced'].reset_index(drop=True)

    ### process batters faced ---------------------------
    batters_faced_values = gamebox_batters_faced['value'].str.split(";", expand=True)
    batters_faced_base = gamebox_batters_faced.drop(['value'], axis=1)
    batters_faced_processed = batters_faced_base.join(batters_faced_values)
    batters_faced_processed = batters_faced_processed.melt(id_vars=['label', 'gamepk'])

    batters_faced_processed = batters_faced_processed.dropna(subset=['value']).reset_index(drop=True)

    batters_faced_processed_values = batters_faced_processed['value'].str.split('(\d+)', expand=True)
    batters_faced_processed_values.columns = ['pitcher_name', 'batters_faced', 'remove_a']

    batters_faced_processed = batters_faced_processed.join(batters_faced_processed_values)
    batters_faced_processed.drop(['variable', 'value', 'remove_a', 'label'], axis=1, inplace=True)
    batters_faced_processed['pitcher_name'] = batters_faced_processed['pitcher_name'].str.strip()


    ### process gamebox ------------------------------------------
    gamebox_winning_pitcher['value'] = gamebox_winning_pitcher['value'].str.replace('.', '')

    winning_pitcher_names = gamebox_winning_pitcher['value'].str.split(';', expand=True)
    gamebox_winning_pitcher_processed = gamebox_winning_pitcher.join(winning_pitcher_names)
    gamebox_winning_pitcher_processed.drop(['label', 'value'], axis=1, inplace=True)
    gamebox_winning_pitcher_processed = gamebox_winning_pitcher_processed.melt(id_vars=['gamepk'], value_name='pitcher_name')
    gamebox_winning_pitcher_processed.drop(['variable'], axis=1, inplace=True)
    gamebox_winning_pitcher_processed.dropna(subset=['pitcher_name'], inplace=True)
    gamebox_winning_pitcher_processed['pitcher_name'] = gamebox_winning_pitcher_processed['pitcher_name'].str.strip()
    gamebox_winning_pitcher_processed['winning_pitcher'] = True


    ### merge processed df into gamebox 
    gamebox_filtered_processed = (
        batters_faced_processed
        .merge(gamebox_winning_pitcher_processed, on=['pitcher_name', 'gamepk'], how='left')
    )

    gamebox_filtered_processed.fillna({'winning_pitcher': False}, inplace=True)
    gamebox_filtered_processed = gamebox_filtered_processed.drop_duplicates()
    
    return gamebox_filtered_processed


def process_pitcher_boxscore(pitcher_boxscore_df): 

    # PROCESS PITCHER BOXSCORE -------------------------------------------------------
    pitcher_boxscore_processed = pitcher_boxscore_df[pitcher_boxscore_df['personId']!=0]
    pitcher_boxscore_processed['teamname'] = pitcher_boxscore_processed['teamname'].str.replace(' Pitchers', '')
    pitcher_boxscore_processed = pitcher_boxscore_processed.drop(['namefield'], axis=1)
    pitcher_boxscore_processed.rename({'name':'pitcher_name'}, axis=1, inplace=True)

    return pitcher_boxscore_processed


def process_batter_boxscore(batter_boxscore_df):
    # PROCESS BATTER BOXSCORE -------------------------------------------------
    batter_boxscore_processed = batter_boxscore_df[~batter_boxscore_df['person_id'].isnull()]
    batter_boxscore_processed['gamepk'] = batter_boxscore_processed['gamepk'].astype(int)
    batter_boxscore_processed['batting_order'] =  batter_boxscore_processed['namefield'].str[0]
    batter_boxscore_processed['teamname'] = batter_boxscore_processed['teamname'].str.replace(' Batters', '')

    return batter_boxscore_processed

def create_pitcher_df(game_processed_df, pitcher_boxscore_df):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'Creating Pitcher df at {current_time}')

    pitcher_final_df = pd.merge(pitcher_boxscore_df, game_processed_df, left_on=['teamname', 'gamepk'], right_on=['teamName','gamepk'], how='left')

    return pitcher_final_df


def create_batter_df(game_processed_df, batter_boxscore_processed):

    game_processed_filtered = game_processed_df[['gamepk','dateTime']]

    batter_final_df = (
        game_processed_filtered
        .merge(batter_boxscore_processed, on=['gamepk'], how='left')
    )

    return batter_final_df

def lag_columns(df, group_cols, cols_to_lag, lag, drop_original_cols=True):
    """
    Create lagged versions of selected columns in a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    group_cols : list of str
        Columns to groupby
    cols_to_lag : list of str
        Columns to create lags for
    lag : int
        Number of lags to create

    Returns
    -------
    pandas.DataFrame
        DataFrame with lagged columns added
    """
    lagged = df.groupby(group_cols)[cols_to_lag].shift(periods=lag, fill_value=np.nan)
    lagged.columns = [f"{col}_lag{lag}" for col in cols_to_lag]

    df_lagged = pd.concat([df, lagged], axis=1)

    if drop_original_cols:
        df_lagged = df_lagged.drop(cols_to_lag, axis=1)
    df_lagged = df_lagged.dropna()

    return df_lagged


def rolling_summary_stats(df, group_col, time_col, window, stats={'min', 'max', 'std', 'mean', 'median', 'sum'}, window_type='observations', cols_to_exclude=None):
    """
    Calculate rolling summary statistics for columns with 'lag' over a specified window for each group in a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with lagged columns.
    group_col : str
        Name of column to group by.
    time_col : str
        Name of column with time series data (usually date or game_id).
    window : int or str
        Rolling window size. If window_type is 'observations', this should be an integer representing the number of observations. If window_type is 'time', this should be a string representing the window size (e.g. '7D' for 7-day rolling window).
    stats : list of str
        List of summary statistics to calculate. Defaults to ['min', 'max', 'std', 'mean', 'median', 'sum'].
    window_type : str, optional
        Type of rolling window. Default is 'observations'. Can be set to 'time' to use a time-based rolling window.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with summary statistics calculated over rolling window for each group.
    """
    # Create an empty DataFrame to store the summary statistics
    summary = pd.DataFrame()
    df = df.sort_values(time_col)
    lag_cols = [col for col in df.columns if 'lag' in col]
    if cols_to_exclude != None:
        lag_cols = [col for col in lag_cols if col not in game_cols_not_summarized]

    filtered_df = df[group_col + lag_cols]
    
    if window_type == 'observations':
        # Loop through each statistic and calculate it over the rolling window for each group
        summary = filtered_df.groupby(group_col).rolling(window, min_periods=1)[lag_cols].agg(stats)
        # Rename columns
        summary.columns = [f"{col}_{stat}_obs_{window}" for col in lag_cols for stat in stats]
    elif window_type == 'time':
        summary = filtered_df.groupby(group_col).rolling(f"{window}d", on=time_col, min_periods=1)[lag_cols].agg(stats)
        # Rename columns, adjusting for the time window
        summary.columns = [f"{col}_{stat}_time_{window}" for col in lag_cols for stat in stats]
    
    # Reset the index and drop unnecessary columns
    summary = summary.reset_index()
    summary = summary.rename(columns={'level_3': 'orig_index'})

    df = df.reset_index()
    df = df.rename(columns={'index':'orig_index'})

    # Merge summary statistics back into original DataFrame
    df = df.merge(summary, on=['orig_index'] + group_col, how='left')
    df = df.drop(columns='orig_index')
    
    return df


# Create team level game stats ---------------------------------------

team_lookup_df = team_lookup_df.rename(columns={'id':'team_id'})
team_lookup_df = team_lookup_df[['team_id', 'teamName']]


game_processed = process_game_df(season_playoff_game)

game_processed = pd.merge(game_processed, team_lookup_df, on='team_id', how='left')
game_processed['teamName'] = game_processed['teamName'].astype(str)
game_processed = game_processed.drop(['team_name'], axis=1)

game_lookup = game_processed[['gamepk', 'dateTime', 'teamName', 'season']].drop_duplicates()


rel_num_cols_team_old = ['gamesplayed', 'wins', 'losses', 'ties', 'pct',
       'flyOuts', 'groundOuts', 'runs', 'doubles', 'triples', 'homeRuns',
       'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits', 'hitByPitch',
       'atBats', 'obp', 'caughtStealing', 'stolenBases', 'sacBunts',
       'sacFlies', 'avg', 'slg', 'ops', 'groundIntoDoublePlay',
       'plateAppearances', 'totalBases', 'leftOnBase', 'atBatsPerHomeRun']

rel_num_cols_team_new = ['team_' + col for col in rel_num_cols_team_old]

game_processed = game_processed.rename(columns=dict(zip(rel_num_cols_team_old, rel_num_cols_team_new)))
game_processed['team_atBatsPerHomeRun'] = game_processed['team_atBatsPerHomeRun'].str.replace('-.--', '0')
game_processed[rel_num_cols_team_new] = game_processed[rel_num_cols_team_new].astype(float)

game_processed = lag_columns(game_processed, group_cols=['teamName', 'team_id', 'season'], cols_to_lag=rel_num_cols_team_new, lag=1)


game_cols_not_summarized = ['team_gamesplayed_lag1', 'team_wins_lag1', 'team_losses_lag1', 'team_ties_lag1', 'team_pct_lag1']
game_processed = rolling_summary_stats(game_processed, ['teamName', 'team_id', 'season'], time_col='dateTime', window=30, window_type='observations', cols_to_exclude=game_cols_not_summarized)






# run query to make sure we have all the game_ids!!!!!!!
# pitcher -------------------------------------------




# gamebox_pitcher_processed = process_pitcher_gamebox(gamebox_summary) unable to join at pitcher_name, have to get batters_faced elsewher
pitcher_boxscore_processed = process_pitcher_boxscore(pitcher_boxscore)


pitcher_boxscore_processed['ip'] = pitcher_boxscore_processed['ip'].astype(float)
pitcher_df = pitcher_boxscore_processed[(pitcher_boxscore_processed['note'] !='') & (pitcher_boxscore_processed['ip'] > 1)] # filter down to pitchers that have a note = starting/close


rel_num_cols_player = ['ip', 'h', 'r', 'er', 'bb', 'k', 'hr', 'era', 'p', 's']
pitcher_df[rel_num_cols_player] = pitcher_df[rel_num_cols_player].astype(float)

def calculate_pitcher_fp(df):
    df['ip_round'] = df['ip'].round()
    df['out'] = df['ip_round']*3
    df['fp'] = df['ip_round']*2.25 + df['out']*0.75 + df['er']*-2 + df['h']*-0.6 

    df = df.drop(['ip_round'],axis=1)

    return df

pitcher_df = calculate_pitcher_fp(pitcher_df)
rel_num_cols_player.append('fp')


pitcher_df = pd.merge(pitcher_df, game_lookup[['gamepk', 'dateTime', 'teamName', 'season']], left_on=['gamepk', 'teamname'], right_on=['gamepk','teamName'], how='left')

pitcher_group_cols = ['personId', 'pitcher_name', 'season']


# create lagged stats 
pitcher_df = lag_columns(pitcher_df, pitcher_group_cols, rel_num_cols_player, 1, drop_original_cols=False)

pitcher_df = pitcher_df.reset_index(drop=True)

cols_to_exclude = ['team_gamesplayed_lag1', 'team_wins1', 'team_losses1', 'team_ties1', 'team_pct']

pitcher_df = rolling_summary_stats(pitcher_df, group_col=pitcher_group_cols, time_col='dateTime', window=30, window_type='observations', cols_to_exclude=cols_to_exclude)



# pitcher_df_processed = create_pitcher_df(game_processed, pitcher_df)

pitcher_df_processed = pd.merge(pitcher_df, game_processed, on=['gamepk','dateTime', 'teamName', 'season'])
pitcher_df_processed = pitcher_df_processed.dropna()



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import mlflow

from category_encoders import TargetEncoder

# split training and test 

pitcher_train = pitcher_df_processed[pitcher_df_processed['season'].astype(int) <= 2017].sort_values('dateTime')

pitcher_train_subset = pitcher_train[pitcher_train['season'] != '2017']
pitcher_train_valid = pitcher_train[pitcher_train['season'] == '2017']


pitcher_test = pitcher_df_processed[pitcher_df_processed['season'].astype(int) > 2017]


rel_num_cols_to_pred = ['ip', 'h', 'r', 'er', 'bb', 'k', 'hr', 'era', 'p', 's']


# base model just predicts the average ---------------------------------------------------

## mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlruns/ 
remote_server_uri = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(remote_server_uri)


pitcher_exp_name = 'mlb_pitcher_fantasy_regression'
mlflow.set_experiment(pitcher_exp_name)

with mlflow.start_run() as run:
    base_model = pitcher_train_valid[['season', 'pitcher_name','personId','fp_lag1_mean_obs_30','fp']]
    base_model['pred_residual'] = base_model['fp'] - base_model['fp_lag1_mean_obs_30']

    run_name = 'base_rolling_avg'
    base_rmse = np.sqrt(mean_squared_error(base_model['fp'], base_model['fp_lag1_mean_obs_30']))
    base_r2 = r2_score(base_model['fp'],  base_model['fp_lag1_mean_obs_30'])

    mlflow.log_metric('rmse', base_rmse)
    mlflow.log_metric('r2', base_r2)

    mlflow.set_tag('mlflow.runName', run_name) # set tag with run name so we can search for it later



# residual plot is there any discernible trend?
## it looks like there are a lot of negative residuals, meaning it predicts a high value and ends up being much lower
px.scatter(base_model, x='fp_lag1_mean_obs_30', y='pred_residual')




# numeric features
rel_num_cols = [col for col in pitcher_train.columns if 'lag' in col] + ['season']

# cat features 

rel_cat_cols = ['personId', 'team_id']

cat_pipeline_high_card = Pipeline(steps=[
    ('encoder', TargetEncoder(smoothing=2))
])



## custom date transformer  
date_feats = ['dayofweek', 'dayofyear',  'is_leap_year', 'quarter', 'weekofyear', 'year']

class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
          return self

    def transform(self, x):

        x['dateTime'] = pd.to_datetime(x['dateTime'])

        dayofweek = x.dateTime.dt.dayofweek
        dayofyear= x.dateTime.dt.dayofyear
        is_leap_year =  x.dateTime.dt.is_leap_year
        quarter =  x.dateTime.dt.quarter
        weekofyear = x.dateTime.dt.weekofyear
        year = x.dateTime.dt.year

        df_dt = pd.concat([dayofweek, dayofyear,  is_leap_year, quarter, weekofyear, year], axis=1)

        return df_dt


preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), rel_cat_cols),
        ('standard_scaler', StandardScaler(), rel_num_cols),
        ('date', DateTransformer(),  ['dateTime'])
    ]
)

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_jobs=-1))
])



X_pitcher_train_subset = pitcher_train_subset[rel_num_cols + rel_cat_cols + ['dateTime']]
y_pitcher_train_subset = pitcher_train_subset['fp']

X_pitcher_train_valid = pitcher_train_valid[rel_num_cols + rel_cat_cols + ['dateTime']]
y_pitcher_train_valid = pitcher_train_valid['fp']


with mlflow.start_run() as run:
        
    # lr train test and predict 
    lr_pipeline.fit(X_pitcher_train_subset, y_pitcher_train_subset)
    y_pitcher_pred_lr = lr_pipeline.predict(X_pitcher_train_valid)

    lr_r2_score = r2_score(y_pitcher_train_valid, y_pitcher_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_pitcher_train_valid, y_pitcher_pred_lr))

    run_name = 'lr_v0_onehot_scale_date'

    mlflow.set_tag('mlflow.runName', run_name)
    mlflow.log_metric('r2', lr_r2_score)
    mlflow.log_metric('rmse', lr_rmse)


lr_r2_score
# RF PIPELINE --------------------------------------

rf_pipeline.fit(X_pitcher_train, y_pitcher_train)
y_pitcher_pred_rf = rf_pipeline.predict(X_pitcher_train)

rf_r2_score = r2_score(y_pitcher_train, y_pitcher_pred_rf)
rf_mse = mean_squared_error(y_pitcher_train, y_pitcher_pred_rf)

rf_model_pred = pd.DataFrame(y_pitcher_train)
rf_model_pred['fp_pred'] = y_pitcher_pred_rf
rf_model_pred['residual'] = rf_model_pred['fp'] - rf_model_pred['fp_pred']
px.scatter(rf_model_pred, x='fp_pred', y='residual')











tscv = TimeSeriesSplit(n_splits=5)
lr_cros_val_scores = cross_val_score(lr_pipeline, X_pitcher_train, y_pitcher_train, cv=tscv)










aws_server_uri = 's3://mlbdk-model/model-artifacts/'
mlflow.set_tracking_uri(aws_server_uri)









# batter stats 
single, double, triple, homerun, rbi, run, walk



batter_boxscore_processed = process_batter_boxscore(batter_boxscore_stats)
batter_boxscore_processed.columns

batter_df = create_batter_df(game_processed, batter_boxscore_processed)
batter_df = batter_df[batter_df['ab'] > 0 ] # filter out batters that didn't play


pitcher_df

pitcher_df[pitcher_df['gamepk']==345633]
batter_df[batter_df['gamepk']==345633]



# APPENDIX ------------------------------------------------------------

