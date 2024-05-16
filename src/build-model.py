import pandas as pd
import awswrangler as wr
import logging
from datetime import datetime
import numpy as np
from statsapi import lookup_team
import statsapi
import re

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
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
from category_encoders import TargetEncoder, CountEncoder
from sklearn.model_selection import cross_val_score, cross_validate

from processdataapp.module import GetData

# READ IN DATA ------------------------------------------------------------------


unique_team_ids = season_playoff_game.away_id.unique()

team_lookup_df = []
for team in unique_team_ids:
    df = lookup_team(team)[0]
    team_lookup_df.append(df)

team_lookup_df = pd.DataFrame(team_lookup_df)



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
        lag_cols = [col for col in lag_cols if col not in cols_to_exclude]

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

# SO WE START HERE -------------------------------------------------


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



game_processed.gamepk.value_counts(sort=True)


# run query to make sure we have all the game_ids!!!!!!!
# pitcher -------------------------------------------




# gamebox_pitcher_processed = process_pitcher_gamebox(gamebox_summary) unable to join at pitcher_name, have to get batters_faced elsewher


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


pitcher_df = pd.merge(pitcher_df, game_lookup[['gamepk', 'dateTime', 'teamName', 'season']], left_on=['gamepk', 'teamname'], right_on=['gamepk','teamName'], how='left')

pitcher_group_cols = ['personId', 'pitcher_name', 'season']

# create lagged stats 
pitcher_df = lag_columns(pitcher_df, pitcher_group_cols, rel_num_cols_player, 1, drop_original_cols=False)

pitcher_df = pitcher_df.reset_index(drop=True)

pitcher_df = rolling_summary_stats(pitcher_df, group_col=pitcher_group_cols, time_col='dateTime', window=30, window_type='observations')


# pitcher_df_processed = create_pitcher_df(game_processed, pitcher_df)

pitcher_df_processed = pd.merge(pitcher_df, game_processed, on=['gamepk','dateTime', 'teamName', 'season'])
pitcher_df_processed = pitcher_df_processed.dropna()

team_game_cols = [col for col in game_processed.columns if 'team' in col]
game_processed_opposing = game_processed[['gamepk'] + team_game_cols]
game_processed_opposing = game_processed_opposing.add_prefix('opposing_')

pitcher_df_processed = pd.merge(pitcher_df_processed, game_processed_opposing, left_on='gamepk', right_on='opposing_gamepk', how='left')
pitcher_df_processed = pitcher_df_processed[pitcher_df_processed['teamName'] != pitcher_df_processed['opposing_teamName']]
pitcher_df_processed = pitcher_df_processed.dropna()


# numeric features
rel_num_cols = [col for col in pitcher_df_processed.columns if 'lag' in col] + ['season']

# cat features 


rel_cat_cols = ['personId', 'team_id']
pitcher_df_processed['personId'] = pd.Categorical(pitcher_df_processed['personId'])
pitcher_df_processed['team_id'] = pd.Categorical(pitcher_df_processed['team_id'])


# split training and test 
pitcher_train = pitcher_df_processed[pitcher_df_processed['season'].astype(int) <= 2017].sort_values('dateTime')

pitcher_train_subset = pitcher_train[pitcher_train['season'] != '2017']
pitcher_train_valid = pitcher_train[pitcher_train['season'] == '2017']

pitcher_test = pitcher_df_processed[pitcher_df_processed['season'].astype(int) > 2017]


# ideas
## choose multiple players to plot line with predictions versus actual with dots
## exponential moving average
## polynomial
## library that utomatically finds time series features: tsfresh

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
        ('count_encode', CountEncoder(), rel_cat_cols),
        ('target_encode', TargetEncoder(), rel_cat_cols),
        ('standard_scaler', StandardScaler(), rel_num_cols),
        ('date', DateTransformer(),  ['dateTime'])
    ]
)



lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_jobs=-1, min_samples_leaf=10))
])

pitcher_train_subset = pitcher_train_subset.reset_index(drop=True)

X_pitcher_train_subset = pitcher_train_subset[rel_num_cols + rel_cat_cols + ['dateTime']]
y_pitcher_train_subset = np.log(pitcher_train_subset['fp'] + 25)

pitcher_train_valid = pitcher_train_valid.reset_index(drop=True)
X_pitcher_train_valid = pitcher_train_valid[rel_num_cols + rel_cat_cols + ['dateTime']]
y_pitcher_train_valid = np.log(pitcher_train_valid['fp']+ 20)



# LINEAR REGRESSION ----------------------------------------------------
with mlflow.start_run() as run:
        
    # lr train test and predict 
    lr_pipeline.fit(X_pitcher_train_subset, y_pitcher_train_subset)
    y_pitcher_pred_lr = lr_pipeline.predict(X_pitcher_train_valid)

    lr_r2_score = r2_score(y_pitcher_train_valid, y_pitcher_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_pitcher_train_valid, y_pitcher_pred_lr))

    run_name = 'ridge_v2_log'

    mlflow.set_tag('mlflow.runName', run_name)
    mlflow.log_metric('r2', lr_r2_score)
    mlflow.log_metric('rmse', lr_rmse)





        
    # lr train test and predict 
lr_pipeline.fit(X_pitcher_train_subset, y_pitcher_train_subset)

lr_cv_scores = cross_validate(lr_pipeline, X_pitcher_train_subset, y_pitcher_train_subset, cv=5, return_train_score=True)
lr_cv_scores = tr


y_pitcher_pred_lr = lr_pipeline.predict(X_pitcher_train_valid, include)

cols_w_nulls = X_pitcher_train_subset.isnull().sum() > 0
cols_w_nulls[cols_w_nulls]



# RF PIPELINE SINGLE OUTCOME --------------------------------------

with mlflow.start_run() as run:

    rf_pipeline.fit(X_pitcher_train_subset, y_pitcher_train_subset)

    y_pitcher_pred_rf_train = rf_pipeline.predict(X_pitcher_train_subset)
    y_pitcher_pred_valid_rf = rf_pipeline.predict(X_pitcher_train_valid)

    rf_r2_score_train = r2_score(y_pitcher_train_subset, y_pitcher_pred_lr_train)
    rf_rmse_train = np.sqrt(mean_squared_error(y_pitcher_train_subset, y_pitcher_pred_lr_train))
    
    rf_r2_score = r2_score(y_pitcher_train_valid, y_pitcher_pred_valid_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_pitcher_train_valid, y_pitcher_pred_valid_rf))

    run_name = 'rf_v3_min_sample_leafs25'

    mlflow.set_tag('mlflow.runName', run_name)
    mlflow.log_metric('r2_train', rf_r2_score_train)
    mlflow.log_metric('rmse_train', rf_rmse_train)

    mlflow.log_metric('r2', rf_r2_score)
    mlflow.log_metric('rmse', rf_rmse)


rf_cv_scores = cross_validate(rf_pipeline, X_pitcher_train_subset, y_pitcher_train_subset, cv=5, return_train_score=True)
rf_cv_scores 

# residual plot ---------------
rf_pitcher_train_residual = X_pitcher_train_subset[['dateTime', 'personId']]
rf_pitcher_train_residual['fp'] = y_pitcher_train_subset
rf_pitcher_train_residual['fp_pred'] = y_pitcher_pred_rf_train
rf_pitcher_train_residual['residual'] = rf_pitcher_train_residual['fp'] - rf_pitcher_train_residual['fp_pred'] 

px.scatter(rf_pitcher_train_residual, x='fp_pred', y='residual')

# top feature plot -------------------------------------
date_feats = ['dayofweek', 'dayofyear',  'is_leap_year', 'quarter', 'weekofyear', 'year']
# rf_feats = rf_pipeline['preprocessor'].named_transformers_['onehot'].get_feature_names_out().tolist() + rel_num_cols + date_feats

rf_feats = ['personId_countencode' + 'team_id_countencode'] + ['personId_target' + 'team_id_target'] + rel_num_cols + date_feats

rf_feat_importances = rf_pipeline['regressor'].feature_importances_
len(rf_feats)
len(rf_feat_importances)
rf_feats_df = pd.DataFrame({'feature': rf_feats, 'rf_feat_importances': rf_feat_importances})

top_10_feats_pos = rf_feats_df.sort_values('rf_feat_importances', ascending=False).head(10)
top_10_feats_pos.plot(kind='barh', x='feature', y='rf_feat_importances')

top_10_feats_neg = rf_feats_df.sort_values('rf_feat_importances').head(10)
top_10_feats_neg.plot(kind='barh', x='feature', y='rf_feat_importances')







# h20 random forest -----------------------------------------


import h2o
from h2o.estimators import H2ORandomForestEstimator
from h2o.transforms.preprocessing import H2OColumnTransformer

h2o.init()

# Convert the pandas DataFrames to H2OFrame
h2o_train = h2o.H2OFrame(pd.concat([X_pitcher_train_subset, y_pitcher_train_subset], axis=1))


h2o_valid = h2o.H2OFrame(pd.concat([X_pitcher_train_valid, y_pitcher_train_valid]), axis=1)

# Specify the column names for input features and target variables
feature_cols = X_pitcher_train_subset.columns.tolist()
target_col = ['fp']

# Train the DRF model
pitcher_rf = h2o.randomForest(x = feature_cols, y = target_col,
                            training_frame = h2o_train, nfolds = 5,
                            seed = 1234)

# Define the H2O Random Forest model
rf_model = H2ORandomForestEstimator(seed=42)

# Shut down the H2O cluster
h2o.shutdown(prompt=False)




from merf import MERF
merf = MERF()

X_pitcher_train_clusters = X_pitcher_train_subset['personId'].astype(str)
X_pitcher_train_subset_merf = X_pitcher_train_subset.drop(['personId'], axis=1)
Z_train = np.ones(shape=(X_pitcher_train_subset_merf.shape[0],1))

X_pitcher_train_valid_clusters = X_pitcher_train_valid['personId'].astype(str)
X_pitcher_train_valid_merf = X_pitcher_train_valid.drop(['personId'], axis=1)
Z_train_valid = np.ones(shape=(X_pitcher_train_valid.shape[0],1))


def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

X_pitcher_train_subset_merf['dateTime'] = pd.to_datetime(X_pitcher_train_subset_merf['dateTime'])
X_pitcher_train_valid_merf['dateTime'] = pd.to_datetime(X_pitcher_train_valid_merf['dateTime'])

add_datepart(X_pitcher_train_subset_merf, 'dateTime')
add_datepart(X_pitcher_train_valid_merf, 'dateTime')


merf.fit(X_pitcher_train_subset_merf, Z_train, X_pitcher_train_clusters,  y_pitcher_train_subset)

merf_valid_pred = merf.predict(X_pitcher_train_valid_merf, Z_train_valid, X_pitcher_train_valid_clusters)

merf_r2_score = r2_score(y_pitcher_train_valid, merf_valid_pred)
merf_rmse = np.sqrt(mean_squared_error(y_pitcher_train_valid, merf_valid_pred))

merf_r2_score
merf_rmse



# each pitcher has different consistencies, so you just want to program those consistencies into it
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
clusters = X_pitcher_train_subset['personId']
n_clusters = clusters.nunique()
n_obs = len(y_pitcher_train_subset)
q = Z_train.shape[1]  # random effects dimension
Z = np.array(Z_train)  # cast Z to numpy array (required if it's a dataframe, otw, the matrix mults later fail)

cluster_counts = clusters.value_counts()









        # Do expensive slicing operations only once
Z_by_cluster = {}
y_by_cluster = {}
n_by_cluster = {}
I_by_cluster = {}
indices_by_cluster = {}

279824
indices_i
cluster_id = 279824
indices_i = clusters == cluster_id
indices_by_cluster[cluster_id] = indices_i

Z_by_cluster['cluster_id'] = Z[indices_i]

Z_by_cluster[cluster_id] = Z[indices_i]
y_by_cluster[cluster_id] = y[indices_i]

            # Get the counts for each cluster and create the appropriately sized identity matrix for later computations
n_by_cluster[cluster_id] = cluster_counts[cluster_id]
I_by_cluster[cluster_id] = np.eye(cluster_counts[cluster_id])


for cluster_id in cluster_counts.index:
            # Find the index for all the samples from this cluster in the large vector
    indices_i = clusters == cluster_id
    indices_by_cluster[cluster_id] = indices_i

            # Slice those samples from Z and y
    Z_by_cluster[cluster_id] = Z[indices_i]
    y_by_cluster[cluster_id] = y[indices_i]

            # Get the counts for each cluster and create the appropriately sized identity matrix for later computations
    n_by_cluster[cluster_id] = cluster_counts[cluster_id]
    I_by_cluster[cluster_id] = np.eye(cluster_counts[cluster_id])









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


rel_num_cols_to_pred = ['ip', 'h', 'r', 'er', 'bb', 'k', 'hr', 'era', 'p', 's']
from sklearn.multioutput import MultiOutputRegressor

y_pitcher_train_subset_multioutput = pitcher_train_subset[rel_num_cols_to_pred]
y_pitcher_train_valid_multioutput = pitcher_train_valid[rel_num_cols_to_pred]


rf_multioutput_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(RandomForestRegressor(n_jobs=-1)))
])


rf_multioutput_pipeline.fit(X_pitcher_train_subset, y_pitcher_train_subset_multioutput)


with mlflow.start_run() as run:

    rf_pipeline.fit(X_pitcher_train_subset, y_pitcher_train_subset)
    y_pitcher_pred_valid_rf = rf_pipeline.predict(X_pitcher_train_valid)

    rf_r2_score = r2_score(y_pitcher_train_valid, y_pitcher_pred_valid_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_pitcher_train_valid, y_pitcher_pred_valid_rf))

    run_name = 'rf_multioutcome_v1_opposing_team_stats'

    mlflow.set_tag('mlflow.runName', run_name)
    mlflow.log_metric('r2', rf_r2_score)
    mlflow.log_metric('rmse', rf_rmse)