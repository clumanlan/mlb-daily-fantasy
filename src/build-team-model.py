import awswrangler as wr
import logging
from datetime import datetime
import numpy as np
from statsapi import lookup_team
import statsapi
import re
from unidecode import unidecode
from time import sleep
import json

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

import plotly.express as px
import mlflow
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier  
from sklearn.inspection import permutation_importance
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.summarize import WindowSummarizer
import boto3 

import pandas as pd
from processdataapp.module import GetData, ProcessData
pd.set_option('display.max_columns', None)

def get_secret():

    secret_name = "dkuser_aws_keys"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    get_secret_value_response = client.get_secret_value(SecretId=secret_name)

    secret_response = get_secret_value_response['SecretString']

    return json.loads(secret_response)


secret_dict = get_secret()
aws_key_id = secret_dict['aws_access_key_id']
aws_secret = secret_dict['aws_secret_access_key']

session = boto3.Session(
    aws_access_key_id=aws_key_id,
    aws_secret_access_key=aws_secret)

del aws_key_id, aws_secret, secret_dict

# CHANGES
## PULL DATA FUNCTION
### REMOVE PERSON ID
### CHANGE TIMING OF WHEN FUNCTION RUNS, IT'S PULLING IN PLAYER INFO DIFFERENTLY EXAPMLE GAMEPK: 662780



# EXTRA POTENTIAL DATA:

# gamebox_summary = GetData().get_gamebox_summary() ## reevaluate how to get pitcher batter's faced, game boxsummary doesn't have player team id linked 
# gamebox_processed = ProcessData().process_gamebox_df(gamebox_summary)



# game_playbyplay = statsapi.get("game_playByPlay", {"gamePk":6809}) #gametype parameter doesn't work
# game_playbyplay['allPlays'][0]

# gamePace




# READ IN DATA ---------------------------------------------------------------------------------------------------------

season_playoff_games = GetData().get_season_playoff_games()
batter_boxscore = GetData().get_batter_boxscore()


# REG SEASON GAME PROCESSING ------------------------------------------------------------------------------------------------

season_playoff_games_processed = ProcessData().process_game_df(season_playoff_games)

# could add back to 2001 but would have to account for expos
reg_season_games_processed = season_playoff_games_processed[(season_playoff_games_processed['game_type']=='R') & (season_playoff_games_processed['season'].astype(int)>2004)].reset_index(drop=True)

reg_season_games_processed = ( # adress these later, maybe figure out how to hold on to flyout
    reg_season_games_processed
    .drop(['wind_speed','wind_direction', 'atBatsPerHomeRun', 'capacity', 'leftCenter','rightCenter',
            'leftLine', 'rightLine', 'center','flyOuts', 'dateTime'], axis=1)
    .assign(
        officialDate = lambda x: pd.to_datetime(x['officialDate']),
        officialDate_n_Time = lambda x: pd.to_datetime(x['officialDate'].astype(str) + ' ' + x['time'] + ' ' + x['ampm'],  format='%Y-%m-%d %I:%M %p')
    )
    .drop_duplicates(subset=['gamepk', 'team_id', 'atBats'])
    .reset_index(drop=True)
)


team_cols = ['wins', 'losses', 'ties', 'pct', 'groundOuts', 'runs',
    'doubles', 'triples', 'homeRuns', 'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits',
    'hitByPitch', 'atBats','obp','caughtStealing', 'stolenBases','sacBunts',
    'sacFlies','avg','slg','ops','groundIntoDoublePlay','plateAppearances','totalBases',
     'leftOnBase']


reg_season_games_processed[team_cols] = reg_season_games_processed[team_cols].astype(float)
reg_season_games_processed.columns = ['team_' + col if col in team_cols else col for col in reg_season_games_processed.columns ]

reg_season_games_processed = pd.merge(reg_season_games_processed, reg_season_games_processed[['gamepk', 'team_id']], suffixes=['_batter', '_pitcher'], on='gamepk', how='left')
reg_season_games_processed = reg_season_games_processed[reg_season_games_processed['team_id_batter']!=reg_season_games_processed['team_id_pitcher']].drop_duplicates()
reg_season_games_processed = reg_season_games_processed[reg_season_games_processed['season'].astype(int)<2020]

del  season_playoff_games, season_playoff_games_processed 


# CREATE TEAM LOOKUP TABLE AND pLAYER LOOK UP TABLE --------------------------------------------------------------------------------

import statsapi
team_ids_list = reg_season_games_processed.team_id_batter.unique()

team_names = []
team_locations = []
team_abbrs = []

for team_id in team_ids_list:
    team =statsapi.get("team", {"teamId":int(team_id)}) #gametype parameter doesn't work


    team_names.append(team['teams'][0]['teamName'])
    team_locations.append(team['teams'][0]['locationName'])
    team_abbrs.append(team['teams'][0]['abbreviation'])

    sleep(1.1)


team_lookup = pd.DataFrame({'team_id':team_ids_list,'teamname':team_names, 'team_location':team_locations, 'team_abbr':team_abbrs})

del team_ids_list, team_locations, team_names, team_abbrs, team_cols, team, team_id


# PITCHER FEATS ---------------------------------------------------------------------------------------------------------------------
## PROCESS PITCHER SEASON -----------------------------------------------------------------

pitcher_stats_season = wr.s3.read_parquet(path='s3://mlbdk-model/features/pitcher/season_stats/pitcher_season_stats_to_2023.parquet')

pitcher_stats_season.columns = [col.replace('stats.', 'prev_season_stats_') for col in pitcher_stats_season.columns] 


drop_cols = ['first_name', 'last_name',  'active', 'current_team','prev_season_stats_stolenBasePercentage', 'prev_season_stats_winPercentage',
             'position','nickname', 'last_played', 'group', 'type', 'personId','season_lagged', 'season', 'prev_season_stats_strikeoutWalkRatio',
             'prev_season_stats_groundOutsToAirouts']


pitcher_stats_season[pitcher_stats_season['prev_season_stats_era'] == '-.--'][['first_name', 'last_name', 'season']]

pitcher_stats_season = (pitcher_stats_season
                        .assign(mlb_debut = lambda x: pd.to_datetime(x['mlb_debut']),
                                mlb_debut_season = lambda x: pd.to_datetime(x['mlb_debut']).dt.year.astype(str),
                                pitcher_season = lambda x: x['season'],
                                season_lagged = lambda x: x['season'].astype(int)+1,
                                pitcher_personId = lambda x: x['personId'].astype(str),
                                personId_n_prev_season_pitcher = lambda x: x['pitcher_personId'] + '_' + x['season_lagged'].astype(str))
                        .drop(drop_cols, axis=1)
)

pitcher_stats_season = pitcher_stats_season[pitcher_stats_season['prev_season_stats_era'] != '-.--']

string_cols_to_convert_to_num = ['prev_season_stats_avg', 'prev_season_stats_obp', 'prev_season_stats_slg',
                                 'prev_season_stats_ops', 'prev_season_stats_era',
                                 'prev_season_stats_inningsPitched', 'prev_season_stats_whip', 'prev_season_stats_strikePercentage',
                                  'prev_season_stats_pitchesPerInning', 'prev_season_stats_strikeoutsPer9Inn', 'prev_season_stats_walksPer9Inn',
                                 'prev_season_stats_hitsPer9Inn', 'prev_season_stats_runsScoredPer9','prev_season_stats_homeRunsPer9']


pitcher_stats_season[string_cols_to_convert_to_num] = pitcher_stats_season[string_cols_to_convert_to_num].astype(float)

cols_to_check_why_null = ['prev_season_stats_era', 'prev_season_stats_whip',
       'prev_season_stats_groundOutsToAirouts',
       'prev_season_stats_pitchesPerInning',
       'prev_season_stats_strikeoutWalkRatio',
       'prev_season_stats_strikeoutsPer9Inn', 'prev_season_stats_walksPer9Inn',
       'prev_season_stats_hitsPer9Inn', 'prev_season_stats_runsScoredPer9',
       'prev_season_stats_homeRunsPer9']

pitcher_stats_season.columns[(pitcher_stats_season == '-.--').any()]


pitcher_stats_season[pitcher_stats_season['prev_season_stats_strikeoutWalkRatio'] == '-.--'][['mlb_debut','personId_n_prev_season_pitcher']]




# for players that played for mulitple teams in one season there's a total row that you can grab if if you just grab the max of the games played 
pitcher_season_totals_idx = pitcher_stats_season.groupby(['personId_n_prev_season_pitcher'])['prev_season_stats_gamesPlayed'].transform(max) == pitcher_stats_season['prev_season_stats_gamesPlayed']
pitcher_stats_season = pitcher_stats_season[pitcher_season_totals_idx].reset_index(drop=True)

pitcher_detail_cols = ['mlb_debut', 'mlb_debut_season', 'pitcher_personId', 'pitch_hand']
pitcher_details = pitcher_stats_season[pitcher_detail_cols].drop_duplicates()

# we create first year stats for median fill - there are players that are few years into the league that go in and out of minors, but they're kinda the same playeres
## we could bin this a bit differently based on how many games htat person's played but let's just do it basic now and se ewhat we get
first_year_pitcher_season = pitcher_stats_season[(pitcher_stats_season['mlb_debut_season']==pitcher_stats_season['pitcher_season']) & (pitcher_stats_season['pitcher_season'].astype(int)<2013)]
first_year_pitcher_season_num_feats = first_year_pitcher_season.select_dtypes(exclude='number')

first_year_pitcher_season_num_feats_median = first_year_pitcher_season_num_feats.median().round().astype(int)



[col for col in first_year_pitcher_season_num_feats if 'avg' in col]
first_year_pitcher_season_num_feats.columns.tolist()


px.histogram(first_year_pitcher_season['prev_season_stats_gamesPlayed']) # could change based on games played


pitcher_stats_season = pitcher_stats_season.drop(pitcher_detail_cols, axis=1)

non_rel_cols = ['personId_n_prev_season_pitcher',  'bat_side', 'pitcher_season', 'season_lagged']
pitcher_prev_season_num_feats = [col for col in pitcher_stats_season.columns if col not in non_rel_cols]


del pitcher_season_totals_idx, non_rel_cols, first_year_pitcher_season, first_year_pitcher_season_num_feats


## PITCHER BOXSCORE PROCESS -----------------------------------------------------------------------------

pitcher_boxscore = GetData().get_pitcher_boxscore()
pitcher_boxscore_processed = ProcessData().process_pitcher_boxscore(pitcher_boxscore)

pitcher_boxscore_processed = (
    pitcher_boxscore_processed
    .assign(pitcher_name = lambda x: x['pitcher_name'].str.replace(r'[,.].+', '').apply(lambda y : unidecode(y)), # since we have personid we don't need first name
                    teamname = lambda x: x['teamname'].str.replace('Indians','Guardians').str.replace('Devil Rays', 'Rays').str.replace('Diamondbacks','D-backs'))
    .drop_duplicates(subset=['gamepk','personId']) # there are some state duplicates that are .03 off but not many
    .merge(team_lookup, on='teamname', how='left')
)

pitcher_boxscore_reg_season = (
    reg_season_games_processed[['gamepk', 'team_id_pitcher', 'season', 'officialDate_n_Time']]
    .merge(pitcher_boxscore_processed, left_on=['gamepk', 'team_id_pitcher'], right_on=['gamepk', 'team_id'], how='left')
    .assign(
        personId_n_season_pitcher = lambda x:x['personId']+ '_' + x['season'],
        teamId_n_season_pitcher = lambda x: x['team_id'] + '_' + x['season'],
        previous_season = lambda x: (x['season'].astype(int)-1).astype(str))
)


pitcher_cols = ['ip', 'h', 'r', 'er', 'bb', 'k', 'hr',  'p', 's'] # could add 'era' later

pitcher_boxscore_reg_season[pitcher_cols] = pitcher_boxscore_reg_season[pitcher_cols].astype(float)
pitcher_boxscore_reg_season.columns = ['pitcher_' + col if col in pitcher_cols else col for col in pitcher_boxscore_reg_season.columns]

del pitcher_boxscore, pitcher_boxscore_processed, pitcher_cols



# PITCHER STARTER ROLLING SEASON STATS (ASSUME MOST pitches IS STARTER) ----------------------------------------------------------

main_pitcher_idx = pitcher_boxscore_reg_season.groupby(['gamepk', 'team_id_pitcher'])['pitcher_p'].transform(max) == pitcher_boxscore_reg_season['pitcher_p']
pitcher_main_rolling_stats = pitcher_boxscore_reg_season[main_pitcher_idx].reset_index(drop=True)

rel_num_cols_pitcher = ['pitcher_ip','pitcher_h', 'pitcher_r', 'pitcher_er', 'pitcher_bb','pitcher_k', 'pitcher_hr',  
                        'pitcher_p', 'pitcher_s']


pitcher_main_rolling_stats = (
    pitcher_main_rolling_stats[['personId', 'personId_n_season_pitcher', 'teamId_n_season_pitcher', 'officialDate_n_Time'] + rel_num_cols_pitcher]
    .sort_values(by=['officialDate_n_Time', 'personId_n_season_pitcher'])
    .reset_index(drop=True)
)

pitcher_main_rolling_stats = (
    pitcher_main_rolling_stats
    .assign(
        pitcher_h_per_ip = lambda x: x['pitcher_h']/x['pitcher_ip'],
        pitcher_r_per_ip = lambda x: x['pitcher_r']/x['pitcher_ip'],
        pitcher_er_per_ip = lambda x: x['pitcher_er']/x['pitcher_ip'],
        pitcher_bb_per_ip = lambda x: x['pitcher_bb']/x['pitcher_ip'],
        pitcher_hr_per_ip = lambda x: x['pitcher_hr']/x['pitcher_ip'],
        pitcher_p_per_ip = lambda x: x['pitcher_p']/x['pitcher_ip'],
        pitcher_s_per_ip = lambda x: x['pitcher_s']/x['pitcher_ip'],
        pitcher_whip = lambda x: (x['pitcher_h'] + x['pitcher_bb']) / x['pitcher_ip']
    )
)

pitcher_main_rel_cols = ['pitcher_ip', 'pitcher_h', 'pitcher_r', 'pitcher_er', 'pitcher_bb',
       'pitcher_k', 'pitcher_hr',  'pitcher_p', 'pitcher_s', 'pitcher_h_per_ip']


pitcher_main_stats_lag = (
    pitcher_main_rolling_stats
    .sort_values(by='officialDate_n_Time')
    .groupby(['personId_n_season_pitcher', 'teamId_n_season_pitcher'], as_index=False)
    [pitcher_main_rel_cols].transform(lambda x: x.shift(1))
    .sort_index()
 )


pitcher_main_stats_lag.columns = [col + '_lag1' for col in pitcher_main_stats_lag.columns]
pitcher_main_rolling_stats = pd.merge(pitcher_main_rolling_stats, pitcher_main_stats_lag, left_index=True, right_index=True)

pitcher_main_rolling_stats = (
    pitcher_main_rolling_stats
    .drop(pitcher_main_rel_cols, axis=1)
)


def create_aggregate_rolling_functions(window_num = 300, window_min = 1):
    ## aggregate rolling functions to create summary stats
    f_min = lambda x: x.rolling(window=window_num, min_periods=window_min).min() 
    f_max = lambda x: x.rolling(window=window_num, min_periods=window_min).max()
    f_mean = lambda x: x.rolling(window=window_num, min_periods=window_min).mean()
    f_std = lambda x: x.rolling(window=window_num, min_periods=window_min).std()
    f_sum = lambda x: x.rolling(window=window_num, min_periods=window_min).sum()

    return f_min, f_max, f_mean, f_std, f_sum

f_min, f_max, f_mean, f_std, f_sum = create_aggregate_rolling_functions()

function_list = [f_min, f_max, f_mean, f_std, f_sum]
function_name = ['min', 'max', 'mean', 'std', 'sum']

for col in pitcher_main_rolling_stats.columns[pitcher_main_rolling_stats.columns.str.contains('lag1')]:
    print(col)
    for i in range(len(function_list)):
        pitcher_main_rolling_stats.loc[:,(col + '_rolling_season_%s' % function_name[i])] = pitcher_main_rolling_stats.groupby(['personId_n_season_pitcher'], group_keys=False)[col].apply(function_list[i])
        print(function_name[i])

pitcher_main_rolling_stats_feats = [col for col in pitcher_main_rolling_stats.columns if col not in ['personId_n_season_pitcher', 'teamId_n_season_pitcher', 'officialDate_n_Time']]

del pitcher_main_rel_cols, pitcher_main_stats_lag, rel_num_cols_pitcher, function_list, function_name, main_pitcher_idx


## PTICHER ROLLING SHORT WINDOW!!!! --------------------------------------------------------



## CREATE FINAL PITCHER DF --------------------------------------------------------------------------
pitcher_team_gamepks = pitcher_boxscore_reg_season[['gamepk',  'officialDate_n_Time', 'teamId_n_season_pitcher', 'team_id_pitcher', 'season']].drop_duplicates()
pitcher_stats_complete = pd.merge(pitcher_team_gamepks, pitcher_main_rolling_stats, on=['teamId_n_season_pitcher', 'officialDate_n_Time'], how='left')
pitcher_stats_complete = pd.merge(pitcher_stats_complete, pitcher_stats_season, left_on=['personId_n_season_pitcher'], right_on=['personId_n_prev_season_pitcher'], how='left')
pitcher_stats_complete = pd.merge(pitcher_stats_complete, pitcher_details, left_on='personId', right_on='pitcher_personId')


pitcher_stats_complete['prev_season_stats_na_filled'] = np.where(pitcher_stats_complete.prev_season_stats_gamesPlayed.isnull(), True, False)
first_year_pitcher_season_num_feats_median

pitcher_stats_complete = (
     pitcher_stats_complete
     .fillna(first_year_pitcher_season_num_feats_median)
     .assign(mlb_debut_season = lambda x: np.where(x['mlb_debut_season'].isnull(), x['season'], x['mlb_debut_season']),
             pitcher_years_in_league = lambda x: x['season'].astype(int) - x['mlb_debut_season'].astype(int))
)

pitcher_stats_complete[['season','pitcher_years_in_league','mlb_debut_season']]

first_year_pitcher_season_num_feats_median

pitcher_stats_complete[pitcher_stats_complete['prev_season_stats_avg'].isnull()]
pitcher_stats_complete.columns[pitcher_stats_complete.isnull().sum()>0].tolist()



pitcher_stats_complete['mlb_debut_season'].isnull().sum()

pitcher_stats_complete[pitcher_stats_complete['mlb_debut_season'].isnull()][['mlb_debut_season', 'season']]
pitcher_stats_complete.columns.tolist()


# dedupe check
pitcher_main_rolling_stats[['teamId_n_season_pitcher', 'officialDate_n_Time', 'personId_n_season_pitcher']].value_counts()


del pitcher_boxscore, pitcher_boxscore_processed, pitcher_boxscore_reg_season, pitcher_team_gamepks, col, drop_cols



# what i'm trying tobuild to is pitcher + last season stats + rolling season stats (short and whole season)
# MEDIAN FILL FOR FIRST SEASON OR JUST PUT TO 0?











# BATTER FEATS  --------------------------------------------------------------------------------------------------------------------

batter_boxscore_processed = ProcessData().process_batter_boxscore(batter_boxscore)

num_batter_cols = [
    'ab', 'r', 'h', 'doubles', 'triples', 'hr', 'rbi', 'sb', 'bb',
    'k', 'lob', 'avg', 'ops', 'obp', 'slg', 'battingOrder'
]

batter_cols_to_remove = ['note', 'namefield', 'person_id'] # person_id won't need to be removed later this is because we pulled it in wrong

batter_boxscore_processed = batter_boxscore_processed.drop(batter_cols_to_remove, axis=1)
batter_boxscore_processed[num_batter_cols] = batter_boxscore_processed[num_batter_cols].astype(float)
batter_boxscore_processed.columns = ['batter_' + col  if col in num_batter_cols else col for col in batter_boxscore_processed.columns]

batter_boxscore_processed = (
    batter_boxscore_processed
    .assign(
        gamepk = lambda x: x['gamepk'].astype(str),
        teamname = lambda x: x['teamname'].str.replace('Indians','Guardians').str.replace('Devil Rays', 'Rays').str.replace('Diamondbacks','D-backs'),
        batting_order = lambda x: x['batting_order'].str.replace(r'^\s*$', '', regex=True).replace('', np.nan).astype(float))
    .merge(team_lookup, on='teamname', how='left')
    .drop_duplicates()
)



batter_stats_season_df_a = wr.s3.read_parquet(path='s3://mlbdk-model/features/batter/season_stats/batter_season_stats_to_2023_part_a.parquet')
batter_stats_season_df_b = wr.s3.read_parquet(path='s3://mlbdk-model/features/batter/season_stats/batter_season_stats_to_2023_part_b.parquet')


batter_stats_season = pd.concat([batter_stats_season_df_a, batter_stats_season_df_b])


# need to dedupe based on at bats since recent years (2022 + ) the way we're pulling in data is pulling two different values for players

batter_boxscore_processed = (
    batter_boxscore_processed.
    sort_values(by='batter_ab')
    .drop_duplicates(subset=['personId', 'gamepk'], keep='last')
)

batter_reg_season_df = pd.merge(reg_season_games_processed[['gamepk', 'team_id_batter', 'officialDate_n_Time', 'season']], batter_boxscore_processed, left_on=['gamepk', 'team_id_batter'], right_on=['gamepk', 'team_id'], how='left')


batter_reg_season_df = (
    batter_reg_season_df
    .assign(
        personId_n_season_batter = lambda x: x['personId'] + '_' + x['season'],
        team_id_n_season_batter = lambda x: x['team_id'] + '_' + x['season'],
    )
)


batter_reg_season_starters = (
    batter_reg_season_df[batter_reg_season_df['batter_ab']>0]
    .dropna(subset='batting_order')
)


# BATTER ROLLING STATS ----------------------------------------------------------------------------

rel_num_cols_batter = ['batter_h', 'batter_ab', 'batter_obp', 'batter_hr', 'batter_rbi', 'batter_sb', 'batter_lob', 'batter_r', 'batter_doubles', 'batter_triples', 'batter_bb',
                'batter_k', 'batter_slg', 'batter_avg', 'batter_ops', 'batter_battingOrder']


batter_rolling_stats_short = (
    batter_reg_season_starters[['personId_n_season_batter','officialDate_n_Time'] + rel_num_cols_batter]
    .sort_values(by=['personId_n_season_batter','officialDate_n_Time'])
    .set_index(['personId_n_season_batter',  'officialDate_n_Time'])
)


kwargs = {
    "lag_feature": {
        "lag": [1,2,3,4,5,6],
        "mean": [[1,2], [1, 3], [1, 4], [1, 5], [1, 6]],
        "median": [[1,2], [1, 3], [1, 4], [1, 5], [1, 6]],
        "std": [[1, 2],[1, 3],[1, 4],[1, 5],[1, 6]],
        "sum": [[1,2], [1, 3], [1, 4], [1, 5], [1, 6]]

    }
}


transformer = WindowSummarizer(**kwargs, target_cols=rel_num_cols_batter)
batter_rolling_stats_short = transformer.fit_transform(batter_rolling_stats_short)

rel_num_cols_batter_transformed = batter_rolling_stats_short.columns.tolist()
batter_rolling_stats_short = batter_rolling_stats_short.reset_index()



batter_stats_complete = pd.merge(batter_reg_season_starters, batter_rolling_stats_short, on=['personId_n_season_batter', 'officialDate_n_Time'], how='left')




del batter_reg_season_starters,  batter_boxscore_processed, batter_boxscore, batter_reg_season_df, batter_rolling_stats_short







# TEAM FEATS ---------------------------------------------------------------------------------------------------------------------------



# PTICHER TEAM ROLLING STATS -------------------------------------------------

team_pitcher_boxscore = (
    pitcher_boxscore_reg_season
    .groupby(['teamId_n_season_pitcher', 'gamepk', 'officialDate_n_Time'], as_index=False)
    .agg(
         pitcher_team_ip = ('pitcher_ip', sum),
         pitcher_team_h = ('pitcher_h', sum),
         pitcher_team_r = ('pitcher_r', sum),
         pitcher_team_er = ('pitcher_er', sum),
         pitcher_team_bb = ('pitcher_bb', sum),
         pitcher_team_k = ('pitcher_k', sum),
         pitcher_team_hr = ('pitcher_hr', sum),
         pitcher_team_p = ('pitcher_p', sum),
         pitcher_team_s = ('pitcher_s', sum))
    .assign(
        pitcher_team_ip_adjusted = lambda x: x['pitcher_team_ip'].round(),
        pitcher_team_whip = lambda x: (x['pitcher_team_h'] + x['pitcher_team_bb'])/x['pitcher_team_ip_adjusted'],
        pitcher_team_hit_ip = lambda x: x['pitcher_team_h']/x['pitcher_team_ip_adjusted'],
        pitcher_team_walk_ip = lambda x: x['pitcher_team_bb']/x['pitcher_team_ip_adjusted'],
        pitcher_team_hr_ip = lambda x: x['pitcher_team_hr']/x['pitcher_team_ip_adjusted'],
        pitcher_team_strikeout_ip = lambda x: x['pitcher_team_k']/x['pitcher_team_ip_adjusted'],
        pitcher_er_ip = lambda x: x['pitcher_team_er']/x['pitcher_team_ip_adjusted']
    )
    .reset_index(drop=True)
)


rel_num_cols_pitcher_team = ['pitcher_team_ip_adjusted','pitcher_team_h', 'pitcher_team_r', 'pitcher_team_er', 'pitcher_team_bb','pitcher_team_k', 'pitcher_team_hr', 
                        'pitcher_team_p', 'pitcher_team_s']


team_pitcher_rolling_stats_short = (
    team_pitcher_boxscore[['teamId_n_season_pitcher','officialDate_n_Time'] + rel_num_cols_pitcher_team]
    .sort_values(by=['teamId_n_season_pitcher','officialDate_n_Time'])
    .set_index(['teamId_n_season_pitcher',  'officialDate_n_Time'])
)

kwargs = {
    "lag_feature": {
        "lag": [1,2,3,4,5,6],
        "mean": [[1,2], [1, 3], [1, 4], [1, 5], [1, 6]],
        "median": [[1,2], [1, 3], [1, 4], [1, 5], [1, 6]],
        "std": [[1, 2],[1, 3],[1, 4],[1, 5],[1, 6]],
        "sum": [[1,2], [1, 3], [1, 4], [1, 5], [1, 6]]

    }
}


transformer = WindowSummarizer(**kwargs, target_cols=rel_num_cols_pitcher_team)
team_pitcher_rolling_stats_short = transformer.fit_transform(team_pitcher_rolling_stats_short)
rel_num_cols_pitcher_team_transformed = team_pitcher_rolling_stats_short.columns.tolist()

team_pitcher_rolling_stats_short = team_pitcher_rolling_stats_short.reset_index()



# BATTER TEAM ROLLING STATS -------------------------------------------------

rel_num_cols_team_off = ['team_groundOuts', 'team_runs', 'team_doubles', 'team_triples', 'team_homeRuns', 'team_strikeOuts', 'team_baseOnBalls',
 'team_intentionalWalks', 'team_hits', 'team_hitByPitch', 'team_atBats', 'team_obp', 'team_caughtStealing', 'team_stolenBases', 'team_sacBunts',
 'team_sacFlies', 'team_avg','team_slg','team_ops','team_groundIntoDoublePlay','team_plateAppearances','team_totalBases','team_leftOnBase']


team_off_rolling_stats_short = (reg_season_games_processed[['team_id_batter', 'season', 'officialDate_n_Time'] + rel_num_cols_team_off]
                       .drop_duplicates()
                       .assign(team_id_n_season_batter = lambda x: x['team_id_batter'] + '_' + x['season'])
                       .drop(['team_id_batter', 'season'], axis=1)
                       .sort_values(['team_id_n_season_batter', 'officialDate_n_Time'])
                       .set_index(['team_id_n_season_batter', 'officialDate_n_Time'])
)


transformer = WindowSummarizer(**kwargs, target_cols=rel_num_cols_team_off)
team_off_rolling_stats_short = transformer.fit_transform(team_off_rolling_stats_short)
rel_num_cols_team_off_transformed = team_off_rolling_stats_short.columns.tolist()

team_off_rolling_stats_short = team_off_rolling_stats_short.reset_index()




# MERGE AND CREATE FINAL DATAFRAME --------------------------------------------------------------------------------------------------

pitcher_stats_complete = pd.merge(pitcher_stats_complete, team_pitcher_rolling_stats_short, on=['teamId_n_season_pitcher', 'officialDate_n_Time'], how='left')

batter_stats_complete = pd.merge(batter_stats_complete, team_off_rolling_stats_short, on=['team_id_n_season_batter', 'officialDate_n_Time'],  how='left')



batter_train = pd.merge(batter_stats_complete, reg_season_games_processed[['gamepk', 'team_id_batter', 'team_id_pitcher', 'home_visitor']], on=['gamepk', 'team_id_batter'], how='left')
batter_train = pd.merge(batter_train, pitcher_stats_complete, on=['gamepk', 'team_id_pitcher'], how='left')
batter_train = batter_train.dropna(subset='pitcher_team_s_sum_1_6') # check this later


batter_train = batter_train.dropna(subset='batter_h_lag_6') # we lose observations here for some teams so we don't have opposing pitchers

batter_train.isnull().sum()

batter_train.shape[0]



batter_train[batter_train['pitcher_s_median_1_3'].isnull()]




batter_train['batter_at_least_one_hit'] = np.where(batter_train['batter_h']>=1, True, False)

del batter_stats_complete, pitcher_stats_complete



# TEAM HOME /AWAY VISISTOR STATS FROM LAST YEAR AVERAGES, ROLLING AVERAGES 


# TRAIN MODEL -----------------------------------------------------------------------------------------
# CHANGE TO 2015 
#, 'home_visitor'

rel_cols = ['batting_order', 'season_x'] + ['batter_at_least_one_hit'] + rel_num_cols_batter_transformed + pitcher_main_rolling_stats_feats + first_year_pitcher_num_cols + rel_num_cols_team_off_transformed +  rel_num_cols_pitcher_team_transformed 

id_cols_to_drop = ['personId_n_season']

train = batter_train[rel_cols]
train['season_x'] = train['season_x'].astype(int)

classifier = RandomForestClassifier(random_state=24, n_jobs=-1, max_depth=3)


train, test = train[train['season_x']<=2013], train[train['season_x']>2013]

X_train, y_train = train.drop('batter_at_least_one_hit', axis=1),  train['batter_at_least_one_hit']
X_test, y_test = test.drop('batter_at_least_one_hit', axis=1),  test['batter_at_least_one_hit']

num_feats = X_train.select_dtypes(np.number).columns.tolist()
cat_feats =  X_train.select_dtypes(object).columns.tolist()





categorical_transformer = OneHotEncoder()

# Combine preprocessing steps for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_feats),
        ('cat', categorical_transformer, cat_feats)
    ])


X_train_transfomred = preprocessor.fit_transform(X_train)
classifier.fit(X_train_transfomred, y_train)


y_train_pred = classifier.predict(X_train_transfomred)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)


X_test_transformed = preprocessor.transform(X_test)
y_test_pred = classifier.predict(X_test_transformed)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)


# train eval metrics
print("train accuracy:", train_accuracy)
print("train precision:", train_precision)
print("train recall:", train_recall)

# test eval metrics
print("test accuracy:", test_accuracy)
print("test precision:", test_precision)
print("test recall:", test_recall)



# BASELINE OF SIMPLE TREE W MAX DEPTH OF 3
train accuracy: 0.657818636605435
train precision: 0.6675550337175676
train recall: 0.9392350185301727
test accuracy: 0.6447045304619837
test precision: 0.652483264146004
test recall: 0.9385145039605158


perm_importance = permutation_importance(classifier, X_test_transformed, y_test, n_repeats=3, random_state=42)

print('Permutation importance: ', perm_importance.importances_mean)
print('permutation importance std: ', perm_importance.importances_std )



# ROLLING AVERAGE OVER A GIVEN WINDOW 
# SPREAD TOO OF DIFFERENCE BETWEEN CURRENT PERFORMANCE VERSUS HISTORICAL TREND
# GAME IN SERIES, if 2 GAME WHAT HAPPEND IN LAST GAME OF SERIES HOW DO YOU NA FILL?
# THIS WHOLE TIME WE'VE BEEN ASSUMING INDEPENDENCE I THINK WE SHOULD THINK ABOUT HOW IF THINGS ARE NOT INDEPENEDENT


# how would we could this dependence acrsos trials 
## the easy way to go bout it would be alag of the last event, 

### in seeing trends and dependence you're assuming a function that in part is explained by a portion fo a trend 
## that's the cool thing about it s we're not assumign independence or dependence
## it is all dependent really, not causally so but probabilyt so since all facotrs are historical

# we'll add pitcher defense next -> then team defense (like pace of game)



# PITCHER (DEFENSE) + TEAM DEFENSE AND OPPOSING TEAM OFFENSE 











# PLOTTING ---------------------------------------------------------------------

batter_reg_season_starters['batter_at_least_one_hit'].value_counts(normalize=True)


batter_starters_hit_pct = batter_reg_season_starters.groupby(['batting_order'], as_index=False)['batter_at_least_one_hit'].value_counts(normalize=True)

px.line(batter_starters_hit_pct[batter_starters_hit_pct['batter_at_least_one_hit']==True], x='batting_order', y='proportion')
























# APPENDIX -------------------------------------------------------------------------------------------------------------------------


# PITCHER STARTER STATS (ASSUME MOST pitches IS STARTER) ----------------------------------------------------------

main_pitcher_idx = pitcher_boxscore_reg_season.groupby(['gamepk', 'team_id_pitcher'])['pitcher_p'].transform(max) == pitcher_boxscore_reg_season['pitcher_p']
pitcher_main_rolling_stats_short = pitcher_boxscore_reg_season[main_pitcher_idx].reset_index(drop=True)


rel_num_cols_pitcher = ['pitcher_ip','pitcher_h', 'pitcher_r', 'pitcher_er', 'pitcher_bb','pitcher_k', 'pitcher_hr',  
                        'pitcher_p', 'pitcher_s']


pitcher_main_rolling_stats_short = (
    pitcher_main_rolling_stats_short[['personId_n_season_pitcher', 'teamId_n_season_pitcher', 'officialDate_n_Time'] + rel_num_cols_pitcher]
    .sort_values(by=['personId_n_season_pitcher','officialDate_n_Time'])
    .set_index(['personId_n_season_pitcher', 'teamId_n_season_pitcher',  'officialDate_n_Time'])
)

kwargs = {
    "lag_feature": {
        "lag": [1,2,3],
        "mean": [[1,2], [1, 3]],
        "median": [[1,2], [1, 3]],
        "std": [[1, 2],[1, 3]],
        "sum": [[1,2], [1, 3]]

    }
}


transformer = WindowSummarizer(**kwargs, target_cols=rel_num_cols_pitcher)
pitcher_main_rolling_stats_short = transformer.fit_transform(pitcher_main_rolling_stats_short)
rel_num_cols_pitcher_transformed = pitcher_main_rolling_stats_short.columns.tolist()

pitcher_main_rolling_stats_short = pitcher_main_rolling_stats_short.reset_index()











# BASIC PROCESSING OF ROLLING AVERAGES -----------------------------------------------

# BUILD BASELINE MODEL -----------------------------------------------------


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







