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
        'away_sacFlies', 'away_flyOuts', 'away_avg', 'away_slg', 'away_ops',
        'away_groundIntoDoublePlay', 'away_plateAppearances', 'away_totalBases', 'away_leftOnBase', 'away_atBatsPerHomeRun']

    game_home_rel_cols = ['pk', 'home_id', 'home_name', 'home_leaguename', 
        'home_divisionname', 'home_gamesplayed', 'home_wins', 'home_losses', 'home_ties', 'home_pct',
        'home_flyOuts', 'home_groundOuts', 'home_runs', 'home_doubles', 'home_triples',
        'home_homeRuns', 'home_strikeOuts', 'home_baseOnBalls', 'home_intentionalWalks', 'home_hits',
        'home_hitByPitch', 'home_atBats', 'home_obp', 'home_caughtStealing', 'home_stolenBases', 'home_sacBunts',
        'home_sacFlies',  'home_flyOuts', 'home_avg', 'home_slg', 'home_ops',
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

def lag_columns(df, cols, lag):
    """
    Create lagged versions of selected columns in a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    cols : list of str
        Columns to create lags for
    lag : int
        Number of lags to create

    Returns
    -------
    pandas.DataFrame
        DataFrame with lagged columns added
    """
    lagged = df.groupby(['personId', 'pitcher_name'])[cols].shift(periods=lag, fill_value=np.nan)
    lagged.columns = [f"{col}_lag{lag}" for col in cols]

    df_lagged = pd.concat([df, lagged], axis=1)
    df_lagged = df_lagged.drop(cols, axis=1)
    df_lagged = df_lagged.dropna()

    return df_lagged


def rolling_summary_stats(df, group_col, time_col, window, stats={'min', 'max', 'std', 'mean', 'median', 'sum'}, window_type='observations', cols_to_exclude=None):
    """
    Calculate rolling summary statistics over a specified window for each group in a DataFrame.
    
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
        lag_cols.remove(cols_to_exclude)
    filtered_df = df[[group_col] + lag_cols]
    
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
    summary = summary.rename(columns={'level_1': 'orig_index'})

    df = df.reset_index()
    df = df.rename(columns={'index':'orig_index'})

    # Merge summary statistics back into original DataFrame
    df = df.merge(summary, on=[group_col, 'orig_index'], how='left')
    df = df.drop(columns='orig_index')
    
    return df




# pitcher -------------------------------------------


team_lookup_df = team_lookup_df.rename(columns={'id':'team_id'})
team_lookup_df = team_lookup_df[['team_id', 'teamName']]



game_processed = process_game_df(season_playoff_game)
game_processed = pd.merge(game_processed, team_lookup_df, on='team_id', how='left')




# gamebox_pitcher_processed = process_pitcher_gamebox(gamebox_summary) unable to join at pitcher_name, have to get batters_faced elsewher
pitcher_boxscore_processed = process_pitcher_boxscore(pitcher_boxscore)

game_processed['teamName'] = game_processed['teamName'].astype(str)

team_mapping = {'Indians': 'Guardians',
           'Montreal Expos': 'Nationals',
           'Expos':'Nationals',
           'Devil Rays': 'Rays',
           'Diamondbacks': 'D-backs'}

pitcher_boxscore_processed['teamname'] = pitcher_boxscore_processed['teamname'].astype(str)
pitcher_boxscore_processed['teamname'] = pitcher_boxscore_processed['teamname'].replace(team_mapping)

pitcher_df = create_pitcher_df(game_processed, pitcher_boxscore_processed)

pitcher_df['ip'] = pitcher_df['ip'].astype(float)
pitcher_df = pitcher_df[(pitcher_df['note'] !='') & (pitcher_df['ip'] > 1)] # filter down to pitchers that have a note = starting/close


rel_num_cols_player = ['ip', 'h', 'r', 'er', 'bb', 'k', 'hr', 'era', 'p', 's']
rel_num_cols_team = ['gamesplayed', 'wins', 'losses', 'ties', 'pct',
       'flyOuts', 'groundOuts', 'runs', 'doubles', 'triples', 'homeRuns',
       'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits', 'hitByPitch',
       'atBats', 'obp', 'caughtStealing', 'stolenBases', 'sacBunts',
       'sacFlies', 'flyOuts', 'avg', 'slg', 'ops', 'groundIntoDoublePlay',
       'plateAppearances', 'totalBases', 'leftOnBase', 'atBatsPerHomeRun',
       'teamName']


pitcher_df[rel_num_cols_player] = pitcher_df[rel_num_cols_player].astype(float)

def calculate_pitcher_fp(df):
    df['ip_round'] = df['ip'].round()
    df['out'] = df['ip_round']*3
    df['fp'] = df['ip_round']*2.25 + df['out']*0.75 + df['er']*-2 + df['h']*-0.6 

    df = df.drop(['ip_round'],axis=1)

    return df

pitcher_df = calculate_pitcher_fp(pitcher_df)


# create lagged stats 
pitcher_df_processed = lag_columns(pitcher_df, rel_num_cols_player + rel_num_cols_team, 1)

pitcher_df_processed = pitcher_df_processed.reset_index(drop=True)

cols_to_exclude = ['gamesplayed_lag1']
pitcher_df_processed = rolling_summary_stats(pitcher_df_processed, group_col='personId', time_col='dateTime', window=30, window_type='observations', cols_to_exclude=cols_to_exclude)



rays_test = game_processed[(game_processed['season']=='2016') & (game_processed['teamName']=='Rays')].sort_values('dateTime')

# split training and test 

pitcher_train = pitcher_df_processed[pitcher_df_processed['season'].astype(int) <= 2017]
pitcher_test = pitcher_df_processed[pitcher_df_processed['season'].astype(int) > 2017]

pitcher_train


pitcher_train['ip_lag1'].round()








# batter stats 
single, double, triple, homerun, rbi, run, walk



batter_boxscore_processed = process_batter_boxscore(batter_boxscore_stats)
batter_boxscore_processed.columns

batter_df = create_batter_df(game_processed, batter_boxscore_processed)
batter_df = batter_df[batter_df['ab'] > 0 ] # filter out batters that didn't play


pitcher_df

pitcher_df[pitcher_df['gamepk']==345633]
batter_df[batter_df['gamepk']==345633]