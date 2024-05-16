import boto3
from datetime import datetime, timedelta
import awswrangler as wr
import pandas as pd
import logging
from datetime import datetime
from functools import wraps
from time import time, sleep


def timing(f):
    @wraps(f)
    def wrap(*arg,**kw):

        time_start = time()
        print(f"Starting function: {f.__name__}..........")

        result = f (*arg, **kw)
        time_end = time()
        
        print(f"Completed function: {f.__name__} in {round((time_end-time_start)/60,2)} minutes")

        return result
    return wrap
    

# BREAK UP GET DATA FUNCTIONS AND HOW LONG I READ FOR, I GUESS I CAN OFFER OPTIOJN FOR SEASON OR NOT
# CHANGE GET DATA FUNCION


class GetData():

    def __init__(self) -> None:
        
        s3 = boto3.resource('s3')
        mlb_bucket = s3.Bucket('mlbdk-model')

        pass
        

    @timing
    def get_season_playoff_games(self):

        season_playoff_game_dfs = wr.s3.read_parquet(path='s3://mlbdk-model/season_playoff_game_details/', chunked=True)

        season_playoff_game_list = []

        for df in season_playoff_game_dfs:

            df_adjusted = df.astype(str)
            season_playoff_game_list.append(df_adjusted)

        season_playoff_game_complete = pd.concat(season_playoff_game_list)

        return season_playoff_game_complete
    
    @timing
    def get_gamebox_summary(self):
        
        gamebox_summary_dfs = wr.s3.read_parquet(path='s3://mlbdk-model/gamebox_summary/', chunked=True)

        gamebox_summary_list = []

        for df in gamebox_summary_dfs:

            df_adjusted = df.astype(str)
            gamebox_summary_list.append(df_adjusted)

        gamebox_summary_complete = pd.concat(gamebox_summary_list)

        return gamebox_summary_complete
    
    @timing
    def get_batter_boxscore(self):

        batter_boxscore_dfs = wr.s3.read_parquet(path='s3://mlbdk-model/batter_boxscore_stats/', chunked=True)

        batter_boxscore_list = []

        for df in batter_boxscore_dfs:

            df_adjusted = df.astype(str)
            batter_boxscore_list.append(df_adjusted)

        batter_boxscore_complete = pd.concat(batter_boxscore_list)

        return batter_boxscore_complete
          
    @timing
    def get_pitcher_boxscore(self):

        pitcher_boxscore_dfs = wr.s3.read_parquet(path='s3://mlbdk-model/pitcher_boxscore_stats/', chunked=True)

        pitcher_boxscore_list = []

        for df in pitcher_boxscore_dfs:

            df_adjusted = df.astype(str)
            pitcher_boxscore_list.append(df_adjusted)

        pitcher_boxscore_complete = pd.concat(pitcher_boxscore_list)

        return pitcher_boxscore_complete
    
    
          
    
class ProcessData():

    def __init__(self) -> None:
        pass
        
    @timing
    def process_game_df(self, game_df):
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
        game_home_processed['home_visitor'] = 'home'

        game_away_filtered = game_df[game_away_rel_cols]
        game_away_filtered.columns = game_away_filtered.columns.str.replace('away_', '')
        game_away_processed = game_base_processed.merge(game_away_filtered, on=['pk']).reset_index(drop=True)
        game_away_processed['home_visitor'] = 'visitor'

        game_combined_processed = pd.concat([game_home_processed, game_away_processed], ignore_index=True)
        game_combined_processed = game_combined_processed.rename({'pk': 'gamepk', 'type': 'game_type', 'id':'team_id', 'name': 'team_name'}, axis=1)

        return game_combined_processed



    @timing
    def process_gamebox_df(self, gamebox_df):

        gamebox_winning_pitcher = gamebox_df[gamebox_df['label'] == 'WP'].reset_index(drop=True)
        gamebox_pitches_strikes = gamebox_df[gamebox_df['label'] == 'Pitches-strikes'].reset_index(drop=True)
        gamebox_batters_faced = gamebox_df[gamebox_df['label'] == 'Batters faced'].reset_index(drop=True)

        ## process pitchers and strikes -------------
        gamebox_values_ps = gamebox_pitches_strikes['value'].str.split(";", expand=True)
        pitches_strikes_base = gamebox_pitches_strikes.drop(['value'], axis=1)
        pitches_strikes_processed = pitches_strikes_base.join(gamebox_values_ps)
        pitches_strikes_processed = pitches_strikes_processed.melt(id_vars=['gamepk', 'label']) 

        pitches_strikes_processed = pitches_strikes_processed.dropna(subset=['value']).reset_index(drop=True)

        pitches_strikes_values = pitches_strikes_processed['value'].str.split('(\d+)', expand=True)
        pitches_strikes_values.columns = ['pitcher_name', 'pitches', 'remove_a', 'strikes', 'remove_b']
        pitches_strikes_values = pitches_strikes_values[['pitcher_name', 'pitches', 'strikes']]

        pitches_strikes_processed.drop(['value', 'variable', 'label'], axis=1, inplace=True)
        pitches_strikes_processed = pitches_strikes_processed.join(pitches_strikes_values)
        pitches_strikes_processed['pitcher_name'] = pitches_strikes_processed['pitcher_name'].str.strip()


        ## process batters faced ---------------------------
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



        ## process gamebox ------------------------------------------
        gamebox_winning_pitcher['value'] = gamebox_winning_pitcher['value'].str.replace('.', '')

        winning_pitcher_names = gamebox_winning_pitcher['value'].str.split(';', expand=True)
        gamebox_winning_pitcher_processed = gamebox_winning_pitcher.join(winning_pitcher_names)
        gamebox_winning_pitcher_processed.drop(['label', 'value'], axis=1, inplace=True)
        gamebox_winning_pitcher_processed = gamebox_winning_pitcher_processed.melt(id_vars=['gamepk'], value_name='pitcher_name')
        gamebox_winning_pitcher_processed.drop(['variable'], axis=1, inplace=True)
        gamebox_winning_pitcher_processed.dropna(subset=['pitcher_name'], inplace=True)
        gamebox_winning_pitcher_processed['pitcher_name'] = gamebox_winning_pitcher_processed['pitcher_name'].str.strip()
        gamebox_winning_pitcher_processed['winning_pitcher'] = True


        ## merge processed df into gamebox 
        gamebox_filtered_processed = (
            batters_faced_processed
            .merge(pitches_strikes_processed, on=['gamepk', 'pitcher_name'])
            .merge(gamebox_winning_pitcher_processed, on=['pitcher_name', 'gamepk'], how='left')
        )

        gamebox_filtered_processed.fillna({'winning_pitcher': False}, inplace=True)

        return gamebox_filtered_processed

    @timing
    def process_pitcher_boxscore(self, pitcher_boxscore_df): 

        pitcher_boxscore_processed = pitcher_boxscore_df[pitcher_boxscore_df['personId']!='0']
        pitcher_boxscore_processed['teamname'] = pitcher_boxscore_processed['teamname'].str.replace(' Pitchers', '')
        pitcher_boxscore_processed.drop(['namefield'], axis=1, inplace=True)
        pitcher_boxscore_processed.rename({'name':'pitcher_name'}, axis=1, inplace=True)

        return pitcher_boxscore_processed

    @timing
    def process_batter_boxscore(self, batter_boxscore_df):

        batter_boxscore_processed = batter_boxscore_df[~batter_boxscore_df['namefield'].str.contains('Batters')]

        batter_boxscore_processed = (
            batter_boxscore_processed
            .assign(
                gamepk = lambda x: x['gamepk'].apply(lambda y: int(float(y))),
                batting_order =  lambda x: x['namefield'].str[0],
                teamname = lambda x: x['teamname'].str.replace(' Batters', '')
            )
        )

        return batter_boxscore_processed

    @timing
    def create_pitcher_df(self, game_processed_df, gamebox_processed_df, pitcher_boxscore_df):

        pitcher_final_df = (
            game_processed_df
            .merge(pitcher_boxscore_df, on=['gamepk', 'team_id'], how='left')
        )

        return pitcher_final_df

    @timing
    def create_batter_df(self, game_processed_df, batter_boxscore_processed):

        batter_final_df = (
            game_processed_df
            .merge(batter_boxscore_processed, on=['gamepk', 'team_id'], how='left')
        )

        return batter_final_df