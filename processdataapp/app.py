import boto3
from datetime import datetime, timedelta
import awswrangler as wr
import pandas as pd
import logging
from datetime import datetime
from functools import wraps
from time import time, sleep
from module import GetData #FIGURE OUT IF THIS WORKS? 

game, gamebox, batter_boxscore, pitcher_boxscore = get_rel_dfs()
game_processed = process_game_df(game)
gamebox_processed = process_gamebox_df(gamebox)

pitcher_boxscore_processed = process_pitcher_boxscore(pitcher_boxscore)
batter_boxscore_processed = process_batter_boxscore(batter_boxscore)

pitcher_df = create_pitcher_df(game_processed, gamebox_processed, pitcher_boxscore_processed)

