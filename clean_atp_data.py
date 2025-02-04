from constants import MATCH_DF_PATH
from constants import PLAYER_DF_PATH   
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
PLAYER1_SERV1_WON_COL = "PlayerTeam1.Sets[0].Stats.ServiceStats.FirstServe.Dividend"
PLAYER1_SERV1_TOTAL_COL = "PlayerTeam1.Sets[0].Stats.ServiceStats.FirstServe.Divisor"
PLAYER1_SERV2_WON_COL = "PlayerTeam1.Sets[0].Stats.ServiceStats.SecondServe.Dividend"
PLAYER1_SERV2_TOTAL_COL = "PlayerTeam1.Sets[0].Stats.ServiceStats.SecondServe.Divisor"

PLAYER2_SERV1_WON_COL = "PlayerTeam2.Sets[0].Stats.ServiceStats.FirstServe.Dividend"
PLAYER2_SERV1_TOTAL_COL = "PlayerTeam2.Sets[0].Stats.ServiceStats.FirstServe.Divisor"
PLAYER2_SERV2_WON_COL = "PlayerTeam2.Sets[0].Stats.ServiceStats.SecondServe.Dividend"
PLAYER2_SERV2_TOTAL_COL = "PlayerTeam2.Sets[0].Stats.ServiceStats.SecondServe.Divisor"

class PlayerRank:
    """
    'RankDate': '1995-01-09T00:00:00', 'SglRollRank': 2, 'SglRollTie': False, 'SglRollPoints': 0, 'SglRaceRank': 0, 'SglRaceTie': False, 'SglRacePoints': 0, 'DblRollRank': 626, 'DblRollTie': False, 'DblRollPoints'
    """
    def __init__(self, object):
        self.RankDate = object["RankDate"]
        self.SglRollRank = object["SglRollRank"]
        self.SglRollTie = object["SglRollTie"]
        self.SglRollPoints = object["SglRollPoints"]
        self.SglRaceRank = object["SglRaceRank"]
        self.SglRaceTie = object["SglRaceTie"]
        self.SglRacePoints = object["SglRacePoints"]
        self.DblRollRank = object["DblRollRank"]
        self.DblRollTie = object["DblRollTie"]
        self.DblRollPoints = object["DblRollPoints"]
    
    def __str__(self):
        return json.dumps(self.__dict__, indent=4)
def handle_na(my_df):
    my_df = my_df.dropna()
    return my_df

def convert_to_ints(df):
    categorical_columns = ["EventType", "Court"]

    df = pd.get_dummies(df, columns=categorical_columns, dtype=int)
    object_int_columns = [
        "PlayerTeam1.SglRollTie", "PlayerTeam1.SglRaceTie", "PlayerTeam1.DblRollTie",
        "PlayerTeam2.SglRollTie", "PlayerTeam2.SglRaceTie", "PlayerTeam2.DblRollTie"
    ]
    for column in object_int_columns:
        df[column] = df[column].astype(int)
        #print(df[column].unique())
    return df

def pre_drop_cols(my_df):
    cols_to_drop = [
        "IsDoubles", "RoundName", "CourtName", "LastServer", "DateSeq", "IsQualifier", "ScoringSystem", "EntryStatusPlayerTeam", "GamePointsPlayerTeam", "PlayerTeam1.SeedPlayerTeam"
    ]
    for column in my_df.columns:
        for col in  cols_to_drop:
            if(col in column):
                my_df = my_df.drop(columns= [column])
    my_df = my_df[~my_df['Reason'].isin(['RET', 'DEF'])]
    my_df = my_df[my_df['PlayerTeam1.Sets[1].Stats'].isna()]
    my_df = my_df.loc[:, my_df.nunique() > 1]

    return my_df

def get_swap_dictionary(my_df):
    result = {}
    for column in my_df.columns:
        if("PlayerTeam1" in column):
            result[column] = "PlayerTeam2" + column[11:]
        elif("PlayerTeam2" in column):
            result[column] = "PlayerTeam1" + column[11:]
    return result        

def swap_and_add(my_df):
    new_df = my_df.copy(deep=True)
    swap_dict = get_swap_dictionary(my_df)
    new_df = new_df.rename(columns = swap_dict)
    new_df["PlayerTeam1.won"] = 1
    new_df.loc[new_df["PlayerTeam2.won"] == 1, "PlayerTeam1.won"] = 0
    new_df = new_df.drop(columns = ["PlayerTeam2.won"])
    result = pd.concat([my_df, new_df])
    # print(len(result))
    # print(result.isna().sum())
    # print(result["PlayerTeam1.won"].unique())
    return result    

def retrieve_latest_ranking(player_rankings, player_id, current_date):
    if(player_id in player_rankings):  
        player_history = player_rankings[player_id]["History"]
    else:
        #print("player id ", player_id, " not found in player rankings")
        return None
    for item in player_history:
        rank_date = datetime.strptime(item["RankDate"][:10], "%Y-%m-%d")
        if(rank_date < current_date):
            return PlayerRank(item)
    return None

def append_score_columns(player_ranking, prefix, row):
    for item in player_ranking.__dict__:
        row[prefix + item] = player_ranking.__dict__[item]
    return row

def process_row(row, player_rankings):
    """Process a single row using apply() instead of iterrows()."""
    current_start_date = row["StartDate"]
    
    player_1_id = row["PlayerTeam1.PlayerId"]
    player_2_id = row["PlayerTeam2.PlayerId"]

    player_1_ranking = retrieve_latest_ranking(player_rankings, player_1_id, current_start_date)
    player_2_ranking = retrieve_latest_ranking(player_rankings, player_2_id, current_start_date)
    
    
    row_dict = row.to_dict()  # Convert row to dict for modification
    if(not player_1_ranking is None):
        row_dict = append_score_columns(player_1_ranking, "PlayerTeam1.", row_dict)
    if(not player_2_ranking is None):
        row_dict = append_score_columns(player_2_ranking, "PlayerTeam2.", row_dict)

    return row_dict  # Return modified row as a dictionary

def add_rankings_to_dataset(my_df, player_rankings):
    """Optimized function using apply() instead of iterrows()."""
    # Convert StartDate column to datetime for faster comparison
    my_df["StartDate"] = pd.to_datetime(my_df["StartDate"].str[:10], format="%Y-%m-%d")

    # Apply the function to all rows efficiently and collect results
    updated_rows = my_df.apply(lambda row: process_row(row, player_rankings), axis=1, result_type="expand")

    # Convert list of dictionaries to DataFrame
    updated_df = pd.DataFrame(updated_rows)

    #print(list(updated_df.columns))  # Debugging: Print columns
    joblib.dump(updated_df, "./data/atp_with_rankings.pkl")
    return updated_df

def cols_to_remove_before_training(df):
    bad_words = [
        "Sets[", "TournamentName", "Doubles", "Singles", "Date", "EventYear", "EventId", "Round", "Time", "Winner", "Winning", "NumberOfSets", "MatchId", "TournamentCity", "Id", "Name", "Country"
    ]
    orig_columns = list(df.columns)
    for column in orig_columns:
        for word in bad_words:
            if word in column and column in df.columns:
                df = df.drop(columns = column)
    return df
def add_features(df):
    df["PlayerTeam1.serve_pct"] = (df[PLAYER1_SERV1_WON_COL] + df[PLAYER1_SERV2_WON_COL]) / (df[PLAYER1_SERV1_TOTAL_COL] + df[PLAYER1_SERV2_TOTAL_COL])
    df["PlayerTeam2.serve_pct"] = (df["l_1stWon"] + match_df["l_2ndWon"]) / match_df["l_svpt"]
def main():
    write_player_ranking_db = False
    if(write_player_ranking_db):
        my_df = joblib.load("./data/atp_stats.pkl")    
        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns (if needed)
        my_df = pre_drop_cols(my_df)
        player_rankings = joblib.load("./data/player_rankings.pkl")
        add_rankings_to_dataset(my_df, player_rankings)
        return;

    df = joblib.load("./data/atp_with_rankings.pkl")
    pd.set_option('display.max_rows', 500)
    print(df.isna().sum())
    df["PlayerTeam1.won"] = 0  
    df["PlayerTeam1.PlayerId"] = df["PlayerTeam1.PlayerId"].astype(str)
    df["WinningPlayerId"] = df["WinningPlayerId"].astype(str)

    df.loc[df["PlayerTeam1.PlayerId"] == df["WinningPlayerId"], "PlayerTeam1.won"] = 1
    # print(len(df.loc[df["PlayerTeam1.won"] == 1]))
    # print(df[["PlayerTeam1.PlayerId", "PlayerTeam2.PlayerId", "WinningPlayerId", "PlayerTeam1.won"]])
    df = cols_to_remove_before_training(df)
    df = handle_na(df)
    # print(df.isna().sum())
    # print(df.dtypes)
    df = convert_to_ints(df)
    df = swap_and_add(df)
    joblib.dump(df, "./data/preprocessed_df.pkl")

if __name__ == "__main__":
    main();