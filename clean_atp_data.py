from constants import MATCH_DF_PATH
from constants import PLAYER_DF_PATH   
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
PLAYER1_SERVE_WON="PlayerTeam1.Sets[0].Stats.PointStats.TotalServicePointsWon.Dividend"
PLAYER1_SERVE_TOTAL="PlayerTeam1.Sets[0].Stats.PointStats.TotalServicePointsWon.Divisor"

PLAYER2_SERVE_WON="PlayerTeam2.Sets[0].Stats.PointStats.TotalServicePointsWon.Dividend"
PLAYER2_SERVE_TOTAL="PlayerTeam2.Sets[0].Stats.PointStats.TotalServicePointsWon.Divisor"

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

def retrieve_player_stats(match_df, num_years):
    date_column = "StartDate"
    # Sort by tourney_date
    match_df = match_df.sort_values(date_column).reset_index(drop=True)

    # Create a long-format DataFrame for easier aggregation
    print("CREATING PLAYER STATS DATAFRAME")
    player_stats = pd.DataFrame({
        "player_id": match_df["PlayerTeam1.PlayerId"].tolist() + match_df["PlayerTeam2.PlayerId"].tolist(),
        "match_id": match_df["match_id"].tolist() + match_df["match_id"].tolist(),  # Use match_id for merging
        date_column: match_df[date_column].tolist() + match_df[date_column].tolist(),
        "serve_pct": match_df["w_serve_pct"].tolist() + match_df["l_serve_pct"].tolist(),
        "bp_saved_pct": match_df["w_bp_save_pct"].tolist() + match_df["l_bp_save_pct"].tolist(),
        "is_winner": [1] * len(match_df) + [0] * len(match_df)
    })

    # Sort player_stats by player_id and tourney_date
    print("SORTING PLAYER STATS")
    player_stats = player_stats.sort_values(["player_id", date_column]).reset_index(drop=True)

    # Add columns for the rolling sum and count within the last num_years
    player_stats["rolling_serve_pct_sum"] = 0.0
    player_stats["rolling_match_count"] = 0
    player_stats["matches_won"] = 0
    player_stats["rolling_bp_pct_sum"] = 0.0
    print("SLIDING WINDOW")
    # Use a sliding window to calculate the rolling stats
    for player_id, group in player_stats.groupby("player_id"):
        serve_pct_sum = 0
        match_count = 0
        matches_won = 0
        bp_pct_sum = 0
        j = 0  # Sliding window pointer

        for i in range(len(group)):
            current_date = group.iloc[i][date_column]
            cutoff_date = current_date.replace(year=current_date.year - num_years)

            # Slide the window to exclude matches older than num_years
            while j < i and group.iloc[j][date_column] < cutoff_date:
                serve_pct_sum -= group.iloc[j]["serve_pct"]
                bp_pct_sum -= group.iloc[j]["bp_saved_pct"]
                matches_won -= group.iloc[j]["is_winner"]
                match_count -= 1
                j += 1

            # Assign rolling sum and count (excluding current match)
            player_stats.loc[group.index[i], "rolling_serve_pct_sum"] = serve_pct_sum
            player_stats.loc[group.index[i], "rolling_match_count"] = match_count
            player_stats.loc[group.index[i], "matches_won"] = matches_won
            player_stats.loc[group.index[i], "rolling_bp_pct_sum"] = bp_pct_sum
            # Update rolling sum and count for the next iteration (excluding current match)
            serve_pct_sum += group.iloc[i]["serve_pct"]
            bp_pct_sum += group.iloc[i]["bp_saved_pct"]
            matches_won += group.iloc[i]["is_winner"]
            match_count += 1

    # Calculate rolling averages
    print("ROLLING AVERAGES")
    player_stats["rolling_avg_serve_pct"] = (
        player_stats["rolling_serve_pct_sum"] / player_stats["rolling_match_count"]
    ).fillna(0)
    player_stats["rolling_avg_bp_pct"] = (
        player_stats["rolling_bp_pct_sum"] / player_stats["rolling_match_count"]
    ).fillna(0)
    new_stat_columns = ["rolling_avg_serve_pct", "rolling_avg_bp_pct", "rolling_match_count", "matches_won"]
    return player_stats, new_stat_columns

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
    df["PlayerTeam1.serve_pct"] = df[PLAYER1_SERVE_WON] / df[PLAYER1_SERVE_TOTAL]
    df["PlayerTeam2.serve_pct"] = df[PLAYER2_SERVE_WON] / df[PLAYER2_SERVE_TOTAL]
    df["match_id"] = df["MatchId"] + "-" + df[""]
    print(df["PlayerTeam1.serve_pct"].mean())
    print(df["PlayerTeam2.serve_pct"].mean())
    return df

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
    df = add_features(df)
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