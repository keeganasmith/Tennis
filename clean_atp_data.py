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

PLAYER1_BP_WON="PlayerTeam1.Sets[0].Stats.ServiceStats.BreakPointsSaved.Dividend"
PLAYER1_BP_TOTAL="PlayerTeam1.Sets[0].Stats.PointStats.TotalPointsWon.Divisor"

PLAYER2_BP_WON="PlayerTeam2.Sets[0].Stats.ServiceStats.BreakPointsSaved.Dividend"
PLAYER2_BP_TOTAL="PlayerTeam2.Sets[0].Stats.PointStats.TotalPointsWon.Divisor"


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
    
def replace_divide_by_zero(df, divisor_col, resulting_col):
    print(f"DIVIDEND MEAN FOR {resulting_col} IS: ", df[resulting_col].median())
    df.loc[df[divisor_col] == 0, resulting_col] = 0
    return df

def fill_na_columns(match_df, columns, value):
    values = {}
    for column in columns:
        values[column] = value
    match_df = match_df.fillna(value = values)
    return match_df

def get_new_column_names(columns, prefix):
    result = []
    for column in columns:
        result.append(prefix + "." + column)
    return result

def compute_surface_stats(match_df, num_years):
    new_df = match_df.copy()
    
    for surface_type, group in match_df.groupby("Court"):
        player_stats, stats_columns = retrieve_player_stats(group, num_years)
        w_prefix = "PlayerTeam1." + surface_type
        l_prefix = "PlayerTeam2." + surface_type

        winner_columns = get_new_column_names(stats_columns, w_prefix)
        loser_columns = get_new_column_names(stats_columns, l_prefix)
        winner_rename_mapping = get_rename_mapping(stats_columns, w_prefix)
        loser_rename_mapping = get_rename_mapping(stats_columns, l_prefix)
        columns_to_merge_on = stats_columns + ["match_id"]
        new_df = merge_dataframes(new_df, player_stats, columns_to_merge_on, winner_rename_mapping, 1)
        new_df = merge_dataframes(new_df, player_stats, columns_to_merge_on, loser_rename_mapping, 0)
        new_df = fill_na_columns(new_df, winner_columns + loser_columns, 0)
        
    match_df = new_df
    return match_df

def retrieve_player_stats(match_df, num_years):
    date_column = "StartDate"
    
    # Convert StartDate to datetime format (important for rolling calculations)
    match_df[date_column] = pd.to_datetime(match_df[date_column], errors="coerce")

    # Drop rows where StartDate conversion failed (if any)
    match_df = match_df.dropna(subset=[date_column])

    # Sort DataFrame by date
    match_df = match_df.sort_values(date_column).reset_index(drop=True)

    # Create long-format player stats DataFrame
    print("CREATING PLAYER STATS DATAFRAME")
    player_stats = pd.DataFrame({
        "player_id": match_df["PlayerTeam1.PlayerId"].tolist() + match_df["PlayerTeam2.PlayerId"].tolist(),
        "match_id": match_df["match_id"].tolist() + match_df["match_id"].tolist(),
        date_column: match_df[date_column].tolist() + match_df[date_column].tolist(),
        "serve_pct": match_df["PlayerTeam1.serve_pct1"].tolist() + match_df["PlayerTeam2.serve_pct1"].tolist(),
        "bp_saved_pct": match_df["PlayerTeam1.bp_save_pct1"].tolist() + match_df["PlayerTeam2.bp_save_pct1"].tolist(),
        "is_winner": [1] * len(match_df) + [0] * len(match_df)
    })

    # Ensure StartDate is datetime in player_stats too
    player_stats[date_column] = pd.to_datetime(player_stats[date_column], errors="coerce")
    player_stats = player_stats.dropna(subset=[date_column])

    # Sort player_stats by player_id and date
    print("SORTING PLAYER STATS")
    player_stats.sort_values(["player_id", date_column], inplace=True)

    # Rolling calculations using Pandas
    print("ROLLING CALCULATIONS")

    # Group by player_id and calculate rolling statistics
    def rolling_stats(group):
        # Ensure StartDate is a DatetimeIndex
        group = group.set_index(date_column)  # Set StartDate as index

        # Convert years to days (approximate rolling window)
        window_days = num_years * 365

        # **Shift by 1 to exclude the current row from rolling calculations**
        group["rolling_match_count"] = group["is_winner"].shift(1).rolling(f"{window_days}D").count().fillna(0)
        group["rolling_serve_pct_sum"] = group["serve_pct"].shift(1).rolling(f"{window_days}D").sum().fillna(0)
        group["rolling_bp_pct_sum"] = group["bp_saved_pct"].shift(1).rolling(f"{window_days}D").sum().fillna(0)
        group["matches_won"] = group["is_winner"].shift(1).rolling(f"{window_days}D").sum().fillna(0)

        # Rolling averages (dividing by match count, avoiding divide by zero)
        group["rolling_avg_serve_pct"] = (
            group["rolling_serve_pct_sum"] / group["rolling_match_count"]
        ).fillna(0)
        group["rolling_avg_bp_pct"] = (
            group["rolling_bp_pct_sum"] / group["rolling_match_count"]
        ).fillna(0)

        # Reset index back to normal
        group = group.reset_index()

        return group




    # Apply rolling stats to each player_id
    player_stats = player_stats.groupby("player_id", group_keys=False).apply(rolling_stats)

    # Handle divide-by-zero cases
    print("REPLACING NaN VALUES")
    player_stats["rolling_avg_serve_pct"].fillna(0, inplace=True)
    player_stats["rolling_avg_bp_pct"].fillna(0, inplace=True)

    new_stat_columns = ["rolling_avg_serve_pct", "rolling_avg_bp_pct", "rolling_match_count", "matches_won"]

    return player_stats, new_stat_columns

# def retrieve_player_stats(match_df, num_years):
#     date_column = "StartDate"
#     # Sort by tourney_date
#     match_df = match_df.sort_values(date_column).reset_index(drop=True)

#     # Create a long-format DataFrame for easier aggregation
#     print("CREATING PLAYER STATS DATAFRAME")
#     player_stats = pd.DataFrame({
#         "player_id": match_df["PlayerTeam1.PlayerId"].tolist() + match_df["PlayerTeam2.PlayerId"].tolist(),
#         "match_id": match_df["match_id"].tolist() + match_df["match_id"].tolist(),  # Use match_id for merging
#         date_column: match_df[date_column].tolist() + match_df[date_column].tolist(),
#         "serve_pct": match_df["PlayerTeam1.serve_pct1"].tolist() + match_df["PlayerTeam2.serve_pct1"].tolist(),
#         "bp_saved_pct": match_df["PlayerTeam1.bp_save_pct1"].tolist() + match_df["PlayerTeam2.bp_save_pct1"].tolist(),
#         "is_winner": [1] * len(match_df) + [0] * len(match_df)
#     })

#     # Sort player_stats by player_id and tourney_date
#     print("SORTING PLAYER STATS")
#     player_stats = player_stats.sort_values(["player_id", date_column]).reset_index(drop=True)

#     # Add columns for the rolling sum and count within the last num_years
#     player_stats["rolling_serve_pct_sum"] = 0.0
#     player_stats["rolling_match_count"] = 0
#     player_stats["matches_won"] = 0
#     player_stats["rolling_bp_pct_sum"] = 0.0
#     for column in match_df.columns:
#         if("Rank" in column):
#             player_stats[column + "_rolling"] = 0.0
#     print("SLIDING WINDOW")
#     # Use a sliding window to calculate the rolling stats
#     for player_id, group in player_stats.groupby("player_id"):
#         serve_pct_sum = 0
#         match_count = 0
#         matches_won = 0
#         bp_pct_sum = 0
#         rank_sums = {}
        
#         for column in player_stats.columns:
#             if("Rank" in column):
#                 rank_sums[column] = 0.0
                
#         j = 0  # Sliding window pointer

#         for i in range(len(group)):
#             current_date = group.iloc[i][date_column]
#             cutoff_date = current_date.replace(year=current_date.year - num_years)

#             # Slide the window to exclude matches older than num_years
#             while j < i and group.iloc[j][date_column] < cutoff_date:
#                 serve_pct_sum -= group.iloc[j]["serve_pct"]
#                 bp_pct_sum -= group.iloc[j]["bp_saved_pct"]
#                 matches_won -= group.iloc[j]["is_winner"]
#                 for key in list(rank_sums.keys()):
#                     rank_sums[key] -= group.iloc[j][key]
                        
#                 match_count -= 1
#                 j += 1

#             # Assign rolling sum and count (excluding current match)
#             player_stats.loc[group.index[i], "rolling_serve_pct_sum"] = serve_pct_sum
#             player_stats.loc[group.index[i], "rolling_match_count"] = match_count
#             player_stats.loc[group.index[i], "matches_won"] = matches_won
#             player_stats.loc[group.index[i], "rolling_bp_pct_sum"] = bp_pct_sum
#             for key in list(rank_sums.keys()):
#                 player_stats.loc[group.index[i], key] = rank_sums[key]
#             # Update rolling sum and count for the next iteration (excluding current match)
#             serve_pct_sum += group.iloc[i]["serve_pct"]
#             bp_pct_sum += group.iloc[i]["bp_saved_pct"]
#             matches_won += group.iloc[i]["is_winner"]
#             for key in list(rank_sums.keys()):
#                 rank_sums[key] += group.iloc[i][key]
#             match_count += 1

#     # Calculate rolling averages
#     print("ROLLING AVERAGES")
#     player_stats["rolling_avg_serve_pct"] = (
#         player_stats["rolling_serve_pct_sum"] / player_stats["rolling_match_count"]
#     ).fillna(0)
#     player_stats["rolling_avg_bp_pct"] = (
#         player_stats["rolling_bp_pct_sum"] / player_stats["rolling_match_count"]
#     ).fillna(0)
#     for column in match_df.columns:
#         if("Rank" in column):
#             player_stats[column + "_avg"] = (
#                 player_stats[column + "_rolling"] / player_stats["rolling_match_count"]
#             ).fillna(0)
#     print("AVERAGE BP BEFORE REPLACING WITH 0: ", player_stats["rolling_avg_bp_pct"].median())
#     player_stats = replace_divide_by_zero(player_stats, "rolling_match_count", "rolling_avg_serve_pct")
#     player_stats = replace_divide_by_zero(player_stats, "rolling_match_count", "rolling_avg_bp_pct")    
#     new_stat_columns = ["rolling_avg_serve_pct", "rolling_avg_bp_pct", "rolling_match_count", "matches_won"]
#     for column in match_df.columns:
#         if("Rank" in column):
#             new_stat_columns.append(column + "_avg")
#     return player_stats, new_stat_columns

def get_rename_mapping(original_columns, prefix):
    result = {}
    for column in original_columns:
        result[column] = prefix +"." + column
    return result

def merge_dataframes(match_df, other_df, columns_to_merge_on, rename_mapping, is_winner):
    match_df = match_df.merge(
        other_df[other_df["is_winner"] == is_winner][columns_to_merge_on],
        left_on="match_id",
        right_on="match_id",
        how="left",
        suffixes=("", "_winner")
    ).rename(columns=rename_mapping)
    return match_df

    
def add_features(df):
    df["PlayerTeam1.serve_pct1"] = df[PLAYER1_SERVE_WON] / df[PLAYER1_SERVE_TOTAL]
    df["PlayerTeam2.serve_pct1"] = df[PLAYER2_SERVE_WON] / df[PLAYER2_SERVE_TOTAL]
    df["PlayerTeam1.bp_save_pct1"] = df[PLAYER1_BP_WON] / df[PLAYER1_BP_TOTAL]
    df["PlayerTeam2.bp_save_pct1"] = df[PLAYER2_BP_WON] / df[PLAYER2_BP_TOTAL]
    df = replace_divide_by_zero(df, PLAYER1_BP_TOTAL, "PlayerTeam1.bp_save_pct1")
    df = replace_divide_by_zero(df, PLAYER2_BP_TOTAL, "PlayerTeam2.bp_save_pct1")
    df["match_id"] = df["MatchId"].astype(str) + "-" + df["EventId"].astype(str) + "-" + df["EventYear"].astype(str)
    player_stats, new_stats_columns = retrieve_player_stats(df, 3)    
    # Merge rolling averages back into the original DataFrame using match_id
    columns_to_merge_on = new_stats_columns + ["match_id"]
    winner_rename_mapping = get_rename_mapping(new_stats_columns, "PlayerTeam1")
    loser_rename_mapping = get_rename_mapping(new_stats_columns, "PlayerTeam2")
    print("MERGING ROLLING AVERAGES")
    # Merge for winners
    df = merge_dataframes(df, player_stats, columns_to_merge_on, winner_rename_mapping, 1)
    print("winner rename mappings: ", winner_rename_mapping)
    print("loser rename mappings: ", loser_rename_mapping)
    # Merge for losers
    df = merge_dataframes(df, player_stats, columns_to_merge_on, loser_rename_mapping, 0)
    print("AVG SERVE PCT", df["PlayerTeam1.rolling_avg_serve_pct"].median())
    print("AVG SERVE PCT", df["PlayerTeam2.rolling_avg_serve_pct"].median())
    print("AVG BP PCT:", df["PlayerTeam1.rolling_avg_bp_pct"].mean())
    print("AVG BP PCT:", df["PlayerTeam2.rolling_avg_bp_pct"].mean())
    return df

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
        "Sets[", "TournamentName", "Doubles", "Singles", "Date", "EventYear", "EventId", "Round", "Time", "Winner", "Winning", "NumberOfSets", "MatchId", "TournamentCity", "Id", "Name", "Country", "match_id",
        "serve_pct1", "bp_save_pct1"
    ]
    orig_columns = list(df.columns)
    for column in orig_columns:
        for word in bad_words:
            if word in column and column in df.columns:
                df = df.drop(columns = column)
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
    print(df.isna().sum())

    df = add_features(df)
    df = compute_surface_stats(df, 3)
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
    print(df.isna().sum())
    joblib.dump(df, "./data/preprocessed_df.pkl")

if __name__ == "__main__":
    main();