from constants import MATCH_DF_PATH
from constants import PLAYER_DF_PATH   
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.simplefilter(action='ignore', category=Warning)

PLAYER1_SERVE_WON="PlayerTeam1.Sets[0].Stats.PointStats.TotalServicePointsWon.Dividend"
PLAYER1_SERVE_TOTAL="PlayerTeam1.Sets[0].Stats.PointStats.TotalServicePointsWon.Divisor"

PLAYER2_SERVE_WON="PlayerTeam2.Sets[0].Stats.PointStats.TotalServicePointsWon.Dividend"
PLAYER2_SERVE_TOTAL="PlayerTeam2.Sets[0].Stats.PointStats.TotalServicePointsWon.Divisor"

PLAYER1_BP_WON="PlayerTeam1.Sets[0].Stats.ServiceStats.BreakPointsSaved.Dividend"
PLAYER1_BP_TOTAL="PlayerTeam1.Sets[0].Stats.PointStats.TotalPointsWon.Divisor"

PLAYER2_BP_WON="PlayerTeam2.Sets[0].Stats.ServiceStats.BreakPointsSaved.Dividend"
PLAYER2_BP_TOTAL="PlayerTeam2.Sets[0].Stats.PointStats.TotalPointsWon.Divisor"

PLAYER1_BP_CONVERTED="PlayerTeam1.Sets[0].Stats.ReturnStats.BreakPointsConverted.Dividend"
PLAYER1_BP_CONVERTED_TOTAL="PlayerTeam1.Sets[0].Stats.ReturnStats.BreakPointsConverted.Divisor"

PLAYER2_BP_CONVERTED="PlayerTeam2.Sets[0].Stats.ReturnStats.BreakPointsConverted.Dividend"
PLAYER2_BP_CONVERTED_TOTAL="PlayerTeam2.Sets[0].Stats.ReturnStats.BreakPointsConverted.Divisor"

PLAYER1_RETURN = "PlayerTeam1.Sets[0].Stats.PointStats.TotalReturnPointsWon.Dividend"
PLAYER1_RETURN_TOTAL = "PlayerTeam2.Sets[0].Stats.PointStats.TotalReturnPointsWon.Divisor"

PLAYER2_RETURN = "PlayerTeam2.Sets[0].Stats.PointStats.TotalReturnPointsWon.Dividend"
PLAYER2_RETURN_TOTAL = "PlayerTeam2.Sets[0].Stats.PointStats.TotalReturnPointsWon.Divisor"

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

def fill_na_columns(df, columns, value):
    values = {}
    for column in columns:
        values[column] = value
    df = df.fillna(value = values)
    return df

def get_new_column_names(columns, prefix):
    result = []
    for column in columns:
        result.append(prefix + "." + column)
    return result

def compute_surface_stats(df, num_years):
    new_df = df.copy()
    
    for surface_type, group in df.groupby("Court"):
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
        
    df = new_df
    return df

def generate_adjusted_rolling_columns(df, columns):
    for team in ["PlayerTeam1", "PlayerTeam2"]:
        for column in columns:
            df[team + "." + column + ".adjusted"] = df[team + "." + column] * df[team + ".opponent_factor"]
    return df

def compute_rolling(group, col, window_days, func, shift=1):
    """
    Compute a rolling statistic on a column with a shifted window.
    
    Parameters:
        group (DataFrame): The group to operate on (with a DatetimeIndex).
        col (str): The source column name.
        window_days (int): Window size in days.
        func (str): Aggregation function as a string (e.g., 'sum', 'count').
        shift (int): How many periods to shift (default is 1 to exclude current row).
    
    Returns:
        Series: The rolling aggregation.
    """
    return group[col].shift(shift).rolling(f"{window_days}D").agg(func).fillna(0)

def retrieve_player_stats(
    df,
    num_years,
    date_column="StartDate",
    include_adjusted=True,
    # Rolling specs for the base columns
    rolling_specs=None,
    rolling_avg_specs=None
):
    """
    Converts a wide-format match DataFrame into a long-format player stats DataFrame
    and computes rolling statistics for each player over a given window (num_years).
    
    If include_adjusted is True, additional columns (which are multiplied by the opponent factor)
    are included and their rolling statistics are computed.
    """
    # Default rolling specifications for base stats if none are provided.
    if rolling_specs is None:
        rolling_specs = [
            {"new_col": "rolling_match_count", "source_col": "is_winner", "func": "count"},
            {"new_col": "rolling_serve_pct_sum", "source_col": "serve_pct", "func": "sum"},
            {"new_col": "rolling_bp_pct_sum", "source_col": "bp_saved_pct", "func": "sum"},
            {"new_col": "rolling_bp_conv_pct_sum", "source_col": "bp_conv_pct", "func": "sum"},
            {"new_col": "rolling_return_pct_sum", "source_col": "return_pct", "func": "sum"},
            {"new_col": "matches_won", "source_col": "is_winner", "func": "sum"},
        ]
    if rolling_avg_specs is None:
        rolling_avg_specs = [
            {"new_col": "rolling_avg_serve_pct", "sum_col": "rolling_serve_pct_sum"},
            {"new_col": "rolling_avg_bp_pct", "sum_col": "rolling_bp_pct_sum"},
            {"new_col": "rolling_avg_bp_conv_pct", "sum_col": "rolling_bp_conv_pct_sum"},
            {"new_col": "rolling_avg_return_pct", "sum_col": "rolling_return_pct_sum"},
        ]
    
    # Define adjusted rolling specifications if we want adjusted stats.
    adjusted_rolling_specs = [
        {"new_col": "rolling_serve_pct_adjusted_sum", "source_col": "serve_pct_adjusted", "func": "sum"},
        {"new_col": "rolling_bp_saved_pct_adjusted_sum", "source_col": "bp_saved_pct_adjusted", "func": "sum"},
        {"new_col": "rolling_bp_conv_pct_adjusted_sum", "source_col": "bp_conv_pct_adjusted", "func": "sum"},
        {"new_col": "rolling_return_pct_adjusted_sum", "source_col": "return_pct_adjusted", "func": "sum"},
    ]
    adjusted_rolling_avg_specs = [
        {"new_col": "rolling_avg_serve_pct_adjusted", "sum_col": "rolling_serve_pct_adjusted_sum"},
        {"new_col": "rolling_avg_bp_saved_pct_adjusted", "sum_col": "rolling_bp_saved_pct_adjusted_sum"},
        {"new_col": "rolling_avg_bp_conv_pct_adjusted", "sum_col": "rolling_bp_conv_pct_adjusted_sum"},
        {"new_col": "rolling_avg_return_pct_adjusted", "sum_col": "rolling_return_pct_adjusted_sum"},
    ]
    
    # --- Preprocessing: convert date and sort
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)
    
    # --- Create long-format DataFrame for player stats.
    # For each match, we create two rows (one for each player).
    # For base stats we pull from the appropriate team columns.
    long_data = {
        "player_id": df["PlayerTeam1.PlayerId"].tolist() + df["PlayerTeam2.PlayerId"].tolist(),
        "match_id": df["match_id"].tolist() + df["match_id"].tolist(),
        date_column: df[date_column].tolist() + df[date_column].tolist(),
        "serve_pct": df["PlayerTeam1.serve_pct1"].tolist() + df["PlayerTeam2.serve_pct1"].tolist(),
        "bp_saved_pct": df["PlayerTeam1.bp_save_pct1"].tolist() + df["PlayerTeam2.bp_save_pct1"].tolist(),
        "bp_conv_pct": df["PlayerTeam1.bp_conv_pct1"].tolist() + df["PlayerTeam2.bp_conv_pct1"].tolist(),
        "return_pct": df["PlayerTeam1.return_pct1"].tolist() + df["PlayerTeam2.return_pct1"].tolist(),
        "opponent_factor": df["PlayerTeam1.opponent_factor"].tolist() + df["PlayerTeam2.opponent_factor"].tolist(),
        "is_winner": [1] * len(df) + [0] * len(df)
    }
    
    # If we want adjusted stats, add them as well.
    if include_adjusted:
        long_data["serve_pct_adjusted"] = (
            df["PlayerTeam1.serve_pct1.adjusted"].tolist() +
            df["PlayerTeam2.serve_pct1.adjusted"].tolist()
        )
        long_data["bp_saved_pct_adjusted"] = (
            df["PlayerTeam1.bp_save_pct1.adjusted"].tolist() +
            df["PlayerTeam2.bp_save_pct1.adjusted"].tolist()
        )
        long_data["bp_conv_pct_adjusted"] = (
            df["PlayerTeam1.bp_conv_pct1.adjusted"].tolist() +
            df["PlayerTeam2.bp_conv_pct1.adjusted"].tolist()
        )
        long_data["return_pct_adjusted"] = (
            df["PlayerTeam1.return_pct1.adjusted"].tolist() +
            df["PlayerTeam2.return_pct1.adjusted"].tolist()
        )
    
    player_stats = pd.DataFrame(long_data)
    
    # Ensure the date column is datetime and sort by player_id and date.
    player_stats[date_column] = pd.to_datetime(player_stats[date_column], errors="coerce")
    player_stats = player_stats.dropna(subset=[date_column])
    print("SORTING PLAYER STATS")
    player_stats.sort_values(["player_id", date_column], inplace=True)
    
    # --- Rolling Calculations
    print("ROLLING CALCULATIONS")
    window_days = num_years * 365  # approximate conversion to days

    def rolling_stats(group):
        # Set the date column as the index for time-based rolling.
        group = group.set_index(date_column)
        
        # Compute rolling statistics for the base stats.
        for spec in rolling_specs:
            new_col = spec["new_col"]
            source_col = spec["source_col"]
            func = spec["func"]
            group[new_col] = compute_rolling(group, source_col, window_days, func, shift=1)
        
        for spec in rolling_avg_specs:
            new_avg_col = spec["new_col"]
            sum_col = spec["sum_col"]
            group[new_avg_col] = (group[sum_col] / group["rolling_match_count"]).fillna(0)
        
        # If adjusted stats are to be included, compute their rolling values.
        if include_adjusted:
            for spec in adjusted_rolling_specs:
                new_col = spec["new_col"]
                source_col = spec["source_col"]
                func = spec["func"]
                group[new_col] = compute_rolling(group, source_col, window_days, func, shift=1)
            
            for spec in adjusted_rolling_avg_specs:
                new_avg_col = spec["new_col"]
                sum_col = spec["sum_col"]
                group[new_avg_col] = (group[sum_col] / group["rolling_match_count"]).fillna(0)
        
        # Reset the index so that the date becomes a column again.
        return group.reset_index()

    # Apply rolling calculations for each player.
    player_stats = player_stats.groupby("player_id", group_keys=False).apply(rolling_stats)
    
    # Collect names of new statistic columns.
    base_new_stats = [spec["new_col"] for spec in rolling_avg_specs] + ["rolling_match_count", "matches_won"]
    if include_adjusted:
        adjusted_new_stats = [spec["new_col"] for spec in adjusted_rolling_avg_specs]
        new_stat_columns = base_new_stats + adjusted_new_stats
    else:
        new_stat_columns = base_new_stats

    return player_stats, new_stat_columns

def get_rename_mapping(original_columns, prefix):
    result = {}
    for column in original_columns:
        result[column] = prefix +"." + column
    return result

def merge_dataframes(df, other_df, columns_to_merge_on, rename_mapping, is_winner):
    df = df.merge(
        other_df[other_df["is_winner"] == is_winner][columns_to_merge_on],
        left_on="match_id",
        right_on="match_id",
        how="left",
        suffixes=("", "_winner")
    ).rename(columns=rename_mapping)
    return df

    
def add_features(df):
    df["PlayerTeam1.serve_pct1"] = df[PLAYER1_SERVE_WON] / df[PLAYER1_SERVE_TOTAL]
    df["PlayerTeam2.serve_pct1"] = df[PLAYER2_SERVE_WON] / df[PLAYER2_SERVE_TOTAL]
    df["PlayerTeam1.bp_save_pct1"] = df[PLAYER1_BP_WON] / df[PLAYER1_BP_TOTAL]
    df["PlayerTeam2.bp_save_pct1"] = df[PLAYER2_BP_WON] / df[PLAYER2_BP_TOTAL]
    df["PlayerTeam1.bp_conv_pct1"] = df[PLAYER1_BP_CONVERTED] / df[PLAYER1_BP_CONVERTED_TOTAL]
    df["PlayerTeam2.bp_conv_pct1"] = df[PLAYER2_BP_CONVERTED] / df[PLAYER2_BP_CONVERTED_TOTAL]
    df["PlayerTeam1.return_pct1"] = df[PLAYER1_RETURN] / df[PLAYER1_RETURN_TOTAL]
    df["PlayerTeam2.return_pct1"] = df[PLAYER2_RETURN] / df[PLAYER2_RETURN_TOTAL]
    df = replace_divide_by_zero(df, PLAYER1_BP_TOTAL, "PlayerTeam1.bp_save_pct1")
    df = replace_divide_by_zero(df, PLAYER2_BP_TOTAL, "PlayerTeam2.bp_save_pct1")
    df = replace_divide_by_zero(df, PLAYER1_BP_CONVERTED, "PlayerTeam1.bp_conv_pct1")
    df = replace_divide_by_zero(df, PLAYER2_BP_CONVERTED, "PlayerTeam2.bp_conv_pct1")
    df = replace_divide_by_zero(df, PLAYER1_RETURN, "PlayerTeam1.return_pct1")
    df = replace_divide_by_zero(df, PLAYER2_RETURN, "PlayerTeam2.return_pct1")
    
    df["PlayerTeam1.opponent_factor"] = 1 / df["PlayerTeam2.SglRaceRank"]
    df["PlayerTeam2.opponent_factor"] = 1 / df["PlayerTeam1.SglRaceRank"]
    df = replace_divide_by_zero(df, "PlayerTeam2.SglRaceRank", "PlayerTeam1.opponent_factor")
    df = replace_divide_by_zero(df, "PlayerTeam1.SglRaceRank", "PlayerTeam2.opponent_factor")
    
    rolling_columns = ["serve_pct1", "bp_save_pct1", "bp_conv_pct1", "return_pct1"]
    df = generate_adjusted_rolling_columns(df, rolling_columns)
    #print(df.isna().sum())
    
    df["match_id"] = df["MatchId"].astype(str) + "-" + df["EventId"].astype(str) + "-" + df["EventYear"].astype(str)
    player_stats, new_stats_columns = retrieve_player_stats(df, 3)    
    # Merge rolling averages back into the original DataFrame using match_id
    columns_to_merge_on = new_stats_columns + ["match_id"]
    winner_rename_mapping = get_rename_mapping(new_stats_columns, "PlayerTeam1")
    loser_rename_mapping = get_rename_mapping(new_stats_columns, "PlayerTeam2")
    print("MERGING ROLLING AVERAGES")
    # Merge for winners
    df = merge_dataframes(df, player_stats, columns_to_merge_on, winner_rename_mapping, 1)
    # Merge for losers
    df = merge_dataframes(df, player_stats, columns_to_merge_on, loser_rename_mapping, 0)
    print("AVG SERVE PCT", df["PlayerTeam1.rolling_avg_serve_pct"].median())
    print("AVG SERVE PCT", df["PlayerTeam2.rolling_avg_serve_pct"].median())
    print("AVG BP PCT:", df["PlayerTeam1.rolling_avg_bp_pct"].mean())
    print("AVG BP PCT:", df["PlayerTeam2.rolling_avg_bp_pct"].mean())
    print("AVG SERVE PCT:", df["PlayerTeam1.rolling_avg_return_pct"].mean())
    print("AVG SERVE PCT:", df["PlayerTeam2.rolling_avg_return_pct"].mean())
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
        "serve_pct1", "pct1"
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
    #print(df.isna().sum())

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
    # print("Player 1 rolling average: ", df["PlayerTeam1.rolling_avg_bp_conv_pct"].mean())
    # print("Player 2 rolling average: ", df["PlayerTeam2.rolling_avg_bp_conv_pct"].mean())

    df = convert_to_ints(df)
    df = swap_and_add(df)
    print(len(df))
    print(df.isna().sum())
    print(df.describe(include="all"))

    joblib.dump(df, "./data/preprocessed_df.pkl")

if __name__ == "__main__":
    main();