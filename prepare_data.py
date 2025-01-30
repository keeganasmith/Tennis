from constants import MATCH_DF_PATH
from constants import PLAYER_DF_PATH   
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
def reverse_columns(columns):
    column_dict = {}
    for i in range(0, len(columns)):
        new_column_name = None
        if(columns[i] == "player1_won"):
            continue
        if(columns[i][:7] == "player1"):
            new_column_name = "player2" + columns[i][7:]
        elif(columns[i][:7] == "player2"):
            new_column_name = "player1" + columns[i][7:]
        else:
            continue
        column_dict[columns[i]] = new_column_name
    print(column_dict)
    return column_dict
def retrieve_player_stats(match_df, num_years):
    match_df["tourney_date"] = pd.to_datetime(match_df["tourney_date"], format="%Y%m%d")

    # Sort by tourney_date
    match_df = match_df.sort_values("tourney_date").reset_index(drop=True)

    # Create a long-format DataFrame for easier aggregation
    print("CREATING PLAYER STATS DATAFRAME")
    player_stats = pd.DataFrame({
        "player_id": match_df["winner_id"].tolist() + match_df["loser_id"].tolist(),
        "match_id": match_df["match_id"].tolist() + match_df["match_id"].tolist(),  # Use match_id for merging
        "tourney_date": match_df["tourney_date"].tolist() + match_df["tourney_date"].tolist(),
        "serve_pct": match_df["w_serve_pct"].tolist() + match_df["l_serve_pct"].tolist(),
        "bp_saved_pct": match_df["w_bp_save_pct"].tolist() + match_df["l_bp_save_pct"].tolist(),
        "is_winner": [1] * len(match_df) + [0] * len(match_df)
    })

    # Sort player_stats by player_id and tourney_date
    print("SORTING PLAYER STATS")
    player_stats = player_stats.sort_values(["player_id", "tourney_date"]).reset_index(drop=True)

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
            current_date = group.iloc[i]["tourney_date"]
            cutoff_date = current_date.replace(year=current_date.year - num_years)

            # Slide the window to exclude matches older than num_years
            while j < i and group.iloc[j]["tourney_date"] < cutoff_date:
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

def get_rename_mapping(original_columns, prefix):
    result = {}
    for column in original_columns:
        result[column] = prefix +"_" + column
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
def compute_service_percentage_per_game_prefix(match_df, num_years):
    player_stats, new_stats_columns = retrieve_player_stats(match_df, num_years)    
    # Merge rolling averages back into the original DataFrame using match_id
    columns_to_merge_on = new_stats_columns + ["match_id"]
    winner_rename_mapping = get_rename_mapping(new_stats_columns, "w")
    loser_rename_mapping = get_rename_mapping(new_stats_columns, "l")
    print("MERGING ROLLING AVERAGES")
    # Merge for winners
    match_df = merge_dataframes(match_df, player_stats, columns_to_merge_on, winner_rename_mapping, 1)
    print("winner rename mappings: ", winner_rename_mapping)
    print("loser rename mappings: ", loser_rename_mapping)
    # Merge for losers
    match_df = merge_dataframes(match_df, player_stats, columns_to_merge_on, loser_rename_mapping, 0)
    # Fill NaN values with 0 (for new players or missing data)
    # match_df["w_avg_serve_pct"] = match_df["w_avg_serve_pct"].fillna(0)
    # match_df["l_avg_serve_pct"] = match_df["l_avg_serve_pct"].fillna(0)
    
    return match_df

def compute_surface_stats(match_df, num_years):
    new_df = match_df.copy()
    for surface_type, group in match_df.groupby("surface"):
        player_stats, stats_columns = retrieve_player_stats(group, num_years)
        winner_rename_mapping = get_rename_mapping(stats_columns, "w_" + surface_type + "_")
        loser_rename_mapping = get_rename_mapping(stats_columns, "l_" + surface_type + "_")

        

def label_data(match_df):
    match_df.loc[match_df["player2_entry"] == "Alt", "player2_entry"] = "ALT"
    match_df.loc[match_df["player1_entry"] == "Alt", "player1_entry"] = "ALT"
    columns_to_encode = [
        "player1_entry",
        "player1_hand",
        "player2_entry",
        "player2_hand",
        "surface",
        "tourney_level"
    ]

    # Perform one-hot encoding on these columns
    encoded_df = pd.get_dummies(match_df, columns=columns_to_encode, drop_first=True)

    # Identify newly created encoded columns
    encoded_columns = [col for col in encoded_df.columns if col not in match_df.columns]

    # Convert only the encoded columns to integers
    encoded_df[encoded_columns] = encoded_df[encoded_columns].astype(int)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    return encoded_df
    
def handle_na(match_df):
    #print(match_df.isna().sum())
    match_df["player1_entry"] = match_df["player1_entry"].fillna("O")
    match_df["player2_entry"] = match_df["player2_entry"].fillna("O")
    match_df["player1_age"] = match_df["player1_age"].fillna(26.15)
    match_df["player2_age"] = match_df["player2_age"].fillna(26.15)
    match_df["player1_ht"] = match_df["player1_ht"].fillna(185.0)
    match_df["player2_ht"] = match_df["player2_ht"].fillna(185.0)
    match_df["player1_hand"] = match_df["player1_hand"].fillna("U")
    match_df["player2_hand"] = match_df["player2_hand"].fillna("U")
    match_df["player1_rank"] = match_df["player1_rank"].fillna(0)
    match_df["player2_rank"] = match_df["player2_rank"].fillna(0)
    match_df.replace([np.inf, -np.inf], 0, inplace=True)
    return match_df

def drop_cols(match_df):
    drop_columns = [
        "winner_id", "loser_id", "winner_name", "loser_name", "score", "match_num",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
        "tourney_id", "tourney_date", "winner_seed", "loser_seed", "winner_rank_points", "loser_rank_points",
        "round", "minutes", "w_serve_pct", "l_serve_pct", "match_id", "tourney_name", "w_bp_save_pct", "l_bp_save_pct",
        "winner_ioc", "loser_ioc"
    ]
    match_df = match_df.drop(columns = drop_columns)
    return match_df

def check_for_correct_columns(match_df, reversed_df):
    for i in range(0, len(match_df.columns)):
        if(not match_df.columns[i] in reversed_df.columns):
            print("column ", match_df.columns[i], " not in reversed df")
    for i in range(0, len(reversed_df.columns)):
        if(not reversed_df.columns[i] in match_df.columns):
            print("column ", reversed_df.columns[i], " not in match df")
    
def pretty_print_columns(match_df):
    for column in match_df.columns:
        print(column)

def rename_mapping_for_conversion_to_player(columns):
    result = {}
    for column in columns:
        if(column[0:2] == "w_"):
            result[column] = "player1_" + column[2:]
        elif(column[0:2] == "l_"):
            result[column] = "player2_" + column[2:]
        elif(column[0:7] == "winner_"):
            result[column] = "player1_" + column[7:]
        elif(column[0:6] == "loser_"):
            result[column] = "player2_" + column[6:]
    return result

def prepare_data():
    # Load the match and player DataFrames
    match_df = joblib.load(MATCH_DF_PATH)
    player_df = joblib.load(PLAYER_DF_PATH)

    # Convert tourney_date to datetime
    match_df["tourney_date"] = pd.to_datetime(match_df["tourney_date"], format="%Y%m%d")
    match_df["match_id"] = match_df["tourney_id"] + "-" + match_df["match_num"].astype(str)

    # Compute serve percentages
    match_df["w_serve_pct"] = (match_df["w_1stWon"] + match_df["w_2ndWon"]) / match_df["w_svpt"]
    match_df["l_serve_pct"] = (match_df["l_1stWon"] + match_df["l_2ndWon"]) / match_df["l_svpt"]
    match_df["w_serve_pct"] = match_df["w_serve_pct"].fillna(0.6)
    match_df["l_serve_pct"] = match_df["l_serve_pct"].fillna(0.6)
    
    match_df["w_bp_save_pct"] = (match_df["w_bpSaved"] / match_df["w_bpFaced"])
    match_df["l_bp_save_pct"] = (match_df["l_bpSaved"] / match_df["l_bpFaced"])
    match_df.loc[match_df["w_bpSaved"] == 0, "w_bp_save_pct"] = .5
    match_df.loc[match_df["l_bpSaved"] == 0, "l_bp_save_pct"] = .5

    print("COMPUTING SERVE WIN PERCENTAGES")
    match_df = compute_service_percentage_per_game_prefix(match_df, 3)
    #print(match_df.columns)
    match_df = drop_cols(match_df)
    
    # Rename columns to player1/player2 format

    # rename_mapping = {
    #     "winner_entry": "player1_entry",
    #     "winner_hand": "player1_hand",
    #     "winner_ht": "player1_ht",
    #     "winner_age": "player1_age",
    #     "winner_rank": "player1_rank",

    #     "loser_entry": "player2_entry",
    #     "loser_hand": "player2_hand",
    #     "loser_ht": "player2_ht",
    #     "loser_age": "player2_age",
    #     "loser_rank": "player2_rank",

    #     "w_avg_serve_pct": "player1_avg_serve_pct",
    #     "l_avg_serve_pct": "player2_avg_serve_pct",
    #     "w_avg_bp_pct": "player1_avg_bp_pct",
    #     "l_avg_bp_pct": "player2_avg_bp_pct",
    #     "w_matches_played": "player1_matches_played",
    #     "l_matches_played": "player2_matches_played",
    #     "w_matches_won": "player1_matches_won",
    #     "l_matches_won": "player2_matches_won"
    # }
    rename_mapping = rename_mapping_for_conversion_to_player(match_df.columns)
    match_df = match_df.rename(columns=rename_mapping)
    match_df["player1_won"] = 1  

    # Handle missing values and label data
    match_df = handle_na(match_df)
    match_df = label_data(match_df)
    match_df["player1_entry_S"] = 0
    #print(match_df.head())
    # Create the reversed DataFrame (switch player1 and player2)
    reversed_columns = reverse_columns(match_df.columns)
    reversed_df = match_df.rename(columns=reversed_columns)

    #print(reversed_df.head())
    # _zero_entries = len(match_df.loc[match_df["player1_avg_bp_pct"] != 0])
    # num_non_zero_entries_l = len(match_df.loc[match_df["player2_avg_bp_pct"] != 0])
    # print("num non zero entries (1): ", num_non_zero_entries)
    # print("num non zero entries (2): ", num_non_zero_entries_lnum_non)
    # print("max of player1_avg_bp_pct: ", reversed_df["player1_avg_bp_pct"].max())
    # print("min of player1_avg_bp_pct: ", reversed_df["player1_avg_bp_pct"].min())
    reversed_df["player1_won"] = 0  # Reverse the label for reversed matches
    # Concatenate the original and reversed datasets
    check_for_correct_columns(match_df, reversed_df)
    doubled_df = pd.concat([match_df, reversed_df], ignore_index=True)

    print(f"Original dataset size: {len(match_df)}")
    print(f"Doubled dataset size: {len(doubled_df)}")
    #print(doubled_df.isin([np.inf, -np.inf]).values.sum()) 
    #print(doubled_df.isna().sum())
    print("columns:")
    pretty_print_columns(doubled_df)
    return doubled_df

def main():
    match_df = prepare_data()
    joblib.dump(match_df, "processed_df.pkl")
    
if __name__ == "__main__":
    main();