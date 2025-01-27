from constants import MATCH_DF_PATH
from constants import PLAYER_DF_PATH   
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
def compute_service_percentage_per_game_prefix(match_df, num_years):
    # Convert tourney_date to datetime if not already
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
        "is_winner": [1] * len(match_df) + [0] * len(match_df)
    })

    # Sort player_stats by player_id and tourney_date
    print("SORTING PLAYER STATS")
    player_stats = player_stats.sort_values(["player_id", "tourney_date"]).reset_index(drop=True)

    # Add columns for the rolling sum and count within the last num_years
    player_stats["rolling_serve_pct_sum"] = 0.0
    player_stats["rolling_match_count"] = 0
    print("SLIDING WINDOW")
    # Use a sliding window to calculate the rolling stats
    for player_id, group in player_stats.groupby("player_id"):
        serve_pct_sum = 0
        match_count = 0
        j = 0  # Sliding window pointer

        for i in range(len(group)):
            current_date = group.iloc[i]["tourney_date"]
            cutoff_date = current_date.replace(year=current_date.year - num_years)

            # Slide the window to exclude matches older than num_years
            while j < i and group.iloc[j]["tourney_date"] < cutoff_date:
                serve_pct_sum -= group.iloc[j]["serve_pct"]
                match_count -= 1
                j += 1

            # Assign rolling sum and count (excluding current match)
            player_stats.loc[group.index[i], "rolling_serve_pct_sum"] = serve_pct_sum
            player_stats.loc[group.index[i], "rolling_match_count"] = match_count

            # Update rolling sum and count for the next iteration (excluding current match)
            serve_pct_sum += group.iloc[i]["serve_pct"]
            match_count += 1

    # Calculate rolling averages
    print("ROLLING AVERAGES")
    player_stats["rolling_avg_serve_pct"] = (
        player_stats["rolling_serve_pct_sum"] / player_stats["rolling_match_count"]
    ).fillna(0)
    # Merge rolling averages back into the original DataFrame using match_id
    print("MERGING ROLLING AVERAGES")
    # Merge for winners
    match_df = match_df.merge(
        player_stats[player_stats["is_winner"] == 1][["match_id", "rolling_avg_serve_pct"]],
        left_on="match_id",
        right_on="match_id",
        how="left",
        suffixes=("", "_winner")
    ).rename(columns={"rolling_avg_serve_pct": "w_avg_serve_pct"})

    # Merge for losers
    match_df = match_df.merge(
        player_stats[player_stats["is_winner"] == 0][["match_id", "rolling_avg_serve_pct"]],
        left_on="match_id",
        right_on="match_id",
        how="left",
        suffixes=("", "_loser")
    ).rename(columns={"rolling_avg_serve_pct": "l_avg_serve_pct"})

    # Fill NaN values with 0 (for new players or missing data)
    # match_df["w_avg_serve_pct"] = match_df["w_avg_serve_pct"].fillna(0)
    # match_df["l_avg_serve_pct"] = match_df["l_avg_serve_pct"].fillna(0)

    return match_df
def label_data(match_df):
    match_df.loc[match_df["player2_entry"] == "Alt", "player2_entry"] = "ALT"
    match_df.loc[match_df["player1_entry"] == "Alt", "player1_entry"] = "ALT"
    columns_to_encode = [
        "player1_entry",
        "player1_hand",
        "player1_ioc",
        "player2_entry",
        "player2_hand",
        "player2_ioc",
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
        "round", "minutes", "w_serve_pct", "l_serve_pct", "match_id", "tourney_name"
    ]
    match_df = match_df.drop(columns = drop_columns)
    return match_df
    
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
    
    print("COMPUTING SERVE WIN PERCENTAGES")
    match_df = compute_service_percentage_per_game_prefix(match_df, 3)
    match_df = drop_cols(match_df)

    # Rename columns to player1/player2 format
    rename_mapping = {
        "winner_entry": "player1_entry",
        "winner_hand": "player1_hand",
        "winner_ht": "player1_ht",
        "winner_ioc": "player1_ioc",
        "winner_age": "player1_age",
        "winner_rank": "player1_rank",

        "loser_entry": "player2_entry",
        "loser_hand": "player2_hand",
        "loser_ht": "player2_ht",
        "loser_ioc": "player2_ioc",
        "loser_age": "player2_age",
        "loser_rank": "player2_rank",
        
        "w_avg_serve_pct": "player1_avg_serve_pct",
        "l_avg_serve_pct": "player2_avg_serve_pct"
    }
    match_df = match_df.rename(columns=rename_mapping)
    match_df["player1_won"] = 1  # Set the winner label for the original dataset

    # Handle missing values and label data
    match_df = handle_na(match_df)
    match_df = label_data(match_df)

    # Create the reversed DataFrame (switch player1 and player2)
    reversed_df = match_df.rename(columns={
        "player1_entry": "player2_entry",
        "player1_hand": "player2_hand",
        "player1_ht": "player2_ht",
        "player1_ioc": "player2_ioc",
        "player1_age": "player2_age",
        "player1_rank": "player2_rank",
        "player1_avg_serve_pct": "player2_avg_serve_pct",
        "player2_entry": "player1_entry",
        "player2_hand": "player1_hand",
        "player2_ht": "player1_ht",
        "player2_ioc": "player1_ioc",
        "player2_age": "player1_age",
        "player2_rank": "player1_rank",
        "player2_avg_serve_pct": "player1_avg_serve_pct",
    })

    reversed_df["player1_won"] = 0  # Reverse the label for reversed matches
    # Concatenate the original and reversed datasets
    doubled_df = pd.concat([match_df, reversed_df], ignore_index=True)

    print(f"Original dataset size: {len(match_df)}")
    print(f"Doubled dataset size: {len(doubled_df)}")

    return doubled_df

def main():
    match_df = prepare_data()
    joblib.dump(match_df, "processed_df.pkl")
    
if __name__ == "__main__":
    main();