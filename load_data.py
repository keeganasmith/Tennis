import pandas as pd
import joblib
from constants import MATCH_DF_PATH
from constants import PLAYER_DF_PATH
def load_data(years=[2000, 2020]):
    all_match_data = []
    for year in range(years[0], years[1]):
        file_path = f"./data/atp_matches_{year}.csv"
        df = pd.read_csv(file_path)
        
        all_match_data.append(df)
    
    match_df = pd.concat(all_match_data, ignore_index=True)
    player_df = pd.read_csv("./data/atp_players.csv")
    joblib.dump(match_df, MATCH_DF_PATH)
    joblib.dump(player_df, PLAYER_DF_PATH)

def main():
    load_data()

if __name__ == "__main__":
    main()
