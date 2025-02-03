from constants import MATCH_DF_PATH
from constants import PLAYER_DF_PATH   
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

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

    print(len(my_df))
    print(my_df.isna().sum())
    return my_df
def main():
    my_df = joblib.load("./data/atp_stats.pkl")    
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns (if needed)
    my_df = pre_drop_cols(my_df)
if __name__ == "__main__":
    main();