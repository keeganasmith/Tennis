import requests
import json
import pandas as pd
import joblib
import random
import string
import time

def random_three_letter_string():
    return ''.join(random.choices(string.ascii_lowercase, k=3))

def search_players(three_digit_search, players_df):
    url = f"https://www.atptour.com/en/-/www/players/find/byname/{three_digit_search}/en"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    # Check for successful response
    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        return None  # Return None on failure

    players = response.json()
    my_players = pd.DataFrame(players)
    if players_df is None or players_df.empty:
        players_df = my_players
    else:
        players_df = pd.concat([players_df, my_players], ignore_index=True)

    players_df = players_df.drop_duplicates(subset=['PlayerId'], keep='first')
    joblib.dump(players_df, "atp_players_df.pkl")
    return players_df

def populate_players():
    already_searched = []
    try:
        already_searched = joblib.load("searched_terms.pkl")
    except:
        print("searched_terms.pkl doesn't exist yet")
    
    players_df = None
    try:
        players_df = joblib.load("atp_players_df.pkl")
    except:
        print("players df doesn't exist yet")

    for i in range(0, 1000):
        three_letter_string = random_three_letter_string()
        print("three letter string is: ", three_letter_string)
        if(not (three_letter_string in already_searched)):
            players_df = search_players(three_letter_string, players_df)
            already_searched.append(three_letter_string)
            joblib.dump(already_searched, "searched_terms.pkl")
            print("players df size now ", len(players_df))
            time.sleep(1)



def main():
    populate_players()

if __name__ == "__main__":
    main()