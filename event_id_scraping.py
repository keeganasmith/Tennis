import requests
from bs4 import BeautifulSoup
import joblib
import time
import pandas as pd
import json
import flatdict
import numpy as np
irrelevant_keys = [
    # Identifiers & Metadata
    "EventDisplayName",  "TournamentImage",
    "TournamentLogo", "TournamentWhiteLogo", "TournamentUrl", "PrizeMoney",
    "CurrencySymbol",

    # Broadcast & Display Info (Not Game-Related)
    "IsWatchLive", "ExtendedMessage", "Message", "ScoreCentreMatchSortOrder",
    "RightRailMatchSortOrder", "Head2HeadUrl",

    # Umpire Details (Irrelevant to Outcome)
    "UmpireFirstName", "UmpireLastName",


    # Doubles Partner Details (Not Relevant for Singles Matches)
    "PlayerTeam1.PartnerId", "PlayerTeam1.PartnerFirstName",
    "PlayerTeam1.PartnerFirstNameFull", "PlayerTeam1.PartnerLastName",
    "PlayerTeam1.PartnerCountryCode", "PlayerTeam1.PartnerScRelativeUrlPlayerProfile",
    "PlayerTeam1.PartnerScRelativeUrlPlayerCountryFlag",
    "PlayerTeam1.PartnerHeadshotImageUrl", "PlayerTeam1.PartnerGladiatorImageUrl",
    "PlayerTeam2.PartnerId", "PlayerTeam2.PartnerFirstName",
    "PlayerTeam2.PartnerFirstNameFull", "PlayerTeam2.PartnerLastName",
    "PlayerTeam2.PartnerCountryCode", "PlayerTeam2.PartnerScRelativeUrlPlayerProfile",
    "PlayerTeam2.PartnerScRelativeUrlPlayerCountryFlag",
    "PlayerTeam2.PartnerHeadshotImageUrl", "PlayerTeam2.PartnerGladiatorImageUrl",
]

def retrieve_player_historical_rankings(player_id):
    url = f"https://www.atptour.com/en/-/www/rank/history/{player_id}?v=1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest"
    }
    max_retries = 5
    retries = 0
    while retries < max_retries:
        start_time = time.time()
        try:
            response = requests.get(url, headers=headers, timeout=2)
            elapsed_time = time.time() - start_time
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    print("Response is not JSON. Printing raw text:")
                    print(response.text)
                    return None
            else:
                print(f"Request failed with status code {response.status_code}")

        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"Request timed out or failed: {e}")
            time.sleep(10)
        
        retries += 1
        print(f"Retrying... Attempt {retries}/{max_retries}")
    
    print("Max retries reached. Returning None.")
    return None

def map_player_rankings():
    my_df = joblib.load("./data/atp_stats.pkl")
    unique_player1_ids = my_df["PlayerTeam1.PlayerId"].unique()
    unique_player2_ids = my_df["PlayerTeam2.PlayerId"].unique()
    unique_players = np.union1d(unique_player1_ids, unique_player2_ids)
    current_player_mappings = {}
    try:
        current_player_mappings = joblib.load("./data/player_rankings.pkl")
    except:
        print("player rankings not found")
        
    for id in unique_players:
        if(id in current_player_mappings):
            print("already exists")
            continue
        ranking_results = retrieve_player_historical_rankings(id)
        current_player_mappings[id] = ranking_results
        joblib.dump(current_player_mappings, "./data/player_rankings.pkl")
        print("progress: ", len(current_player_mappings.keys()), "/", len(unique_players))
        time.sleep(.1)
    
def remove_irrelevant_keys(my_dict):
    keys = list(my_dict.keys())
    for key in keys:
        if(key in irrelevant_keys):
            del my_dict[key]
        elif(("IsStatBetter" in key) or ("YearToDateStats" in key) or ("Url" in key)):
            del my_dict[key]
    return my_dict

def flatten_dict(d, parent_key='', sep='.'):
    """Recursively flattens a nested dictionary, handling lists properly."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, elem in enumerate(v):
                if isinstance(elem, dict):
                    items.extend(flatten_dict(elem, f"{new_key}[{i}]", sep=sep).items())
                else:
                    items.append((f"{new_key}[{i}]", elem))
        else:
            items.append((new_key, v))
    return dict(items)

def retrieve_match_stats(event_id, match_id, year, max_retries=5):

    url = f"https://www.atptour.com/-/Hawkeye/MatchStats/Complete/{year}/{event_id}/{match_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest"
    }

    retries = 0
    while retries < max_retries:
        start_time = time.time()
        try:
            response = requests.get(url, headers=headers, timeout=2)  # Set timeout to 1 second
            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    print("Response is not JSON. Printing raw text:")
                    print(response.text)
                    return None
            else:
                print(f"Request failed with status code {response.status_code}")

        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"Request timed out or failed: {e}")
            time.sleep(10)
        
        retries += 1
        print(f"Retrying... Attempt {retries}/{max_retries}")
    
    print("Max retries reached. Returning None.")
    return None

def check_row_exists(df, event_id, match_id, year):
    if(df.empty):
        return False
    result = df[
        (df["EventId"] == event_id) & 
        (df["MatchId"] == match_id) & 
        (df["EventYear"] == year)
    ]
    return not result.empty
def pretty_print(my_list):
    for i in range(0, len(my_list)):
        print(my_list[i])
def populate_match_statistics():
    events_and_matches = joblib.load("event_and_match_ids.pkl")
    stats_df = pd.DataFrame();
    
    try:
        stats_df = joblib.load("./data/atp_stats.pkl")
        pretty_print(list(stats_df.columns))
    except:
        print("stats df does not exist")
    for event in events_and_matches:
        match_ids = event["match_ids"]
        needs_to_be_appended = False
        for match_id in match_ids:
            if(check_row_exists(stats_df, event["id"], match_id.upper(), event["year"])):
                print("row already exists..")
                continue;
            needs_to_be_appended = True            
            print("event id: ", event["id"], ", match id: ", match_id, " year: ", event["year"])

            data = retrieve_match_stats(event["id"], match_id, event["year"])
            num_attempts = 0
            while(data is None and num_attempts < 3):
                data = retrieve_match_stats(event["id"], match_id, event["year"])
                num_attempts += 1
            if(data is None):
                print("data was none, so continuing")
                continue
            element_data = {}
            tournament_info = data["Tournament"]
            element_data.update(tournament_info)
            del data["Tournament"]
            match_info = data["Match"]
            if(match_info["IsDoubles"]):
                print("skipping because IsDoubles was: ", match_info["IsDoubles"])
                continue
            del match_info["TeamTieResults"]
            del match_info["PlayerTeam"]
            del match_info["OpponentTeam"]
            match_info.update(tournament_info)
            #print(match_info)
            flat_dict = flatten_dict(match_info)
            flat_dict = remove_irrelevant_keys(flat_dict)
            #print((flat_dict))
            #print(flat_dict.keys())
            print(pd.DataFrame([flat_dict]))
            stats_df = pd.concat([stats_df, pd.DataFrame([flat_dict])], ignore_index=True)
            print("appended a match")
            print("total length of df: ", len(stats_df))
        if(needs_to_be_appended):
            print("whole event appended")
            joblib.dump(stats_df, "./data/atp_stats.pkl")
            #print(json.dumps(match_info, indent=4))

def get_match_ids(event_id, year, event_name):
    # Construct the URL for the tournament's results page
    url = f"https://www.atptour.com/en/scores/archive/{event_name}/{event_id}/{year}/results"
    print(url)
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve data: {response.status_code}")
        return []
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all 'Stats' links inside divs with class 'match-cta'
    match_ids = []
    for match_cta in soup.find_all('div', class_='match-cta'):
        stats_link = match_cta.find('a', href=True, text="Stats")  # Find 'Stats' link
        if stats_link:
            href = stats_link['href']
            match_id = href.split('/')[-1]  # Extract match ID from the URL
            match_ids.append(match_id)
    print(match_ids)
    return match_ids

def fetch_event_ids(year):
    url = f"https://www.atptour.com/en/scores/results-archive?year={year}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    event_ids = []

    # Find all tournament profile links
    for link in soup.find_all('a', class_="tournament__profile", href=True):
        href = link['href']
        parts = href.split('/')
        if len(parts) >= 5:
            print(parts)
            event_id = parts[4]  # Extract event ID from the URL
            event_name = parts[3]
            event_ids.append({"id":event_id, "name": event_name, "year": year})
            
    return event_ids

def write_all_event_ids(start_year = 2000):
    event_ids =[]
    for i in range(start_year, 2025):
        new_event_ids = fetch_event_ids(i)
        event_ids += new_event_ids
        joblib.dump(event_ids, "event_ids.pkl")
        print("number of event ids: ", len(event_ids))
        time.sleep(1)

def check_if_already_present(result, event_id):
    my_event_id = event_id["id"]
    for item in result:
        if(item["id"] == my_event_id and item["year"] == event_id["year"]):
            return True
    return False

def write_all_match_ids():

    event_ids = joblib.load("event_ids.pkl")
    result = []

    try:
        result = joblib.load("event_and_match_ids.pkl")
    except:
        print("event and match ids not found")

    for event_id in event_ids:
        present = check_if_already_present(result, event_id)
        if(present):
            print("event id already present")
            continue
        match_ids = get_match_ids(event_id["id"], event_id["year"], event_id["name"])
        new_event_id = event_id
        new_event_id["match_ids"] = match_ids
        result.append(new_event_id)
        joblib.dump(result, "event_and_match_ids.pkl")
        print("length of event and match ids: ", len(result))
        time.sleep(.1)

def print_match_ids():
    event_and_match_ids = joblib.load("event_and_match_ids.pkl")
    print(event_and_match_ids)

if __name__ == "__main__":
    #print_match_ids()
    #write_all_event_ids()
    #write_all_match_ids()
    #populate_match_statistics()
    map_player_rankings()

    