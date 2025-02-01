import requests
from bs4 import BeautifulSoup
import joblib
import time
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
        time.sleep(.5)

def print_match_ids():
    event_and_match_ids = joblib.load("event_and_match_ids.pkl")
    print(event_and_match_ids)

if __name__ == "__main__":
    #print_match_ids()
    write_all_match_ids()

    