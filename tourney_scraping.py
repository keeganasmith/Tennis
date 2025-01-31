import requests
import json

# Define the URL
url = "https://www.atptour.com/en/-/tournaments/calendar/tour"

# Define the headers (as extracted from curl)
headers = {
    "sec-ch-ua-platform": '"Windows"',
    "Referer": "https://www.atptour.com/en/tournaments",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0",
    "Accept": "application/json, text/plain, */*",
    "sec-ch-ua": '"Not A(Brand";v="8", "Chromium";v="132", "Microsoft Edge";v="132"',
    "sec-ch-ua-mobile": "?0"
}

# Send the GET request
response = requests.get(url, headers=headers)

# Check if request was successful
if response.status_code == 200:
    # Save response as JSON
    with open("atp_tournaments.json", "w", encoding="utf-8") as f:
        json.dump(response.json(), f, indent=4)

    print("✅ ATP tournament data saved as 'atp_tournaments.json'")
else:
    print(f"❌ Failed to fetch data. Status Code: {response.status_code}")
    print(response.text)  # Print error response