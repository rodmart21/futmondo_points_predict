import requests
import json
import time

# --- 1. CONFIGURATION (Verified Data) ---

BASE_URL = 'https://api.futmondo.com'

# CONFIRMED VALUES from your account data
YOUR_AUTH_TOKEN = '5e65_20f468a19ad19ef73979642df99d6603' 
YOUR_USER_ID = '56c6b62085617f9b1dc7d061'
YOUR_CHAMPIONSHIP_ID = '5f7b19924dcd0d43e8a092dd4'
YOUR_USERTEAM_ID = '5f7b33ea6f9120324dedcb15'

# --- 2. ENDPOINTS & HEADERS ---

ENDPOINTS = {
    'GET_TEAM_PLAYERS': '/1/userteam/roster',
}

# The final, correct set of headers to satisfy API security checks
HEADERS = {
    'Authorization': f'Bearer {YOUR_AUTH_TOKEN}', 
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'x-futmondo-token': YOUR_AUTH_TOKEN,       # Custom required header
    'x-futmondo-userid': YOUR_USER_ID,         # Custom required header
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# --- 3. CORE REQUEST FUNCTION ---

def make_request(endpoint_name, payload):
    """Makes API request and handles the response."""
    
    endpoint = ENDPOINTS.get(endpoint_name)
    if not endpoint:
        print(f"❌ ERROR: Unknown endpoint name {endpoint_name}")
        return None

    full_url = f"{BASE_URL}{endpoint}"
    
    print(f"\n{'='*70}")
    print(f"📡 {endpoint_name}")
    print(f"{'='*70}")
    print(f"URL: {full_url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            full_url,
            headers=HEADERS, # Use the global HEADERS dictionary
            json=payload,
            timeout=20
        )
        
        print(f"Status: {response.status_code}")
        response.raise_for_status() 

        data = response.json()
        
        # Safely handle the response structure (list vs. dict, and checking for API errors)
        if isinstance(data, dict):
            answer = data.get('answer', data)
            if isinstance(answer, dict) and answer.get('error'):
                print(f"❌ API ERROR: {answer.get('code', 'Unknown Error')}")
                return None
            
            # Successful dict or list within the 'answer' key
            if isinstance(answer, list):
                print(f"✅ SUCCESS - Got {len(answer)} items in answer")
            
            return data
            
        elif isinstance(data, list):
            print(f"✅ SUCCESS - Got {len(data)} items directly")
            return data

        return data
            
    except requests.exceptions.HTTPError as errh:
        print(f"❌ HTTP ERROR {response.status_code}: {response.text}")
        return None
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return None

# --- 4. EXECUTION ---

if __name__ == '__main__':
    
    print("\n" + "="*70)
    print("🚀 FUTMONDO ROSTER DATA RETRIEVAL")
    print("="*70)

    # Payload for the Roster endpoint
    roster_payload = {
        'query': {
            'championshipId': YOUR_CHAMPIONSHIP_ID,
            'userteamId': YOUR_USERTEAM_ID
        }
    }
    
    # Execute the Roster request
    roster_data_response = make_request('GET_TEAM_PLAYERS', roster_payload)

    if roster_data_response:
        print("\n👥 ROSTER DATA SUCCESSFULLY RECEIVED:")
        
        # Extract the list of players, which is often under the 'answer' key
        player_list = roster_data_response.get('answer', roster_data_response)

        if isinstance(player_list, list) and len(player_list) > 0:
            print(f"Found {len(player_list)} players.")
            print("\n--- SAMPLE PLAYER DATA ---")
            print(json.dumps(player_list[0], indent=2))
        elif isinstance(player_list, list) and len(player_list) == 0:
            print("⚠️ WARNING: Roster request successful, but the player list is empty.")
        else:
            print(json.dumps(roster_data_response, indent=2))
    else:
        print("\n❌ FAILED to retrieve roster data. Check the error message above.")
    
    print("\n" + "="*70)
    print("📋 TASK COMPLETE")
    print("="*70)