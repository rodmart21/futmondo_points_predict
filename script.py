import requests
import json
import pandas as pd
from datetime import datetime

# --- CORRECT CONFIGURATION ---
BASE_URL = 'https://api.futmondo.com'

YOUR_AUTH_TOKEN = '5e65_20f468a19ad19ef73979642df99d6603' 
YOUR_USER_ID = '56c6b62085617f9b1dc7d061'
YOUR_CHAMPIONSHIP_ID = '5f7b19924dcd043e8a092dd4'  # CORRECTED!
YOUR_USERTEAM_ID = '5f7b33ea6f9120324dedcb15'
YOUR_LEAGUE_ID = '504e4f584d8bec9a67000079'

headers = {
    'Authorization': f'Bearer {YOUR_AUTH_TOKEN}', 
    'Content-Type': 'application/json',
    'x-futmondo-token': YOUR_AUTH_TOKEN,
    'x-futmondo-userid': YOUR_USER_ID,
}

def api_call(endpoint, payload):
    """Make API call"""
    try:
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ {endpoint}: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

print("="*80)
print("🚀 FUTMONDO DATA SCRAPER - COMPLETE PLAYER DATABASE")
print("="*80)

# ============================================================================
# 1. GET ALL MARKET PLAYERS (Complete Database)
# ============================================================================
print("\n📊 Fetching ALL market players...")

market_data = api_call('/1/market/players', {
    'query': {
        'championshipId': YOUR_CHAMPIONSHIP_ID
    }
})

all_players = []

if market_data and 'answer' in market_data:
    all_players = market_data['answer']
    print(f"✅ Retrieved {len(all_players)} players from market")
    
    # Save raw JSON
    with open('futmondo_all_players.json', 'w', encoding='utf-8') as f:
        json.dump(market_data, f, indent=2, ensure_ascii=False)
    print("💾 Saved: futmondo_all_players.json")
    
    # Show sample
    if len(all_players) > 0:
        print("\n📋 Sample Player:")
        print(json.dumps(all_players[0], indent=2, ensure_ascii=False)[:600])
else:
    print("❌ Failed to get market data")

# ============================================================================
# 2. GET YOUR TEAM ROSTER
# ============================================================================
print("\n\n👥 Fetching your team roster...")

roster_data = api_call('/1/userteam/roster', {
    'query': {
        'championshipId': YOUR_CHAMPIONSHIP_ID,
        'userteamId': YOUR_USERTEAM_ID
    }
})

my_players = []

if roster_data and 'answer' in roster_data:
    my_players = roster_data['answer']
    print(f"✅ Your team has {len(my_players)} players")
    
    with open('futmondo_my_roster.json', 'w', encoding='utf-8') as f:
        json.dump(roster_data, f, indent=2, ensure_ascii=False)
    print("💾 Saved: futmondo_my_roster.json")
else:
    print("❌ Failed to get roster")

# ============================================================================
# 3. GET LEAGUE STANDINGS
# ============================================================================
print("\n\n🏆 Fetching league standings...")

standings_data = api_call('/2/championship/teams', {
    'query': {
        'championshipId': YOUR_CHAMPIONSHIP_ID
    }
})

if standings_data:
    with open('futmondo_standings.json', 'w', encoding='utf-8') as f:
        json.dump(standings_data, f, indent=2, ensure_ascii=False)
    print("✅ Saved: futmondo_standings.json")

# ============================================================================
# 4. GET DETAILED STATS FOR SAMPLE PLAYERS
# ============================================================================
print("\n\n📈 Fetching detailed player stats (sample)...")

detailed_stats = []

if all_players and len(all_players) > 0:
    # Get stats for first 20 players as sample
    sample_size = min(20, len(all_players))
    print(f"Getting detailed stats for {sample_size} players...")
    
    for i, player in enumerate(all_players[:sample_size]):
        player_id = player.get('_id')
        player_name = player.get('name', 'Unknown')
        
        print(f"  {i+1}/{sample_size}: {player_name}...", end='')
        
        stats = api_call('/1/player/summary', {
            'query': {
                'playerId': player_id,
                'championshipId': YOUR_CHAMPIONSHIP_ID
            }
        })
        
        if stats:
            detailed_stats.append(stats)
            print(" ✅")
        else:
            print(" ❌")
    
    if detailed_stats:
        with open('futmondo_detailed_stats.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_stats, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved: futmondo_detailed_stats.json ({len(detailed_stats)} players)")

# ============================================================================
# 5. CREATE PANDAS DATAFRAMES FOR ANALYSIS
# ============================================================================
print("\n\n📊 Creating DataFrames for analysis...")

if all_players:
    # Convert to DataFrame
    df_players = pd.json_normalize(all_players)
    
    print(f"\n✅ Players DataFrame: {df_players.shape[0]} rows × {df_players.shape[1]} columns")
    print("\nColumns available:")
    for col in sorted(df_players.columns):
        print(f"  - {col}")
    
    # Save to CSV
    df_players.to_csv('futmondo_players.csv', index=False, encoding='utf-8')
    print("\n💾 Saved: futmondo_players.csv")
    
    # Show basic stats
    print("\n" + "="*80)
    print("📊 DATASET OVERVIEW")
    print("="*80)
    
    print(f"\nTotal Players: {len(df_players)}")
    
    if 'team' in df_players.columns:
        print(f"\nPlayers by Team (Top 10):")
        print(df_players['team'].value_counts().head(10))
    
    if 'position' in df_players.columns:
        print(f"\nPlayers by Position:")
        print(df_players['position'].value_counts())
    
    if 'points' in df_players.columns:
        print(f"\nPoints Statistics:")
        print(df_players['points'].describe())
        
        print(f"\n🌟 Top 10 Players by Points:")
        top_players = df_players.nlargest(10, 'points')[['name', 'team', 'position', 'points', 'value']]
        print(top_players.to_string(index=False))
    
    if 'value' in df_players.columns:
        print(f"\n💰 Most Valuable Players:")
        valuable = df_players.nlargest(10, 'value')[['name', 'team', 'position', 'points', 'value']]
        print(valuable.to_string(index=False))

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n\n" + "="*80)
print("✅ DATA COLLECTION COMPLETE!")
print("="*80)

print(f"""
📁 Files Created:
  1. futmondo_all_players.json - Complete market data ({len(all_players)} players)
  2. futmondo_players.csv - CSV for data analysis
  3. futmondo_my_roster.json - Your team roster
  4. futmondo_standings.json - League standings
  5. futmondo_detailed_stats.json - Detailed stats (sample)

🎯 Next Steps for Your Data Science Project:
  
  1. EXPLORATORY ANALYSIS:
     - Load futmondo_players.csv in Jupyter
     - Analyze point distributions, value vs performance
     - Identify undervalued players
  
  2. FEATURE ENGINEERING:
     - Points per match
     - Value efficiency (points/value)
     - Form trends (recent games)
  
  3. MODELING IDEAS:
     - Predict player points for next gameweek
     - Recommend optimal team selection
     - Find bargain players (high points, low value)
  
  4. GET MORE DATA:
     - Historical gameweek data
     - Match-by-match performance
     - Injury/suspension status

🔄 To get ALL detailed stats for all players, modify the script to:
   - Remove the sample_size limit
   - Add delay between requests (time.sleep(0.5))
   - This will take ~10-30 minutes for all players
""")

print("\n" + "="*80)