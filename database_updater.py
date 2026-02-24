import requests
import json
import pandas as pd
import numpy as np
import logging
import os
from sqlalchemy import create_engine
from src.utils import (
    create_round_features,
    add_matchup_probabilities,
    create_advanced_features,
    get_team_stats,
    standardize_team_names,
    predict_upcoming_matches,
    TEAM_ID_MAPPING,
    TEAM_NAME_MAPPING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_championship_players(auth_token, user_id, championship_id):
    """Fetch all players from championship endpoint"""
    headers = {
        'Authorization': f'Bearer {auth_token}',
        'x-futmondo-token': auth_token,
        'x-futmondo-userid': user_id
    }
    
    endpoint_url = 'https://api.futmondo.com/5/league/championshipplayers'
    query_params = {'championshipId': championship_id}
    
    logging.info(f"Fetching all players from: {endpoint_url}")
    
    try:
        response = requests.post(endpoint_url, headers=headers, json={'query': query_params})
        
        if response.status_code == 200:
            full_response = response.json()
            player_list = full_response.get('answer', {}).get('players')
            
            if isinstance(player_list, list):
                logging.info(f"Successfully extracted {len(player_list)} player records")
                return player_list
            else:
                logging.error("Player list not found in expected structure")
                return None
        else:
            logging.error(f"Failed to fetch players: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        return None


def clean_numpy_values(df):
    """Convert numpy types to native Python types"""
    def clean_value(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    
    return df.applymap(clean_value)


def main():
    AUTH_TOKEN = '5e65_20f468a19ad19ef73979642df99d6603'
    USER_ID = '56c6b62085617f9b1dc7d061'
    CHAMPIONSHIP_ID = '5f7b19924dcd043e8a092dd4'
    
    # Connection to database
    db_config = {
        'username': 'rodrigo',
        'host': 'localhost',
        'port': '5432',
        'database': 'futmondo_full_players_info'}
    
    engine = create_engine(
        f"postgresql+psycopg2://{db_config['username']}@{db_config['host']}:"
        f"{db_config['port']}/{db_config['database']}")
    
    # Load existing player points data
    query = "SELECT * FROM player_points_def"  # Use player_points_def, single player_points is 
    df = pd.read_sql(query, engine)
    df.shape
    
    # 1. Fetch players from the last round. Important to run after the end of the round.
    player_list = fetch_championship_players(AUTH_TOKEN, USER_ID, CHAMPIONSHIP_ID)
    if not player_list:
        logging.error("Failed to fetch player data. Aborting.")
        return
    
    # 2. Create rolling features and update player_points table
    df_rolling = create_round_features(player_list, target_rounds=[18, 19, 20, 21])
    
    # Clean specific IDs
    ids_to_remove = ['51ffb6b7113981890700003a', '5211d81592d57d145a0000ce', '520e4ee4a776cc826b00004b']
    df_rolling = df_rolling[~df_rolling['team_id'].isin(ids_to_remove)]
    df_rolling['team'] = df_rolling['team_id'].map(TEAM_ID_MAPPING)

    # Include new round values: 2 per round (those round values and next one with target_points=nan)
    try:
        final_df = pd.concat([df, df_rolling], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['round', 'name'], keep='last')
        final_df['target_points'] = final_df.groupby('player_id')['match_minus_1'].shift(-1)  # Update target points
    except:
        final_df = df_rolling
    
    # Save updated player points data
    final_df.to_sql('player_points', engine, if_exists='replace', index=False)
    logging.info("Saved player_points to database")
    
    # 3. Load matches data and predict upcoming matches
    df_la_liga = pd.read_sql("SELECT * FROM la_liga_matches", engine) # Data with probabilities
    df_liga_next = pd.read_csv('/Users/rodrigo/football-data-analytics/futmondo_points_predict/data/la_liga_next_rounds copy.csv')
    
    # Clean and standardize team names
    df_la_liga = standardize_team_names(df_la_liga, is_historical=True)
    df_liga_next = standardize_team_names(df_liga_next, is_historical=False)
    
    historical_teams = set(df_la_liga['HomeTeam'].unique()) | set(df_la_liga['AwayTeam'].unique())
    upcoming_teams = set(df_liga_next['Home Team'].unique()) | set(df_liga_next['Away Team'].unique())
    missing_teams = upcoming_teams - historical_teams

    if missing_teams:
        logging.warning(f"Teams in upcoming matches not found in historical data: {missing_teams}")
    
    team_stats = get_team_stats(df_la_liga)
    predictions_df = predict_upcoming_matches(df_liga_next, df_la_liga, team_stats)
    updated_df = pd.concat([df_la_liga, predictions_df], ignore_index=True)  # combine historical and predicted matches
    
    # 4. Add matchup probabilities and create advanced features
    df_with_matchups = add_matchup_probabilities(final_df, updated_df)
    df_enriched = create_advanced_features(df_with_matchups)
    df_enriched = clean_numpy_values(df_enriched)
    
    # 5. Save enriched training data
    df_enriched.to_sql('full_training_data', engine, if_exists='replace', index=False)
    logging.info("Saved full_training_data to database")
    logging.info("Database update completed successfully")


if __name__ == "__main__":
    main()