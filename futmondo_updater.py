import requests
import time
import pandas as pd
import hashlib
from sqlalchemy import create_engine, text
from datetime import datetime
import logging
from functions.utils import extract_individual_player_features

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.FileHandler('futmondo_updater.log'),
        logging.StreamHandler()
    ]
)

class FutmondoUpdater:
    def __init__(self, auth_token, user_id, championship_id, userteam_id, db_config):
        self.auth_token = auth_token
        self.user_id = user_id
        self.championship_id = championship_id
        self.userteam_id = userteam_id
        
        self.headers = {
            'Authorization': f'Bearer {auth_token}',
            'x-futmondo-token': auth_token,
            'x-futmondo-userid': user_id
        }
        
        # Database connection
        self.engine = create_engine(
            f"postgresql+psycopg2://{db_config['username']}@{db_config['host']}:"
            f"{db_config['port']}/{db_config['database']}"
        )
    
    def get_all_players(self):
        """Fetch all player IDs from market"""
        logging.info("Fetching all players from market...")
        
        response = requests.post(
            'https://api.futmondo.com/1/market/players',
            headers=self.headers,
            json={
                'query': {
                    'championshipId': self.championship_id,
                    'userteamId': self.userteam_id
                }
            }
        )
        
        if response.status_code == 200:
            market_data = response.json()
            players = market_data['answer']
            logging.info(f"Found {len(players)} players")
            return players
        else:
            logging.error(f"Failed to fetch players: {response.status_code}")
            return []
    
    def get_player_details(self, player_id, player_name):
        """Get detailed data for a specific player"""
        response = requests.post(
            'https://api.futmondo.com/1/player/summary',
            headers=self.headers,
            json={
                'query': {
                    'playerId': player_id,
                    'championshipId': self.championship_id
                }
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logging.warning(f"Failed to fetch details for {player_name}: {response.status_code}")
            return None
    
    def generate_unique_id(self, player_id, round_number):
        """Generate unique ID for a player-round combination"""
        unique_string = f"{player_id}_{round_number}"
        return hashlib.sha256(unique_string.encode()).hexdigest()
    
    def get_existing_ids(self):
        """Get all existing unique_ids from database"""
        try:
            query = "SELECT unique_id FROM futmondo_points"
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                existing_ids = {row[0] for row in result}
            logging.info(f"Found {len(existing_ids)} existing records in database")
            return existing_ids
        except Exception as e:
            logging.warning(f"Could not fetch existing IDs (table might not exist): {e}")
            return set()
    
    def get_existing_player_ids(self):
        """Get all existing player_ids from players_features table"""
        try:
            query = "SELECT player_id FROM players_features"
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                existing_player_ids = {row[0] for row in result}
            logging.info(f"Found {len(existing_player_ids)} existing players in features table")
            return existing_player_ids
        except Exception as e:
            logging.warning(f"Could not fetch existing player IDs (table might not exist): {e}")
            return set()
    
    def get_existing_prediction_ids(self):
        """Get all existing unique_ids from futmondo_predictions table"""
        try:
            query = "SELECT unique_id FROM futmondo_predictions"
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                existing_ids = {row[0] for row in result}
            logging.info(f"Found {len(existing_ids)} existing prediction records in database")
            return existing_ids
        except Exception as e:
            logging.warning(f"Could not fetch existing prediction IDs (table might not exist): {e}")
            return set()
    
    def process_player_data(self, player_data, existing_ids, existing_prediction_ids):
        """Process player data and create training samples + prediction sample"""
        training_samples = []
        prediction_sample = None
        
        answer = player_data['answer']
        player_info = answer['data']
        historical_matches = answer.get('points', [])
        
        if len(historical_matches) < 3:
            return training_samples, prediction_sample
        
        # Extract price history
        prices = [p['price'] for p in answer['prices']]
        max_price = max(prices)
        min_price = min(prices)
        
        # Create training samples (sliding windows with target)
        if len(historical_matches) >= 4:
            for i in range(len(historical_matches) - 3):
                match_window = historical_matches[i:i+4]
                feature_matches = match_window[:3]
                target_match = match_window[3]
                
                round_number = target_match['round']
                unique_id = self.generate_unique_id(player_info['id'], round_number)
                
                # Skip if already exists
                if unique_id in existing_ids:
                    continue
                
                feature_points = [m['points'] for m in feature_matches]
                last_3_average = sum(feature_points) / 3
                
                sample_dict = {
                    'unique_id': unique_id,
                    'player_id': player_info['id'],
                    'name': player_info['name'],
                    'team': player_info['team'],
                    'role': player_info['role'],
                    'round': round_number,
                    'home_average': player_info['average']['homeAverage'],
                    'away_average': player_info['average']['awayAverage'],
                    'overall_average': player_info['average']['average'],
                    'last_3_average': last_3_average,
                    'current_price': player_info['value'],
                    'max_price': max_price,
                    'min_price': min_price,
                    'is_home_target': target_match['isHomeTeam'],
                    'match_minus_1': feature_matches[2]['points'],
                    'match_minus_2': feature_matches[1]['points'],
                    'match_minus_3': feature_matches[0]['points'],
                    'target_points': target_match['points']
                }
                training_samples.append(sample_dict)
        
        # Create prediction sample for next round (last 3 matches, no target)
        if len(historical_matches) >= 3:
            last_3_matches = historical_matches[-3:]
            
            # The next round is the last played round + 1
            next_round = historical_matches[-1]['round'] + 1
            unique_id = self.generate_unique_id(player_info['id'], next_round)
            
            # Skip if already exists in predictions
            if unique_id not in existing_prediction_ids:
                feature_points = [m['points'] for m in last_3_matches]
                last_3_average = sum(feature_points) / 3
                
                # We don't know if next match is home or away, so we'll set it to None
                # You may need to fetch this from a fixtures/schedule API if available
                prediction_dict = {
                    'unique_id': unique_id,
                    'player_id': player_info['id'],
                    'name': player_info['name'],
                    'team': player_info['team'],
                    'role': player_info['role'],
                    'round': next_round,
                    'home_average': player_info['average']['homeAverage'],
                    'away_average': player_info['average']['awayAverage'],
                    'overall_average': player_info['average']['average'],
                    'last_3_average': last_3_average,
                    'current_price': player_info['value'],
                    'max_price': max_price,
                    'min_price': min_price,
                    'is_home_target': None,  # Unknown for next match
                    'match_minus_1': last_3_matches[2]['points'],
                    'match_minus_2': last_3_matches[1]['points'],
                    'match_minus_3': last_3_matches[0]['points']
                }
                prediction_sample = prediction_dict
        
        return training_samples, prediction_sample
    
    def update_database(self):
        """Main method to fetch data and update database"""
        logging.info("=" * 60)
        logging.info(f"Starting daily update at {datetime.now()}")
        logging.info("=" * 60)
        
        # Get existing IDs to avoid duplicates
        existing_ids = self.get_existing_ids()
        existing_player_ids = self.get_existing_player_ids()
        existing_prediction_ids = self.get_existing_prediction_ids()
        
        # Fetch all players
        all_players = self.get_all_players()
        if not all_players:
            logging.error("No players found. Aborting.")
            return
        
        # Process each player
        all_training_samples = []
        all_prediction_samples = []
        current_features = []
        
        for i, player in enumerate(all_players):
            player_id = player['id']
            player_name = player['name']
            
            logging.info(f"Processing {i+1}/{len(all_players)}: {player_name}")
            
            player_data = self.get_player_details(player_id, player_name)
            if player_data:
                # Process historical data for training + prediction data
                training_samples, prediction_sample = self.process_player_data(
                    player_data, existing_ids, existing_prediction_ids
                )
                all_training_samples.extend(training_samples)
                
                if prediction_sample:
                    all_prediction_samples.append(prediction_sample)
                
                # Extract current features for prediction (only if player doesn't exist)
                if player_id not in existing_player_ids:
                    try:
                        features = extract_individual_player_features(player_data)
                        current_features.append(features)
                    except Exception as e:
                        logging.warning(f"Failed to extract features for {player_name}: {e}")
                else:
                    logging.debug(f"Player {player_name} already exists in features table, skipping")
            
            time.sleep(0.5)  # Rate limiting
        
        # Save historical training data to futmondo_points table
        if all_training_samples:
            df_historical = pd.DataFrame(all_training_samples)
            logging.info(f"Found {len(df_historical)} new historical records to insert")
            df_historical.to_sql('futmondo_points', self.engine, if_exists='append', index=False)
            logging.info(f"Successfully inserted {len(df_historical)} historical records")
        else:
            logging.info("No new historical records to insert")
        
        # Save prediction data to futmondo_predictions table
        if all_prediction_samples:
            df_predictions = pd.DataFrame(all_prediction_samples)
            logging.info(f"Found {len(df_predictions)} new prediction records to insert")
            df_predictions.to_sql('futmondo_predictions', self.engine, if_exists='append', index=False)
            logging.info(f"Successfully inserted {len(df_predictions)} prediction records")
        else:
            logging.info("No new prediction records to insert")
        
        # Save current player features to players_features table
        if current_features:
            df_features = pd.DataFrame(current_features)
            logging.info(f"Found {len(df_features)} new player features to insert")
            
            # Remove duplicates by player_id (keep first occurrence)
            df_features.drop_duplicates(subset='player_id', inplace=True)
            
            # Append new players only (don't replace)
            df_features.to_sql('players_features', self.engine, if_exists='append', index=False)
            logging.info(f"Successfully inserted {len(df_features)} new player features")
        else:
            logging.info("No new player features to insert")
        
        logging.info("Daily update completed successfully")
        logging.info("=" * 60)


def main():
    # Configuration
    AUTH_TOKEN = '5e65_20f468a19ad19ef73979642df99d6603'
    USER_ID = '56c6b62085617f9b1dc7d061'
    CHAMPIONSHIP_ID = '5f7b19924dcd043e8a092dd4'
    USERTEAM_ID = '5f7b33ea6f9120324dedcb15'
    
    db_config = {
        'username': 'rodrigo',
        'host': 'localhost',
        'port': '5432',
        'database': 'futmondo'
    }
    
    # Create updater and run
    updater = FutmondoUpdater(
        AUTH_TOKEN, USER_ID, CHAMPIONSHIP_ID, USERTEAM_ID, db_config
    )
    
    try:
        updater.update_database()
    except Exception as e:
        logging.error(f"Error during update: {e}", exc_info=True)


if __name__ == "__main__":
    main()