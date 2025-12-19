import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor # Assuming this is correctly imported
import os

def get_db_connection():
    """
    Get database connection using Neon credentials.
    """
    
        # Try Streamlit secrets (for deployment)
    try:
        import streamlit as st
        return psycopg2.connect(
            host=st.secrets["DATABASE_HOST"],
            port=st.secrets["DATABASE_PORT"],
            database=st.secrets["DATABASE_NAME"],
            user=st.secrets["DATABASE_USER"],
            password=st.secrets["DATABASE_PASSWORD"],
            sslmode='require',
            cursor_factory=RealDictCursor)
    except:
            # Fall back to .env (for local development)
            from dotenv import load_dotenv
            load_dotenv()
            return psycopg2.connect(
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                cursor_factory=RealDictCursor
            )

def get_player_features(player_name: str, round_number: int):
    """
    Query to get all necessary features for a player in a specific round
    from the full_training_data table
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT 
        home_average,
        away_average,
        overall_average,
        last_2_average,
        current_price,
        is_home,
        match_minus_1,
        match_minus_2,
        matchup_prob_win,
        matchup_prob_draw,
        matchup_prob_loss,
        form_trend,
        home_away_diff,
        price_per_point,
        price_efficiency,
        recent_momentum,
        home_form_interaction,
        away_form_interaction,
        location_adjusted_average,
        matchup_strength,
        team_expected_performance,
        delantero_matchup_bonus,
        centrocampista_matchup_bonus,
        defensa_matchup_bonus,
        portero_matchup_bonus,
        home_matchup_boost,
        difficult_matchup,
        easy_matchup
    FROM full_training_data
    WHERE name = %s 
      AND round = %s;
    """
    
    cursor.execute(query, (player_name, round_number))
    result = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return result


def get_all_players_for_round(round_num: int) -> dict:
    """
    Get features for all players in a specific round from full_training_data table
    
    Args:
        round_num: The round number (1-38)
    
    Returns:
        Dictionary mapping player_name -> features_dict
        Example: {
            "Player Name 1": {"feature1": val1, "feature2": val2, ...},
            "Player Name 2": {"feature1": val1, "feature2": val2, ...}
        }
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Query to get all players and their features for the round
        query = """
            SELECT
            name,
            home_average,
            away_average,
            overall_average,
            last_2_average,
            current_price,
            is_home,
            match_minus_1,
            match_minus_2,
            matchup_prob_win,
            matchup_prob_draw,
            matchup_prob_loss,
            form_trend,
            home_away_diff,
            price_per_point,
            price_efficiency,
            recent_momentum,
            home_form_interaction,
            away_form_interaction,
            location_adjusted_average,
            matchup_strength,
            team_expected_performance,
            delantero_matchup_bonus,
            centrocampista_matchup_bonus,
            defensa_matchup_bonus,
            portero_matchup_bonus,
            home_matchup_boost,
            difficult_matchup,
            easy_matchup
            FROM full_training_data
            WHERE round = %s
        """
        
        cursor.execute(query, (round_num,))
        results = cursor.fetchall()
        
        # Convert to dictionary format
        players_dict = {}
        for row in results:
            player_name = row['name']
            # Create a copy without the 'name' field
            features = {k: v for k, v in row.items() if k != 'name'}
            players_dict[player_name] = features
        
        cursor.close()
        conn.close()
        
        return players_dict
        
    except Exception as e:
        print(f"Error fetching players for round {round_num}: {e}")
        return {}