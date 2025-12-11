import psycopg2
from psycopg2.extras import RealDictCursor
import os

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "futmondo"),
        user=os.getenv("DB_USER", "rodrigo"),
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
        last_3_average,
        current_price,
        max_price,
        min_price,
        is_home_target,
        match_minus_1,
        match_minus_2,
        match_minus_3,
        matchup_prob_win,
        matchup_prob_draw,
        matchup_prob_loss,
        is_home,
        form_trend,
        home_away_diff,
        price_vs_max,
        price_volatility,
        recent_momentum,
        home_form_interaction,
        away_form_interaction,
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
      AND round = %s
    """
    
    cursor.execute(query, (player_name, round_number))
    result = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return result