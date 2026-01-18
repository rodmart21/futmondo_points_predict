import pandas as pd
import json
import hashlib

# Team mappings
TEAM_ID_MAPPING = {
    '504e581e4d8bec9a670000cf': 'Levante',
    '520347e4b8d07d930b00000f': 'Girona',
    '504e581e4d8bec9a670000d9': 'Celta',
    '504e581e4d8bec9a670000c7': 'Barcelona',
    '51ffb00e78b20d7f0700003f': 'Oviedo',
    '504e581e4d8bec9a670000c9': 'Ath Bilbao',
    '504e581e4d8bec9a670000cb': 'Valencia',
    '504e581e4d8bec9a670000ca': 'Vallecano',
    '504e581e4d8bec9a670000c6': 'Real Madrid',
    '504e581e4d8bec9a670000c8': 'Ath Madrid',
    '51b890f5b986415a2c000012': 'Villarreal',
    '504e581e4d8bec9a670000d1': 'Osasuna',
    '504e581e4d8bec9a670000cd': 'Getafe',
    '51b889b1e401a15f2c0000f0': 'Elche',
    '504e581e4d8bec9a670000d0': 'Espanol',
    '504e581e4d8bec9a670000d5': 'Sevilla',
    '504e581e4d8bec9a670000ce': 'Sociedad',
    '504e581e4d8bec9a670000cc': 'Betis',
    '52038563b8d07d930b00008a': 'Alaves',
    '504e581e4d8bec9a670000d2': 'Mallorca'
}

TEAM_NAME_MAPPING = {
    'Real Sociedad': 'Sociedad',
    'Atlético Madrid': 'Ath Madrid',
    'Celta Vigo': 'Celta',
    'Alavés': 'Alaves',
    'Rayo Vallecano': 'Vallecano',
    'Real Betis': 'Betis',
    'Real Oviedo': 'Oviedo',
    'Athletic Bilbao': 'Ath Bilbao',
    'Athletic Club': 'Ath Bilbao',
    'Espanyol': 'Espanol'
}

# Enhanced feature engineering
def create_advanced_features(df):
    """Create additional features for better predictions"""
    df_enhanced = df.copy()
    
    # Form indicators
    df_enhanced['form_trend'] = df_enhanced['last_2_average'] - df_enhanced['overall_average']
    df_enhanced['home_away_diff'] = df_enhanced['home_average'] - df_enhanced['away_average']
    
    # Price features
    df_enhanced['price_per_point'] = df_enhanced['current_price'] / (df_enhanced['overall_average'] + 0.1)
    df_enhanced['price_efficiency'] = df_enhanced['overall_average'] / df_enhanced['current_price']
    
    # Recent form momentum
    df_enhanced['recent_momentum'] = (
        df_enhanced['match_minus_1'] * 0.6 + 
        df_enhanced['match_minus_2'] * 0.4
    )
    
    # Interaction features
    df_enhanced['home_form_interaction'] = (
        df_enhanced['is_home'] * df_enhanced['home_average']
    )
    df_enhanced['away_form_interaction'] = (
        (1 - df_enhanced['is_home']) * df_enhanced['away_average']
    )
    
    # Location-adjusted average (use home avg when home, away avg when away)
    df_enhanced['location_adjusted_average'] = (
        df_enhanced['is_home'] * df_enhanced['home_average'] + 
        (1 - df_enhanced['is_home']) * df_enhanced['away_average']
    )
    
    # Matchup features
    if 'matchup_prob_win' in df_enhanced.columns:
        # Overall matchup strength
        df_enhanced['matchup_strength'] = (
            df_enhanced['matchup_prob_win'] - df_enhanced['matchup_prob_loss']
        )
        
        # Expected team performance (higher win prob = better for all players)
        df_enhanced['team_expected_performance'] = (
            df_enhanced['matchup_prob_win'] * 3 + 
            df_enhanced['matchup_prob_draw'] * 1
        )
        
        # Role-specific matchup bonuses
        # Delanteros (forwards) benefit most from high win probability
        delantero_mask = df_enhanced['role'].str.contains('delantero', case=False, na=False)
        df_enhanced['delantero_matchup_bonus'] = 0.0
        df_enhanced.loc[delantero_mask, 'delantero_matchup_bonus'] = (
            df_enhanced.loc[delantero_mask, 'matchup_prob_win'] * 2.0
        ).fillna(0.0)
        
        # Centrocampistas (midfielders) benefit moderately
        centro_mask = df_enhanced['role'].str.contains('centrocampista', case=False, na=False)
        df_enhanced['centrocampista_matchup_bonus'] = 0.0
        df_enhanced.loc[centro_mask, 'centrocampista_matchup_bonus'] = (
            df_enhanced.loc[centro_mask, 'matchup_prob_win'] * 1.2
        ).fillna(0.0)
        
        # Defensas benefit from high win prob (clean sheet bonus) but also from draw
        defensa_mask = df_enhanced['role'].str.contains('defensa', case=False, na=False)
        df_enhanced['defensa_matchup_bonus'] = 0.0
        df_enhanced.loc[defensa_mask, 'defensa_matchup_bonus'] = (
            df_enhanced.loc[defensa_mask, 'matchup_prob_win'] * 1.0 + 
            df_enhanced.loc[defensa_mask, 'matchup_prob_draw'] * 0.5
        ).fillna(0.0)
        
        # Porteros (goalkeepers) benefit most from low loss probability (clean sheets)
        portero_mask = df_enhanced['role'].str.contains('portero', case=False, na=False)
        df_enhanced['portero_matchup_bonus'] = 0.0
        df_enhanced.loc[portero_mask, 'portero_matchup_bonus'] = (
            (1 - df_enhanced.loc[portero_mask, 'matchup_prob_loss']) * 1.5
        ).fillna(0.0)
        
        # Home advantage interaction with matchup
        df_enhanced['home_matchup_boost'] = (
            df_enhanced['is_home'] * df_enhanced['matchup_prob_win'] * 0.5
        )
        
        # Difficult matchup indicator (high loss probability)
        df_enhanced['difficult_matchup'] = (df_enhanced['matchup_prob_loss'] > 0.5).astype(int)
        
        # Easy matchup indicator (high win probability)
        df_enhanced['easy_matchup'] = (df_enhanced['matchup_prob_win'] > 0.5).astype(int)
    
    return df_enhanced


def add_matchup_probabilities(df_players, df_liga):
    """
    Add matchup probabilities to player data based on their team and round
    """
    df_enriched = df_players.copy()
    
    # Initialize new columns
    df_enriched['matchup_prob_win'] = None
    df_enriched['matchup_prob_draw'] = None
    df_enriched['matchup_prob_loss'] = None
    df_enriched['is_home'] = None
    df_enriched['opponent'] = None
    
    for idx, player in df_enriched.iterrows():
        team = player['team']
        round_num = player['round']
        
        # Find the matchup for this team in this round
        home_match = df_liga[(df_liga['Round'] == round_num) & (df_liga['HomeTeam'] == team)]
        away_match = df_liga[(df_liga['Round'] == round_num) & (df_liga['AwayTeam'] == team)]
        
        if not home_match.empty:
            # Team is playing at home
            match = home_match.iloc[0]
            df_enriched.at[idx, 'matchup_prob_win'] = match['Prob_Home_Norm']
            df_enriched.at[idx, 'matchup_prob_draw'] = match['Prob_Draw_Norm']
            df_enriched.at[idx, 'matchup_prob_loss'] = match['Prob_Away_Norm']
            df_enriched.at[idx, 'is_home'] = 1
            df_enriched.at[idx, 'opponent'] = match['AwayTeam']
            
        elif not away_match.empty:
            # Team is playing away
            match = away_match.iloc[0]
            df_enriched.at[idx, 'matchup_prob_win'] = match['Prob_Away_Norm']
            df_enriched.at[idx, 'matchup_prob_draw'] = match['Prob_Draw_Norm']
            df_enriched.at[idx, 'matchup_prob_loss'] = match['Prob_Home_Norm']
            df_enriched.at[idx, 'is_home'] = 0
            df_enriched.at[idx, 'opponent'] = match['HomeTeam']
    
    return df_enriched


# Calculate average probabilities for each team when playing home and away
def get_team_stats(df):
    stats = {}
    
    for team in df['HomeTeam'].unique():
        home_matches = df[df['HomeTeam'] == team]
        if len(home_matches) > 0:
            stats[team] = {
                'home_win_prob': home_matches['Prob_Home_Norm'].mean(),
                'home_draw_prob': home_matches['Prob_Draw_Norm'].mean(),
                'home_lose_prob': home_matches['Prob_Away_Norm'].mean()
            }
    
    for team in df['AwayTeam'].unique():
        away_matches = df[df['AwayTeam'] == team]
        if len(away_matches) > 0:
            if team not in stats:
                stats[team] = {}
            stats[team]['away_win_prob'] = away_matches['Prob_Away_Norm'].mean()
            stats[team]['away_draw_prob'] = away_matches['Prob_Draw_Norm'].mean()
            stats[team]['away_lose_prob'] = away_matches['Prob_Home_Norm'].mean()
    
    return stats


def standardize_team_names(df, is_historical=True):
    """Standardize team names and strip whitespace"""
    df = df.copy()
    
    if is_historical:
        df['HomeTeam'] = df['HomeTeam'].str.strip()
        df['AwayTeam'] = df['AwayTeam'].str.strip()
    else:
        df['Home Team'] = df['Home Team'].str.strip().replace(TEAM_NAME_MAPPING)
        df['Away Team'] = df['Away Team'].str.strip().replace(TEAM_NAME_MAPPING)
    
    return df


def predict_upcoming_matches(upcoming_df, historical_df, team_stats):
    """Create predictions for upcoming matches based on historical team stats"""
    predictions = []
    
    league_avg_home_win = historical_df['Prob_Home_Norm'].mean()
    league_avg_draw = historical_df['Prob_Draw_Norm'].mean()
    league_avg_away_win = historical_df['Prob_Away_Norm'].mean()
    
    for idx, row in upcoming_df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        round_num = row['Round']
        
        # Get home team's home performance
        if home_team in team_stats and 'home_win_prob' in team_stats[home_team]:
            home_win_strength = team_stats[home_team]['home_win_prob']
            home_draw_strength = team_stats[home_team]['home_draw_prob']
        else:
            home_win_strength = league_avg_home_win
            home_draw_strength = league_avg_draw
        
        # Get away team's away performance
        if away_team in team_stats and 'away_win_prob' in team_stats[away_team]:
            away_win_strength = team_stats[away_team]['away_win_prob']
            away_draw_strength = team_stats[away_team]['away_draw_prob']
        else:
            away_win_strength = league_avg_away_win
            away_draw_strength = league_avg_draw
        
        # Calculate probabilities
        prob_home = (home_win_strength + (1 - away_win_strength)) / 2
        prob_away = (away_win_strength + (1 - home_win_strength)) / 2
        prob_draw = (home_draw_strength + away_draw_strength) / 2
        
        # Normalize probabilities
        total = prob_home + prob_draw + prob_away
        prob_home_norm = prob_home / total
        prob_draw_norm = prob_draw / total
        prob_away_norm = prob_away / total
        
        # Convert to odds
        avg_h = 1 / prob_home_norm if prob_home_norm > 0 else 999
        avg_d = 1 / prob_draw_norm if prob_draw_norm > 0 else 999
        avg_a = 1 / prob_away_norm if prob_away_norm > 0 else 999
        
        predictions.append({
            'Date': None,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'FTHG': None,
            'FTAG': None,
            'FTR': None,
            'AvgH': round(avg_h, 2),
            'AvgD': round(avg_d, 2),
            'AvgA': round(avg_a, 2),
            'Prob_Home': round(prob_home, 6),
            'Prob_Draw': round(prob_draw, 6),
            'Prob_Away': round(prob_away, 6),
            'Total': round(total, 6),
            'Prob_Home_Norm': round(prob_home_norm, 6),
            'Prob_Draw_Norm': round(prob_draw_norm, 6),
            'Prob_Away_Norm': round(prob_away_norm, 6),
            'Round': round_num
        })
    
    return pd.DataFrame(predictions)

# Function to extract individual player features from JSON data
def extract_individual_player_features(json_data):
    answer = json_data['answer']
    data = answer['data']
    
    # Prices are at answer level, not data level
    prices_list = answer.get('prices', [])
    if prices_list:
        prices = [p.get('price') for p in prices_list if 'price' in p]
        max_price = max(prices) if prices else None
        min_price = min(prices) if prices else None
    else:
        max_price = None
        min_price = None
    
    # Current price is in data.market
    current_price = data.get('market', {}).get('p') if data.get('market') else None
    
    # Match is at answer level, not data level
    match = answer.get('match', {})
    
    # Default values for missing data
    is_home = None
    opponent = None
    team_win_prob = None
    draw_prob = None
    opponent_win_prob = None
    
    # Check if match data exists
    if match:
        # Get player's team name
        player_team = data.get('team')
        
        # Get match teams
        home_team = match.get('h', {}).get('id', {}).get('name', None)
        away_team = match.get('a', {}).get('id', {}).get('name', None)
        
        # Determine if player is home or away by comparing team names
        if player_team == home_team:
            is_home = True
            opponent = away_team
            team_win_prob = match.get('wc', {}).get('h', None)
            opponent_win_prob = match.get('wc', {}).get('a', None)
        elif player_team == away_team:
            is_home = False
            opponent = home_team
            team_win_prob = match.get('wc', {}).get('a', None)
            opponent_win_prob = match.get('wc', {}).get('h', None)
        else:
            # Player's team doesn't match either team in the match
            print(f"Warning: Player team '{player_team}' doesn't match home '{home_team}' or away '{away_team}'")
        
        draw_prob = match.get('wc', {}).get('d', None)
    
    # Get last 5 match points (fitness array)
    fitness = data.get('average', {}).get('fitness', [])
    
    # Ensure we have 5 elements
    while len(fitness) < 5:
        fitness.insert(0, None)
    
    return {
        'player_id': data.get('id'),
        'name': data.get('name'),
        'team': data.get('team'),
        'role': data.get('role'),
        'total_points': data.get('total', {}).get('points'),
        'matches_played': data.get('total', {}).get('played'),
        'average': data.get('average', {}).get('average'),
        'home_average': data.get('average', {}).get('homeAverage'),
        'away_average': data.get('average', {}).get('awayAverage'),
        'last_5_average': data.get('average', {}).get('averageLastFive'),
        'current_price': current_price,
        'max_price': max_price,
        'min_price': min_price,
        'is_home_next': is_home,
        'opponent_next': opponent,
        'team_win_prob': team_win_prob,
        'draw_prob': draw_prob,
        'opponent_win_prob': opponent_win_prob,
        'match_minus_1': fitness[-1] if len(fitness) >= 1 else None,
        'match_minus_2': fitness[-2] if len(fitness) >= 2 else None,
        'match_minus_3': fitness[-3] if len(fitness) >= 3 else None,
        'match_minus_4': fitness[-4] if len(fitness) >= 4 else None,
        'match_minus_5': fitness[-5] if len(fitness) >= 5 else None
    }


def create_round_features(players_list, target_rounds, lookback_window=2):
    """
    Create rolling features for specified rounds from player data.
    
    The fitness array ALWAYS contains the last 5 completed rounds.
    fitness = [10, 2, 16, 3, 12] means:
    - Index 0 = 5 rounds ago
    - Index 1 = 4 rounds ago  
    - Index 2 = 3 rounds ago
    - Index 3 = 2 rounds ago
    - Index 4 = 1 round ago (most recent)
    
    Examples:
    - For round 18 (next/prediction): match_minus_1=12 (idx 4), match_minus_2=3 (idx 3), target=None
    - For round 17 (last available): match_minus_1=3 (idx 3), match_minus_2=16 (idx 2), target=12 (idx 4)
    - For round 16: match_minus_1=16 (idx 2), match_minus_2=2 (idx 1), target=3 (idx 3)
    
    Parameters:
    - json_file_path: Path to JSON file with player data
    - target_rounds: List of rounds (e.g., [18] for next round, [14,15,16,17] for training)
    - lookback_window: Number of previous rounds as features (default: 2)
    
    Returns:
    - DataFrame with rolling features
    """

    # Flatten nested dictionaries
    flattened = []
    for player in players_list:
        flat = {k: v for k, v in player.items() 
                if k not in ['userteamId', 'userteam', 'userteamSlug']}
        
        if 'average' in player and isinstance(player['average'], dict):
            for k, v in player['average'].items():
                flat[f'average_{k}'] = v
        
        flattened.append(flat)
    
    df = pd.DataFrame(flattened)
    
    # The fitness array has 5 elements representing the last 5 completed rounds
    # fitness[4] is always the most recent completed round
    # We need to know what round number that is
    
    # Determine the most recent completed round (fitness[4])
    # If we're predicting the next round, it's max(target_rounds) - 1
    # If we're training/analyzing completed rounds, it's max(target_rounds)
    most_recent_completed = max(target_rounds) - 1
    
    # Map fitness array indices to round numbers
    # fitness[4] = most_recent_completed
    # fitness[3] = most_recent_completed - 1
    # fitness[2] = most_recent_completed - 2
    # fitness[1] = most_recent_completed - 3
    # fitness[0] = most_recent_completed - 4
    
    all_rounds = []
    
    for target_round in target_rounds:
        round_df = pd.DataFrame({
            'player_id': df['id'],
            'name': df['name'],
            'role': df['role'],
            'round': target_round,
            'team_id': df['teamId'],
            'home_average': df['average_homeAverage'],
            'away_average': df['average_awayAverage'],
            'overall_average': df['average_average'],
            'current_price': df['value'],
            'matches_played': df['average_matches'],
            'rating': df['rating']
        })
        
        # Calculate lookback features
        lookback_values = []
        for i in range(1, lookback_window + 1):
            lookback_round = target_round - i
            # Convert round number to fitness array index
            # fitness_idx = 4 - (most_recent_completed - lookback_round)
            fitness_idx = 4 - (most_recent_completed - lookback_round)
            
            round_df[f'match_minus_{i}'] = df['average_fitness'].apply(
                lambda x: x[fitness_idx] if isinstance(x, list) and 0 <= fitness_idx < len(x) else 0
            )
            lookback_values.append(round_df[f'match_minus_{i}'])
        
        # Calculate average of lookback matches
        round_df[f'last_{lookback_window}_average'] = pd.concat(lookback_values, axis=1).mean(axis=1)
        
        # Determine target points
        # Target is available only if target_round <= most_recent_completed
        if target_round <= most_recent_completed:
            target_fitness_idx = 4 - (most_recent_completed - target_round)
            round_df['target_points'] = df['average_fitness'].apply(
                lambda x: x[target_fitness_idx] if isinstance(x, list) and 0 <= target_fitness_idx < len(x) else None
            )
        else:
            # Future round - no target available
            round_df['target_points'] = None
        
        # Create unique identifier
        round_df['unique_id'] = (df['id'].astype(str) + '_' + str(target_round)).apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()
        )
        
        all_rounds.append(round_df)
    
    return pd.concat(all_rounds, ignore_index=True)
