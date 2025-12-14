import pandas as pd
import json
import hashlib

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


def create_round_features(json_file_path, rounds=[13, 14, 15, 16]):
    """
    Load player data from JSON and create rolling features for multiple rounds.
    
    Parameters:
    - json_file_path: Path to the JSON file with player data
    - rounds: List of rounds to generate features for (default: [13, 14, 15, 16])
    
    Returns:
    - DataFrame with rolling features for all specified rounds
    """
    
    # Helper function to flatten player data
    def flatten_player_data(player):
        flat_data = {}
        excluded_fields = ['userteamId', 'userteam', 'userteamSlug']
        
        for key, value in player.items():
            if key in excluded_fields:
                continue
                
            if key == 'average' and isinstance(value, dict):
                for avg_key, avg_value in value.items():
                    flat_data[f'average_{avg_key}'] = avg_value
            elif key == 'clause' and isinstance(value, dict):
                for clause_key, clause_value in value.items():
                    flat_data[f'clause_{clause_key}'] = clause_value
            else:
                flat_data[key] = value
        
        return flat_data
    
    # Load players data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        players_data = json.load(f)
    
    if not isinstance(players_data, list):
        players_data = [players_data]
    
    # Create DataFrame
    flattened_players = [flatten_player_data(player) for player in players_data]
    df = pd.DataFrame(flattened_players)
    
    # Keep only needed columns and preserve all original values
    keep_cols = ['id', 'name', 'slug', 'role', 'points', 'value',
                 'rating', 'average_average', 'average_homeAverage', 
                 'average_awayAverage', 'average_averageLastFive', 
                 'average_matches', 'average_fitness', 'change', 'teamId',
                 'clause_price', 'clause_suggestedClause']
    
    df_clean = df[keep_cols].copy()
    
    # Create features for each round
    all_rounds = []
    
    for round_num in rounds:
        df_round = pd.DataFrame()
        
        df_round['player_id'] = df_clean['id']
        df_round['name'] = df_clean['name']
        df_round['role'] = df_clean['role']
        df_round['round'] = round_num
        df_round['home_average'] = df_clean['average_homeAverage']
        df_round['away_average'] = df_clean['average_awayAverage']
        df_round['overall_average'] = df_clean['average_average']
        df_round['current_price'] = df_clean['value']
        df_round['matches_played'] = df_clean['average_matches']
        df_round['rating'] = df_clean['rating']
        # df_round['change_clause'] = df_clean['change']
        # df_round['clause_price'] = df_clean['clause_price']
        df_round['team_id'] = df_clean['teamId']
        # df_round['value'] = df_clean['value']
        
        # Fitness array indices: [round 11, round 12, round 13, round 14, round 15]
        # For each round, we use 2 previous matches and predict the current round
        # Round 13: use rounds 12, 11 (fitness[1,0]) → target is round 13 (fitness[2])
        # Round 14: use rounds 13, 12 (fitness[2,1]) → target is round 14 (fitness[3])
        # Round 15: use rounds 14, 13 (fitness[3,2]) → target is round 15 (fitness[4])
        # Round 16: use rounds 15, 14 (fitness[4,3]) → target is unknown
        
        round_to_idx = {11: 0, 12: 1, 13: 2, 14: 3, 15: 4}
        target_idx = round_to_idx.get(round_num, None)
        
        if round_num >= 13 and round_num <= 15:
            # For training rounds, get previous 2 matches
            idx_minus_1 = round_to_idx.get(round_num - 1, 0)
            idx_minus_2 = round_to_idx.get(round_num - 2, 0)
        else:  # round 16
            # For prediction round, use last 2 available matches
            idx_minus_1 = 4  # round 15
            idx_minus_2 = 3  # round 14
            target_idx = None
        
        df_round['match_minus_1'] = df_clean['average_fitness'].apply(
            lambda x: x[idx_minus_1] if isinstance(x, list) and len(x) > idx_minus_1 else 0
        )
        df_round['match_minus_2'] = df_clean['average_fitness'].apply(
            lambda x: x[idx_minus_2] if isinstance(x, list) and len(x) > idx_minus_2 else 0
        )
        
        # Calculate last_2_average from the 2 previous matches
        df_round['last_2_average'] = (df_round['match_minus_1'] + df_round['match_minus_2']) / 2
        
        # Target points
        if target_idx is not None:
            df_round['target_points'] = df_clean['average_fitness'].apply(
                lambda x: x[target_idx] if isinstance(x, list) and len(x) > target_idx else None
            )
        else:
            df_round['target_points'] = None
        
        # Create unique_id
        df_round['unique_id'] = (df_clean['id'].astype(str) + '_' + 
                                 str(round_num)).apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()
        )
        
        all_rounds.append(df_round)
    
    # Combine all rounds
    df_final = pd.concat(all_rounds, ignore_index=True)
    
    return df_final
