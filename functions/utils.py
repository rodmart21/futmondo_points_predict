

# Enhanced feature engineering
def create_advanced_features(df):
    """Create additional features for better predictions"""
    df_enhanced = df.copy()
    
    # Form indicators
    df_enhanced['form_trend'] = df_enhanced['last_3_average'] - df_enhanced['overall_average']
    df_enhanced['home_away_diff'] = df_enhanced['home_average'] - df_enhanced['away_average']
    
    # Price features
    df_enhanced['price_vs_max'] = df_enhanced['current_price'] / df_enhanced['max_price']
    df_enhanced['price_volatility'] = (df_enhanced['max_price'] - df_enhanced['min_price']) / df_enhanced['min_price']
    
    # Recent form momentum
    df_enhanced['recent_momentum'] = (
        df_enhanced['match_minus_1'] * 0.5 + 
        df_enhanced['match_minus_2'] * 0.3 + 
        df_enhanced['match_minus_3'] * 0.2
    )
    
    # Interaction features
    df_enhanced['home_form_interaction'] = (
        df_enhanced['is_home_target'] * df_enhanced['home_average']
    )
    df_enhanced['away_form_interaction'] = (
        (1 - df_enhanced['is_home_target']) * df_enhanced['away_average']
    )
    
    # Matchup features (if columns exist)
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
        ).fillna(0.0)  # Fill NaN with 0.0
        
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