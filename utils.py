
# Function to extract individual player features for next match. Including prices and next match probabilities.
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
        is_home = match.get('ho', None)
        
        # Try to get opponent and probabilities
        if is_home is not None:
            try:
                if is_home:
                    opponent = match.get('a', {}).get('id', {}).get('name', None)
                    team_win_prob = match.get('wc', {}).get('h', None)
                    opponent_win_prob = match.get('wc', {}).get('a', None)
                else:
                    opponent = match.get('h', {}).get('id', {}).get('name', None)
                    team_win_prob = match.get('wc', {}).get('a', None)
                    opponent_win_prob = match.get('wc', {}).get('h', None)
                
                draw_prob = match.get('wc', {}).get('d', None)
            except Exception as e:
                print(f"Error parsing match data for {data.get('name')}: {e}")
    
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
        'match_minus_5': fitness[-5] if len(fitness) >= 5 else None}
