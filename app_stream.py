import streamlit as st
import pandas as pd
import joblib
from app_utils import get_player_features, get_all_players_for_round

st.set_page_config(page_title="Fantasy Points Predictor", page_icon="⚽", layout="wide")

st.title("⚽ Fantasy Points Predictor")

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('data/model/fantasy_model_complete.pkl')
        st.success("✓ Model Loaded Successfully")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

MODEL = load_model()

def classify_points(points: float) -> str:
    """Classify predicted points into categories"""
    if points <= 3:
        return "low-point-round"
    elif points <= 6:
        return "medium-point-round"
    else:
        return "high-point-round"

def predict_player_points(player_name: str, round_num: int):
    """Predict fantasy points for a player in a specific round"""
    if MODEL is None:
        return None, "Model not loaded"
    
    features_dict = get_player_features(player_name, round_num)
    
    if not features_dict:
        return None, f"Player '{player_name}' not found for round {round_num}"
    
    features_df = pd.DataFrame([features_dict])
    features_df = features_df[MODEL['feature_columns']]
    
    if features_df.isna().any().any():
        return None, "Missing required features for prediction"
    
    features_scaled = MODEL['scaler'].transform(features_df)
    prediction = MODEL['model'].predict(features_scaled)[0]
    
    return round(float(prediction), 1), None

def get_team_lineup(team: str, round_num: int):
    """Get best lineup for a specific team in a specific round"""
    if MODEL is None:
        return None, "Model not loaded"
    
    if round_num < 1 or round_num > 38:
        return None, "Round must be between 1 and 38"
    
    players_data = get_all_players_for_round(round_num)
    
    if not players_data:
        return None, f"No players found for round {round_num}"
    
    predictions = []
    
    for player_name, features_dict in players_data.items():
        try:
            if features_dict.get('team') != team:
                continue
            
            features_df = pd.DataFrame([features_dict])
            features_df = features_df[MODEL['feature_columns']]
            
            if features_df.isna().any().any():
                continue
            
            features_scaled = MODEL['scaler'].transform(features_df)
            prediction = MODEL['model'].predict(features_scaled)[0]
            
            predictions.append({
                "player_name": player_name,
                "position": features_dict.get('role', 'Unknown'),
                "predicted_points": round(float(prediction), 1)})
        except Exception:
            continue
    
    if not predictions:
        return None, f"No players found for team '{team}' in round {round_num}"
    
    df = pd.DataFrame(predictions).sort_values('predicted_points', ascending=False)
    
    lineup = {
        'portero': [],
        'defensa': [],
        'centrocampista': [],
        'delantero': []
    }
    
    for position in ['portero', 'defensa', 'centrocampista', 'delantero']:
        position_players = df[df['position'] == position]
        
        if position == 'portero':
            lineup[position] = position_players.head(1).to_dict('records')
        elif position == 'defensa':
            lineup[position] = position_players.head(4).to_dict('records')
        elif position == 'centrocampista':
            lineup[position] = position_players.head(3).to_dict('records')
        elif position == 'delantero':
            lineup[position] = position_players.head(3).to_dict('records')
    
    total_points = sum(p['predicted_points'] for pos in lineup.values() for p in pos)
    
    return {'lineup': lineup, 'total_points': round(total_points, 1)}, None

def get_available_teams(round_num: int):
    """Get list of teams for a specific round"""
    players_data = get_all_players_for_round(round_num)
    
    if not players_data:
        return []
    
    teams = set()
    for features_dict in players_data.values():
        team = features_dict.get('team')
        if team:
            teams.add(team)
    
    return sorted(list(teams))

st.divider()

tab1, tab2 = st.tabs(["🔍 Player Prediction", "⚽ Team Lineup"])

with tab1:
    st.header("Predict Individual Player Points")
    
    col1, col2 = st.columns(2)
    
    with col1:
        player_name = st.text_input("Player Name", placeholder="e.g., John Smith")
    
    with col2:
        round_num = st.number_input("Round", min_value=1, max_value=38, value=1, key="player_round")
    
    if st.button("Predict Points", type="primary"):
        if not player_name:
            st.warning("Please enter a player name")
        else:
            with st.spinner("Predicting..."):
                points, error = predict_player_points(player_name, round_num)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    classification = classify_points(points)
                    
                    if classification == "high-point-round":
                        st.success(f"### 🔥 HIGH POINT ROUND")
                    elif classification == "medium-point-round":
                        st.info(f"### 📊 MEDIUM POINT ROUND")
                    else:
                        st.warning(f"### 📉 LOW POINT ROUND")
                    
                    st.metric(label="Predicted Points", value=f"{points}")
                    st.progress(min(points / 10, 1.0))

with tab2:
    st.header("Best Team Lineup")
    
    lineup_round = st.number_input("Select Round", min_value=1, max_value=38, value=1, key="lineup_round_input")
    
    teams = get_available_teams(lineup_round)
    
    if not teams:
        st.warning(f"No teams found for round {lineup_round}")
    else:
        selected_team = st.selectbox("Select Team", teams, key="team_selector")
        
        if st.button("Get Lineup", type="primary"):
            with st.spinner("Building lineup..."):
                result, error = get_team_lineup(selected_team, lineup_round)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.session_state['current_lineup'] = result
                    st.session_state['current_team'] = selected_team
                    st.session_state['current_round'] = lineup_round
        
        if 'current_lineup' in st.session_state:
            st.divider()
            st.subheader(f"{st.session_state['current_team']} - Round {st.session_state['current_round']}")
            
            lineup = st.session_state['current_lineup']['lineup']
            
            st.markdown("#### 🥅 Goalkeeper (1)")
            for player in lineup['portero']:
                col_name, col_pts = st.columns([3, 1])
                with col_name:
                    st.write(f"**{player['player_name']}**")
                with col_pts:
                    st.metric("", f"{player['predicted_points']}")
            
            st.divider()
            
            st.markdown("#### 🛡️ Defenders (4)")
            for player in lineup['defensa']:
                col_name, col_pts = st.columns([3, 1])
                with col_name:
                    st.write(f"**{player['player_name']}**")
                with col_pts:
                    st.metric("", f"{player['predicted_points']}")
            
            st.divider()
            
            st.markdown("#### ⚙️ Midfielders (3)")
            for player in lineup['centrocampista']:
                col_name, col_pts = st.columns([3, 1])
                with col_name:
                    st.write(f"**{player['player_name']}**")
                with col_pts:
                    st.metric("", f"{player['predicted_points']}")
            
            st.divider()
            
            st.markdown("#### ⚡ Forwards (3)")
            for player in lineup['delantero']:
                col_name, col_pts = st.columns([3, 1])
                with col_name:
                    st.write(f"**{player['player_name']}**")
                with col_pts:
                    st.metric("", f"{player['predicted_points']}")
            
            st.divider()