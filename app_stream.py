import streamlit as st
import pandas as pd
import joblib
from app_utils import get_player_features, get_all_players_for_round

st.set_page_config(page_title="Fantasy Points Predictor", page_icon="⚽", layout="wide")

st.title("⚽ Fantasy Points Predictor")

# Load model at startup
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
    
    # Get features from database
    features_dict = get_player_features(player_name, round_num)
    
    if not features_dict:
        return None, f"Player '{player_name}' not found for round {round_num}"
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features_dict])
    
    # Ensure correct column order
    features_df = features_df[MODEL['feature_columns']]
    
    # Handle missing values
    if features_df.isna().any().any():
        return None, "Missing required features for prediction"
    
    # Scale and predict
    features_scaled = MODEL['scaler'].transform(features_df)
    prediction = MODEL['model'].predict(features_scaled)[0]
    
    return float(prediction), None

def get_top_players_for_round(round_num: int):
    """Get top 10 predicted players for a specific round"""
    if MODEL is None:
        return None, "Model not loaded"
    
    if round_num < 1 or round_num > 38:
        return None, "Round must be between 1 and 38"
    
    # Get all players for the round
    players_data = get_all_players_for_round(round_num)
    
    if not players_data:
        return None, f"No players found for round {round_num}"
    
    predictions = []
    
    for player_name, features_dict in players_data.items():
        try:
            # Convert to DataFrame
            features_df = pd.DataFrame([features_dict])
            
            # Ensure correct column order
            features_df = features_df[MODEL['feature_columns']]
            
            # Skip if missing values
            if features_df.isna().any().any():
                continue
            
            # Scale and predict
            features_scaled = MODEL['scaler'].transform(features_df)
            prediction = MODEL['model'].predict(features_scaled)[0]
            
            predictions.append({
                "player_name": player_name,
                "predicted_points": float(prediction),
                "classification": classify_points(prediction)
            })
        except Exception:
            continue
    
    # Sort by predicted points and get top 10
    top_10 = sorted(predictions, key=lambda x: x['predicted_points'], reverse=True)[:10]
    
    return top_10, None

st.divider()

# Create tabs for different views
tab1, tab2 = st.tabs(["🔍 Player Prediction", "🏆 Top 10 Players"])

# TAB 1: Individual Player Prediction
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
                    
                    # Display classification with color coding
                    if classification == "high-point-round":
                        st.success(f"### 🔥 HIGH POINT ROUND")
                    elif classification == "medium-point-round":
                        st.info(f"### 📊 MEDIUM POINT ROUND")
                    else:
                        st.warning(f"### 📉 LOW POINT ROUND")
                    
                    # Show detailed points in expander
                    with st.expander("📈 See exact predicted points"):
                        st.metric(label="Predicted Points", value=f"{points:.2f}")
                    
                    # Visual progress bar
                    st.progress(min(points / 10, 1.0))

# TAB 2: Top 10 Players
with tab2:
    st.header("Top 10 Players by Round")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        top_round = st.number_input("Select Round", min_value=1, max_value=38, value=1, key="top_round")
        
        if st.button("Get Top 10", type="primary"):
            with st.spinner("Loading top players..."):
                top_players, error = get_top_players_for_round(top_round)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    if not top_players:
                        st.info("No players found for this round")
                    else:
                        # Store in session state
                        st.session_state['top_players'] = top_players
                        st.session_state['display_round'] = top_round
    
    # Display results
    if 'top_players' in st.session_state:
        with col2:
            st.subheader(f"Round {st.session_state['display_round']} - Top Performers")
            
            for i, player in enumerate(st.session_state['top_players'], 1):
                classification = player['classification']
                points = player['predicted_points']
                name = player['player_name']
                
                # Color coding
                if classification == "high-point-round":
                    emoji = "🔥"
                elif classification == "medium-point-round":
                    emoji = "📊"
                else:
                    emoji = "📉"
                
                # Display player card
                with st.container():
                    col_rank, col_name, col_class, col_points = st.columns([0.5, 2, 1.5, 1])
                    
                    with col_rank:
                        st.markdown(f"### {i}")
                    
                    with col_name:
                        st.markdown(f"**{name}**")
                    
                    with col_class:
                        if classification == "high-point-round":
                            st.success(f"{emoji} High")
                        elif classification == "medium-point-round":
                            st.info(f"{emoji} Medium")
                        else:
                            st.warning(f"{emoji} Low")
                    
                    with col_points:
                        st.metric("Points", f"{points:.1f}")
                    
                    st.divider()
            
            # Export option
            if st.button("📥 Export to CSV"):
                df = pd.DataFrame(st.session_state['top_players'])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"top_10_round_{st.session_state['display_round']}.csv",
                    mime="text/csv"
                )