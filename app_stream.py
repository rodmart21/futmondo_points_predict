import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Fantasy Points Predictor", page_icon="⚽", layout="wide")

st.title("⚽ Fantasy Points Predictor")

# Check API health
try:
    health = requests.get(f"{API_URL}/health").json()
    if health["status"] == "healthy":
        st.success("✓ API Connected")
    else:
        st.error("API Unhealthy")
except:
    st.error("⚠️ Cannot connect to API. Make sure FastAPI is running on port 8000")

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
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"player_name": player_name, "round": round_num}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        points = data['predicted_points']
                        classification = data['classification']
                        
                        # Display classification with color coding
                        if classification == "high-point-round":
                            st.success(f"### 🔥 HIGH POINT ROUND")
                            color = "#28a745"
                        elif classification == "medium-point-round":
                            st.info(f"### 📊 MEDIUM POINT ROUND")
                            color = "#ffc107"
                        else:
                            st.warning(f"### 📉 LOW POINT ROUND")
                            color = "#dc3545"
                        
                        # Show detailed points in expander
                        with st.expander("📈 See exact predicted points"):
                            st.metric(label="Predicted Points", value=f"{points:.2f}")
                        
                        # Visual progress bar
                        st.progress(min(points / 10, 1.0))
                        
                    else:
                        error = response.json()
                        st.error(f"Error: {error['detail']}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")

# TAB 2: Top 10 Players
with tab2:
    st.header("Top 10 Players by Round")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        top_round = st.number_input("Select Round", min_value=1, max_value=38, value=1, key="top_round")
        
        if st.button("Get Top 10", type="primary"):
            with st.spinner("Loading top players..."):
                try:
                    response = requests.get(f"{API_URL}/top-players/{top_round}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        top_players = data['top_players']
                        
                        if not top_players:
                            st.info("No players found for this round")
                        else:
                            # Store in session state
                            st.session_state['top_players'] = top_players
                            st.session_state['display_round'] = top_round
                    else:
                        error = response.json()
                        st.error(f"Error: {error['detail']}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
    
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
                    color = "#d4edda"
                elif classification == "medium-point-round":
                    emoji = "📊"
                    color = "#fff3cd"
                else:
                    emoji = "📉"
                    color = "#f8d7da"
                
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