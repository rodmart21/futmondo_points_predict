# streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

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

# Input form
col1, col2 = st.columns(2)

with col1:
    player_name = st.text_input("Player Name", placeholder="e.g., John Smith")

with col2:
    round_num = st.number_input("Round", min_value=1, max_value=38, value=1)

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
                    st.success(f"**Predicted Points:** {data['predicted_points']:.2f}")
                else:
                    error = response.json()
                    st.error(f"Error: {error['detail']}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")