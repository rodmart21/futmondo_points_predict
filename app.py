from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from app_utils import get_player_features, get_all_players_for_round

app = FastAPI(title="Fantasy Points Predictor")

# Load model once at startup
MODEL = joblib.load('data/model/fantasy_model_complete.pkl')

class PredictionRequest(BaseModel):
    player_name: str
    round: int

class PredictionResponse(BaseModel):
    player_name: str
    round: int
    predicted_points: float
    classification: str

class TopPlayersResponse(BaseModel):
    round: int
    top_players: list[dict]

def classify_points(points: float) -> str:
    """Classify predicted points into categories"""
    if points <= 3:
        return "low-point-round"
    elif points <= 6:
        return "medium-point-round"
    else:
        return "high-point-round"

@app.get("/")
def read_root():
    return {"message": "Fantasy Points Predictor API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_points(request: PredictionRequest):
    """
    Predict fantasy points for a player in a specific round
    """
    # Get features from database
    features_dict = get_player_features(request.player_name, request.round)
    
    if not features_dict:
        raise HTTPException(
            status_code=404, 
            detail=f"Player '{request.player_name}' not found for round {request.round}"
        )
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features_dict])
    
    # Ensure correct column order
    features_df = features_df[MODEL['feature_columns']]
    
    # Handle missing values
    if features_df.isna().any().any():
        raise HTTPException(
            status_code=400,
            detail="Missing required features for prediction"
        )
    
    # Scale and predict
    features_scaled = MODEL['scaler'].transform(features_df)
    prediction = MODEL['model'].predict(features_scaled)[0]
    
    return PredictionResponse(
        player_name=request.player_name,
        round=request.round,
        predicted_points=float(prediction),
        classification=classify_points(prediction)
    )

@app.get("/top-players/{round_num}", response_model=TopPlayersResponse)
def get_top_players(round_num: int):
    """
    Get top 10 predicted players for a specific round
    """
    if round_num < 1 or round_num > 38:
        raise HTTPException(
            status_code=400,
            detail="Round must be between 1 and 16"
        )
    
    # Get all players for the round
    players_data = get_all_players_for_round(round_num)
    
    if not players_data:
        raise HTTPException(
            status_code=404,
            detail=f"No players found for round {round_num}"
        )
    
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
    
    return TopPlayersResponse(
        round=round_num,
        top_players=top_10
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None}