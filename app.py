from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from app_utils import get_player_features, get_all_players_for_round

app = FastAPI(title="Fantasy Points Predictor")

MODEL = joblib.load('data/model/fantasy_model_complete.pkl')

class PredictionRequest(BaseModel):
    player_name: str
    round: int

class PredictionResponse(BaseModel):
    player_name: str
    round: int
    predicted_points: float
    classification: str

class LineupResponse(BaseModel):
    round: int
    team: str
    lineup: dict
    total_points: float

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
    """Predict fantasy points for a player in a specific round"""
    features_dict = get_player_features(request.player_name, request.round)
    
    if not features_dict:
        raise HTTPException(
            status_code=404, 
            detail=f"Player '{request.player_name}' not found for round {request.round}"
        )
    
    features_df = pd.DataFrame([features_dict])
    features_df = features_df[MODEL['feature_columns']]
    
    if features_df.isna().any().any():
        raise HTTPException(
            status_code=400,
            detail="Missing required features for prediction"
        )
    
    features_scaled = MODEL['scaler'].transform(features_df)
    prediction = MODEL['model'].predict(features_scaled)[0]
    
    return PredictionResponse(
        player_name=request.player_name,
        round=request.round,
        predicted_points=round(float(prediction), 1),
        classification=classify_points(prediction)
    )

@app.get("/lineup/{team}/{round_num}", response_model=LineupResponse)
def get_team_lineup(team: str, round_num: int):
    """Get best lineup for a specific team in a specific round"""
    if round_num < 1 or round_num > 38:
        raise HTTPException(
            status_code=400,
            detail="Round must be between 1 and 38"
        )
    
    players_data = get_all_players_for_round(round_num)
    
    if not players_data:
        raise HTTPException(
            status_code=404,
            detail=f"No players found for round {round_num}"
        )
    
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
                "position": features_dict.get('position', 'Unknown'),
                "predicted_points": round(float(prediction), 1)
            })
        except Exception:
            continue
    
    if not predictions:
        raise HTTPException(
            status_code=404,
            detail=f"No players found for team '{team}' in round {round_num}"
        )
    
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
    
    return LineupResponse(
        round=round_num,
        team=team,
        lineup=lineup,
        total_points=round(total_points, 1)
    )

@app.get("/teams/{round_num}")
def get_available_teams(round_num: int):
    """Get list of all teams available for a specific round"""
    if round_num < 1 or round_num > 38:
        raise HTTPException(
            status_code=400,
            detail="Round must be between 1 and 38"
        )
    
    players_data = get_all_players_for_round(round_num)
    
    if not players_data:
        raise HTTPException(
            status_code=404,
            detail=f"No players found for round {round_num}"
        )
    
    teams = set()
    for features_dict in players_data.values():
        team = features_dict.get('team')
        if team:
            teams.add(team)
    
    return {"round": round_num, "teams": sorted(list(teams))}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None}