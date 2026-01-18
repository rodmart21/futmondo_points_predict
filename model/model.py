import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Database connection
username = os.getenv('DB_USER')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
database = os.getenv('DB_NAME')

engine = create_engine(f'postgresql+psycopg2://{username}@{host}:{port}/{database}')

# Load data
logger.info("Loading data from database...")
query = "SELECT * FROM full_training_data"
df = pd.read_sql(query, engine)
logger.info(f"Data shape: {df.shape}")

# Feature columns
feature_columns = [
    'home_average', 'away_average', 'rating', 'overall_average', 'last_3_average', 
    'current_price', 'max_price', 'min_price', 'is_home_target', 
    'match_minus_1', 'match_minus_2', 'match_minus_3',
    'matchup_prob_win', 'matchup_prob_draw', 'matchup_prob_loss', 'is_home',
    'form_trend', 'home_away_diff', 'price_vs_max', 'price_volatility', 
    'recent_momentum', 'home_form_interaction', 'away_form_interaction', 
    'matchup_strength', 'team_expected_performance', 'delantero_matchup_bonus',
    'centrocampista_matchup_bonus', 'defensa_matchup_bonus', 'portero_matchup_bonus', 
    'home_matchup_boost', 'difficult_matchup', 'easy_matchup'
]

# Clean data
df_clean = df.dropna(subset=['target_points'] + feature_columns)
logger.info(f"Clean data shape: {df_clean.shape}")

X = df_clean[feature_columns]
y = df_clean['target_points']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

logger.info("Training model...")

# Models
gb_model = GradientBoostingRegressor(
    n_estimators=150, max_depth=5, learning_rate=0.05, 
    subsample=0.8, min_samples_split=10, min_samples_leaf=4, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=150, max_depth=12, min_samples_split=8, 
    min_samples_leaf=3, max_features='sqrt', random_state=42
)

ridge_model = Ridge(alpha=10.0, random_state=42)

# Voting ensemble
model = VotingRegressor(
    estimators=[('gb', gb_model), ('rf', rf_model), ('ridge', ridge_model)],
    weights=[2, 2, 1]
)

model.fit(X_train, y_train)

# Evaluate
y_pred_test = model.predict(X_test)
mae = np.mean(np.abs(y_test - y_pred_test))
rmse = np.sqrt(np.mean((y_test - y_pred_test)**2))
r2 = model.score(X_test, y_test)

logger.info(f"\nModel Performance:")
logger.info(f"MAE: {mae:.3f}")
logger.info(f"RMSE: {rmse:.3f}")
logger.info(f"R² Score: {r2:.3f}")

# Save model
logger.info("\nSaving model...")
model_package = {
    'model': model,
    'scaler': scaler,
    'feature_columns': feature_columns
}

joblib.dump(model_package, 'data/model/fantasy_model_complete.pkl')
logger.info("Model saved successfully!")