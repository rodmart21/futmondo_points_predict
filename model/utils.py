import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

def train_ensemble_poisson_regressor(
    X, 
    y, 
    feature_columns,
    test_size=0.2,
    offset=10,
    random_state=42,
    hgb_params=None,
    lgbm_params=None,
    rf_params=None,
    voting_weights=[2, 2, 1],
    show_plot=True
):
    """
    Trains a VotingRegressor (HGB, LGBM, RF) with a Poisson-shifted target.
    Returns the model, scaler, metrics, and feature importance dataframe.
    """
    
    # --- 1. Defaults & Config ---
    hgb_params = hgb_params or {
        'loss': 'poisson', 'learning_rate': 0.05, 'max_iter': 500,
        'max_leaf_nodes': 50, 'min_samples_leaf': 15, 'l2_regularization': 0.1
    }
    lgbm_params = lgbm_params or {
        'objective': 'poisson', 'n_estimators': 1000, 'learning_rate': 0.05,
        'num_leaves': 50, 'subsample': 0.8, 'min_samples_leaf':15, 'colsample_bytree': 0.8, 'verbose': -1
    }
    rf_params = rf_params or {
        'n_estimators': 1000, 'max_depth': 12, 'min_samples_split': 8,
        'min_samples_leaf': 4, 'max_features': 'sqrt'
    }

    # --- 2. Scaling & Splitting ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # Apply Offset to training target
    y_train_shifted = y_train + offset

    # --- 3. Model Definition ---
    hgb = HistGradientBoostingRegressor(**hgb_params, random_state=random_state)
    lgbm = LGBMRegressor(**lgbm_params, random_state=random_state)
    rf = RandomForestRegressor(**rf_params, random_state=random_state)

    model = VotingRegressor(
        estimators=[('hgb', hgb), ('lgbm', lgbm), ('rf', rf)],
        weights=voting_weights
    )

    # --- 4. Training ---
    model.fit(X_train, y_train_shifted)

    # --- 5. Evaluation ---
    raw_preds = model.predict(X_test)
    y_pred_real = raw_preds - offset

    metrics = {
        'mae': mean_absolute_error(y_test, y_pred_real),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_real)),
        'r2': r2_score(y_test, y_pred_real)
    }

    # --- 6. Feature Importance ---
    perm_importance = permutation_importance(
        model, X_test, y_test + offset, n_repeats=10, 
        random_state=random_state, n_jobs=-1
    )

    fi_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': perm_importance.importances_mean,
        'std_dev': perm_importance.importances_std
    }).sort_values(by='importance', ascending=False)

    # --- 7. Visualization ---
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=fi_df.head(15), palette='viridis')
            plt.title('Feature Importance (Permutation)')
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass

    return model, scaler, metrics, fi_df