import joblib
import os
from feature_engineering import apply_feature_engineering

def predict_anomalies(df_new):
    """
    Takes a new DataFrame, applies feature engineering, scales it,
    and returns predictions using the saved K-Means model.
    """
    scaler_path = 'models/scaler.pkl'
    model_path = 'models/kmeans.pkl'
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Models not found. Train models first.")
        
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    
    # Feature engineering
    df_features = apply_feature_engineering(df_new.copy())
    
    # Scale
    X_scaled = scaler.transform(df_features.values)
    
    # Predict clusters
    clusters = model.predict(X_scaled)
    return clusters, X_scaled
