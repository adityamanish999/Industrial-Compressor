import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from feature_engineering import apply_feature_engineering

def load_and_preprocess(filepath='data/MetroPT3(AirCompressor).csv', sample_frac=0.01):
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found. Please run download_data.py first.")
    
    # Due to dataset size (15M rows), we'll read and sample it. 
    # For a production pipeline we might resample time series, but here we read and downsample or aggregate.
    # The dataset has a 'timestamp' column.
    
    df = pd.read_csv(filepath)
    
    print("Preprocessing data...")
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'Unnamed: 0' in df.columns:
        # Sometimes the index is unnamed
        df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Missing values
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    # Resample to 1-minute frequency to reduce data size from 1Hz (60x reduction)
    print("Resampling to 1-minute frequency...")
    # Select only numeric columns for mean resampling
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_resampled = df[numeric_cols].resample('1min').mean().dropna()
    
    # Feature Engineering
    df_resampled = apply_feature_engineering(df_resampled)
    
    # Normalizing
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_resampled)
    df_scaled = pd.DataFrame(scaled_data, columns=df_resampled.columns, index=df_resampled.index)
    
    return df_resampled, df_scaled, scaler

if __name__ == '__main__':
    df, df_scaled, scaler = load_and_preprocess()
    print("Data preprocessed successfully. Shape:", df_scaled.shape)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Saved scaler to models/scaler.pkl")
    
    df_resampled_path = 'cleaned_data/compressor_cleaned.csv'
    df.to_csv(df_resampled_path)
    print(f"Saved unscaled processed data to {df_resampled_path}")
