import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import zipfile
import joblib
import requests
from feature_engineering import apply_feature_engineering

def download_data(dataset_path='data/MetroPT3(AirCompressor).csv'):
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return

    zip_url = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
    zip_path = "data/metropt3.zip"
    
    print("Downloading MetroPT-3 Dataset from UCI...")
    response = requests.get(zip_url, stream=True)
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data/')
        os.remove(zip_path)
        print(f"Extracted dataset. Ensure the CSV is named properly at {dataset_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")


def load_and_preprocess(filepath='data/MetroPT3(AirCompressor).csv'):
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found. Run data_preprocessing.py to download it first.")
    
    # The dataset is large (15M rows at 1Hz), so it is resampled to 1-minute means below.

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

    # The raw CSV carries an unnamed row-index column that must not become a model feature
    df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')], inplace=True)

    # Duplicates (identical timestamp + readings) and missing values
    before = len(df)
    df = df[~df.index.duplicated(keep='first')]
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")
    df = df.sort_index()
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
    download_data()
    df, df_scaled, scaler = load_and_preprocess()
    print("Data preprocessed successfully. Shape:", df_scaled.shape)
    os.makedirs('models', exist_ok=True)
    os.makedirs('cleaned_data', exist_ok=True)
    
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Saved scaler to models/scaler.pkl")
    
    df_resampled_path = 'cleaned_data/compressor_cleaned.csv'
    df.to_csv(df_resampled_path)
    print(f"Saved unscaled processed data to {df_resampled_path}")
