import pandas as pd

def apply_feature_engineering(df):
    """
    Applies feature engineering to the compressor dataset.
    """
    print("Engineering features...")
    
    if 'TP3' in df.columns and 'Reservoirs' in df.columns:
        df['pressure_gap'] = df['TP3'] - df['Reservoirs']
        
    if 'Oil_temperature' in df.columns:
        df['temp_rolling_mean'] = df['Oil_temperature'].rolling(window=10).mean()
        df['temp_rolling_std'] = df['Oil_temperature'].rolling(window=10).std()
        
    # Drop rows with NaN values created by rolling windows
    df.dropna(inplace=True)
    return df
