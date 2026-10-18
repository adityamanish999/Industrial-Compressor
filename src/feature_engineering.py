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
        
    if 'Motor_current' in df.columns:
        df['motor_rolling_mean'] = df['Motor_current'].rolling(window=10).mean()
        df['motor_rolling_std'] = df['Motor_current'].rolling(window=10).std()
        # Motor load state: 1 = loaded (current above 1A), 0 = idle/off
        df['motor_load_state'] = (df['Motor_current'] > 1.0).astype(int)

    if 'TP3' in df.columns:
        df['pressure_moving_avg'] = df['TP3'].rolling(window=30).mean()

    # Drop rows with NaN values created by rolling windows
    df.dropna(inplace=True)
    return df
