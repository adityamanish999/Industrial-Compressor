import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score
import joblib
import os

def train_models(filepath='cleaned_data/compressor_cleaned.csv'):
    print("Loading processed data...")
    if not os.path.exists(filepath):
        raise FileNotFoundError("Processed data not found. Run data_preprocessing.py first.")
        
    df = pd.read_csv(filepath)
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
        
    scaler = joblib.load('models/scaler.pkl')
    X_unscaled = df.values
    X = scaler.transform(X_unscaled)
    print(f"Data shape: {X.shape}")
    
    os.makedirs('models', exist_ok=True)
    
    # 1. K-Means
    print("Training K-Means...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X)
    joblib.dump(kmeans, 'models/kmeans.pkl')
    print("K-Means saved.")
    
    # 2. One-Class SVM (trained on a subsample to avoid memory/time issues)
    print("Training One-Class SVM (on a 10,000 row subsample)...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=min(10000, len(X)), replace=False)
    X_sample = X[sample_indices]
    
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    ocsvm.fit(X_sample)
    joblib.dump(ocsvm, 'models/ocsvm.pkl')
    print("One-Class SVM saved.")
    
    # 3. DBSCAN (also on a subsample, predicting new points with DBSCAN is not straightforward, 
    # but we can fit it to see anomalies in the subsample, or use a different approach).
    # Since DBSCAN doesn't have a 'predict' method for new data out of the box, we just train it.
    print("Training DBSCAN (on the same 10,000 row subsample)...")
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_sample)
    joblib.dump(dbscan, 'models/dbscan.pkl')
    print("DBSCAN trained.")

    print("Model training complete!")

if __name__ == '__main__':
    train_models()
