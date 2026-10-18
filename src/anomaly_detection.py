import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
import joblib
import os
from evaluation import evaluate

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
    # Fixed alert threshold: 95th percentile of training distances (reused by the dashboard)
    threshold = np.percentile(np.min(kmeans.transform(X), axis=1), 95)
    joblib.dump(threshold, 'models/threshold.pkl')
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
    
    # 3. DBSCAN (same subsample; it has no predict(), so only the subsample labels are kept)
    print("Training DBSCAN (on the same 10,000 row subsample)...")
    # eps from the k-distance curve: 90th percentile of each point's distance to its 10th neighbour
    k_dist = NearestNeighbors(n_neighbors=10).fit(X_sample).kneighbors(X_sample)[0][:, -1]
    eps = float(np.percentile(k_dist, 90))
    print(f"DBSCAN eps from k-distance curve: {eps:.2f}")
    dbscan = DBSCAN(eps=eps, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_sample)
    joblib.dump(dbscan, 'models/dbscan.pkl')
    joblib.dump(dbscan_labels, 'models/dbscan_labels.pkl')
    joblib.dump(sample_indices, 'models/sample_idx.pkl')
    print("DBSCAN trained.")

    print("Model training complete!")

    print("Evaluating models against known failure windows...")
    index = pd.to_datetime(df.index)
    evaluate(X, index, kmeans, ocsvm, dbscan_labels, sample_indices, threshold)

if __name__ == '__main__':
    train_models()
