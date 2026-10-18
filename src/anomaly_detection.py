import json
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import joblib
import os

# Known air-leak failure periods of the MetroPT-3 compressor (from the dataset documentation).
# Confirm these against the dataset page before quoting results.
FAILURE_WINDOWS = [
    ('2020-04-18 00:00', '2020-04-18 23:59'),
    ('2020-05-29 23:30', '2020-05-30 06:00'),
    ('2020-06-05 10:00', '2020-06-07 14:30'),
    ('2020-07-15 14:30', '2020-07-15 19:00'),
]

RESULTS_PATH = 'reports/evaluation_results.json'


def in_failure_window(index):
    """Boolean array: True where the timestamp falls inside a known failure window."""
    mask = np.zeros(len(index), dtype=bool)
    for start, end in FAILURE_WINDOWS:
        mask |= (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    return mask


def window_metrics(flags, index):
    """Compare anomaly flags against the known failure windows."""
    truth = in_failure_window(index)
    flagged = int(flags.sum())
    true_pos = int((flags & truth).sum())
    false_pos = flagged - true_pos
    n_normal = int((~truth).sum())
    windows_hit = 0
    for start, end in FAILURE_WINDOWS:
        in_win = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        windows_hit += int(flags[in_win].any())
    return {
        'rows_evaluated': int(len(flags)),
        'anomalies_flagged': flagged,
        'anomaly_rate': round(flagged / len(flags), 4),
        'precision_on_failure_windows': round(true_pos / flagged, 4) if flagged else 0.0,
        'recall_on_failure_windows': round(true_pos / truth.sum(), 4) if truth.sum() else 0.0,
        'false_positive_rate': round(false_pos / n_normal, 4) if n_normal else 0.0,
        'failure_windows_detected': f'{windows_hit}/{len(FAILURE_WINDOWS)}',
    }


def evaluate(X, index, kmeans, ocsvm, dbscan_labels, sample_idx, threshold):
    """Compute silhouette, failure-window overlap and timing for all three models."""
    results = {'failure_windows': FAILURE_WINDOWS, 'kmeans_threshold': float(threshold)}
    rng = np.random.RandomState(42)

    # K-Means: distance to nearest centroid vs. the training-data threshold
    t0 = time.time()
    km_dist = np.min(kmeans.transform(X), axis=1)
    km_time = time.time() - t0
    km_flags = km_dist > threshold
    sil_idx = rng.choice(len(X), size=min(10000, len(X)), replace=False)
    results['silhouette_kmeans'] = round(float(silhouette_score(X[sil_idx], kmeans.labels_[sil_idx])), 4)
    results['kmeans'] = window_metrics(km_flags, index)
    results['kmeans']['inference_seconds'] = round(km_time, 3)

    # One-Class SVM: -1 = outlier
    t0 = time.time()
    oc_flags = ocsvm.predict(X) == -1
    oc_time = time.time() - t0
    results['ocsvm'] = window_metrics(oc_flags, index)
    results['ocsvm']['inference_seconds'] = round(oc_time, 3)

    # DBSCAN: only exists for the training subsample, label -1 = noise
    db_flags = dbscan_labels == -1
    results['dbscan'] = window_metrics(db_flags, index[sample_idx])
    results['dbscan']['clusters_found'] = int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0))
    if results['dbscan']['clusters_found'] > 1:
        mask = dbscan_labels != -1
        results['silhouette_dbscan'] = round(float(silhouette_score(X[sample_idx][mask], dbscan_labels[mask])), 4)

    os.makedirs('reports', exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {RESULTS_PATH}")
    return results


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
