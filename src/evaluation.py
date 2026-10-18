import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

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


def main(filepath='cleaned_data/compressor_cleaned.csv'):
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    scaler = joblib.load('models/scaler.pkl')
    X = scaler.transform(df.values)
    results = evaluate(
        X, df.index,
        joblib.load('models/kmeans.pkl'),
        joblib.load('models/ocsvm.pkl'),
        joblib.load('models/dbscan_labels.pkl'),
        joblib.load('models/sample_idx.pkl'),
        joblib.load('models/threshold.pkl'),
    )
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
