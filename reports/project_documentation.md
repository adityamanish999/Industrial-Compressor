# Project Documentation & Model Evaluation

## Pipeline
1. `data_preprocessing.py` first downloads MetroPT-3 from UCI if missing, then drops the unnamed index column and duplicate rows, forward-fills missing values, resamples to 1-minute means, engineers features and fits a `StandardScaler`.
2. `anomaly_detection.py` trains K-Means (k=2), One-Class SVM (nu=0.05, 10k subsample) and DBSCAN (eps from k-distance curve, min_samples=10, same subsample), saves a fixed K-Means anomaly threshold, then runs its `evaluate()` step.
3. `streamlit_app.py` serves the K-Means model with the saved threshold.

## Engineered Features
- `pressure_gap` (TP3 - Reservoirs), `pressure_moving_avg` (30-sample TP3 average)
- `temp_rolling_mean`, `temp_rolling_std` (10-sample oil temperature)
- `motor_rolling_mean`, `motor_rolling_std` (10-sample motor current) and `motor_load_state` (current > 1A; threshold is an assumption to tune)

## Evaluation Methodology
The data is largely unlabeled, so accuracy is not applicable. `evaluate()` in `src/anomaly_detection.py` measures, and writes to `reports/evaluation_results.json`:
- **Silhouette score** for K-Means (10,000-row sample) and DBSCAN (non-noise points, if more than one cluster is found).
- **Failure window comparison**: precision, recall, false positive rate and number of failure windows hit, using the known air-leak periods defined in `FAILURE_WINDOWS`.
- **Distance from centroid**: K-Means flags rows whose distance to the nearest centroid exceeds the 95th percentile of *training* distances.
- **Computational cost**: inference time per model on the full cleaned dataset.

Run `python src/anomaly_detection.py` to regenerate the results; they are also shown in the dashboard's "Model Comparison" tab.

### Measured Results (242,475 one-minute rows after cleaning; 57,475 duplicate rows removed)
| Model | Anomaly rate | Precision (failure windows) | Recall | False positive rate | Windows hit | Inference time |
|---|---|---|---|---|---|---|
| K-Means (k=2) | 5.0% | 0.297 | 0.729 | 0.036 | 4/4 | 0.015 s |
| One-Class SVM | 5.2% | 0.073 | 0.186 | 0.049 | 4/4 | 2.9 s |
| DBSCAN (10k subsample) | 5.9% | 0.053 | 0.147 | 0.057 | 4/4 | no predict |

Silhouette score: K-Means 0.490, DBSCAN 0.257 (13 clusters, noise excluded).

Reading the results: K-Means gives the best precision/recall balance, the lowest false positive rate and is about 190x faster than One-Class SVM at inference. DBSCAN's `eps` is chosen from the k-distance curve (90th percentile of each point's distance to its 10th neighbour, eps=1.49); an earlier fixed eps=0.5 labelled 62% of points as noise on the 22 standardized features. Precision is low for every model because failure windows are a tiny share of the data and anomalies also occur during other unusual operation. The failure-window timestamps were entered from the MetroPT-3 documentation and should be verified against the dataset page.

## Model Comparison
- **K-Means:** deployed in the dashboard; fast inference on new rows via distance to centroid. Assumes roughly spherical clusters.
- **One-Class SVM:** learns a boundary around normal behaviour; trained on a 10,000-row subsample because training scales poorly with data size.
- **DBSCAN:** finds noise points without a preset cluster count, but has no `predict()` and is memory intensive, so it is only applied to the training subsample.

## Limitations & Future Work
- Anomaly rate is set by percentile/`nu` choices, not by labeled data.
- Failure-window metrics depend on the accuracy of the window timestamps.
- Possible extensions: autoencoder reconstruction error (TensorFlow), streaming with Kafka/Spark, alert notifications.
