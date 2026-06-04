# Project Documentation & Model Evaluation

## Model Evaluation Metrics
Because this is an Unsupervised Machine Learning project with largely unlabeled data, traditional classification accuracy metrics are inapplicable. We utilized the following evaluation methodologies:

### 1. Silhouette Score Analysis
While K-Means relies heavily on spherical cluster assumptions, evaluating its cluster cohesiveness revealed that a simple 2-cluster separation effectively delineates normal operating conditions from extreme outlier occurrences. The algorithm achieves high silhouette scores when distinguishing massive pressure drops from steady-state operations.

### 2. Distance from Cluster Centroid
For anomaly detection using K-Means, we calculated the Euclidean distance from every data point to its assigned cluster centroid. The 95th percentile threshold was utilized to flag anomalous behavior, creating a robust baseline for alerting extreme deviations.

### 3. Model Comparison
- **K-Means Clustering:** Deployed in our Streamlit dashboard because it enables $O(1)$ real-time inference on new, incoming data points, successfully grouping common operations and flagging high-distance anomalies.
- **One-Class SVM:** Trained effectively on subset data ($N=10,000$) to learn boundaries of typical operations, however, inference time scales poorly to production for 1Hz sampling at 15M records without aggressive batching or subsampling.
- **DBSCAN:** Useful for identifying noise points without predefined clusters, but suffers extensively in $O(N^2)$ memory allocation on large datasets.

## Verification & Outcomes
The Streamlit application successfully plots live pressure gaps and temperature variations alongside historical subsets, clearly highlighting out-of-bounds anomaly points. Future work entails collecting fully labeled failure windows to transition this into a semi-supervised architecture.
