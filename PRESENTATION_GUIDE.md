# Presentation guide: Industrial Compressor Anomaly Detection

## Context
Your friend must present this project to a professor. This is a read-only explanation, so no code changes are planned. It is based on every file in `Industrial-Compressor/`. Approving it just means delivering this guide.

---

## 1. Problem statement (say this in 30 seconds)
- Metro trains make compressed air in an Air Production Unit (APU) compressor. Air leaks, pressure drops, overheating and motor problems cause downtime and safety risk.
- Today maintenance is reactive (fix after it breaks) or scheduled (service even when not needed).
- The MetroPT-3 dataset has 15.1M sensor readings at 1Hz (Feb–Aug 2020), 15 signals, almost no failure labels.
- **Goal:** learn what "normal" looks like and flag anything unusual using **unsupervised ML**, then show it in a Streamlit dashboard.
- Why unsupervised: there are no labels for every state, so we can't train a classifier. The model learns the shape of normal data.

## 2. Tech stack and why
| Tool | Used for | Why |
|---|---|---|
| Python | everything | standard for data science |
| pandas | load CSV, datetime index, ffill, resample, rolling | time-series handling |
| NumPy | percentile, argmin, sampling, arrays | fast math |
| scikit-learn | StandardScaler, KMeans, OneClassSVM, DBSCAN | the three required algorithms |
| joblib | save/load `.pkl` scaler and models | train once, reuse in the app |
| matplotlib / seaborn | EDA plots, correlation heatmap (notebook) | static analysis |
| Plotly | interactive line charts in the dashboard | zoom and hover, red anomaly markers |
| Streamlit | web dashboard | Python-only UI, no frontend code |
| requests + zipfile | download and unzip data from UCI | reproducible data step |
| nbformat | `create_notebook.py` builds the EDA notebook | notebook generated from code |
| Git/GitHub | version control, deliverable | required by the assignment |

**Be honest about the following:**
- **SQL is not used anywhere.** There is no database. If asked, say data is stored as CSV and `.pkl` files.
- **TensorFlow is in `requirements.txt` but no code uses it.** It was listed for possible autoencoder or reconstruction-error work, which is future work.
- `ucimlrepo` is also listed but unused. Download goes through `requests`.

## 3. File-by-file
```
data/                       raw CSV (git-ignored, created by download_data.py)
cleaned_data/               compressor_cleaned.csv (git-ignored)
models/                     scaler.pkl, kmeans.pkl, ocsvm.pkl, dbscan.pkl (git-ignored)
src/download_data.py        step 1
src/feature_engineering.py  helper used by steps 2 and by prediction
src/data_preprocessing.py   step 2
src/anomaly_detection.py    step 3 (training)
src/prediction.py           inference helper (not called by the dashboard)
app/streamlit_app.py        step 4 (the UI)
notebooks/EDA_...ipynb      exploratory analysis
create_notebook.py          script that generates that notebook
reports/project_documentation.md   evaluation and model-comparison write-up
README.md, requirements.txt, .gitignore, CLAUDE.md
```

## 4. Execution order (start → end)
**Start:** `python src/download_data.py`
1. **download_data.py** → `data/MetroPT3(AirCompressor).csv`
2. **data_preprocessing.py** → reads the CSV, cleans it, resamples to 1 minute, adds features, fits the scaler → `cleaned_data/compressor_cleaned.csv` + `models/scaler.pkl`
3. **anomaly_detection.py** → loads the cleaned CSV and scaler, trains the 3 models → `models/kmeans.pkl`, `ocsvm.pkl`, `dbscan.pkl`
4. **streamlit_app.py** (`streamlit run app/streamlit_app.py`) → loads CSV, scaler and K-Means, shows anomalies.

**End:** the dashboard in the browser. The EDA notebook is a side branch (it reads the raw CSV directly).

Data flow: raw CSV → ffill/dropna → 1-min mean → features → StandardScaler → models → dashboard.

## 5. Function-by-function
- **download_data.main()**: creates `data/`, skips if the file exists, downloads the UCI zip in 8KB chunks, extracts it, deletes the zip. It prints a message if the status code isn't 200.
- **feature_engineering.apply_feature_engineering(df)**: adds `pressure_gap = TP3 − Reservoirs` (leak indicator), `temp_rolling_mean` and `temp_rolling_std` (10-sample window on Oil_temperature), then drops NaN rows created by rolling.
- **data_preprocessing.load_and_preprocess(filepath, sample_frac)**:
  - reads the CSV and converts `timestamp` to datetime index;
  - forward-fills missing values and drops any remaining NaN;
  - resamples to 1-minute mean (about 1.5M raw rows → 242k rows after dedup);
  - calls feature engineering;
  - fits `StandardScaler` (mean 0, std 1) and returns `(df_resampled, df_scaled, scaler)`.
  - `sample_frac` is an unused parameter.
  - The `__main__` block saves the scaler and the **unscaled** cleaned CSV.
- **anomaly_detection.train_models()**:
  - loads the cleaned CSV and scaler, then scales;
  - **K-Means**: k=2, `random_state=42`, `n_init=10`, on all rows;
  - **One-Class SVM**: RBF kernel, `nu=0.05` (about 5% expected outliers), trained on a 10,000-row random sample because it is slow;
  - **DBSCAN**: `eps` chosen from the k-distance curve (90th percentile of 10th-neighbour distance, about 1.49), `min_samples=10`, same sample (label −1 = noise).
  - Saves each with joblib.
- **prediction.predict_anomalies(df_new)**: loads scaler and K-Means, applies feature engineering, scales, and returns `(clusters, X_scaled)`. This is the reusable inference path.
- **streamlit_app.py**:
  - `load_data()` (`@st.cache_data`) reads the cleaned CSV.
  - `load_model()` (`@st.cache_resource`) loads K-Means.
  - A sidebar slider picks the last 100–5000 rows.
  - **Tab 1, Real-Time Monitoring:** scale → `model.transform` gives distance to each centroid → take the min → flag the top 5% (95th percentile) as anomalies. It shows KPI cards, TP3 and Oil_temperature Plotly charts with red anomaly dots, and a recent-records table.
  - **Tab 2, Model Comparison:** anomaly counts for K-Means and OCSVM on the visible window, pros/cons text, and a DBSCAN note that it can't predict new data.
- **create_notebook.py**: builds the 8-cell EDA notebook (head, describe, correlation heatmap, pressure and temperature time series).

## 6. Concepts: "what is X and where is it here"
- **Scaling:** distance-based models (K-Means, SVM, DBSCAN) are dominated by large-valued columns unless features are standardized.
- **Resampling:** 1Hz → 1min cuts size and noise.
- **Rolling window:** captures local trend and volatility, so a temperature spike shows up in the rolling std.
- **K-Means anomaly logic:** the model only gives 2 clusters, so "anomaly" = far from the nearest centroid.
- **One-Class SVM:** learns a boundary around normal data; outside it is −1.
- **DBSCAN:** density-based; sparse points are noise. It is transductive, with no `predict()`.
- **Streamlit caching:** stops reloading the CSV and model on every interaction.
- **joblib/pickle:** persists trained objects, so no retraining in the app.

## 7. How to answer likely questions
- **Why unsupervised?** No labels except a few failure windows.
- **Why K-Means for the dashboard?** It is cheap to infer on new points. OCSVM is about O(N³) to train and slow to predict, and DBSCAN is O(N²) memory with no predict.
- **Why k=2?** Simple split into normal and extreme operation. Acknowledge it is a design choice. Elbow or silhouette tuning is future work.
- **Why 95th percentile?** An adjustable alert-rate assumption (about 5% flagged). The threshold is fixed from the training data (`models/threshold.pkl`), so it flags 5% of training data and more or fewer on new data; the dashboard banner turns Warning/Alert when the recent anomaly rate climbs.
- **How did you evaluate?** Visual inspection, distance-from-centroid, model comparison (cost, real-time fit). Silhouette (K-Means 0.49) and failure-window precision/recall are computed in `src/evaluation.py`.
- **Failure-window precision?** Implemented: K-Means precision 0.30, recall 0.73, hits 4/4 windows.
- **Does it handle duplicates?** Yes, 57,475 duplicate rows are removed.
- **Is it truly real-time?** It simulates monitoring on the tail of historical data. A live feed (Kafka/Spark) is listed as future work.
- **Why only 3 engineered features?** Minimal set. Motor-load states and moving averages are natural extensions.

## 8. What was added after review (the final version)
- Duplicate removal, plus dropping the unnamed index column.
- Extra features: motor-current rolling mean/std, motor load state, 30-sample pressure moving average (22 features total).
- `src/evaluation.py`: silhouette score, precision/recall/false-positive rate against known failure windows, and inference time, saved to `reports/evaluation_results.json`.
- Fixed alert threshold saved at training time (`models/threshold.pkl`), so the dashboard no longer flags a fixed 5% of any window.
- Dashboard: Normal/Warning/Alert banner, PCA cluster plot, real results table.
- DBSCAN `eps` tuned from the k-distance curve. Notebook extended and executed.

**Measured results (242,475 one-minute rows):**
| Model | Anomaly rate | Precision | Recall | FPR | Inference |
|---|---|---|---|---|---|
| K-Means | 5.0% | 0.297 | 0.729 | 0.036 | 0.015 s |
| One-Class SVM | 5.2% | 0.073 | 0.186 | 0.049 | 2.9 s |
| DBSCAN (10k sample) | 5.9% | 0.053 | 0.147 | 0.057 | no predict |

Silhouette: K-Means 0.49, DBSCAN 0.26. K-Means wins on every measure, which is why it is deployed.

**Be honest about the following if asked:**
1. No SQL; data is CSV/pkl. No TensorFlow model (an autoencoder is future work, and the brief names only the 3 algorithms).
2. Failure windows were entered from the MetroPT-3 documentation and approximate; precision is low for all models because failures are a tiny share of the data.
3. "Real-time" is simulated on the latest rows of historical data; streaming (Kafka/Spark) is future work.
4. `motor_load_state` uses a 1A cutoff chosen by us.
5. The dataset's actual sampling is every 10 seconds, so the 1-minute resample is a ~6x reduction here.
6. The sample folder in the brief lists `isolation_forest.pkl`, but we use the three algorithms the problem statement names.

## 9. Suggested 5-minute talk flow
Problem → data → pipeline diagram (section 4) → features → 3 models and why K-Means is deployed → live dashboard demo → limitations and future work (labels, failure-window check, streaming, autoencoder).

## Verification
Run the 4 commands in section 4 and open the dashboard. The EDA notebook can be checked by running `python create_notebook.py` and opening the `.ipynb`.
