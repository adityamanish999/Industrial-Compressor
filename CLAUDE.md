# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Unsupervised anomaly detection on the MetroPT-3 metro-train air compressor dataset (~15.1M rows, 1Hz sensor readings). A script pipeline in `src/` produces a scaler and models; a Streamlit dashboard in `app/` consumes them. There are no tests and no linter configured.

## Commands

All scripts must be run from the repo root (paths like `data/`, `cleaned_data/`, `models/` are relative to the CWD, and none of them are committed — see `.gitignore`).

```bash
pip install -r requirements.txt
python src/data_preprocessing.py   # downloads the UCI zip into data/ if missing -> cleaned_data/compressor_cleaned.csv, models/scaler.pkl
python src/anomaly_detection.py    # -> models/*.pkl (+ threshold.pkl) and reports/evaluation_results.json
streamlit run app/streamlit_app.py
python create_notebook.py          # regenerates notebooks/EDA_Industrial_Anomaly.ipynb (needs nbformat)
```

Steps must run in that order; each raises `FileNotFoundError` if its inputs are missing.

## Architecture

Pipeline data flow: raw CSV -> forward-fill/dropna -> resample to 1-minute means -> `apply_feature_engineering` -> `StandardScaler` -> models.

- `src/feature_engineering.py` is shared by training and inference (`data_preprocessing.py` and `prediction.py` both import it). It adds `pressure_gap`, rolling mean/std of `Oil_temperature` and `Motor_current`, a `motor_load_state` flag and a 30-sample TP3 moving average, then drops NaN rows. Changing the feature set changes the scaler's expected column count/order, so the scaler and models must be regenerated.
- `src/data_preprocessing.py` also holds `download_data()` (runs first if the raw CSV is missing). It saves the **unscaled** engineered data to `cleaned_data/compressor_cleaned.csv` and the fitted scaler separately. Downstream consumers (`anomaly_detection.py`, the dashboard) reload the CSV and apply `models/scaler.pkl` themselves. It also drops the `Unnamed` index column and duplicate rows.
- `src/anomaly_detection.py` trains K-Means (k=2, full data), One-Class SVM (nu=0.05) and DBSCAN (both on a 10,000-row random subsample). DBSCAN has no `predict`, so its saved model is only useful for inspecting the subsample.
- `src/prediction.py` is the inference entry point: applies feature engineering, scaling, and K-Means `predict`. Only K-Means is used for serving.
- `app/streamlit_app.py` loads the cleaned CSV, scaler and `kmeans.pkl`, and shows two tabs (real-time monitoring, model comparison) over the last N rows. Anomalies are defined as distance-to-centroid above the fixed `models/threshold.pkl` (95th percentile of training distances) (see `reports/project_documentation.md` for the rationale behind choosing K-Means for serving).
- `src/anomaly_detection.py` also holds `evaluate()` (called at the end of `train_models`), which computes silhouette score, precision/recall against the hard-coded `FAILURE_WINDOWS`, and timings; the dashboard's Model Comparison tab reads `reports/evaluation_results.json`.
- `create_notebook.py` programmatically builds the EDA notebook with `nbformat`; edit the script rather than the `.ipynb` when changing it.
