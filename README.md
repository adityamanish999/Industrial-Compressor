# Industrial Compressor Anomaly Detection

## Business Use Cases
Industrial compressors are critical components in railway air production systems. Failures such as air leaks, abnormal pressure drops, overheating, and motor issues can lead to operational downtime, maintenance costs, and safety concerns. This project provides an automated, sensor-based Unsupervised Machine Learning monitoring system that can:
- Detect abnormal compressor behaviour before critical failures occur.
- Reduce unexpected equipment shutdowns through early anomaly detection.
- Prevent operational failures that may affect railway safety systems.
- Reduce maintenance costs and prevent expensive breakdowns.

## Dataset
The MetroPT-3 dataset contains multivariate time-series sensor readings collected from the Air Production Unit (APU) compressor system of a metro train, sampled at 1Hz frequency. It contains ~15.1 million records.

## Project Structure
- `data/` : Raw datasets (downloaded automatically)
- `cleaned_data/` : Preprocessed and aggregated data
- `notebooks/` : Exploratory Data Analysis
- `models/` : Saved Machine Learning models & Scaler
- `src/` : Core data pipeline (Download, Feature Engineering, Anomaly Detection)
- `app/` : Streamlit real-time monitoring dashboard
- `reports/` : Deep dive documentation and evaluation metrics

## How to Run

1. **Install Requirements**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. **Execute Data Pipeline**
```bash
python src/download_data.py
python src/data_preprocessing.py
python src/anomaly_detection.py
```

3. **Start Dashboard**
```bash
streamlit run app/streamlit_app.py
```
