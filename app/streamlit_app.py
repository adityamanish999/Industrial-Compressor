import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
from sklearn.decomposition import PCA

st.set_page_config(page_title="Industrial Compressor Anomaly Detection", layout="wide", page_icon="⚙️")

st.title("⚙️ Industrial Compressor Anomaly Detection")
st.markdown("Monitor real-time sensor data and detect operational anomalies.")

@st.cache_data
def load_data():
    filepath = 'cleaned_data/compressor_cleaned.csv'
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    return df

@st.cache_resource
def load_model():
    model_path = 'models/kmeans.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_results():
    path = 'reports/evaluation_results.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

df = load_data()
model = load_model()
results = load_results()

if df is None:
    st.warning("Processed data not found. Please run the data pipeline (download, preprocess).")
else:
    if model is None:
        st.warning("Model not found. Please run the training script.")
    else:
        st.sidebar.header("Settings")
        num_rows = st.sidebar.slider("Number of data points to visualize", min_value=100, max_value=5000, value=1000, step=100)
        st.sidebar.caption("Simulated live monitoring: replays the latest N one-minute readings from the dataset.")
        
        # Take a subset for visualization
        df_viz = df.tail(num_rows).copy()
        
        tab1, tab2 = st.tabs(["Real-Time Monitoring", "Model Comparison"])
        
        with tab1:
            st.header("Real-Time Anomaly Monitoring")
            # Predict anomalies (For K-Means, we can find distances to centroids or use the smaller cluster as anomalies)
            # Load scaler
            scaler_path = 'models/scaler.pkl'
            if not os.path.exists(scaler_path):
                st.error("Scaler not found!")
                st.stop()
            scaler = joblib.load(scaler_path)
            
            # Scale visualization data
            X_viz_scaled = scaler.transform(df_viz.values)
            
            # Predict clusters
            clusters = model.predict(X_viz_scaled)
            
            # Calculate distance to centroid
            distances = np.min(model.transform(X_viz_scaled), axis=1)
            # Fixed threshold learned on the training data (95th percentile of training distances)
            threshold_path = 'models/threshold.pkl'
            if not os.path.exists(threshold_path):
                st.error("Threshold not found! Run src/anomaly_detection.py.")
                st.stop()
            threshold = joblib.load(threshold_path)
            
            df_viz['Anomaly'] = distances > threshold
            df_viz['Distance_to_centroid'] = distances
            anomaly_rate = df_viz['Anomaly'].mean()
            
            # Compressor health status (training baseline flags ~5% of rows)
            if anomaly_rate < 0.10:
                st.success(f"Compressor health: NORMAL ({anomaly_rate:.1%} of recent readings anomalous)")
            elif anomaly_rate < 0.25:
                st.warning(f"Compressor health: WARNING ({anomaly_rate:.1%} of recent readings anomalous)")
            else:
                st.error(f"Compressor health: ALERT ({anomaly_rate:.1%} of recent readings anomalous)")
            st.caption("Normal: under 10% anomalous | Warning: 10-25% | Alert: above 25%. "
                       "Normal operation flags about 5% of readings.")
            
            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records Visualized", len(df_viz))
            col2.metric("Anomalies Detected", df_viz['Anomaly'].sum())
            col3.metric("Avg Pressure (TP3)", f"{df_viz['TP3'].mean():.2f}")
            col4.metric("Avg Temp", f"{df_viz['Oil_temperature'].mean():.2f}")
            
            # Visualizations
            st.subheader("Pressure Analysis (TP3)")
            fig_pressure = px.line(df_viz.reset_index(), x='timestamp', y='TP3', title="Pressure over Time")
            # Add anomaly markers
            anomalies = df_viz[df_viz['Anomaly']]
            fig_pressure.add_trace(go.Scatter(x=anomalies.index, y=anomalies['TP3'], mode='markers', marker=dict(color='red', size=8), name='Anomaly'))
            st.plotly_chart(fig_pressure, width='stretch')
            st.caption("Red dots are readings flagged as anomalous.")

            st.subheader("Oil Temperature Analysis")
            fig_temp = px.line(df_viz.reset_index(), x='timestamp', y='Oil_temperature', title="Oil Temperature over Time")
            fig_temp.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Oil_temperature'], mode='markers', marker=dict(color='red', size=8), name='Anomaly'))
            st.plotly_chart(fig_temp, width='stretch')
            st.caption("Red dots are readings flagged as anomalous.")

            st.subheader("Motor Current Analysis")
            fig_motor = px.line(df_viz.reset_index(), x='timestamp', y='Motor_current', title="Motor Current over Time")
            fig_motor.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Motor_current'], mode='markers', marker=dict(color='red', size=8), name='Anomaly'))
            st.plotly_chart(fig_motor, width='stretch')
            st.caption("Red dots are readings flagged as anomalous.")

            st.subheader("Anomaly Alerts")
            if anomalies.empty:
                st.success("No anomalies in this window.")
            else:
                st.dataframe(anomalies[['TP3', 'Oil_temperature', 'Motor_current', 'Distance_to_centroid']]
                             .sort_index(ascending=False))
            
            st.subheader("Cluster Visualization (PCA, 2 components)")
            pcs = PCA(n_components=2).fit_transform(X_viz_scaled)
            fig_pca = px.scatter(x=pcs[:, 0], y=pcs[:, 1], color=np.where(df_viz['Anomaly'], 'Anomaly', 'Normal'),
                                 color_discrete_map={'Normal': 'steelblue', 'Anomaly': 'red'},
                                 labels={'x': 'PC1', 'y': 'PC2', 'color': ''})
            st.plotly_chart(fig_pca, width='stretch')

            st.subheader("Recent Data Records")
            st.dataframe(df_viz.tail(10))

        with tab2:
            st.header("Model Comparison & Evaluation")
            st.markdown("Comparing the anomaly detection algorithms trained on the dataset.")
            
            # X_viz_scaled is already computed in tab1 and doesn't have the 'Anomaly' column appended
            
            col_km, col_ocsvm, col_dbscan = st.columns(3)
            
            # K-Means Evaluation
            with col_km:
                st.subheader("K-Means")
                km_model = joblib.load('models/kmeans.pkl')
                distances = np.min(km_model.transform(X_viz_scaled), axis=1)
                km_anomalies = (distances > threshold).sum()
                st.metric("Detected Anomalies", km_anomalies)
                st.write("**Pros:** Fast, scales well to large data, easy real-time prediction.")
                st.write("**Cons:** Assumes spherical clusters, sensitive to outliers.")
                
            # One-Class SVM Evaluation
            with col_ocsvm:
                st.subheader("One-Class SVM")
                ocsvm_model = joblib.load('models/ocsvm.pkl')
                # OCSVM returns -1 for outliers and 1 for inliers
                ocsvm_preds = ocsvm_model.predict(X_viz_scaled)
                ocsvm_anomalies = (ocsvm_preds == -1).sum()
                st.metric("Detected Anomalies", ocsvm_anomalies)
                st.write("**Pros:** Highly effective for novel anomaly detection, finds complex boundaries.")
                st.write("**Cons:** Computationally expensive ($O(N^3)$), poor scalability for >100k rows.")
                
            # DBSCAN Evaluation
            with col_dbscan:
                st.subheader("DBSCAN")
                st.info("DBSCAN trained on 10k subsample. Real-time prediction not natively supported.")
                st.write("**Pros:** Density-based, ignores noise, doesn't need pre-defined clusters.")
                st.write("**Cons:** Memory intensive ($O(N^2)$), struggles with varying densities.")
                
            st.markdown("### Evaluation on Known Failure Windows (full dataset)")
            if results is None:
                st.info("Run src/anomaly_detection.py to generate evaluation results.")
            else:
                rows = {name: results[key] for name, key in
                        [('K-Means', 'kmeans'), ('One-Class SVM', 'ocsvm'), ('DBSCAN (10k subsample)', 'dbscan')]}
                st.dataframe(pd.DataFrame(rows).T)
                st.write(f"**Silhouette score (K-Means):** {results['silhouette_kmeans']}")
                st.write("Failure windows are the known air-leak periods of MetroPT-3. "
                         "K-Means was chosen for the dashboard because it scores new sensor readings instantly, "
                         "unlike DBSCAN (no predict) and One-Class SVM (slow at scale).")
