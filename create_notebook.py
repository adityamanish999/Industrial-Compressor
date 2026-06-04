import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown cell
text = """\
# Exploratory Data Analysis (EDA) - Industrial Anomaly Detection
This notebook explores the MetroPT-3 compressor dataset, performs statistical analysis, and visualizes trends and correlations.
"""

# Code cell
code1 = """\
import pandas as pd
import numpy as np
import matplotlib.subplots as plt
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../data/MetroPT3(AirCompressor).csv', nrows=100000) # Load subset for EDA speed
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
df.head()
"""

text2 = """\
## Summary Statistics
"""

code2 = """\
df.describe()
"""

text3 = """\
## Correlation Heatmap
"""

code3 = """\
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Sensor Data')
plt.show()
"""

text4 = """\
## Time-Series Trends
"""

code4 = """\
plt.figure(figsize=(15, 5))
plt.plot(df.index, df['TP3'], label='Pressure (TP3)')
plt.title('Pressure over Time')
plt.legend()
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(df.index, df['Oil_temperature'], label='Oil Temperature', color='orange')
plt.title('Oil Temperature over Time')
plt.legend()
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4)
]

with open('notebooks/EDA_Industrial_Anomaly.ipynb', 'w') as f:
    nbf.write(nb, f)
