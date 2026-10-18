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
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../data/MetroPT3(AirCompressor).csv', nrows=100000) # Load subset for EDA speed
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')])
df = df.drop_duplicates()

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

text5 = """\
## Sensor Distributions
"""

code5 = """\
cols = ['TP2', 'TP3', 'H1', 'Reservoirs', 'Oil_temperature', 'Motor_current']
df[cols].hist(bins=50, figsize=(15, 8))
plt.suptitle('Sensor Distributions')
plt.show()
"""

text6 = """\
## Motor Current and Operating States
"""

code6 = """\
plt.figure(figsize=(15, 4))
plt.plot(df.index, df['Motor_current'], color='green')
plt.title('Motor Current over Time')
plt.show()

df['motor_state'] = pd.cut(df['Motor_current'], bins=[-1, 1, 5, 100], labels=['Off', 'Offloaded', 'Loaded'])
df['motor_state'].value_counts().plot(kind='bar', title='Motor Operating States')
plt.show()
"""

text7 = """\
## Hourly Pattern and Known Failure Windows
"""

code7 = """\
df.groupby(df.index.hour)['Oil_temperature'].mean().plot(figsize=(10, 4), title='Mean Oil Temperature by Hour of Day')
plt.show()

# Known air-leak failure windows (EDA reads only the first 100k rows, so most fall outside this sample;
# load the full file or raise nrows to see them)
windows = [('2020-04-18 00:00', '2020-04-18 23:59'), ('2020-05-29 23:30', '2020-05-30 06:00'),
           ('2020-06-05 10:00', '2020-06-07 14:30'), ('2020-07-15 14:30', '2020-07-15 19:00')]
ax = df['TP3'].resample('1min').mean().plot(figsize=(15, 4), title='TP3 with failure windows')
for start, end in windows:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color='red', alpha=0.3)
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
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(text5),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_markdown_cell(text6),
    nbf.v4.new_code_cell(code6),
    nbf.v4.new_markdown_cell(text7),
    nbf.v4.new_code_cell(code7)
]

with open('notebooks/EDA_Industrial_Anomaly.ipynb', 'w') as f:
    nbf.write(nb, f)
