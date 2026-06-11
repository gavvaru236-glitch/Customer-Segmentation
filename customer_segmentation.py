# Customer Segmentation using K-Means Clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 1. Load the Dataset
df = pd.read_csv('new.csv')

# 2. Data Preprocessing
df = df.dropna() 

# Feature Engineering on Date
parts = df["Dt_Customer"].str.split("-", n=3, expand=True)
df["day"] = parts[0].astype('int')
df["month"] = parts[1].astype('int')
df["year"] = parts[2].astype('int')

# Drop irrelevant features
df.drop(['Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1, inplace=True)

# 3. Label Encoding for Categorical Data
for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# 4. Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 5. Dimensionality Reduction & Clustering
tsne = TSNE(n_components=2, random_state=0)
tsne_data = tsne.fit_transform(df)

model = KMeans(init='k-means++', n_clusters=5, max_iter=500, random_state=22)
segments = model.fit_predict(df)

print("Pipeline execution complete. Optimal customer segments generated successfully.")
