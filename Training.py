import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib
import os

# Datset Load 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR,"Dataset","store_customers.csv")
df = pd.read_csv(csv_path)

df = pd.DataFrame(df)

# Data Cleaning
df = df.drop("CustomerID", axis = 1)

df["Gender"]=df["Gender"].fillna(df["Gender"].mode()[0])

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Annual Income (k$)"] = df["Annual Income (k$)"].fillna(df["Annual Income (k$)"].median())

df["Spending Score (1-100)"] = df["Spending Score (1-100)"].fillna(df["Spending Score (1-100)"].median())

# Feature Selection
x = df.copy()

# Encoding
le = LabelEncoder()
x["Gender"] = le.fit_transform(x["Gender"])
#
# print(x.head())

# Scaling
scaler = StandardScaler()
scaled = scaler.fit_transform(x)

kmeans = KMeans(n_clusters = 4, random_state = 42)
kmeans.fit(scaled)

df["Clusters"] = kmeans.labels_

# Scatter Plot
plt.figure(figsize=(8,6))

plt.scatter(df["Annual Income (k$)"],
            df["Spending Score (1-100)"],
            c = df["Clusters"],
            cmap = "viridis")

plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spendig Score")


model_path = os.path.join(BASE_DIR, "models/model.pkl")
scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")
feature_path = os.path.join(BASE_DIR, "models/feature.pkl")
encoder_path = os.path.join(BASE_DIR, "models/encoder.pkl")

joblib.dump(kmeans,model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(x.columns, feature_path)

print(le.classes_)
joblib.dump(le,encoder_path)