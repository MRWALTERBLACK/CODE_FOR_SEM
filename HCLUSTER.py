# Hierarchical Clustering - Loan Dataset (Clean Version)

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree

# Step 2: Load Dataset
loan_data = pd.read_csv(r"C:\Users\MAYUR\OneDrive\Desktop\data_mining sem 5\loan.csv")  
print("First 5 rows of the dataset:")
print(loan_data.head())

# Step 3: Check Missing Values
percent_missing = round(100 * (loan_data.isnull().sum()) / len(loan_data), 2)
print("\nPercentage of missing values in each column:")
print(percent_missing)

# Step 4: Drop Unwanted Columns
drop_cols = [col for col in ['purpose', 'not.fully.paid'] if col in loan_data.columns]
if drop_cols:
    loan_data = loan_data.drop(drop_cols, axis=1)

# Step 5: Select Numeric Columns
numeric_data = loan_data.select_dtypes(include=[np.number])
print("\nNumeric columns used for clustering:")
print(numeric_data.columns)

# Step 6: Boxplot Function
def show_boxplot(df, title="Outliers Distribution"):
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, orient="v")
    plt.title(title, fontsize=16)
    plt.ylabel("Range", fontweight='bold')
    plt.xlabel("Attributes", fontweight='bold')
    plt.show()

show_boxplot(numeric_data, "Before Outlier Removal")

# Step 7: Remove Outliers using IQR
def remove_outliers(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        Q1 = df_clean[col].quantile(0.05)
        Q3 = df_clean[col].quantile(0.95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

numeric_data = remove_outliers(numeric_data)
print(f"\nShape after removing outliers: {numeric_data.shape}")
show_boxplot(numeric_data, "After Outlier Removal")

# Step 8: Normalize Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Step 9: Apply Hierarchical Clustering
linkage_methods = {
    "Single Linkage": linkage(scaled_data, method="single", metric="euclidean"),
    "Average Linkage": linkage(scaled_data, method="average", metric="euclidean"),
    "Complete Linkage": linkage(scaled_data, method="complete", metric="euclidean"),
    "Ward Linkage": linkage(scaled_data, method="ward", metric="euclidean"),
    "Centroid Linkage": linkage(scaled_data, method="centroid", metric="euclidean"),
}

for name, linkage_matrix in linkage_methods.items():
    plt.figure(figsize=(10, 5))
    plt.title(f"Dendrogram - {name}")
    dendrogram(linkage_matrix)
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()

# Step 10: Cut Tree
cluster_labels = cut_tree(linkage_methods["Average Linkage"], n_clusters=2).reshape(-1, )
numeric_data["Cluster"] = cluster_labels

# Step 11: Analyze Clusters
if 'ApplicantIncome' in numeric_data.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Cluster', y='ApplicantIncome', data=numeric_data)
    plt.title("Applicant Income Distribution by Cluster")
    plt.show()
