# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score
)

# Step 2: Load the dataset using CSV
df = pd.read_csv(r"C:\Users\MAYUR\OneDrive\Desktop\data_mining sem 5\iris.csv")

print("First 5 rows of dataset:")
print(df.head())

# Step 3: Data cleaning
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)
print(df.head())

# Step 4: Identify target and features
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target_col = 'Species'
print("\nFeatures:", feature_cols)
print("Target column:", target_col)

# Step 5: Convert target into binary classification
df['binary_target'] = df['Species'].apply(lambda s: 0 if s == 'Iris-setosa' else 1)
print("\nBinary target value counts:")
print(df['binary_target'].value_counts())

# Step 6: Visualize class separability
sns.set(style="whitegrid")

# Pairplot
sns.pairplot(df, vars=feature_cols, hue='Species', corner=True)
plt.suptitle("Pairplot (Iris features) - color = species", y=1.02)
plt.show()
plt.close()

# 2D scatter
plt.figure(figsize=(7, 5))
plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'],
            c=df['binary_target'], cmap='viridis', edgecolor='k')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal length vs petal width (binary target)')
plt.grid(True)
plt.show()
plt.close()

# 3D scatter
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['SepalLengthCm'], df['PetalLengthCm'], df['PetalWidthCm'],
           c=df['binary_target'], depthshade=True)
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Petal Length (cm)')
ax.set_zlabel('Petal Width (cm)')
ax.set_title('3D view (sepal length, petal length, petal width)')
plt.show()
plt.close()

print("\nComment on separability:")
print(" - Petal length and petal width provide the best separation.")
print(" - Sepal dimensions overlap for versicolor & virginica.")

# Step 7: Linear SVM (2 features)
X_2f = df[['PetalLengthCm', 'PetalWidthCm']].values
y_bin = df['binary_target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_2f, y_bin, test_size=0.25, random_state=42, stratify=y_bin
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
linear_svm.fit(X_train_scaled, y_train)

y_pred_lin = linear_svm.predict(X_test_scaled)
print("\nLinear SVM Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_lin))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lin))
print("Classification Report:\n", classification_report(y_test, y_pred_lin, digits=4))

# Function for decision boundary
def plot_decision_boundary(clf, X_s, y_s, title):
    x_min, x_max = X_s[:, 0].min() - 1, X_s[:, 0].max() + 1
    y_min, y_max = X_s[:, 1].min() - 1, X_s[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.2)
    plt.scatter(X_s[:, 0], X_s[:, 1], c=y_s, edgecolors='k')
    plt.xlabel('Petal length (scaled)')
    plt.ylabel('Petal width (scaled)')
    plt.title(title)
    plt.show()
    plt.close()

X_all_scaled = scaler.transform(X_2f)
plot_decision_boundary(linear_svm, X_all_scaled, y_bin,
                       "Linear SVM decision boundary (petal length & petal width)")

# Step 8: Non-linear SVM (RBF)
rbf_svm = SVC(kernel='rbf', gamma='scale', C=1.0, probability=True, random_state=42)
rbf_svm.fit(X_train_scaled, y_train)
y_pred_rbf = rbf_svm.predict(X_test_scaled)

print("\nRBF SVM Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rbf))
print("Classification Report:\n", classification_report(y_test, y_pred_rbf, digits=4))

plot_decision_boundary(rbf_svm, X_all_scaled, y_bin,
                       "RBF SVM decision boundary (petal length & petal width)")

# Step 9: Summary
print("\nEvaluation summary:")
print(f" - Linear SVM accuracy: {accuracy_score(y_test, y_pred_lin):.4f}")
print(f" - RBF SVM accuracy:    {accuracy_score(y_test, y_pred_rbf):.4f}")

prec_lin = precision_score(y_test, y_pred_lin)
rec_lin = recall_score(y_test, y_pred_lin)
prec_rbf = precision_score(y_test, y_pred_rbf)
rec_rbf = recall_score(y_test, y_pred_rbf)

print("\nPrecision / Recall:")
print(f" - Linear SVM: precision={prec_lin:.4f}, recall={rec_lin:.4f}")
print(f" - RBF SVM:    precision={prec_rbf:.4f}, recall={rec_rbf:.4f}")
