import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\MAYUR\OneDrive\Desktop\data_mining sem 5\pimadata.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

# -----------------------------
# Features and Target
# -----------------------------
if 'type' not in df.columns:
    raise KeyError("Column 'type' not found in dataset. Please check your CSV file.")

x = df.drop('type', axis=1)
y = df['type']

# Encode target if not numeric
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("\nTarget Encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# Fill missing numeric values
x = x.apply(lambda col: col.fillna(col.median()) if np.issubdtype(col.dtype, np.number) else col)

# Encode categorical features if any
x = pd.get_dummies(x, drop_first=True)

# Standardize numeric features
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# -----------------------------
# Train-Test Split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=1
)

# -----------------------------
# Logistic Regression (sklearn)
# -----------------------------
logr = LogisticRegression(max_iter=200)
logr.fit(x_train, y_train)

y_pred = logr.predict(x_test)
y_probs = logr.predict_proba(x_test)[:, 1]

# -----------------------------
# Metrics
# -----------------------------
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# -----------------------------
# Logistic Regression Summary (statsmodels)
# -----------------------------
x_train_sm = sm.add_constant(x_train)
logit = sm.Logit(y_train, x_train_sm).fit(disp=False)
print("\n--- Statsmodels Summary ---")
print(logit.summary2())

# -----------------------------
# Precision, Recall, F1 & Threshold
# -----------------------------
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Avoid divide-by-zero warnings
f1 = np.where((precision + recall) != 0, 2 * precision * recall / (precision + recall), 0)
best_idx = np.argmax(f1[:-1])  # last precision, recall don't have threshold
best_threshold = thresholds[best_idx]
print(f"\nBest Threshold = {best_threshold:.4f} with F1 = {f1[best_idx]:.4f}")

# -----------------------------
# Precision-Recall Curve
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Precision / Recall')
plt.legend()
plt.title('Precision & Recall vs Threshold')
plt.grid(True)
plt.show()

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
