# -*- coding: utf-8 -*-
"""
Bagging Classifier - Compact (Stepwise)
"""

# Step 1: Imports & Load Data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.ensemble import BaggingClassifier

customer = pd.read_csv(r"C:\Users\MAYUR\OneDrive\Desktop\data_mining sem 5\customer_churn (1).csv")

# Step 2: Features and Target
X = customer.drop("Churn", axis=1)
y = customer["Churn"]

# Step 3: Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Step 4: Base Pipeline (Scaler + Decision Tree)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred_tree = pipeline.predict(X_test)
print("=== Decision Tree ===")
print(classification_report(y_test, y_pred_tree))
ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# Step 5: Bagging Classifier
bagging_classifier = BaggingClassifier(estimator=pipeline, n_estimators=50, random_state=42)
bagging_classifier.fit(X_train, y_train)
y_pred_bag = bagging_classifier.predict(X_test)
print("=== Bagging Classifier ===")
print(classification_report(y_test, y_pred_bag))
ConfusionMatrixDisplay.from_estimator(bagging_classifier, X_test, y_test)
plt.title("Confusion Matrix - Bagging Classifier")
plt.show()

# Step 6: Compare Metrics & Plots
acc_tree = accuracy_score(y_test, y_pred_tree)
acc_bag = accuracy_score(y_test, y_pred_bag)
f1_tree = f1_score(y_test, y_pred_tree, average='weighted')
f1_bag = f1_score(y_test, y_pred_bag, average='weighted')

plt.figure(figsize=(6,4))
plt.bar(['Decision Tree', 'Bagging'], [acc_tree, acc_bag])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(6,4))
plt.bar(['Decision Tree', 'Bagging'], [f1_tree, f1_bag])
plt.title("Model F1-Score Comparison (weighted)")
plt.ylabel("F1 Score")
plt.show()

print(f"Accuracy (Decision Tree): {acc_tree:.3f}")
print(f"Accuracy (Bagging): {acc_bag:.3f}")
print(f"F1 Score (Decision Tree): {f1_tree:.3f}")
print(f"F1 Score (Bagging): {f1_bag:.3f}")
